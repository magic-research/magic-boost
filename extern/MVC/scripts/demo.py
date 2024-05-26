import os
import sys
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch

from extern.MVC.mvc.camera_utils import get_camera, prepare_relative_embedding
from extern.MVC.mvc.ldm.util import (
    instantiate_from_config, 
    set_seed, 
    add_random_background
)
from extern.MVC.mvc.ldm.models.diffusion.ddim import DDIMSampler
from extern.MVC.mvc.model_zoo import build_model
from torchvision import transforms as T



def i2i(
    model,
    image_size,
    prompt,
    uc,
    sampler,
    ip=None,
    ref=None,
    step=20,
    scale=5.0,
    batch_size=8,
    ddim_eta=0.0,
    dtype=torch.float32,
    device="cuda",
    camera=None,
    num_frames=4,
    transform=None
):
    """ The function supports additional image prompt.
    Args:
        model (_type_): the image dream model
        image_size (_type_): size of diffusion output
        prompt (_type_): text prompt for the image
        uc (_type_): _description_
        sampler (_type_): _description_
        ip (Image, optional): the image prompt. Defaults to None.
        step (int, optional): _description_. Defaults to 20.
        scale (float, optional): _description_. Defaults to 7.5.
        batch_size (int, optional): _description_. Defaults to 8.
        ddim_eta (float, optional): _description_. Defaults to 0.0.
        dtype (_type_, optional): _description_. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to "cuda".
        camera (_type_, optional): _description_. Defaults to None.
        num_frames (int, optional): _description_. Defaults to 4
        pixel_control: whether to use pixel conditioning. Defaults to False.
    """
    if type(prompt) != list:
        prompt = [prompt]
        
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}
        
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            # c_["num_frames"] = uc_["num_frames"] = num_frames

        if ip is not None:
            ips = []
            for ip_ in ip:
                ip_embed = model.get_learned_image_conditioning(ip_).to(device)
                ips.append(ip_embed)

            ips = ips[0:1] + ips

            ip_ = torch.cat(ips, dim=0)
            c_["ip"] = ip_
            uc_["ip"] = torch.zeros_like(ip_)

        ip_imgs = []
        for ri in ref:
            ipt = transform(ri).to(device)
            ip_img = model.get_first_stage_encoding(
                model.encode_first_stage(ipt[None, :, :, :])
            )
            ip_imgs.append(ip_img)
        ip_imgs = torch.cat(ip_imgs, dim=0)[None, :, :, :]

        c_["ext_ip_img"] = ip_imgs
        uc_["ext_ip_img"] = torch.zeros_like(ip_imgs)

        c_["noise_embed"] = torch.zeros((batch_size, 2)).to(ip_imgs.device)
        uc_["noise_embed"] = torch.zeros((batch_size, 2)).to(ip_imgs.device)

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=ddim_eta,
            x_T=None,
            device=ip_imgs.device
        )
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))[:1]


class ImageDreamDiffusion():
    def __init__(self, args) -> None:
        # assert args.num_frames % 4 == 1 if args.mode == "pixel" else True
        
        set_seed(args.seed)
        dtype = torch.float16 if args.fp16 else torch.float32
        device = args.device
        batch_size = args.num_frames
        
        print("load image dream diffusion model ... ")
        model = build_model(args.model_name, 
                            config_path=args.config_path,
                            ckpt_path=args.ckpt_path)
        model.device = device
        model.to(device)
        model.eval()
        
        neg_texts = "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear."
        sampler = DDIMSampler(model)
        uc = model.get_learned_conditioning([neg_texts]).to(device)
        print("image dream model load done . ")

        # pre-compute camera matrices
        camera = get_camera(
            num_frames=1,
            elevation=args.target_elevation,
            offset=args.target_azimuth,
            extra_view=False
        )
        camera = camera.repeat(batch_size // args.num_frames, 1).to(device)
        domain_embedding = torch.zeros((camera.shape[0], 2)).to(device)
        domain_embedding[:, 0] = 1

        if "instantmesh" in args.config_path:
            can_cameras = get_camera(
                num_frames=6,
                elevation=[20.,-10.,20.,-10.,20.,-10.],
                offset=[30., 90., 150., 210., 270.,330.],
                extra_view=False
            )
        elif "instant3d" in args.config_path:
            can_cameras = get_camera(
                num_frames=4,
                elevation=[0.,0.,0.,0.],
                offset=[0.,90.,180.,270.],
                extra_view=False
            )

        can_cameras = can_cameras.repeat(batch_size // args.num_frames, 1).to(device)

        can_domain_embedding = torch.zeros((can_cameras.shape[0], 2)).to(device)
        can_domain_embedding[:, 1] = 1

        camera = prepare_relative_embedding(1, camera, can_cameras, domain_embedding, can_domain_embedding, True, True)

        self.image_transform = T.Compose(
            [
                T.Resize((args.size, args.size)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        
        self.dtype = dtype 
        self.device = device
        self.batch_size = batch_size
        self.args = args
        self.model = model
        self.sampler = sampler
        self.uc = uc
        self.camera = camera

    def diffuse(self, t, ip, ref, n_test=3):
        images = []
        for _ in range(n_test):
            img = i2i(
                self.model,
                self.args.size,
                t,
                self.uc,
                self.sampler,
                ip=ip,
                ref=ref,
                step=50,
                scale=5,
                batch_size=self.batch_size,
                ddim_eta=0.0,
                dtype=self.dtype,
                device=self.device,
                camera=self.camera,
                num_frames=args.num_frames,
                transform=self.image_transform
            )
            img = np.concatenate(img, 1)
            images.append(img)
        return images
       
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="mvc",
        help="load pre-trained model from hugginface",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./extern/MVC/mvc/configs/mvc_instantmesh.yaml",
        help="load model from local config (override model_name)",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="./extern/MVC/checkpoints/magic-boost-4view_all.pt", help="path to local checkpoint"
    )
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--text", type=str, default="a ")
    parser.add_argument("--image_name", type=str, default="debug")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=7)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_elevation", type=float, default=10.)
    parser.add_argument("--target_azimuth", type=float, default=10.)

    args = parser.parse_args()
    
    t = args.text + args.suffix

    if "instantmesh" in args.config_path:
        selected_list = [0,1,2,3,4,5]
    elif "instant3d" in args.config_path:
        selected_list = [0,1,2,3]
    else:
        raise NotImplementedError

    supportimg_dir = os.path.basename(args.image_root)
    supportimg_list = []
    for i in selected_list:
        imgp = args.image_root + '/%d.png' % i
        imgp = Image.open(imgp)
        if np.array(imgp).shape[-1] > 3:
            imgpc = add_random_background(imgp, 255)
        else:
            imgpc = imgp
        supportimg_list.append(imgpc)

    ip = supportimg_list
    ref = supportimg_list

    image_dream = ImageDreamDiffusion(args)

    images = image_dream.diffuse(t, ip, ref, n_test=3)
    
    name = 'demo'
    images = np.concatenate(images, 0)
    Image.fromarray(images).save(f"debug_dream.png")

