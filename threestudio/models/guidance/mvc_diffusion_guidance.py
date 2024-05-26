from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from torchvision import transforms as T
from extern.MVC.mvc.camera_utils import add_margin, get_camera, create_camera_to_world_matrix, convert_opengl_to_blender, normalize_camera, prepare_relative_embedding, prepare_domain_embedding
from extern.MVC.mvc.model_zoo import build_model

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

import time
from extern.MVC.mvc.ldm.models.diffusion.ddim import DDIMSampler

@threestudio.register("mvc-diffusion-guidance")
class MvcdiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model_name: str = (
            "mvc-diffusion"
        )
        ckpt_path: Optional[
            str
        ] = None
        config_path: Optional[
            str
        ] = None
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 1
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5
        ip_mode: str = None
        mode: int = 3

        # whether or not use embedding version 2
        use_embedding_v2: bool = False

        # control label trained with noise augmentation
        cond_strength_tex: float = 1.

        # control label trained with grid distort
        cond_strength_geo: float = 0.

        # attn drop mode for update anchor loss
        du_mode: str = 'drop_src'

        anchor_elevation: float = 0.

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading MVC Diffusion ...")

        self.model = build_model(
            self.cfg.model_name,
            config_path=self.cfg.config_path,
            ckpt_path=self.cfg.ckpt_path)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        self.to(self.device)

        # set the multi-view conditions
        self.support_list = None

        # set the attention mode
        self.mode = self.cfg.mode
        self.model.model.diffusion_model.set_forced_mode_type(self.mode)
        self.sampler = DDIMSampler(self.model)

        threestudio.info(f"Loaded Multiview Diffusion!")

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
    ):
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32] Latent space image

    def decode_images(self,
                      imgs:  Float[Tensor, "B 4 32 32"]):
        latents = self.model.decode_first_stage(
            imgs
        )
        return latents

    def set_anchor_latents(self, support_list):
        image_transform = T.Compose(
            [
                T.Resize((self.cfg.image_size, self.cfg.image_size)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        ip_imgs = []
        for ri in support_list:
            ipt = image_transform(ri).to(self.device)
            ip_imgs.append(ipt)
        ip_imgs = torch.stack(ip_imgs, dim=0)

        ip_imgs = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(ip_imgs)
        )
        self.ip_imgs = ip_imgs
        ip = self.model.get_learned_image_conditioning(support_list[0:1])
        self.ip = ip

    def append_extra_view(self, latent_input, t_expand, anchor_num):
        bs, c, h, w = latent_input.shape
        real_batch_size = bs
        latent_input = latent_input.reshape(real_batch_size, -1, c, h, w)

        zero_tensor = torch.zeros(real_batch_size, anchor_num, c, h, w).to(latent_input)
        latent_input = torch.cat([latent_input, zero_tensor], dim=1)
        latent_input = latent_input.reshape(-1, c, h, w)

        # make time expand here
        t_expand = torch.cat([t_expand, t_expand[-1:].repeat(anchor_num * real_batch_size)])

        return latent_input, t_expand

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        anchor_rgb=None,
        anchor_mask=None,
        anchor_elev=None,
        anchor_azimuth=None,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        bg_color = kwargs.get("comp_rgb_bg")
        bg_color = bg_color.mean().detach().cpu().numpy() * 255

        # the multi-view condition is cached for faster inference
        if self.support_list is None:
            support_list = []

            if torch.sum(anchor_mask < 1) == 0:
                support_img = anchor_rgb
            else:
                support_img = (1 - anchor_mask[:, :, :, None].repeat(1, 1, 1, 3)) * bg_color / 255. + anchor_rgb * anchor_mask[:,:, :,None].repeat(1, 1, 1, 3)

            for img in support_img:
                img = img.detach().cpu().numpy() * 255
                img = Image.fromarray(img.astype(np.uint8))
                support_list.append(img)

            self.support_list = support_list
            self.set_anchor_latents(support_list)

        # set camera for target view
        camera = torch.zeros((batch_size, 3)).to(c2w.device)
        camera[:, 0] = elevation
        camera[:, 1] = azimuth
        camera[:, 2] = 0.

        # set camera for conditional view
        can_camera = []
        for (elev, azimuth) in zip(anchor_elev, anchor_azimuth):
            can_camera.append([elev, azimuth, 0.])
        can_camera = torch.FloatTensor(can_camera).to(c2w.device)[None]
        can_camera = can_camera.repeat(batch_size, 1, 1)

        # set domain embedding. Note this domain embedding has no influence on the performance of our model.
        anchor_num = can_camera.shape[1]
        domain_embedding = torch.zeros((batch_size, 2)).to(c2w.device)
        domain_embedding[:, 0] = 1
        can_domain_embedding = torch.zeros((batch_size * anchor_num, 2)).to(c2w.device)
        can_domain_embedding[:, 1] = 1

        # set camera embedding
        camera = prepare_relative_embedding(batch_size, camera, can_camera, domain_embedding, can_domain_embedding, True, self.cfg.use_embedding_v2)

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        # set ip image
        can_ip = self.ip.repeat(anchor_num*batch_size, 1, 1) #  * batch_size
        batch_ip = self.ip.repeat(batch_size, 1,1) # [self.ip[0] for ind in nearest_anchor]

        bs, h, w = batch_ip.shape
        can_ip = can_ip.reshape(-1, anchor_num, h, w)
        batch_ip = batch_ip.reshape(-1, 1, h, w)

        image_embeddings = torch.cat([batch_ip, can_ip], dim=1).reshape(-1, h, w)
        un_image_embeddings = \
            torch.zeros_like(image_embeddings).to(image_embeddings)

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = (
                    F.interpolate(
                        rgb_BCHW, (32, 32), mode="bilinear", align_corners=False
                    )
                    * 2
                    - 1
                )
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=latents.device,
            )
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = camera.repeat(2, 1).to(text_embeddings)

                # num_frames = self.cfg.n_view + 4
                context = {
                    "context": text_embeddings,
                    "camera": camera
                    # "num_frames": num_frames, # number of frames
                }
            else:
                context = {"context": text_embeddings}

            context["ip"] = torch.cat([image_embeddings, un_image_embeddings], dim=0).to(text_embeddings)

            for key in ["context"]:
                embedding = context[key]  # repeat for last dim features
                features = []
                for feature in embedding.chunk(embedding.shape[0]):
                    features.append(torch.cat([feature, feature.repeat(anchor_num,1,1)], dim=0))
                context[key] = torch.cat(features, dim=0)

            ip_imgs= self.ip_imgs.repeat(batch_size, 1, 1, 1, 1)
            context["ext_ip_img"] = torch.cat([
                ip_imgs,
                torch.zeros_like(ip_imgs)], dim=0)

            if self.cfg.use_embedding_v2:
                cond_rate = 1 - self.cfg.cond_strength_tex
                time_step = 500 * cond_rate
                noise_embedding = torch.zeros(batch_size,anchor_num+1,2).to(ip_imgs.device)
                # set control strength; lowest strength is set as 500 here, but could be higher......
                noise_embedding[:, 2:, 0] = time_step
                # set control strength; lowest strength is 0.5 here, but could be higher......
                noise_embedding[:, 2:, 1] = 0.5 * (1 - self.cfg.cond_strength_geo)

                noise_embedding = noise_embedding.reshape(-1, 2)
            else:
                cond_rate = 1 - self.cfg.cond_strength_tex
                time_step = 500 * cond_rate
                noise_embedding = torch.zeros(batch_size, anchor_num + 1, 2).to(ip_imgs.device)
                noise_embedding[:, 2:, 1] = time_step
                noise_embedding = noise_embedding.reshape(-1, 2)

            context["noise_embed"] = torch.cat([noise_embedding,
                                                    torch.zeros_like(noise_embedding)], dim=0)

            latent_model_input, t_expand = self.append_extra_view(latent_model_input, t_expand, anchor_num)

            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(
            2
        )  # Note: flipped compared to stable-dreamfusion

        _, c, h, w = noise_pred_text.shape
        def remove_extra_view(embedding):
            embedding = embedding.reshape(-1, (anchor_num + 1), c, h, w)
            embedding = embedding[:, :1, :, :, :].reshape(-1, c, h, w)
            return embedding

        noise_pred_text, noise_pred_uncond = \
            remove_extra_view(noise_pred_text), \
            remove_extra_view(noise_pred_uncond)

        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(
                latents_noisy, t, noise_pred
            )

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(
                    latents_noisy, t, noise_pred_text
                )
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(
                    -1, self.cfg.n_view, *latents_recon_nocfg.shape[1:]
                )
                latents_recon_reshape = latents_recon.view(
                    -1, self.cfg.n_view, *latents_recon.shape[1:]
                )
                factor = (
                    latents_recon_nocfg_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8
                ) / (latents_recon_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8)

                latents_recon_adjust = latents_recon.clone() * factor.squeeze(
                    1
                ).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = (
                    self.cfg.recon_std_rescale * latents_recon_adjust
                    + (1 - self.cfg.recon_std_rescale) * latents_recon
                )

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = (
                0.5
                * F.mse_loss(latents, latents_recon.detach(), reduction="sum")
                / latents.shape[0]
            )
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # Original SDS
            # w(t), sigma_t^2
            w = 1 - self.model.alphas_cumprod[t]
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def du_forward_fix(
            self,
            rgb_BCHW_512,
            prompt_utils,
            strength,
            anchor_rgb,
            anchor_mask,
            anchor_elev,
            anchor_azimuth,
            elevation: Float[Tensor, "B"],
            azimuth: Float[Tensor, "B"],
            camera_distances: Float[Tensor, "B"],
            view_dependent_prompting=False,
            inference_step=50,
            text_embeddings=None,
            input_is_latent=False,
            rgb_as_latents: bool = False,
            **kwargs,
    ):
        bs, H, W, _ = rgb_BCHW_512.shape

        # set the drop mode for doing anchor update
        anchor_num = anchor_rgb.shape[0]
        if self.cfg.du_mode == "drop_src":
            drop_mask = torch.zeros((bs,anchor_num-1)).to(rgb_BCHW_512.device)
            relative_azimuth = ((azimuth + 360) % 360) - anchor_azimuth[None, 1:]
            nearest_anchor = torch.topk(torch.abs(relative_azimuth), 1, dim=1, largest=False)[1][:, 0]
            drop_mask[torch.arange(bs), nearest_anchor] = 1
            drop_mask = torch.cat([drop_mask, drop_mask], dim=0)

            if hasattr(self.model.model.diffusion_model, "set_drop_mask"):
                self.model.model.diffusion_model.set_drop_mask(drop_mask)
        # elif self.cfg.du_mode == 'all':
        #     drop_mask = torch.ones((bs,anchor_num)).to(rgb_BCHW_512.device)
        #     if hasattr(self.model.model.diffusion_model, "set_drop_mask"):
        #         self.model.model.diffusion_model.set_drop_mask(drop_mask)

        if self.model.model.diffusion_model.anchor_infer_once:
            self.model.model.diffusion_model.set_forced_reset(True)

        # cache the multi-view input
        if self.support_list is None:
            bg_color = 255.
            support_list = []
            support_img = (1 - anchor_mask[:, :, :, None].repeat(1, 1, 1, 3)) * bg_color / 255. + anchor_rgb * anchor_mask[
                                                                                                               :, :, :,None].repeat(1, 1, 1, 3)
            for img in support_img:
                img = img.detach().cpu().numpy() * 255
                img = Image.fromarray(img.astype(np.uint8))
                support_list.append(img)

            self.support_list = support_list
            self.set_anchor_latents(support_list)

        # set target camera
        camera = torch.cat([elevation, azimuth, torch.zeros_like(camera_distances)], dim=1)
        domain_embedding = torch.zeros((bs, 1, 2)).to(rgb_BCHW_512.device)
        domain_embedding[:, :, 0] = 1

        # set condition cameras
        can_camera = []
        for (elev, azimuth) in zip(anchor_elev, anchor_azimuth):
            can_camera.append([elev, azimuth, 0.])
        can_camera = torch.FloatTensor(can_camera).to(rgb_BCHW_512.device)[None]
        can_domain_embedding = torch.zeros((bs, anchor_num, 2)).to(rgb_BCHW_512.device)
        can_domain_embedding[:, :, 1] = 1

        can_camera = can_camera.to(rgb_BCHW_512.device).repeat(4,1,1)
        batch_size = 4
        camera = prepare_relative_embedding(batch_size, camera, can_camera, domain_embedding, can_domain_embedding, True, self.cfg.use_embedding_v2)

        rgb_BCHW = rgb_BCHW_512
        # text embeddings could be ignore...
        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, False
            )

        # set ip images
        can_ip = self.ip.repeat(4*anchor_num, 1, 1) #  * batch_size
        batch_ip = self.ip.repeat(4, 1,1) # [self.ip[0] for ind in nearest_anchor]

        bs, h, w = batch_ip.shape
        can_ip = can_ip.reshape(-1, anchor_num, h, w)
        batch_ip = batch_ip.reshape(-1, 1, h, w)

        image_embeddings = torch.cat([batch_ip, can_ip], dim=1).reshape(-1, h, w)
        un_image_embeddings = \
            torch.zeros_like(image_embeddings).to(image_embeddings)

        if input_is_latent:
            latents = rgb_BCHW_512
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = (
                    F.interpolate(
                        rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
                    )
                    * 2
                    - 1
                )
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb.to(text_embeddings.dtype))

        camera = camera.to(text_embeddings)

        ip_imgs = self.ip_imgs.repeat(4, 1, 1, 1, 1)

        # num_frames = self.cfg.n_view + 4
        c = {
            "context": text_embeddings[:1].repeat(4*(1+anchor_num),1,1),
            "camera": camera,
            "ip": image_embeddings,
            "ext_ip_img": ip_imgs,
        }
        uc = {
            "context": text_embeddings[-1:].repeat(4*(1+anchor_num),1,1),
            "camera": camera,
            "ip": un_image_embeddings,
            "ext_ip_img": torch.zeros_like(ip_imgs),
        }

        if self.cfg.use_embedding_v2:
            cond_rate = 1 - self.cfg.cond_strength_tex
            time_step = 500 * cond_rate
            noise_embedding = torch.zeros(4, anchor_num+1, 2).to(ip_imgs.device)
            # set control strength; lowest strength is set as 500 here, but could be higher......
            noise_embedding[:, 2:, 0] = time_step
            # set control strength; lowest strength is 0.5 here, but could be higher......
            noise_embedding[:, 2:, 1] = 0.5 * (1 - self.cfg.cond_strength_geo)
            noise_embedding = noise_embedding.reshape(-1, 2)
        else:
            cond_rate = 1 - self.cfg.cond_strength_tex
            time_step = 500 * cond_rate
            noise_embedding = torch.zeros(4, anchor_num+1, 2).to(ip_imgs.device)
            noise_embedding[:, 2:, 1] = time_step
            noise_embedding = noise_embedding.reshape(-1, 2)
        c["noise_embed"] = noise_embedding
        uc["noise_embed"] = torch.zeros_like(noise_embedding)

        shape = [4, latents.shape[-2], latents.shape[-1]]

        latents = latents[:, None, :, :, :].repeat(1,1+anchor_num,1,1,1)
        latents = latents.reshape(-1, *latents.shape[2:])
        samples_ddim, _ = self.sampler.du_sample(
            S=inference_step,
            conditioning=c,
            batch_size=4*(anchor_num +1),
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=5.,
            unconditional_conditioning=uc,
            eta=0.0,
            x_T=latents,
            du_strength=strength,
        )

        if hasattr(self.model.model.diffusion_model, "set_drop_mask"):
            self.model.model.diffusion_model.set_drop_mask(None)

        x_sample = self.model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

        x_sample = x_sample[::(1+anchor_num)]

        return x_sample

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
