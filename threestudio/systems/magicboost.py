import os
from dataclasses import dataclass, field
import cv2
import torch
import shutil
import numpy as np
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.misc import C

from rembg import remove


def add_batch(batchs):
    for batch in batchs:
        for ob in batch.keys():
            obv = batch[ob]
            if isinstance(obv, torch.Tensor):
                batch[ob] = obv[None]

@threestudio.register("magicboost-system")
class MagicboostSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)

        # start step to update anchor
        update_anchor_start: int = 3000

        # end step to update anchor
        update_anchor_end: int = 100000

        # interval to update anchor
        update_anchor_interval: int = 1000

        # strength to update anchor
        update_anchor_strength: float = 0.75

        # resolution of anchor
        anchor_resolution: int = 256

        # recon loss mode : one for using only input view, four for using anchor loss
        view_recon_mode: str = "one"

        # whether or not use mask loss from anchor
        only_ft_mask_loss: bool = True

        # max step for using l1 loss from input view
        ref_max_step: int = 100000

        # start step to use anchor loss
        only_ft_loss_step_min: int = -100000

        # end step to use anchor loss
        only_ft_loss_step_max: int = 100000

    cfg: Config

    def configure(self):
        super().configure()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        # save anchor infomation
        self.anchor_rgb = None
        self.anchor_mask = None
        self.anchor_elev = None
        self.anchor_azimuth = None

        # save multi-view inputs
        self.du_input = None

        # save updated anchor information
        self.update_rgb = None
        self.update_mask = None

    def get_pseudo_anchor_imgs(self, anchor_views, prompt_utils, tv):
        out_rgb = []

        with torch.no_grad():
            # render anchor
            for anchor_v in anchor_views:
                anchor_v["only_albedo"] = True
                anchor_out_i = self(anchor_v)['comp_rgb']
                out_rgb.append(anchor_out_i.permute(0, 3, 1, 2))
                del anchor_v["only_albedo"]
            out_rgb = torch.cat(out_rgb, dim=0)
            out_rgb = torch.nn.functional.interpolate(out_rgb, size=self.cfg.anchor_resolution, mode='area')

            batch = {}
            cam_input = {}

            elevation = []
            azimuth = []
            camera_distance = []
            for item in anchor_views:
                elevation.append(item['elevation'])
                azimuth.append(item['azimuth'])
                camera_distance.append(item['camera_distances'])

            cam_input['elevation'] = torch.stack(elevation) # torch.zeros((1)).to(out_rgb.device)
            cam_input['azimuth'] = torch.stack(azimuth)
            cam_input['camera_distances'] = torch.stack(camera_distance)
            batch["random_camera"] = cam_input

            time = torch.FloatTensor([tv]).to(out_rgb.device)

            self.du_input = out_rgb

            # update anchor
            sdu_img = self.guidance.du_forward_fix(out_rgb, \
                                                   prompt_utils, time, anchor_rgb=self.anchor_rgb,
                anchor_mask=self.anchor_mask, anchor_elev=self.anchor_elev, anchor_azimuth=self.anchor_azimuth, **batch["random_camera"])

            # remove bg
            sdu_img = [remove(du.astype(np.uint8)) for du in sdu_img]
            rgb_list, mask_list = [], []
            for du in sdu_img:
                rgb = du[:, :, :3] / 255.
                mask = du[:, :, 3] / 255.
                rgb_list.append(torch.from_numpy(rgb).unsqueeze(0).contiguous().to(out_rgb.device))
                mask_list.append(torch.from_numpy(mask).unsqueeze(0).contiguous().to(out_rgb.device))
            rgb_list = torch.cat(rgb_list, dim=0)
            mask_list = torch.cat(mask_list, dim=0)

            self.update_rgb = list(rgb_list[:].chunk(len(sdu_img)))
            self.update_mask = list(mask_list[:, :, :, None].chunk(len(sdu_img)))

        return rgb_list, mask_list

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

        # visualize all training images
        all_images = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

        self.pearson = PearsonCorrCoef().to(self.device)

    def training_substep(self, batch, batch_idx, guidance: str, render_type="rgb"):
        if guidance == "guidance":
            batch = batch["random_camera"]

        if guidance == "ref":

            # set the prob for selecting different views
            if self.cfg.view_recon_mode == 'one' or self.true_global_step > self.cfg.only_ft_loss_step_max or self.true_global_step < self.cfg.only_ft_loss_step_min:
                probs = [1., 0, 0, 0, 0]
            else:
                probs = [0.25, 0., 0.25, 0.25, 0.25]

            probs = np.array(probs)
            x = np.arange(0, 5)
            rand_view = np.random.choice(a=x, size=1, replace=True, p=probs)
            print(f"rand_view : {rand_view}")

            # We use the first view of multi-view conditions as a l1 loss similar to zero123
            if self.update_rgb is None or self.update_mask is None:
                rand_view[:] = 0
                rgb = [batch['rgb']]
                mask = [batch['mask']]
            else:
                rgb = [batch['rgb']] + self.update_rgb
                mask = [batch['mask']] + self.update_mask

            gt_anchor_rgb = rgb[int(rand_view)]
            gt_anchor_mask = mask[int(rand_view)][..., 0].float()

            if int(rand_view) != 0:
                anchor_views = batch["anchor_views"]
                batch = anchor_views[int(rand_view) - 1]
            else:
                batch = batch

            # set the "only_albedo=True" to disable light
            batch["only_albedo"] = True

        out = self(batch)
        loss_prefix = f"loss_{guidance}_"

        if guidance == "ref":
            del batch["only_albedo"]

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        if guidance == "ref":
            render_size = out['comp_rgb'].shape[1]
            gt_anchor_mask = F.interpolate(gt_anchor_mask[:, None], (render_size, render_size), mode='nearest')[:, 0][..., None]
            gt_anchor_rgb = F.interpolate(gt_anchor_rgb.permute(0, 3, 1, 2), (render_size, render_size)).permute(0, 2, 3, 1)

            # rgb loss
            if self.C(self.cfg.loss.lambda_rgb) > 0:
                gt_anchor_rgb = gt_anchor_rgb * gt_anchor_mask.float() + out["comp_rgb_bg"] * (
                        1 - gt_anchor_mask.float()
                )
                pred_rgb = out["comp_rgb"]
                set_loss("rgb", F.mse_loss(gt_anchor_rgb.to(pred_rgb.dtype), pred_rgb))

            # mask loss
            mask_weight = 1.
            if self.cfg.only_ft_mask_loss and rand_view != 0:
                mask_weight = 0
            if self.C(self.cfg.loss.lambda_mask) > 0:
                set_loss("mask", F.mse_loss(gt_anchor_mask.float(), out["opacity"]) * mask_weight)

            # mask binary cross loss
            if self.C(self.cfg.loss.lambda_mask_binary) > 0:
                set_loss("mask_binary", F.binary_cross_entropy(
                out["opacity"].clamp(1.0e-5, 1.0 - 1.0e-5),
                batch["mask"].float(),) * mask_weight)

        elif guidance == "guidance" and self.true_global_step > self.cfg.freq.no_diff_steps:
            guidance_inp = out["comp_rgb"]

            # sds loss
            guidance_out = self.guidance(
                guidance_inp,
                self.prompt_utils,
                comp_rgb_bg=out["comp_rgb_bg"],
                anchor_rgb=self.anchor_rgb,
                anchor_mask=self.anchor_mask,
                anchor_elev=self.anchor_elev,
                anchor_azimuth=self.anchor_azimuth,
                **batch,
            )

            for name, value in guidance_out.items():
                if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                    self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                   set_loss(name.split("_")[-1], value)

            # Regularization
            if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
                if "comp_normal" not in out:
                    raise ValueError(
                        "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                    )
                normal = out["comp_normal"]
                set_loss(
                    "normal_smooth",
                    (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                    + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
                )

            if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for normal smooth loss, no normal is found in the output."
                    )
                if "normal_perturb" not in out:
                    raise ValueError(
                        "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                    )
                normals = out["normal"]
                normals_perturb = out["normal_perturb"]
                set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())

            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                set_loss(
                    "orient",
                    (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum()
                    / (out["opacity"] > 0).sum(),
                )

            if guidance != "ref" and self.C(self.cfg.loss.lambda_sparsity) > 0:
                set_loss("sparsity", (out["opacity"] ** 2 + 0.01).sqrt().mean())

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                set_loss(
                    "opaque", binary_cross_entropy(opacity_clamped, opacity_clamped)
                )

            if "lambda_z_variance"in self.cfg.loss and self.C(self.cfg.loss.lambda_z_variance) > 0:
                # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
                # helps reduce floaters and produce solid geometry
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                set_loss("z_variance", loss_z_variance)

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        if self.cfg.freq.ref_or_guidance == "accumulate":
            do_ref = True
            do_guidance = True
        elif self.cfg.freq.ref_or_guidance == "alternate":
            do_ref = (
                self.true_global_step > self.cfg.freq.guidance_only_steps
                and self.true_global_step % self.cfg.freq.n_ref == 0
                and not (self.true_global_step > self.cfg.ref_max_step)
            )

            do_guidance = not do_ref

        render_type = "rgb"
        total_loss = 0.0

        gt_anchor_mask_org = batch["anchor_mask"]
        gt_anchor_rgb_org = batch["anchor_rgb"]
        gt_anchor_elev = batch["anchor_elev"]
        gt_anchor_azimuth = batch["anchor_azimuth"]
        anchor_views = batch["anchor_views"]
        add_batch(anchor_views)

        # set multi-view condition info
        if self.anchor_rgb is None:
            self.anchor_rgb = gt_anchor_rgb_org
            self.anchor_mask = gt_anchor_mask_org / 255.
            self.anchor_elev = gt_anchor_elev
            self.anchor_azimuth = gt_anchor_azimuth

        if self.true_global_step <= self.cfg.update_anchor_end and self.true_global_step >= self.cfg.update_anchor_start and (self.true_global_step - self.cfg.update_anchor_start) % self.cfg.update_anchor_interval == 0:
            do_ref = True
            anchor_views = batch["anchor_views"]
            self.get_pseudo_anchor_imgs(anchor_views, self.prompt_utils, self.update_anchor_strength)
            self.guidance.support_list = None
            # force reset the KV cache
            if self.guidance.model.model.diffusion_model.anchor_infer_once:
                self.guidance.model.model.diffusion_model.set_forced_reset(True)

        if do_guidance:
            out = self.training_substep(batch, batch_idx, guidance="guidance", render_type=render_type)
            total_loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref", render_type=render_type)
            total_loss += out["loss"]

        self.log("train/loss", total_loss, prog_bar=True)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0],
                        "kwargs": {}
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],

            name="validation_step",
            step=self.true_global_step,
        )

        if self.anchor_rgb is not None and batch_idx == 0:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}_du.png",
                [
                    {
                        "type": "rgb",
                        "img": img,
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }

                    for img in self.anchor_rgb
                ],
                name="validation_step",
                step=self.true_global_step,
            )

            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}_du_mask.png",
                [
                    {
                        "type": "rgb",
                        "img": img[:, :, None].repeat(1,1,3),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }

                    for img in self.anchor_mask
                ],
                name="validation_step",
                step=self.true_global_step,
            )

            if self.du_input is not None:
                self.save_image_grid(
                    f"it{self.true_global_step}-{batch['index'][0]}_du_input.png",
                    [
                        {
                            "type": "rgb",
                            "img": img.permute(1,2,0),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }

                        for img in self.du_input
                    ],
                    name="validation_step",
                    step=self.true_global_step,
                )

        if self.update_rgb is not None and batch_idx == 0:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}_other.png",
                [
                    {
                        "type": "rgb",
                        "img": img[0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }

                    for img in self.update_rgb
                ],
                name="validation_step",
                step=self.true_global_step,
            )

            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}_other_mask.png",
                [
                    {
                        "type": "rgb",
                        "img": img[0].repeat(1,1,3),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }

                    for img in self.update_mask
                ],
                name="validation_step",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"

        try:
            self.save_img_sequence(
                filestem,
                filestem,
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="validation_epoch_end",
                step=self.true_global_step,
            )
            shutil.rmtree(
                os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
            )
        except:
            pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale", "img": out["depth"][0], "kwargs": {}
                        }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale", "img": out["opacity_vis"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)}
                        }
                ]
                if "opacity_vis" in out
                else []
            )
            ,
            name="test_step",
            step=self.true_global_step,
        )

        # FIXME: save camera extrinsics
        c2w = batch["c2w"]
        save_path = os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test/{batch['index'][0]}.npy")
        np.save(save_path, c2w.detach().cpu().numpy()[0])

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

    def on_before_optimizer_step(self, optimizer) -> None:
        pass

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {"guidance."+k : v for (k,v) in self.guidance.state_dict().items()}
        checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
        return

    @torch.cuda.amp.autocast(enabled=False)
    def set_strength(self, strength):
        self.update_anchor_strength = strength

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.set_strength(
            strength=C(self.cfg.update_anchor_strength, epoch, global_step),
        )