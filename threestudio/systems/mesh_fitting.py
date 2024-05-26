# modified from https://github.com/DSaurus/threestudio-meshfitting/blob/main/guidance/mesh_guidance.py

from dataclasses import dataclass, field

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *
import torch
from threestudio.utils.ops import binary_cross_entropy, dot


@threestudio.register("mesh-fitting-system")
class MeshFittingSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        refinement: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if 'sdf' in self.cfg.guidance_type:
            self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        out = self(batch)

        if "comp_normal" in out.keys():
            normal = out["comp_normal"]
        else:
            normal = None

        guidance_out = self.guidance(out["comp_rgb"], out["opacity"], normal, **batch, rgb_as_latents=False)
        loss = 0.0
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_") and self.C(self.cfg.loss[name.replace("loss_", "lambda_")]) > 0:
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.cfg.loss.lambda_eikonal > 0:
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        if self.cfg.loss.lambda_sparsity > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.cfg.loss.lambda_geometry > 0:
            points_rand = (
                torch.rand((10000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
            )
            sdf_gt = self.geometry.get_gt_sdf(points_rand)
            sdf_pred = self.geometry.forward_sdf(points_rand)
            loss_geometry = torch.nn.functional.mse_loss(sdf_pred, sdf_gt)
            loss += loss_geometry * self.C(self.cfg.loss.lambda_geometry)

        if self.C(self.cfg.loss.lambda_orient) > 0:
                    if "normal" not in out:
                        raise ValueError(
                            "Normal is required for orientation loss, no normal is found in the output."
                        )
                    loss_orient = \
                        (
                            out["weights"].detach()
                            * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                        ).sum() / (out["opacity"] > 0).sum()
                    loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
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

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
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
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

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
