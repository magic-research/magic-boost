from dataclasses import dataclass, field

import threestudio
import torch.nn.functional as F
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *

import torch
import torch.nn.functional as F

def ranking_loss(error, penalize_ratio=0.7, type='mean'):
    error, indices = torch.sort(error)
    # only sum relatively small errors
    s_error = torch.index_select(error, 0, index=indices[: int(penalize_ratio * indices.shape[0])])
    if type == 'mean':
        return torch.mean(s_error)
    elif type == 'sum':
        return torch.sum(s_error)

def BCE_loss(y_hat, y):
    y_hat = torch.cat((1-y_hat, y_hat), 1)
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


@threestudio.register("mesh-fitting-guidance")
class MeshGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        geometry_type: str = ""
        geometry: dict = field(default_factory=dict)
        renderer_type: str = ""
        renderer: dict = field(default_factory=dict)
        material_type: str = ""
        material: dict = field(default_factory=dict)
        background_type: str = ""
        background: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading obj")
        geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        background = threestudio.find(self.cfg.background_type)(self.cfg.background)
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=geometry,
            material=material,
            background=background,
        )
        threestudio.info(f"Loaded mesh!")

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        mask: Float[Tensor, "B H W C"],
        normal: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        guide_rgb = self.renderer(**kwargs)

        # import cv2
        # tmp_vis = guide_rgb["comp_normal"].detach().cpu().numpy()[0]
        # cv2.imshow('tmp', tmp_vis[:, :, [2,1,0]])
        # cv2.waitKey()

        guidance_out = {"loss_l1": F.l1_loss(rgb, guide_rgb["comp_rgb"])}

        mask_errors = BCE_loss(mask.clip(1e-3, 1.0 - 1e-3).reshape(-1, 1), guide_rgb["opacity"].long().detach().reshape(-1, 1))
        mask_loss = ranking_loss(mask_errors[:, 0], penalize_ratio=0.8)
        guidance_out.update({"loss_mask": mask_loss})

        mask = mask.reshape(-1, 1)

        if normal is not None:
            normal_errors = 1 - F.cosine_similarity(normal.reshape(-1, 3), guide_rgb["comp_normal"].reshape(-1, 3), dim=1)
            normal_loss = ranking_loss(normal_errors[mask[:, 0] > 0], penalize_ratio=0.9, type='sum')
            guidance_out.update({"loss_normal": normal_loss})

        return guidance_out
