import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
from extern.MVC.mvc.camera_utils import add_margin
from PIL import Image


def crop_and_padding(img, crop_size, min_x, min_y, max_x, max_y):
    img = img.crop((min_x, min_y, max_x, max_y))
    h, w = img.height, img.width
    scale = crop_size / max(h, w)
    h_, w_ = int(scale * h), int(scale * w)
    img = img.resize((w_, h_))
    img = add_margin(img, size=256)
    return img

@dataclass
class SingleImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    rays_d_normalize: bool = True
    use_mixed_camera_config: bool = False
    load_anchor_view: bool = False
    anchor_height: int = 256
    anchor_width: int = 256
    mvcond_dir: str = None
    anchor_select_list: Optional = None
    anchor_relative_elev: Optional = None
    anchor_azimuth: Optional = None

class SingleImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: SingleImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )

            self.random_camera_cfg = random_camera_cfg

            if self.cfg.use_mixed_camera_config:
                if self.rank % 2 == 0:
                    random_camera_cfg.camera_distance_range=[self.cfg.default_camera_distance, self.cfg.default_camera_distance]
                    random_camera_cfg.fovy_range=[self.cfg.default_fovy_deg, self.cfg.default_fovy_deg]
                    self.fixed_camera_intrinsic = True
                else:
                    self.fixed_camera_intrinsic = False
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )

                if self.cfg.load_anchor_view:
                    anchor_camera_cfg = random_camera_cfg.copy()
                    anchor_camera_cfg.eval_height = cfg.anchor_height
                    anchor_camera_cfg.eval_width = cfg.anchor_width
                    self.anchor_pose_generator = RandomCameraDataset(
                        anchor_camera_cfg, "anchor"
                    )

            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "1 3"] = camera_position
        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        self.c2w: Float[Tensor, "1 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        self.c2w4x4: Float[Tensor, "B 4 4"] = torch.cat(
            [self.c2w, torch.zeros_like(self.c2w[:, :1])], dim=1
        )
        self.c2w4x4[:, 3, 3] = 1.0

        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.focal_length = self.focal_lengths[0]
        self.set_rays()
        self.load_anchor()

        if self.cfg.anchor_relative_elev is not None:
            self.anchor_elev = torch.FloatTensor(self.cfg.anchor_relative_elev) + self.cfg.default_elevation_deg
        else:
            self.anchor_elev = torch.FloatTensor([self.cfg.default_elevation_deg] * len(self.anchor_rgb))

        if self.cfg.anchor_azimuth is not None:
            self.anchor_azimuth = torch.FloatTensor(self.cfg.anchor_azimuth)
        else:
            self.anchor_azimuth = torch.FloatTensor([0., 90., 180., 270.])

        self.load_images()
        self.prev_height = self.height

    def load_anchor(self):
        anchor_list = self.cfg.anchor_select_list
        if anchor_list is None:
            anchor_list = [0,1,2,3]

        rgb_list = []
        mask_list = []
        for i in anchor_list:
            imgp = self.cfg.mvcond_dir + '/%d.png' % int(i)
            imgp = Image.open(imgp).resize((self.cfg.anchor_width, self.cfg.anchor_height))
            imgp = np.array(imgp)

            rgb = imgp[:, :, :3] / 255.

            if imgp.shape[2] > 3:
                mask = imgp[:, :, 3]
            else:
                mask= np.ones_like(imgp[:, :, -1]) * 255.

            rgb_list.append(torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank))
            mask_list.append(torch.from_numpy(mask).unsqueeze(0).contiguous().to(self.rank))
        rgb_list = torch.cat(rgb_list, dim=0)
        mask_list = torch.cat(mask_list, dim=0)
        self.anchor_rgb = rgb_list
        self.anchor_mask = mask_list

    def set_rays(self):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

        rays_o, rays_d = get_rays(
            directions,
            self.c2w,
            keepdim=True,
            noise_scale=self.cfg.rays_noise_scale,
            normalize=self.cfg.rays_d_normalize,
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.01, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx

    def load_images(self):
        # load image
        assert os.path.exists(
            self.cfg.image_path
        ), f"Could not find image {self.cfg.image_path}!"

        rgba = cv2.cvtColor(
            cv2.imread(self.cfg.image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(
                rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0
        )
        rgb = rgba[..., :3]
        self.rgb: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
        )
        self.mask: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
        )

    def get_all_images(self):
        return self.rgb

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")
        self.set_rays()
        self.load_images()
        # self.load_anchor()


class SingleImageIterableDataset(IterableDataset, SingleImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def collate(self, batch) -> Dict[str, Any]:
        batch = {
            "rays_o": self.rays_o,
            "rays_d": self.rays_d,
            "mvp_mtx": self.mvp_mtx,
            "camera_positions": self.camera_position,
            "light_positions": self.light_position,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "rgb": self.rgb,
            "mask": self.mask,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "c2w": self.c2w,
            "c2w4x4": self.c2w4x4,
            "anchor_rgb": self.anchor_rgb,
            "anchor_mask": self.anchor_mask,
            "anchor_elev": self.anchor_elev,
            "anchor_azimuth": self.anchor_azimuth,
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        if self.cfg.load_anchor_view:
            batch["anchor_views"] = []
            for i in range(self.cfg.random_camera["n_anchor_views"]):
                batch["anchor_views"].append(self.anchor_pose_generator[i])

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class SingleImageDataset(Dataset, SingleImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        batch = self.random_pose_generator[index]
        batch.update(
            {
            "height": self.random_pose_generator.cfg.eval_height,
            "width": self.random_pose_generator.cfg.eval_width,
            "mvp_mtx_ref": self.mvp_mtx[0],
            "c2w_ref": self.c2w4x4,
            }
        )
        return batch


@register("single-image-datamodule")
class SingleImageDataModule(pl.LightningDataModule):
    cfg: SingleImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SingleImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SingleImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = SingleImageDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
