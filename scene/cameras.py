#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

import os
T_THRESH = float(os.getenv("T_THRESH", 0.05))

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, depth=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", **kwargs
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self._R = R
        self._T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.original_depth = depth.to(self.data_device).squeeze() if depth is not None else None
        self._original_mask = (1 - gt_alpha_mask).type(torch.float32).squeeze() if gt_alpha_mask is not None else None  # holes=1
        self.inpaint_mask = (1 - gt_alpha_mask).type(torch.float32).squeeze() if gt_alpha_mask is not None else self.original_image.new_zeros((self.image_height, self.image_width))
        self._inpaint_image = kwargs.get("inpaint_image")
        self._inpaint_depth = kwargs.get("inpaint_depth")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, newT):
        self._T = newT
        self.world_view_transform = torch.tensor(getWorld2View2(self._R, self._T, self.trans, self.scale)).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, newR):
        self._R = newR
        self.world_view_transform = torch.tensor(getWorld2View2(self._R, self._T, self.trans, self.scale)).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def inpaint_image(self):
        return self._inpaint_image

    @inpaint_image.setter
    def inpaint_image(self, image):
        # Assume image ndim is always 3
        image = image[None]
        if image.shape[2:] == (self.image_height, self.image_width):
            self._inpaint_image = image[0].float()
        else:
            self._inpaint_image = torch.nn.functional.interpolate(image, (self.image_height, self.image_width), mode="bilinear")[0].float()

    @property
    def inpaint_depth(self):
        return self._inpaint_depth

    @inpaint_depth.setter
    def inpaint_depth(self, depth):
        # Assume depth ndim is always 2
        depth = depth[None, None]
        if depth.shape[2:] == (self.image_height, self.image_width):
            self._inpaint_depth = depth[0, 0].float()
        else:
            self._inpaint_depth = torch.nn.functional.interpolate(depth, (self.image_height, self.image_width), mode="nearest-exact")[0, 0].float()

    @property
    def original_mask(self):
        return self._original_mask

    @original_mask.setter
    def original_mask(self, mask):
        # Assume mask ndim is always 2
        mask = 1 - mask[None, None]
        if mask.shape[2:] == (self.image_height, self.image_width):
            self._original_mask = mask[0, 0]
        else:
            self._original_mask = torch.nn.functional.interpolate(mask, (self.image_height, self.image_width), mode="bilinear")[0, 0]

@torch.no_grad()
def update_camera(camera, scene, pipeline, render_fn, bg_color=None):
    bg_color = torch.tensor([0,0,0], dtype=torch.float32, device="cuda") if bg_color is None else bg_color

    render_pkg = render_fn(camera, scene.gaussians, pipeline, bg_color)
    camera.original_depth = render_pkg["render_depth"].squeeze()
    camera.original_image = render_pkg["render"].clamp(0, 1)
    camera.original_mask = render_pkg["render_acc"]
    camera.original_image_noBG = (camera.original_image /(1 - camera.original_mask)).clamp(0, 1)
    camera.original_image_noBG[torch.isnan(camera.original_image_noBG)] = 0

    return camera


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

