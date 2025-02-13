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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, nptoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_image_depth = None
    if cam_info.depth is not None:
        resized_image_depth = nptoTorch(cam_info.depth, resolution[::-1])

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, depth=resized_image_depth, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    height = camera.height if hasattr(camera, "height") else camera.image_height
    width = camera.width if hasattr(camera, "width") else camera.image_width

    fovx = camera.FovX if hasattr(camera, "FovX") else camera.FoVx
    fovy = camera.FovY if hasattr(camera, "FovY") else camera.FoVy

    camera_entry = {
        'id' : id,
        'uid': camera.uid,
        'img_name' : camera.image_name,
        'width' : width,
        'height' : height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(fovy, height),
        'fx' : fov2focal(fovx, width)
    }
    return camera_entry

def overlap(cam1:Camera, cam2:Camera):
    H, W = cam1.image_height, cam1.image_width
    num_pix = H*W
    focal_x, focal_y = fov2focal(cam1.FoVx, W), fov2focal(cam1.FoVy, H)

    xx, yy = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    depth = cam1.inpaint_depth if cam1.inpaint_depth is not None else cam1.original_depth
    points_cam1 = np.stack(((xx - W // 2) / focal_x, (yy - H // 2) / focal_y,
                            depth.cpu().numpy(), np.ones_like(xx)))
    points_world = np.einsum("cr,chw->rhw", np.linalg.inv(cam1.world_view_transform.cpu().numpy()), points_cam1)
    points_world /= points_world[[-1]]

    # points_pix2_v2 = np.einsum("cr,chw->rhw", cam2.full_proj_transform.cpu().numpy(), points_world)
    # points_pix2_v2 /= points_pix2_v2[[-1]]
    # points_pix2_v2 = points_pix2_v2[:-1]
    # points_pix2_v2[0] = ((points_pix2_v2[0] + 1.0) * W - 1.0) * 0.5
    # points_pix2_v2[1] = ((points_pix2_v2[1] + 1.0) * H - 1.0) * 0.5
    # z_v2 = points_pix2_v2[2]
    # inside_v2 = np.logical_and.reduce((points_pix2_v2[0] >= 0, points_pix2_v2[0] < W, points_pix2_v2[1] >= 0, points_pix2_v2[1] < H, z_v2 >= 0))

    points_cam2 = np.einsum("cr,chw->rhw", cam2.world_view_transform.cpu().numpy(), points_world)
    points_pix2 = np.stack(((points_cam2[0] / points_cam2[2] * focal_x + W // 2, points_cam2[1] / points_cam2[2] * focal_y + H // 2,
                            points_cam2[2])))

    z = points_pix2[2]
    inside = np.logical_and.reduce((points_pix2[0] >= 0, points_pix2[0] < W, points_pix2[1] >= 0, points_pix2[1] < H, z >= 0))

    return inside.sum() / num_pix


def get_sampling_weights_by_overlap(scene):
    if scene.getTrainCameras("virtual") is None:
        weights = np.ones(len(scene.getTrainCameras())) / len(scene.getTrainCameras())
        return weights

    cameras = scene.getTrainCameras() + scene.getTrainCameras("virtual")
    last_added = cameras[-1]

    overlaps = []
    for camera in cameras[:-1]:
        overlaps.append(overlap(camera, last_added))

    return overlaps

import os
def get_sampling_weights(scene, iteration, init_iter, total_iters=None):
    p_train_init = float(os.getenv("train_view_weight", 0.96))

    num_train = len(scene.getTrainCameras())
    num_virtual = len(scene.getTrainCameras("virtual"))

    weights_train = np.ones(num_train) / num_train
    weights_virtual = []
    if len(scene.getTrainCameras("virtual")) > 0:
        assert total_iters is not None
        m = (0.5 - p_train_init) / (total_iters // 2)
        b = p_train_init
        p_train = np.clip(m * (iteration - init_iter) + b, 0.5, None)
        p_virtual = 1 - p_train
        weights_virtual = np.ones(num_virtual) / num_virtual * p_virtual
    else:
        p_train = 1

    weights = np.concatenate((weights_train * p_train, weights_virtual))
    weights = np.concatenate((weights_train * p_train, weights_virtual))
    weights = np.concatenate((weights_train * 0.5, np.ones(num_virtual) / num_virtual * 0.5))
    weights /= np.sum(weights)
    return weights
