import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import fov2focal, raydepth2zdepth
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates

import submodules._360monodepth.code.python.src.utility.spherical_coordinates as sc

def random_perspective_images(rgb, depth, persp_size=400, randomized_t=False):
    H,W,C = rgb.shape

    xangle = np.random.choice(np.linspace(-90, 90, 10, endpoint=False))
    yangle = np.random.choice(np.linspace(-180, 180, 10, endpoint=False))
    rot = R.from_euler("yx", angles=[yangle, xangle], degrees=True).as_matrix()
    camera = Camera(0, rot, np.zeros(3), np.radians(55), np.radians(55), torch.zeros((3, persp_size, persp_size)),
                    None, "test", 0)

    persp_gt = sample_im(rgb, camera, persp_size, 1)
    persp_depth = sample_im(depth, camera, persp_size, 0)
    # Convert to z depth
    persp_depth = raydepth2zdepth(persp_depth, fov2focal(camera.FoVx, persp_size))

    camera_info = CameraInfo(uid=0, R=camera.R, T=camera.T, FovX=camera.FoVx, FovY=camera.FoVy,
                             image=Image.fromarray(np.array(persp_gt*255.0, dtype=np.byte), "RGB"),
                             depth=persp_depth,
                             image_path="", image_name="", width=persp_size, height=persp_size)

    persp_gt = torch.from_numpy(persp_gt).type(torch.float32).to(camera.data_device).permute(2, 0, 1)

    return persp_gt, camera, camera_info


def interpolate_training_views(rgb, depth, view1:CameraInfo, view2:CameraInfo):
    persp_size = view1.depth.shape[0]
    quat1 = R.from_matrix(view1.R.T).as_quat()
    quat2 = R.from_matrix(view2.R.T).as_quat()

    interp = np.linspace(0, 1, 31, endpoint=False)[1:]
    
    camera_infos = []
    for idx, alpha in enumerate(interp):
        quat = (1-alpha) * quat1 + alpha * quat2
        quat /= np.linalg.norm(quat)
        camera = Camera(0, R.from_quat(quat).as_matrix().T, view1.T, view1.FovX, view1.FovY, 
                        torch.zeros((3, persp_size, persp_size)), None, "test", 0)
        
        persp_gt = sample_im(rgb, camera, persp_size, 1)
        persp_depth = sample_im(depth, camera, persp_size, 0)
        # Convert to z depth
        persp_depth = raydepth2zdepth(persp_depth, fov2focal(camera.FoVx, persp_size))
        camera_info = CameraInfo(uid=idx, R=camera.R, T=camera.T, FovX=camera.FoVx, FovY=camera.FoVy,
                            image=Image.fromarray(np.array(persp_gt*255.0, dtype=np.byte), "RGB"),
                            depth=persp_depth,
                            image_path="", image_name=f"{idx:03}", width=persp_size, height=persp_size)
        camera_infos.append(camera_info)
    
    return camera_infos

def lemniscate(train_cameras, radius):
    aux_camera = train_cameras[0]

    # Parameters
    a = radius  # Scale of the lemniscate

    # Create theta values
    theta1 = np.linspace(0.5*np.pi, 0.75*np.pi, 25)
    theta2 = np.linspace(0.75*np.pi, 1.25*np.pi, 80)
    theta3 = np.linspace(1.25*np.pi, 1.75*np.pi, 45)
    theta4 = np.linspace(1.75*np.pi, 2.25*np.pi, 80)
    theta5 = np.linspace(2.25*np.pi, 2.5*np.pi, 25)
    theta = np.concatenate((theta1, theta2, theta3, theta4, theta5))

    # Convert to Cartesian coordinates
    z = a * np.cos(theta) / (np.sin(theta)**2 + 1)
    x = a * np.cos(theta) * np.sin(theta) / (np.sin(theta)**2 + 1)
    y = a * 0.2 * np.cos(4*theta)
    Cs = np.stack((-x, -y, z)).T # -x because right in mesh is the oposite as in replica

    Cs = []
    with open("/home/manuel/Desktop/PHD/code/Rendering360OpticalFlow-replica/replica_360/office_0_lemniscate_1k_0/camera_traj_lookat_noOff.csv", "r") as f:
        lines = f.readlines()
        lines = [l.splitlines()[0].split(" ") for l in lines]
        Cs = np.array([(-float(l[2]), -float(l[3]), float(l[1])) for l in lines])


    dz_dtheta = -a*(np.sin(theta) * (np.sin(theta) ** 2 + 2*np.cos(theta)**2 + 1)) / (np.sin(theta)**2 + 1)**2
    dx_dtheta = -a*(np.sin(theta)**4 + np.sin(theta)**2 + (np.sin(theta)**2-1)*np.cos(theta)**2) / (np.sin(theta)**2 + 1)**2

    lookat = np.stack((-dx_dtheta, np.zeros_like(x), dz_dtheta)).T
    lookat = lookat / np.linalg.norm(lookat, axis=1, keepdims=True)

    UP = np.array([0, -1, 0])[None, :]
    right = np.cross(lookat, UP)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)

    up = np.array([0, 1, 0])[None, ...].repeat(lookat.shape[0], axis=0)

    Rs = np.concatenate((right, up, lookat), axis=1).reshape(-1, 3, 3)  # world2cam
    camera_infos = []
    for i, (rot, C) in enumerate(zip(Rs, Cs)):
        T = -rot@C
        rot = rot.T
        image = np.zeros_like(np.asarray(aux_camera.image))
        camera = CameraInfo(uid=i, R=rot, T=T, FovX=aux_camera.FovX,
                            FovY=aux_camera.FovY, image_name=f"{i:03}",
                            image=Image.fromarray(image), width=aux_camera.width, height=aux_camera.height)
        camera_infos.append(camera)
    return camera_infos


from submodules._360monodepth.code.python.src.utility.projection_icosahedron import erp2ico_image, get_icosahedron_parameters
import submodules._360monodepth.code.python.src.utility.subimage as subimage


def ico_tangent_images(rgb, depth, persp_size=400):
    camera_infos = []
    padding = 0.1
    params = subimage.erp_ico_cam_intrparams(persp_size, padding)
    params_ref = subimage.erp_ico_cam_intrparams(persp_size, padding)[7]

    FovX = np.arctan((persp_size * 0.5) / params_ref['intrinsics']['focal_length_x']) * 2
    FovY = np.arctan((persp_size * 0.5) / params_ref['intrinsics']['focal_length_y']) * 2

    for i in range(20):
        params_i = params[i]
        # params_i['rotation']  # cam2world
        rot = params_i['rotation'].T
        t = -params_i['rotation'] @ params_i['translation']
        camera = Camera(0, rot, t, FovX, FovY,
                        torch.zeros((3, persp_size, persp_size)),
                        None, "test", 0)

        if depth is None or rgb is None:
            camera_infos.append(camera)
            continue
        persp_rgb = sample_im(rgb, camera, persp_size, 1)
        persp_depth = sample_im(depth, camera, persp_size, 0)
        # Convert to z depth
        persp_depth = raydepth2zdepth(persp_depth, params_ref['intrinsics']['focal_length_x'])

        camera_info = CameraInfo(uid=i, R=camera.R, T=camera.T, FovX=camera.FoVx, FovY=camera.FoVy,
                                 image=Image.fromarray(np.array(persp_rgb * 255.0, dtype=np.byte), "RGB"),
                                 depth=persp_depth,
                                 image_path="", image_name=f"{i:03}", width=persp_size, height=persp_size)
        camera_infos.append(camera_info)
    
    return camera_infos

def sample_im(im, camera, persp_size, order=1):
    if im.ndim == 2:
        im = im[..., None]

    H,W,C = im.shape

    focal_x, focal_y = fov2focal(camera.FoVx, persp_size), fov2focal(camera.FoVy, persp_size)

    xx, yy = np.meshgrid(np.arange(persp_size), np.arange(persp_size), indexing="xy")
    points_cam = np.stack(((xx.flatten() - persp_size // 2) / focal_x, (yy.flatten() - persp_size // 2) / focal_y,
                            np.ones_like(xx.flatten())))
    
    c2w = camera.world_view_transform.transpose(1,0).inverse().cpu().numpy()
    points_world = c2w[:3, :3] @ points_cam + c2w[:3, [3]]

    points_world_sph = sc.car2sph(points_world.T)
    erp_xx, erp_yy = sc.sph2erp(points_world_sph[:, 0], points_world_sph[:, 1], H)
    erp_xx = erp_xx.reshape(persp_size, persp_size) + 0.5
    erp_yy = erp_yy.reshape(persp_size, persp_size) + 0.5

    persp_im = np.zeros((persp_size, persp_size, C))
    for c in range(C):
        persp_im[..., c] = map_coordinates(im[..., c], [erp_yy, erp_xx], order=order, mode='grid-wrap')

    return persp_im.squeeze(-1) if C==1 else persp_im