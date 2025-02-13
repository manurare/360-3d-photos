import os
from tqdm import tqdm

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Union, List
from train import training, prepare_output_and_logger, log_camera
from scene.cameras import Camera, update_camera
from scene import Scene, append_cam_to_json
import submodules._360monodepth.code.python.src.utility.spherical_coordinates as sc
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.io_utils import save_visuals, vis_cam2grid
from utils.poisson_utils import PoissonSolver
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from mono23D.random_tangent_imgs import sample_im, ico_tangent_images

from scipy.spatial.transform.rotation import Rotation as R
from model_scripts.utils import get_sd_pipeline, process_inputs
import utils.depth_utils as depth_utils

T_THRESH = float(os.getenv("T_THRESH", 0.05))

def seed_everything(seed=999) -> int:
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if not (min_seed_value <= seed <= max_seed_value):
        raise Exception(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


# seed_everything(1337)
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# # need to set CUBLAS_WORKSPACE_CONFIG=:4096:8 if CUDA >= 10.2
# # see "https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility"
# # torch.set_deterministic(deterministic)
# torch.use_deterministic_algorithms(True)

ACTION_RADIUS=0.5

@torch.no_grad()
def get_action_poses(scene:Scene, pipeline, nposes, rgbfile="", acc_threshold = 0.5):
    if os.path.isfile(rgbfile):
        rgb = np.array(Image.open(rgbfile))
    else:
        rgb = None

    aux_camera = scene.getTrainCameras()[0]

    Cs = np.random.randn(nposes, 3)
    Cs /= np.linalg.norm(Cs, axis=1, keepdims=True)
    Cs_pix = np.array(sc.sph2erp(*sc.car2sph(Cs).T, scene.depth.shape[0]))
    Cs_depth = scene.depth[Cs_pix[1].astype(int), Cs_pix[0].astype(int)][..., None] * 0.7 # do not hit scene boundaries!
    Cs *= np.minimum(Cs_depth, ACTION_RADIUS)

    thetas = np.random.rand(nposes) * 2 * np.pi - np.pi
    phis = np.random.rand(nposes) * 0.25 * np.pi - 0.25* np.pi

    Rs = R.from_euler("yx", np.stack((-thetas, -phis)).T, degrees=False).as_matrix()

    bg_color = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

    action_poses_list = []
    cont = len(scene.getTrainCameras()) + len(scene.getTestCameras()) + len(scene.getTrainCameras("virtual"))
    for (c, r) in tqdm(zip(Cs, Rs), desc="Generating action poses"):
        T = -r @ c
        camera = Camera(-1, r.T, T, aux_camera.FoVx, aux_camera.FoVy, aux_camera.original_image, None,
                    aux_camera.image_name, -1, depth=None,
                    trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                    )
        render_pkg = render(camera, scene.gaussians, pipeline, bg_color)

        if rgb is not None:
            gt = sample_im(rgb, camera, camera.original_image.shape[1])
        camera = Camera(-1, camera.R, camera.T, aux_camera.FoVx, aux_camera.FoVy, render_pkg["render"], render_pkg["render_acc"],
                    f"{cont:03}", cont, depth=render_pkg["render_depth"],
                    trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                    )
        cont += 1
        action_poses_list.append(camera)
    return action_poses_list

def sort_by_hole_size(cameras:Union[Camera, List[Camera]]):
    if not isinstance(cameras, list):
        cameras = [cameras]

    mean_Ts = []
    for c in cameras:
        mean_Ts.append(c.original_mask.mean().item())

    indices = np.argsort(np.array(mean_Ts))[::-1]
    cameras = np.array(cameras)[indices]
    return cameras.tolist()


def action_poses_anim(scene:Scene, pipeline, nposes, rgb=None, acc_threshold = 0.5):
    Rs = np.stack([np.eye(3) for _ in range(nposes*6)])
    Tsx = np.linspace(0, 1, nposes, endpoint=True)[:, None] * ACTION_RADIUS * np.array([1, 0, 0])
    Tsmx = np.linspace(0, 1, nposes, endpoint=True)[:, None] * ACTION_RADIUS * np.array([-1, 0, 0])
    Tsy = np.linspace(0, 1, nposes, endpoint=True)[:, None] * ACTION_RADIUS * np.array([0, 1, 0])
    Tsmy = np.linspace(0, 1, nposes, endpoint=True)[:, None] * ACTION_RADIUS * np.array([0, -1, 0])
    Tsz = np.linspace(0, 1, nposes, endpoint=True)[:, None] * ACTION_RADIUS * np.array([0, 0, 1])
    Tsmz = np.linspace(0, 1, nposes, endpoint=True)[:, None] * ACTION_RADIUS * np.array([0, 0, -1])
    Ts = np.concatenate((Tsx, Tsmx, Tsy, Tsmy, Tsz, Tsmz))

    bg_color = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

    aux_camera = scene.getTrainCameras()[0]

    action_poses_list = []
    for i, (R, T) in enumerate(zip(Rs, Ts)):
        camera = Camera(-1, R, T, aux_camera.FoVx, aux_camera.FoVy, aux_camera.original_image, None,
                    aux_camera.image_name, -1, depth=None,
                    trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                    )
        render_pkg = render(camera, scene.gaussians, pipeline, bg_color)

        if rgb is not None:
            gt = sample_im(rgb, camera, camera.original_image.shape[1])
        camera = Camera(-1, R, T, aux_camera.FoVx, aux_camera.FoVy, render_pkg["render"], render_pkg["render_acc"],
                    aux_camera.image_name, -1, depth=render_pkg["render_depth"],
                    trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                    )
        action_poses_list.append(camera)
    return action_poses_list

def piecewise_T(scene:Scene, pipeline, camera):
    intervals = np.linspace(0, 1, 5, endpoint=True)[1:]

    bg_color = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

    action_poses_list = []
    T_max = camera.T.copy()
    for t in intervals:
        camera.T = T_max * t
        render_pkg = render(camera, scene.gaussians, pipeline, bg_color)

        camera = Camera(-1, camera.R, camera.T, camera.FoVx, camera.FoVy, render_pkg["render"], render_pkg["render_acc"],
                    camera.image_name, -1, depth=render_pkg["render_depth"],
                    trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                    )
        action_poses_list.append(camera)
    return action_poses_list

@torch.no_grad()
def fill_2D(camera, scene: Scene, dataset : ModelParams, load=False):
    device = scene.gaussians.get_rotation.device

    if getattr(dataset, "inpainting_path", None) is None:
        raise "No inpainting path in model params"
    inpainting_path = dataset.inpainting_path
    os.makedirs(inpainting_path, exist_ok=True)


    uid = int(camera.rank)
    fill_rgb_file_fetch = os.path.join(dataset.inpainting_path_ckpt, f"{uid:05d}_fill_rgb.pt")
    fill_depth_file_fetch = os.path.join(dataset.inpainting_path_ckpt, f"{uid:05d}_fill_depth.pt")
    fill_rgb_file = os.path.join(inpainting_path, f"{uid:05d}_fill_rgb.pt")
    fill_depth_file = os.path.join(inpainting_path, f"{uid:05d}_fill_depth.pt")

    if os.path.isfile(fill_rgb_file_fetch) and os.path.isfile(fill_depth_file_fetch) and load:
        camera.inpaint_image = torch.load(fill_rgb_file_fetch, map_location=scene.gaussians.get_rotation.device)
        camera.inpaint_depth = torch.load(fill_depth_file_fetch, map_location=scene.gaussians.get_rotation.device)
        return camera, True

    # inpaint color
    out = None
    if len(pipe) == 1:
        im, mask, kwargs = process_inputs(camera, pipe[0], strategy=dataset.fill_strategy)
        out = pipe[0](**kwargs).images[0]
    else:
        for ipipe in range(len(pipe)):
            if ipipe == 0:
                im, mask, kwargs = process_inputs(camera, pipe[ipipe], im=out, strategy=dataset.fill_strategy)
            else:
                _, _, kwargs = process_inputs(camera, pipe[ipipe], im=out, strategy=dataset.fill_strategy)

            if ipipe < (len(pipe) - 1):
                kwargs["output_type"] = "latent"

            out = pipe[ipipe](**kwargs).images[0]

    camera.inpaint_image = out.squeeze().float()

    # inpaint depth
    raw_mask = (camera.original_mask > T_THRESH).cpu().numpy()
    raw_mask_dil = cv2.dilate(raw_mask.astype(np.uint8), np.ones((5, 5)), iterations=2) > 0
    camera.inpaint_mask = mask.squeeze().float().to(device)
    camera.dilmask = torch.from_numpy(raw_mask_dil).to(device, camera.original_mask.dtype)

    inpainted_depth = depth_utils.predict_depth(camera.inpaint_image[None]).squeeze()
    inpainted_depth = depth_utils.pred2gt_least_squares(inpainted_depth.cpu().numpy(), camera.original_depth.cpu().numpy(), ((camera.original_depth > 0) & ~camera.inpaint_mask.type(torch.bool)).cpu().numpy())
    inpainted_depth = torch.from_numpy(inpainted_depth).to(device, torch.float32)

    # blend RGB
    poisson_solver_raw_dil = PoissonSolver(raw_mask_dil)
    blended_image_dil = poisson_solver_raw_dil.gradient_blending_vec(out.cpu().float().permute(1,2,0).numpy(), camera.original_image.permute(1,2,0).cpu().numpy())
    camera.inpaint_image = torch.from_numpy(blended_image_dil.squeeze()).to(device, camera.original_depth.dtype).permute(2, 0, 1).clamp(0, 1)

    # blend Depth
    poisson_solver = PoissonSolver(((camera.inpaint_mask > 0) | (camera.original_depth <= 0)).cpu().numpy(), target=camera.original_depth.cpu().numpy()[..., None])
    blended_depth = poisson_solver.gradient_blending_vec(inpainted_depth.cpu().numpy()[..., None], camera.original_depth.cpu().numpy()[..., None])
    camera.inpaint_depth = torch.from_numpy(blended_depth.squeeze()).to(device, camera.original_depth.dtype)
    camera.inpaint_depth = depth_utils.clip_depth_holes(camera.inpaint_depth, poisson_solver.mask)

    # save data
    torch.save(camera.inpaint_image, fill_rgb_file)
    torch.save(camera.inpaint_depth, fill_depth_file)
    save_visuals(inpainting_path, camera, uid)

    return camera, False

def populate_scene_with_virtual_info(scene:Scene, new_cameras: Camera):
    gaussians = scene.gaussians

    rgb = camera.inpaint_image
    depth = camera.inpaint_depth
    mask = camera.dilmask > T_THRESH
    if rgb is None or depth is None or (mask.sum() == 0):
        return True
    scene.train_cameras.setdefault("virtual", []).append(camera)
    gaussians.add_extra_gaussians(rgb, depth, mask, camera)
    return True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start_iteration", default=None, type=int)
    group.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--load_inpaint", action='store_true')
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[*range(0, int(2e9), 3000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 7_000, 15_000, 30_000, 40_000, 50_000, 90_000]+list(range(90_000, int(2e9), 20_000)))
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--first_small", action='store_true')
    parser.add_argument("--save_vis", action='store_true')
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    model_params = model.extract(args)
    pipeline_params = pipeline.extract(args)
    opt_params = op.extract(args)

    if args.load_ckpt is not None:
        args.start_iteration = int(os.path.splitext(os.path.basename(args.load_ckpt))[0].split('chkpnt')[-1])
    elif args.start_iteration is not None:
        args.load_ckpt = os.path.join(model_params.model_path, "chkpnt" + str(args.start_iteration) + ".pth")

    opt_params.iterations = 1000  # each inpainted image trains for this amount of iters
    opt_params.densify_until_iter = 1e10
    opt_params.densify_from_iter = 100

    gaussians = GaussianModel(model_params.sh_degree)
    scene = Scene(model_params, gaussians, load_iteration=-1, shuffle=False)
    scene.current_iter = args.start_iteration
    gaussians.training_setup(opt_params)

    (ckpt_params, _) = torch.load(args.load_ckpt)
    gaussians.restore(ckpt_params, opt_params)
    model_params.inpainting_path_ckpt = os.path.join(os.path.dirname(args.load_ckpt), f"inpainting_7000", model_params.fill_strategy)
    # Reduce opacities LR
    for p_group in gaussians.optimizer.param_groups:
        if p_group["name"] == "opacity":
            p_group["lr"] *= 0.1
    
    # inpainting models
    global pipe
    pipe = get_sd_pipeline(scene.gaussians.get_rotation.device, model_params.fill_strategy)

    # Ugly hack. Init as non-virtual but save as virtual
    if not getattr(model_params, "virtual", False):
        scene.model_path = os.path.join(scene.model_path, model_params.xp_name)
        model_params.model_path = scene.model_path

    # recompute inpainting path
    model_params.inpainting_path = os.path.join(model_params.model_path, f"inpainting_{args.start_iteration}", model_params.fill_strategy)

    tb_writer = prepare_output_and_logger(model_params)

    # save initial point cloud
    scene.save(scene.current_iter)
    
    epochs = 100
    opt_params.total_iterations = epochs * opt_params.iterations
    # Train loop
    for i in range(epochs):
        action_cameras = get_action_poses(scene, pipeline_params, 100, model_params.source_path)
        action_cameras_sorted = sort_by_hole_size(action_cameras)
        if args.first_small:
            camera = action_cameras_sorted[-1]
        else:
            camera = action_cameras_sorted[0]

        print(f"Camera: {i:05}")
        camera.rank = f"{i:03}"
        camera = update_camera(camera, scene, pipeline_params, render)
        camera, loaded = fill_2D(camera, scene, model_params, args.load_inpaint)
        append_cam_to_json(model_params, camera)
        if loaded:
            scene.train_cameras.setdefault("virtual", []).append(camera)
            continue

        # populate scene
        provides_new_info = populate_scene_with_virtual_info(scene, camera)
        if not provides_new_info:
            continue

        training(model_params, opt_params, pipeline_params,
                args.test_iterations, args.save_iterations,
                args.checkpoint_iterations, None, -1, scene=scene, tb_writer=tb_writer)

        assert not torch.any(torch.isinf(gaussians._opacity)), "Infinity values in pipeline opacity"
        assert not torch.any(torch.isinf(gaussians._scaling)), "Infinity values in pipeline scaling"
        assert not torch.any(torch.isinf(gaussians._rotation)), "Infinity values in pipeline rotation"
        print(f"Current iteration: {scene.current_iter}")

    # Manually save
    print("\n[ITER {}] Saving Checkpoint".format(scene.current_iter))
    torch.save((gaussians.capture(), scene.current_iter), scene.model_path + "/chkpnt" + str(scene.current_iter) + ".pth")
    scene.save(scene.current_iter)
