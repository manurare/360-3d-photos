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

import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import update_camera
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.io_utils import vis_cam2grid
from utils.camera_utils import get_sampling_weights
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, scene=None, tb_writer=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, tb_writer)
    if scene is None:
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
    else:
        gaussians = scene.gaussians
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy() + scene.getTrainCameras("virtual").copy()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        tb_writer.add_scalars("LR", {pg['name']:pg['lr'] for pg in gaussians.optimizer.param_groups}, global_step=scene.current_iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if scene.current_iter % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        num_cameras = len(scene.getTrainCameras())
        if scene.getTrainCameras("virtual") is not None:
            num_cameras += len(scene.getTrainCameras("virtual"))
        tb_writer.add_scalars("Num cameras",{"in_loop": num_cameras}, global_step=scene.current_iter)

        weights = get_sampling_weights(scene, scene.current_iter, scene.loaded_iter, getattr(opt, "total_iterations", None))
        viewpoint_cam = np.random.choice(viewpoint_stack, p=weights)
        tb_writer.add_scalar("viewpoint UID", viewpoint_cam.uid, global_step=scene.current_iter)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg.get("render_depth"), render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.inpaint_image.cuda() if viewpoint_cam.inpaint_image is not None else viewpoint_cam.original_image.cuda()

        transparency_mask = None
        if viewpoint_cam.original_depth is not None and depth is not None:
            gt_depth = viewpoint_cam.inpaint_depth.cuda() if viewpoint_cam.inpaint_depth is not None else viewpoint_cam.original_depth.cuda()
            ll1_depth = l1_loss(depth, gt_depth, mask=transparency_mask)
        else:
            ll1_depth = torch.tensor(0, device="cuda")
        Ll1 = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image, mask=transparency_mask)
        ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=transparency_mask))
        ll1_depth = opt.lambda_depth * ll1_depth
        loss =  Ll1 + ssim_loss + ll1_depth
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, scene.current_iter, viewpoint_cam, Ll1, ll1_depth, ssim_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (scene.current_iter in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(scene.current_iter))
                scene.save(scene.current_iter)

            # Densification
            if scene.current_iter < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and scene.current_iter % opt.densification_interval == 0:
                    size_threshold = 20 if scene.current_iter > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, opt.only_pruning)
                    gaussians.decrease_opacity()

                if (opt.opacity_reset_interval > 0) and ((scene.current_iter % opt.opacity_reset_interval == 0) or (dataset.white_background and scene.current_iter == opt.densify_from_iter)):
                    for idx in [0, 1]:
                        try:
                            if scene.getTrainCameras("virtual"):
                                # copy to not modify object
                                check = update_camera(copy.copy(scene.getTrainCameras("virtual")[idx]), scene, pipe, render, bg)
                            elif len(scene.getTrainCameras()) > 20:
                                check = update_camera(copy.copy(scene.getTrainCameras()[idx + 20]), scene, pipe, render, bg)
                            else:
                                check = update_camera(copy.copy(scene.getTrainCameras()[idx]), scene, pipe, render, bg)
                            log_camera(tb_writer, scene.current_iter, check)
                        except:
                            print("Not able to log this camera")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (scene.current_iter in (checkpoint_iterations + saving_iterations)):
                print("\n[ITER {}] Saving Checkpoint".format(scene.current_iter))
                torch.save((gaussians.capture(), scene.current_iter), scene.model_path + "/chkpnt" + str(scene.current_iter) + ".pth")

            # keep track of iterations in scene object
            scene.current_iter += 1

def prepare_output_and_logger(args, tb_writer=None):
    if tb_writer is not None:
        return tb_writer
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def log_camera(tb_writer, iteration, camera):
    if camera.inpaint_image is None or camera.inpaint_depth is None:
        return
    l1_metric = l1_loss(camera.original_image, camera.inpaint_image).mean().double()
    psnr_metric = psnr(camera.original_image, camera.inpaint_image).mean().double()
    ssim_metric = ssim(camera.original_image, camera.inpaint_image)
    l1_depth_metric = l1_loss(camera.original_depth, camera.inpaint_depth).mean().double()

    tb_writer.add_scalar(f"debug_view_{camera.uid}/metrics/l1_rgb", l1_metric.item(), global_step=iteration)
    tb_writer.add_scalar(f"debug_view_{camera.uid}/metrics/l1_depth", l1_depth_metric.item(), global_step=iteration)
    tb_writer.add_scalar(f"debug_view_{camera.uid}/metrics/psnr", psnr_metric.item(), global_step=iteration)
    tb_writer.add_scalar(f"debug_view_{camera.uid}/metrics/ssim", ssim_metric.item(), global_step=iteration)

    render_pkg = vis_cam2grid(camera, save=False)
    tb_writer.add_images(f"debug_view_{camera.uid}/vis", np.asarray(render_pkg["grid"]) / 255., global_step=iteration, dataformats="HWC")

def training_report(tb_writer, iteration, curr_camera, Ll1, ll1_depth, ssim, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalars('train_loss_patches', {"l1_rgb": Ll1.item(), "l1_depth": ll1_depth.item(), "ssim": ssim.item(), "total": loss.item()}, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()},
                              {'name': 'virtual', 'cameras' : np.random.choice(scene.getTrainCameras("virtual"), min(len(scene.getTrainCameras("virtual")), 20), replace=False).tolist()})

        for config in validation_configs:
            debug_dir = os.path.join(tb_writer.log_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                l1_test_depth = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)

                    # metrics
                    image_th = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    if viewpoint.inpaint_image is not None:
                        gt_image_th = viewpoint.inpaint_image
                    else:
                        gt_image_th = viewpoint.original_image
                    gt_image_th = torch.clamp(gt_image_th, 0.0, 1.0).to("cuda")

                    depth_th = render_pkg["render_depth"]
                    if viewpoint.inpaint_depth is not None:
                        gt_depth_th = viewpoint.inpaint_depth
                    else:
                        gt_depth_th = viewpoint.original_depth
                    gt_depth_th = gt_depth_th.to("cuda")
                    l1_test += l1_loss(image_th, gt_image_th).mean().double()
                    psnr_test += psnr(image_th, gt_image_th).mean().double()
                    l1_test_depth += l1_loss(depth_th, gt_depth_th).mean().double()

                    # visualizations
                    gt_depth = gt_depth_th.squeeze().cpu().numpy() # HW
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).cpu().permute(1, 2, 0).numpy() # HW3
                    gt_image = torch.clamp(gt_image_th, 0.0, 1.0).permute(1, 2, 0).cpu().numpy() # HW3
                    depth = render_pkg["render_depth"].squeeze().cpu().numpy()  # HW

                    # normalize depth wrt gt_depth
                    min_gt, max_gt = gt_depth.min(), gt_depth.max()
                    depth = (depth - min_gt) / (max_gt - min_gt)
                    depth = np.clip(depth, 0, 1)
                    gt_depth = (gt_depth - min_gt) / (max_gt - min_gt)
                    depth_l1_error = np.abs(gt_depth - depth)
                    depth_l1_error = np.clip(depth_l1_error, np.percentile(depth_l1_error, 5), np.percentile(depth_l1_error, 95))

                    # apply colormaps
                    gt_depth = plt.cm.turbo(gt_depth)
                    depth = plt.cm.turbo(depth)
                    depth_l1_error = plt.cm.RdPu(depth_l1_error)

                    # if tb_writer and (idx < 5):
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(getattr(viewpoint, "rank", viewpoint.image_name)), image, global_step=iteration, dataformats="HWC")
                    tb_writer.add_images(config['name'] + "_view_{}/render_depth".format(getattr(viewpoint, "rank", viewpoint.image_name)), depth, global_step=iteration, dataformats="HWC")
                    tb_writer.add_images(config['name'] + "_view_{}/depth_abs_error".format(getattr(viewpoint, "rank", viewpoint.image_name)), depth_l1_error, global_step=iteration, dataformats="HWC")
                    tb_writer.add_images(config['name'] + "_view_{}/depth_abs_error".format(getattr(viewpoint, "rank", viewpoint.image_name)), depth_l1_error, global_step=iteration, dataformats="HWC")
                    # add ground-truth with always same global_step
                    tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(getattr(viewpoint, "rank", viewpoint.image_name)), gt_image, global_step=0, dataformats="HWC")
                    tb_writer.add_images(config['name'] + "_view_{}/ground_truth_depth".format(getattr(viewpoint, "rank", viewpoint.image_name)), gt_depth, global_step=0, dataformats="HWC")

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                l1_test_depth /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {:4f} PSNR {:4f} L1 Depth {:4f}".format(iteration, config['name'], l1_test, psnr_test, l1_test_depth))
                if tb_writer:
                    tb_writer.add_scalars('loss_viewpoint/l1_loss', {config['name']: l1_test}, iteration)
                    tb_writer.add_scalars('loss_viewpoint/psnr', {config['name']: psnr_test}, iteration)
                    tb_writer.add_scalars('loss_viewpoint/l1_loss_depth', {config['name']: l1_test_depth}, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000, 4_000, 7_000, 10_000, 15_000, 
                                                                           18_000, 21_000, 25_000, 28_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
