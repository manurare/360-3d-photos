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
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import shutil
import time

render_times = []

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    grids_path = os.path.join(model_path, name, "ours_{}".format(iteration), "grids")

    shutil.rmtree(render_path, ignore_errors=True)
    shutil.rmtree(gts_path, ignore_errors=True)
    shutil.rmtree(grids_path, ignore_errors=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(grids_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering progress {name}")):
        begin = time.time()
        render_pkg = render(view, gaussians, pipeline, background)
        render_times.append(time.time() - begin)
        rendering = render_pkg["render"]

        if view.original_image is None:
            gt = torch.zeros_like(rendering)[0:3, :, :]
        else:
            gt = view.original_image[0:3, :, :]
        gt = gt.clamp(0,1).permute(1,2,0).cpu().numpy()
        rendering = rendering.clamp(0,1).permute(1,2,0).cpu().numpy()
        plt.imsave(os.path.join(render_path, '{0:05d}'.format(idx) + "_rgb.png"), rendering)
        plt.imsave(os.path.join(gts_path, '{0:05d}'.format(idx) + "_rgb.png"), gt)

        # Depth
        rendering_depth = render_pkg["render_depth"].squeeze()
        if view.original_depth is None:
            gt_depth = torch.zeros_like(rendering_depth)
            minv, maxv = rendering_depth.min(), rendering_depth.max()
        else:
            gt_depth = view.original_depth
            minv, maxv = gt_depth.min(), gt_depth.max()

        # save float depth for metrics
        np.save(os.path.join(render_path, '{0:05d}_depth'.format(idx) + ".npy"), render_pkg["render_depth"].squeeze().cpu().numpy())
        np.save(os.path.join(gts_path, '{0:05d}_depth'.format(idx) + ".npy"), gt_depth.cpu().numpy())
        np.save(os.path.join(render_path, '{0:05d}_mask'.format(idx) + ".npy"), render_pkg["render_acc"].cpu().numpy())

        norm = Normalize(minv, maxv)
        gt_depth = plt.cm.turbo(norm(gt_depth.cpu().numpy()))[..., :3]
        rendering_depth = plt.cm.turbo(np.clip(norm(rendering_depth.cpu().numpy()), 0, 1))[..., :3]
        plt.imsave(os.path.join(render_path, '{0:05d}_depth_vis'.format(idx) + ".png"), rendering_depth)
        plt.imsave(os.path.join(gts_path, '{0:05d}_depth_vis'.format(idx) + ".png"), gt_depth)
        
        # save grid
        gt_grid = Image.new('RGB', size=(rendering.shape[0]*2, rendering.shape[1]))
        gt_grid.paste(Image.fromarray((gt*255).astype(np.uint8)), box=(gt.shape[1]*0, 0))
        gt_grid.paste(Image.fromarray((gt_depth*255).astype(np.uint8)), box=(gt.shape[1]*1, 0))
        pred_grid = Image.new('RGB', size=(rendering.shape[0]*2, rendering.shape[1]))
        pred_grid.paste(Image.fromarray((rendering*255).astype(np.uint8)), box=(gt.shape[1]*0, 0))
        pred_grid.paste(Image.fromarray((rendering_depth*255).astype(np.uint8)), box=(gt.shape[1]*1, 0))
        gt_grid.save(os.path.join(grids_path, '{0:05d}'.format(idx) + "_gt_grid.png"))
        pred_grid.save(os.path.join(grids_path, '{0:05d}'.format(idx) + "_pred_grid.png"))

    # RGB video
    files_rgb = os.path.join(render_path, "*_rgb.png")
    output_rgb = os.path.join(render_path, "out_rgb.mp4")
    os.system(f"ffmpeg -framerate 20 -pattern_type glob -i '{files_rgb}' -c:v libx264 -crf 0 {output_rgb}")

    # Depth video
    files_depth = os.path.join(render_path, "*_depth_vis.png")
    output_depth = os.path.join(render_path, "out_depth.mp4")
    os.system(f"ffmpeg -framerate 20 -pattern_type glob -i '{files_depth}' -c:v libx264 -crf 0 {output_depth}")

    print(np.mean(render_times))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)