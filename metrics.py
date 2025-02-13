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

from pathlib import Path
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, mse
from argparse import ArgumentParser, Namespace
from submodules._360monodepth.code.python.src.utility.metrics import abs_rel_error, lin_rms_sq_error, delta_inlier_ratio
from submodules._360monodepth.code.python.src.utility.depthmap_utils import read_dpt

from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance

T_THRESH = float(os.getenv("T_THRESH", 0.05))
lpips_alex = lpips.LPIPS(net='alex').to("cuda")

def readImages(renders_dir, gt_dir):
    renders = {"rgb": [], "depth": []}
    gts = {"rgb": [], "depth": []}
    image_names = {"rgb": [], "depth": []}
    files = sorted(os.listdir(renders_dir))
    for fname in files:
        if not os.path.splitext(fname)[-1] in [".npy", ".dpt", ".png", ".jpg"]:
            continue
        if "depth.npy" in fname:
            k = "depth"
            load_fn = [np.load, read_dpt]
        elif "rgb" in fname or "frame" in fname:
            k = "rgb"
            load_fn = [Image.open]
        else:
            continue
        render = load_fn[0](renders_dir / fname)
        try:
            gt = load_fn[0](gt_dir / fname)
        except:
            gt = load_fn[1](gt_dir / fname.replace(".npy", ".dpt"))
        renders[k].append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts[k].append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names[k].append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            cfgfilepath = os.path.join(scene_dir, "cfg_args")
            with open(cfgfilepath) as cfg_file:
                print("Config file found: {}".format(cfgfilepath))
                cfgfile_dict = vars(eval(cfg_file.read()))
            
            gt_test_dir = Path(os.path.dirname(cfgfile_dict.get("ground_truth", "")))
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt" if not os.path.isdir(gt_test_dir) else gt_test_dir
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                errormaps_dir = method_dir / "errors"
                os.makedirs(errormaps_dir, exist_ok=True)

                ssims = []
                psnrs = []
                lpipss = []
                abs_rel = []
                rms = []
                delta_1 = []
                delta_2 = []
                delta_3 = []

                for idx in tqdm(range(len(renders['rgb'])), desc="Metric evaluation progress"):
                    render = renders['rgb'][idx].to("cuda")
                    gt = gts['rgb'][idx].to("cuda")
                    ssims.append(ssim(render, gt).cpu().item())
                    psnrs.append(psnr(render, gt).cpu().item())
                    lpipss.append(lpips_alex(render, gt).cpu().item())
                    if not os.path.islink(os.path.join(errormaps_dir, f"{idx:05}_rgb_cgt.png")):
                        os.symlink(os.path.abspath(os.path.join(gt_dir, f"{idx:05}_rgb.png")), os.path.join(errormaps_dir, f"{idx:05}_rgb_cgt.png"))
                    if not os.path.islink(os.path.join(errormaps_dir, f"{idx:05}_rgb_cpred.png")):
                        os.symlink(os.path.abspath(os.path.join(renders_dir, f"{idx:05}_rgb.png")), os.path.join(errormaps_dir, f"{idx:05}_rgb_cpred.png"))
                    # plt.figure()
                    # plt.imshow(mse(render, gt, True).squeeze().mean(axis=0).cpu().numpy(), cmap="RdPu_r")
                    # plt.colorbar()
                    # plt.savefig(os.path.join(errormaps_dir, f"{idx:05}_rgb_mse.png"))
                    # plt.close("all")
                    # plt.figure()
                    # plt.imshow(ssim(render, gt, return_map=True).squeeze().mean(axis=0).cpu().numpy(), cmap="RdPu")
                    # plt.colorbar()
                    # plt.savefig(os.path.join(errormaps_dir, f"{idx:05}_rgb_ssim.png"))
                    # plt.close("all")
                
                for idx in tqdm(range(len(renders['depth'])), desc="Depth metric evaluation progress"):
                    pred = renders['depth'][idx].cpu().numpy()
                    gt = gts['depth'][idx].cpu().numpy()
                    mask = gt > 0
                    abs_rel.append(abs_rel_error(pred, gt, mask))
                    rms.append(lin_rms_sq_error(pred, gt, mask))
                    delta_1.append(delta_inlier_ratio(pred, gt, mask, 1))
                    delta_2.append(delta_inlier_ratio(pred, gt, mask, 2))
                    delta_3.append(delta_inlier_ratio(pred, gt, mask, 3))

                print("PSNR, SSIM, LPIPS, ABS REL, RMS, delta1, delta2, delta3")
                print(f"{torch.tensor(psnrs).mean().item()}, {torch.tensor(ssims).mean().item()}, {torch.tensor(lpipss).mean().item()}, {torch.tensor(abs_rel).mean().item()}, {torch.tensor(rms).mean().item()}, {torch.tensor(delta_1).mean().item()}, {torch.tensor(delta_2).mean().item()}, {torch.tensor(delta_3).mean().item()}")
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "ABSREL": torch.tensor(abs_rel).mean().item(),
                                                        "RMS": torch.tensor(rms).mean().item(),
                                                        "delta1": torch.tensor(delta_1).mean().item(),
                                                        "delta2": torch.tensor(delta_2).mean().item(),
                                                        "delta3": torch.tensor(delta_3).mean().item(),
                                                        })
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names['rgb'])},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names['rgb'])},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names['rgb'])},
                                                            "ABSREL": {name: lp for lp, name in zip(abs_rel, image_names['depth'])},
                                                            "RMS": {name: lp for lp, name in zip(rms, image_names['depth'])},
                                                            "delta1": {name: lp for lp, name in zip(delta_1, image_names['depth'])},
                                                            "delta2": {name: lp for lp, name in zip(delta_2, image_names['depth'])},
                                                            "delta3": {name: lp for lp, name in zip(delta_3, image_names['depth'])}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

def metrics_on_the_fly(scene, pipe, render_fn):
    output_file = os.path.join(scene.model_path, f"ablation_camera_weights_{scene.current_iter}.json")
    metrics = {"train": {}, "virtual": {}, "test": {}}
    for cam_set, split in zip([scene.getTrainCameras(), scene.getTrainCameras("virtual"), scene.getTestCameras()], ["train", "virtual", "test"]):
        ssims, psnrs, lpipss, abs_rel, rms, delta_1, delta_2, delta_3 = [], [], [], [], [], [], [], []
        for camera in tqdm(cam_set, desc=f"{split} metrics"):
            with torch.no_grad():
                bg_color = torch.tensor([1., 0, 0], device=scene.gaussians._opacity.device)
                render_pkg = render_fn(camera, scene.gaussians, pipe, bg_color)
            gt_rgb = camera.original_image if camera.inpaint_image is None else camera.inpaint_image
            gt_rgb = gt_rgb[None]
            pred_rgb = render_pkg["render"][None]
            ssims.append(ssim(pred_rgb, gt_rgb).item())
            psnrs.append(psnr(pred_rgb, gt_rgb).item())
            lpipss.append(lpips(pred_rgb, gt_rgb, net_type='vgg').item())

            # depth
            gt_depth = camera.original_depth if camera.inpaint_depth is None else camera.inpaint_depth
            gt_depth = gt_depth.cpu().numpy()
            pred_depth = render_pkg["render_depth"].cpu().numpy()
            mask = np.ones_like(gt_depth, dtype=bool)
            abs_rel.append(abs_rel_error(pred_depth, gt_depth, mask))
            rms.append(lin_rms_sq_error(pred_depth, gt_depth, mask))
            delta_1.append(delta_inlier_ratio(pred_depth, gt_depth, mask, 1).item())
            delta_2.append(delta_inlier_ratio(pred_depth, gt_depth, mask, 2).item())
            delta_3.append(delta_inlier_ratio(pred_depth, gt_depth, mask, 3).item())

        metrics[split].update({"SSIM": torch.tensor(ssims).mean().item(),
                                "PSNR": torch.tensor(psnrs).mean().item(),
                                "LPIPS": torch.tensor(lpipss).mean().item(),
                                "ABSREL": torch.tensor(abs_rel).mean().item(),
                                "RMS": torch.tensor(rms).mean().item(),
                                "delta1": torch.tensor(delta_1).mean().item(),
                                "delta2": torch.tensor(delta_2).mean().item(),
                                "delta3": torch.tensor(delta_3).mean().item(),
                                })

    with open(output_file, 'w') as fp:
                json.dump(metrics, fp, indent=True)


def inpainting_metrics(path, train_cameras, test_cameras):
    import random

    if len(train_cameras) > len(test_cameras):
        train_cameras = random.sample(train_cameras, k=len(test_cameras))
    if len(test_cameras) > len(train_cameras):
        test_cameras = random.sample(test_cameras, k=len(train_cameras))

    # metrics
    FID = FrechetInceptionDistance(normalize=True).to(train_cameras[0].original_image)
    KID = KernelInceptionDistance(normalize=True, subset_size=5).to(train_cameras[0].original_image)

    inpainted_images = []
    for train, test in zip(train_cameras, test_cameras):
        mask = (test.original_mask > T_THRESH).type(torch.float32)
        inpainted_images.append(test.inpaint_image * mask + train.original_image * (1 - mask))

    train_images = torch.stack([c.original_image for c in train_cameras])
    inpainted_images = torch.stack(inpainted_images)
    KID.update(train_images, real=True)
    KID.update(inpainted_images, real=False)
    FID.update(train_images, real=True)
    FID.update(inpainted_images, real=False)

    fid = FID.compute()
    meank, stdk = KID.compute()
    with open(os.path.join(path, "results.txt"), "w") as f:
        f.write(f"FID: {fid.item():.4f}\n")
        f.write(f"KID: {meank.item():.4f}, {stdk.item():.4f}\n")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
