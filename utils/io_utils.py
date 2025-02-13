import os
import torch
import numpy as np

from PIL import Image
from torchvision.utils import make_grid

from scene.cameras import Camera
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def save_visuals(savepath, camera: Camera, icam):
    mask = camera.original_mask.cpu()
    rgb = camera.original_image.cpu()
    rgb *= (1 - mask)
    inp_rgb = camera.inpaint_image.cpu()
    depth = camera.original_depth.cpu()
    mind, maxd = depth.min().item(), depth.max().item()
    inp_depth = camera.inpaint_depth.cpu()
    
    # grid rgb, filled rgb
    grid_rgb = make_grid([rgb, inp_rgb], nrow=2)
    grid_rgb = grid_rgb.permute(1, 2, 0).numpy()
    # grid depth, filled depth
    grid_depth = make_grid([depth[None], inp_depth[None]], nrow=2)
    grid_depth = grid_depth.permute(1, 2, 0).numpy()
    grid_depth = (grid_depth - mind) / (maxd - mind)
    grid_depth = plt.cm.turbo(grid_depth[..., 0])

    # normalize to 0,1 for saving purposes
    depth = (depth - mind) / (maxd - mind)
    inp_depth = torch.clamp((inp_depth - mind) / (maxd - mind), 0, 1)
    depth = plt.cm.turbo(depth)  # This converts to numpy
    inp_depth = plt.cm.turbo(inp_depth)  # This converts to numpy

    # grid_mgt = make_grid([rgb.cpu(), mask.expand(3, -1, -1).cpu()], nrow=2)
    # grid_mgt = grid_mgt.permute(1, 2, 0).numpy()

    Image.fromarray((grid_rgb*255).astype(np.uint8)).save(os.path.join(savepath, f"{icam:05d}_grid_rgb.png"))
    Image.fromarray((grid_depth*255).astype(np.uint8)).save(os.path.join(savepath, f"{icam:05d}_grid_depth.png"))
    Image.fromarray((rgb.permute(1, 2, 0).numpy()*255).astype(np.uint8)).save(os.path.join(savepath, f"{icam:05d}_rgb_gt.png"))
    Image.fromarray((inp_rgb.permute(1, 2, 0).numpy()*255).astype(np.uint8)).save(os.path.join(savepath, f"{icam:05d}_rgb_pred.png"))
    Image.fromarray((depth*255).astype(np.uint8)).save(os.path.join(savepath, f"{icam:05d}_depth_gt.png"))
    Image.fromarray((inp_depth*255).astype(np.uint8)).save(os.path.join(savepath, f"{icam:05d}_depth_pred.png"))
    Image.fromarray((mask.numpy()*255).astype(np.uint8)).save(os.path.join(savepath, f"{icam:05d}_mask.png"))

def vis_cam2grid(camera:Camera, path=None, iter=None, save=True):
    if save and path is None:
        raise "Provide a path to save visualizations"

    if path is not None:
        os.makedirs(path, exist_ok=True)
    
    depth = camera.original_depth.cpu().squeeze().numpy()
    acc = np.zeros_like(depth)[..., 0] if camera.inpaint_mask is None else camera.inpaint_mask.cpu().squeeze().numpy()
    depth_holes = (depth * (1-acc))
    depth_inp = np.zeros_like(depth) if camera.inpaint_depth is None else camera.inpaint_depth.squeeze().cpu().numpy()

    depth_min, depth_max = None, None
    if camera.inpaint_depth is not None:
        depth_min, depth_max = camera.inpaint_depth.min(), camera.inpaint_depth.max()

    norm = Normalize(vmin=depth_min, vmax=depth_max)
    depth = plt.cm.turbo(norm(depth))[..., :3]
    depth_inp = plt.cm.turbo(norm(depth_inp))[..., :3]
    depth_holes = plt.cm.turbo(norm(depth_holes))[..., :3]

    acc = plt.cm.binary(Normalize(vmin=0, vmax=0.7)(acc))[..., :3]

    rgb = getattr(camera, "original_image_noBG", camera.original_image).cpu().permute(1, 2, 0).numpy()
    rgb_holes = rgb * acc
    rgb_inp = np.zeros_like(rgb) if camera.inpaint_image is None else camera.inpaint_image.permute(1, 2, 0).cpu().numpy()

    if iter is None:
        outpath = os.path.join(path, f"{camera.uid:03}_grid.png")
    else:
        outpath = os.path.join(path, f"{camera.uid:03}_iter{iter:04}_grid.png")

    grid = Image.new('RGB', size=(rgb.shape[0]*5, rgb.shape[1]))

    attrs = {"rgb": rgb, "rgb_inp": rgb_inp,
             "depth": depth, "depth_inp": depth_inp,
             "acc": acc, "grid": grid}
    if not save:
        return attrs
    
    if iter is None:
        iter = 0
    uid = camera.uid if not hasattr(camera, "rank") else int(float(camera.rank))
    Image.fromarray((acc*255).astype(np.uint8)).save(os.path.join(path, f"{uid:03}_acc.png"))
    Image.fromarray((rgb_holes*255).astype(np.uint8)).save(os.path.join(path, f"{uid:03}_rgb_holes.png"))
    Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(path, f"{uid:03}_rgb_{iter:04}.png"))
    Image.fromarray((rgb_inp*255).astype(np.uint8)).save(os.path.join(path, f"{uid:03}_rgb_inp.png"))
    Image.fromarray((depth_holes*255).astype(np.uint8)).save(os.path.join(path, f"{uid:03}_depth_holes.png"))
    Image.fromarray((depth*255).astype(np.uint8)).save(os.path.join(path, f"{uid:03}_depth_{iter:04}.png"))
    Image.fromarray((depth_inp*255).astype(np.uint8)).save(os.path.join(path, f"{uid:03}_depth_inp.png"))

    # grid.save(outpath)
