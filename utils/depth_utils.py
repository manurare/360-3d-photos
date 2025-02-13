from transformers import AutoModelForDepthEstimation, AutoImageProcessor
import torch
import numpy as np
from submodules._360monodepth.code.python.src.utility import depthmap_utils
import cv2
from utils.poisson_utils import find_boundaries, get_surrounding

device = "cpu" if not torch.cuda.is_available else "cuda"

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", do_rescale=False)
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)

@torch.no_grad()
def predict_depth(image):
    size = image.shape[2:]
    image = image_processor(image, return_tensors="pt")["pixel_values"].to(image.device)
    outputs = model(image)
    predicted_depth = outputs.predicted_depth
    predicted_depth = 1 / (predicted_depth + 1e-6)

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=size,
        mode="nearest-exact",
    )

    return prediction

def pred2gt_least_squares(pred, gt, mask):
    gt = depthmap_utils.depth2disparity(gt)
    pred = depthmap_utils.depth2disparity(pred)
    a_00 = np.sum(pred[mask] * pred[mask])
    a_01 = np.sum(pred[mask])
    a_11 = np.sum(mask)

    b_0 = np.sum(pred[mask] * gt[mask])
    b_1 = np.sum(gt[mask])

    det = a_00 * a_11 - a_01 * a_01

    s = (a_11 * b_0 - a_01 * b_1) / det
    o = (-a_01 * b_0 + a_00 * b_1) / det

    pred = s * pred + o
    pred = depthmap_utils.disparity2depth(pred)
    return pred

def push_depth_behind(depth, mask):
    H,W = depth.squeeze().shape
    device = depth.device
    if isinstance(depth, torch.Tensor):
        depth_og = depth.cpu().numpy()
        new_depth = np.copy(depth_og)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(np.uint8)

    output = cv2.connectedComponentsWithStats(mask, 4)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    for i in range(1, num_labels):
        blob = np.where(labels == i, 1, 0)
        blob_coords = blob.nonzero()
        inner_ring = find_boundaries(blob, mode="inner")
        inner_ring_coords = np.array(inner_ring.nonzero()).T
        outer_ring = find_boundaries(blob, mode="outer")
        outer_ring = (outer_ring.astype(int) - inner_ring.astype(int)) > 0
        outer_ring_coords = np.array(outer_ring.nonzero()).T
        offsets = []
        for coord in outer_ring_coords:
            nes = np.array(get_surrounding(coord, H, W))
            local_offsets = []
            for ne in nes:
                if np.any(np.all(ne == inner_ring_coords, axis=1)):
                    local_offsets.append(np.abs(depth_og[coord[0], coord[1]] - depth_og[ne[0], ne[1]]))
            offsets.append(max(local_offsets))
        max_offset = min(offsets)
        new_depth[blob_coords] += max_offset

    new_depth = torch.from_numpy(new_depth).to(device, torch.float32)
    return new_depth

def clip_depth_holes(depth, mask):
    H,W = depth.squeeze().shape
    device = depth.device
    if mask.sum() == 0:
        return depth
    if isinstance(depth, torch.Tensor):
        depth_og = depth.cpu().numpy()
        new_depth = np.copy(depth_og)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    output = cv2.connectedComponentsWithStats(mask, 4)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    for i in range(1, num_labels):
        blob = np.where(labels == i, 1, 0)
        blob_coords = blob.nonzero()
        inner_ring = find_boundaries(blob, mode="inner")
        outer_ring = find_boundaries(blob, mode="outer")
        outer_ring = (outer_ring.astype(int) - inner_ring.astype(int)) > 0
        min_depth = np.min(depth_og[outer_ring.nonzero()])
        new_depth[blob_coords] = np.clip(new_depth[blob_coords], min_depth, None)

    new_depth = torch.from_numpy(new_depth).to(device, torch.float32)
    return new_depth

if __name__ == "__main__":
    image = torch.load("utils/in.pt").to(device)
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = np.asarray(Image.open(requests.get(url, stream=True).raw)) / 255.
    # image = image_processor(images=image, return_tensors="pt")
    depth = predict_depth(image)
    print("end")