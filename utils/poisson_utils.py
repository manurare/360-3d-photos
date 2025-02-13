import scipy.sparse
import scipy.signal
from scipy.sparse.linalg import spsolve
import numpy as np
from skimage.segmentation import find_boundaries
from skimage import measure

class PoissonSolver:
    def __init__(self, mask, target=None):
        self.source = None
        self.target = target
        self.mask = self.preprocess_mask(mask).astype(bool)
        self.mask_points = np.array(np.nonzero(self.mask)).T
        self.boundary_points = None
        self.N = np.sum(self.mask) # N = number of points in mask
        self.mat_A = None
        self.b = None

        self.poisson_sparse_matrix_vec()

    def poisson_sparse_matrix_vec(self):
        H, W = self.mask.shape

        pts_indices_ref_img = np.ravel_multi_index(self.mask_points.T, (H, W))
        pts_indices_ref_mask = np.arange(len(pts_indices_ref_img))

        # Left neighbours
        good_nes_l = (self.mask_points[:, 1] - 1) >= 0
        ne_l = np.vstack((self.mask_points[:, 0], (self.mask_points[:, 1] - 1)))

        # Right neighbours
        good_nes_r = (self.mask_points[:, 1] + 1) < W
        ne_r = np.vstack((self.mask_points[:, 0], (self.mask_points[:, 1] + 1)))

        # Top neighbours
        good_nes_t = (self.mask_points[:, 0] - 1) >= 0
        ne_t = np.vstack((self.mask_points[:, 0] - 1, self.mask_points[:, 1]))

        # Bottom neighbours
        good_nes_b = (self.mask_points[:, 0] + 1) < H
        ne_b = np.vstack((self.mask_points[:, 0] + 1, self.mask_points[:, 1]))

        ne_l_indices_ref_img = np.ravel_multi_index(ne_l, (H, W), mode='clip')
        ne_r_indices_ref_img = np.ravel_multi_index(ne_r, (H, W), mode='clip')
        ne_t_indices_ref_img = np.ravel_multi_index(ne_t, (H, W), mode='clip')
        ne_b_indices_ref_img = np.ravel_multi_index(ne_b, (H, W), mode='clip')

        ne_l_in_mask_ids_ref_img = np.in1d(ne_l_indices_ref_img, pts_indices_ref_img) & good_nes_l
        ne_r_in_mask_ids_ref_img = np.in1d(ne_r_indices_ref_img, pts_indices_ref_img) & good_nes_r
        ne_t_in_mask_ids_ref_img = np.in1d(ne_t_indices_ref_img, pts_indices_ref_img) & good_nes_t
        ne_b_in_mask_ids_ref_img = np.in1d(ne_b_indices_ref_img, pts_indices_ref_img) & good_nes_b

        ne_l_in_mask_ref_mask = np.searchsorted(pts_indices_ref_img, ne_l_indices_ref_img[ne_l_in_mask_ids_ref_img])
        ne_r_in_mask_ref_mask = np.searchsorted(pts_indices_ref_img, ne_r_indices_ref_img[ne_r_in_mask_ids_ref_img])
        ne_t_in_mask_ref_mask = np.searchsorted(pts_indices_ref_img, ne_t_indices_ref_img[ne_t_in_mask_ids_ref_img])
        ne_b_in_mask_ref_mask = np.searchsorted(pts_indices_ref_img, ne_b_indices_ref_img[ne_b_in_mask_ids_ref_img])

        mat_A_ne_l = np.vstack((pts_indices_ref_mask[ne_l_in_mask_ids_ref_img], ne_l_in_mask_ref_mask))
        mat_A_ne_r = np.vstack((pts_indices_ref_mask[ne_r_in_mask_ids_ref_img], ne_r_in_mask_ref_mask))
        mat_A_ne_t = np.vstack((pts_indices_ref_mask[ne_t_in_mask_ids_ref_img], ne_t_in_mask_ref_mask))
        mat_A_ne_b = np.vstack((pts_indices_ref_mask[ne_b_in_mask_ids_ref_img], ne_b_in_mask_ref_mask))

        mat_A_vec = 4 * scipy.sparse.eye(self.N, format="coo").tolil()
        mat_A_vec[mat_A_ne_l[0], mat_A_ne_l[1]] = -1
        mat_A_vec[mat_A_ne_r[0], mat_A_ne_r[1]] = -1
        mat_A_vec[mat_A_ne_t[0], mat_A_ne_t[1]] = -1
        mat_A_vec[mat_A_ne_b[0], mat_A_ne_b[1]] = -1

        self.mat_A = mat_A_vec.tocsr()
        self.boundary_points = np.array(find_boundaries(self.mask, mode="inner").nonzero()).T


    def get_b_vector_vec(self):
        H, W, C = self.source.shape
        lap_kernel = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])

        laplacian = np.zeros_like(self.source)
        for c in range(C):
            conv = self.compute_mixed_gradients(self.source[..., c], self.target[..., c], mode="alpha")
            # conv = scipy.signal.correlate2d(self.source[..., c], lap_kernel, mode="same")
            laplacian[..., c] = conv

        for i, index in enumerate(zip(self.boundary_points[:, 0], self.boundary_points[:, 1])):
            for pt in get_surrounding(index, H, W):
                if in_omega(pt, self.mask) == False:
                    laplacian[index] += self.target[pt]

        self.b = laplacian[self.mask]

    def compute_mixed_gradients(self, source, target, mode="max", alpha=1.0):
        if mode == "max":
            Ixf_src, Iyf_src = compute_gradient(source)
            Ixf_target, Iyf_target = compute_gradient(target)
            Ixb_src, Iyb_src = compute_gradient(source, forward=False)
            Ixb_target, Iyb_target = compute_gradient(target, forward=False)
            total = np.where((np.abs(Ixf_target) > np.abs(Ixf_src)) & mask, Ixf_target, Ixf_src) + np.where((np.abs(Ixb_target) > np.abs(Ixb_src)) & mask, Ixb_target, Ixb_src) + \
                    np.where((np.abs(Iyf_target) > np.abs(Iyf_src)) & mask, Iyf_target, Iyf_src) + np.where((np.abs(Iyb_target) > np.abs(Iyb_src)) & mask, Iyb_target, Iyb_src)
            return total
        elif mode == "alpha":
            src_laplacian = compute_laplacian(source)
            target_laplacian = compute_laplacian(target)
            return alpha * src_laplacian + (1 - alpha) * target_laplacian
        else:
            raise ValueError(f"Gradient mixing mode '{mode}' not supported!")

    def preprocess_mask(self, mask):
        if self.target is None:
            return mask
        blobs, nlabels = measure.label(mask.astype(np.uint8), return_num=True)
        for b in range(nlabels):
            if np.all(self.target[blobs == b] > 0):
                mask[blobs == b] = False
        return mask

    def gradient_blending_vec(self, source, target):
        self.source = source
        self.target = target
        H, W, C = self.source.shape

        if self.mat_A is None:
            self.poisson_sparse_matrix_vec()

        self.get_b_vector_vec()

        blended_image = np.copy(self.target)

        for c in range(C):
            x_hat = spsolve(self.mat_A, self.b[:, c])
            blended_image[self.mask, c] = x_hat

        return blended_image

# Get indices above, below, to the left and right
def get_surrounding(index, H, W):
    y, x = index
    nes = []
    if 0 <= (y - 1) < H:
        nes.append((y - 1, x))
    if 0 <= (y + 1) < H: 
        nes.append((y + 1, x))
    if 0 <= (x - 1) < W:
        nes.append((y, (x - 1)))
    if 0 <= (x + 1) < W:
        nes.append((y, (x + 1)))
    
    return nes

# Determine if a given index is either outside or inside omega
def in_omega(index, mask):
    return mask[index] == 1

def compute_laplacian(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    laplacian = scipy.signal.correlate2d(img, kernel, mode="same")
    return laplacian

def compute_gradient(img, forward=True):
    if forward:
        kx = np.array([
            [0, 0, 0],
            [0, 1, -1],
            [0, 0, 0]
        ])
        ky = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, -1, 0]
        ])
    else:
        kx = np.array([
            [0, 0, 0],
            [-1, 1, 0],
            [0, 0, 0]
        ])
        ky = np.array([
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
    Gx = scipy.signal.correlate2d(img, kx, mode="same")
    Gy = scipy.signal.correlate2d(img, ky, mode="same")
    return Gx, Gy

import matplotlib.pyplot as plt
if __name__ == "__main__":
    mask = np.load("/home/manuel/Desktop/PHD/code/gaussian-splatting/utils/mask.npy")
    source_ls = np.load("/home/manuel/Desktop/PHD/code/gaussian-splatting/utils/LS_inpainted.npy")[..., None]
    source_inv = np.load("/home/manuel/Desktop/PHD/code/gaussian-splatting/utils/inv_stitch_inpainted.npy")[..., None]
    target = np.load("/home/manuel/Desktop/PHD/code/gaussian-splatting/utils/target.npy")[..., None]

    poisson_solver = PoissonSolver(mask, target)
    blended_depth_inv = poisson_solver.gradient_blending_vec(source_inv, target)
    blended_depth_ls = poisson_solver.gradient_blending_vec(source_ls, target)
    print("end")
