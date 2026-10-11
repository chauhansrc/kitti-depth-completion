"""Helpers for detailed evaluation (used by colab_eval.ipynb).

Predictions are kept in memory as (N, H, W) float32 arrays: for the ~300 validation
frames at 352x1216 that is ~0.5 GB per array, which is fine on Colab.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .data import IMAGENET_MEAN, IMAGENET_STD, find_samples, read_depth, read_rgb
from .model import DepthCompletionNet

METRICS = ["rmse", "mae", "irmse", "imae", "delta1"]
MIN_DEPTH = 0.9


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = DepthCompletionNet(args["modality"], args["layers"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def val_samples(ckpt, data_root):
    """The held-out frames this checkpoint was validated on."""
    names = set(ckpt["val_names"])
    return [s for s in find_samples(data_root) if s["name"] in names]


class LidarSubsample(Dataset):
    """Keeps a random fraction of the LiDAR points in each frame (same points every time for a given seed)."""

    def __init__(self, base, keep, seed=0):
        self.base, self.keep, self.seed = base, keep, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.keep < 1.0:
            rng = np.random.default_rng(self.seed + idx)
            mask = torch.from_numpy(rng.random(tuple(item["sparse"].shape)) < self.keep)
            item["sparse"] = item["sparse"] * mask
        return item


@torch.no_grad()
def predict(model, dataset, device, batch_size=4, num_workers=2, amp=True):
    """Run the model over a dataset. Returns {"pred", "gt", "sparse"}: (N,H,W) arrays in meters, plus "names"."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    out = {"pred": [], "gt": [], "sparse": [], "names": []}
    model.eval()
    for batch in tqdm(loader, desc="predict", leave=False):
        rgb = batch["rgb"].to(device, non_blocking=True)
        sparse = batch["sparse"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            pred = model(rgb, sparse)
        out["pred"].append(pred.float().cpu().numpy()[:, 0])
        out["gt"].append(batch["gt"].numpy()[:, 0])
        out["sparse"].append(batch["sparse"].numpy()[:, 0])
        out["names"] += list(batch["name"])
    for k in ("pred", "gt", "sparse"):
        out[k] = np.concatenate(out[k])
    return out


# ---------------------------------------------------------------- classical baseline

def nearest_fill(sparse):
    """No-learning baseline: every pixel takes the depth of its nearest LiDAR point."""
    valid = sparse > 0
    if not valid.any():
        return np.full_like(sparse, MIN_DEPTH)
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return np.maximum(sparse[iy, ix], MIN_DEPTH)


def nearest_fill_all(sparse_stack):
    return np.stack([nearest_fill(s) for s in sparse_stack])


def lidar_distance(sparse):
    """Distance in pixels from each pixel to the nearest LiDAR point (0 on the points themselves)."""
    return ndimage.distance_transform_edt(sparse <= 0).astype(np.float32)


# ---------------------------------------------------------------- metrics

def frame_metrics(pred, gt):
    """Same definitions as s2d.metrics.DepthMetrics, for one HxW frame."""
    m = gt > 0
    p, g = pred[m].astype(np.float64), gt[m].astype(np.float64)
    err = p - g
    inv_err = 1.0 / p - 1.0 / g
    return {
        "rmse": np.sqrt((err ** 2).mean()) * 1000.0,
        "mae": np.abs(err).mean() * 1000.0,
        "irmse": np.sqrt((inv_err ** 2).mean()) * 1000.0,
        "imae": np.abs(inv_err).mean() * 1000.0,
        "delta1": (np.maximum(p / g, g / p) < 1.25).mean(),
    }


def per_frame_table(pred, gt, names):
    """One row per frame with GT. Average the metric columns to get KITTI-style (per-image) numbers."""
    rows = []
    for i, (p, g, n) in enumerate(zip(pred, gt, names)):
        if (g > 0).any():
            rows.append({"idx": i, "name": n, "drive": n.split("_sync")[0] + "_sync", **frame_metrics(p, g)})
    return pd.DataFrame(rows)


def binned_errors(pred, gt, values, edges, label):
    """Pool GT pixels from all frames and report RMSE / MAE (mm) per bin of `values` (same shape as gt)."""
    m = gt > 0
    p, g, v = pred[m], gt[m], values[m]
    idx = np.digitize(v, edges) - 1
    rows = []
    for i in range(len(edges) - 1):
        sel = idx == i
        if not sel.any():
            continue
        err = (p[sel] - g[sel]).astype(np.float64)
        rows.append({
            label: f"{edges[i]:g}-{edges[i + 1]:g}",
            "pixels_%": 100.0 * sel.mean(),
            "rmse": np.sqrt((err ** 2).mean()) * 1000.0,
            "mae": np.abs(err).mean() * 1000.0,
        })
    return pd.DataFrame(rows)


def spatial_mae(pred, gt):
    """Mean absolute error (m) at each pixel position over all frames; NaN where GT never exists."""
    m = gt > 0
    err_sum = np.where(m, np.abs(pred - gt), 0.0).sum(0)
    count = m.sum(0)
    return np.where(count > 0, err_sum / np.maximum(count, 1), np.nan), count


def row_mae(pred, gt):
    """Mean absolute error (m) per image row, pooled over frames and columns."""
    m = gt > 0
    err_sum = np.where(m, np.abs(pred - gt), 0.0).sum(axis=(0, 2))
    count = m.sum(axis=(0, 2))
    return np.where(count > 0, err_sum / np.maximum(count, 1), np.nan)


# ---------------------------------------------------------------- speed

@torch.no_grad()
def benchmark(model, device, height=352, width=1216, batch_size=1, iters=50, warmup=10, amp=True):
    rgb = torch.randn(batch_size, 3, height, width, device=device)
    sparse = torch.rand(batch_size, 1, height, width, device=device) * 80
    model.eval()
    use_amp = amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            model(rgb, sparse)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000.0
    result = {"batch_size": batch_size, "ms_per_batch": ms, "frames_per_s": batch_size * 1000.0 / ms}
    if device.type == "cuda":
        result["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    return result


# ---------------------------------------------------------------- KITTI test set

def _test_dir(data_root):
    for c in (os.path.join(data_root, "depth_selection", "test_depth_completion_anonymous"),
              os.path.join(data_root, "test_depth_completion_anonymous")):
        if os.path.isdir(os.path.join(c, "image")):
            return c
    raise FileNotFoundError(f"test_depth_completion_anonymous not found under {data_root}")


@torch.no_grad()
def predict_test_set(model, data_root, out_dir, device, amp=True):
    """Write KITTI-format predictions (uint16 PNG, depth * 256) for the 1000 anonymous test frames."""
    test_dir = _test_dir(data_root)
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(os.listdir(os.path.join(test_dir, "image")))
    model.eval()
    for name in tqdm(names, desc="test set"):
        rgb = read_rgb(os.path.join(test_dir, "image", name))
        sparse = read_depth(os.path.join(test_dir, "velodyne_raw", name))
        h, w = sparse.shape
        if h % 16 or w % 16:
            raise ValueError(f"{name}: size {h}x{w} is not divisible by 16")
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(device)
        sparse_t = torch.from_numpy(sparse)[None, None].to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            pred = model(rgb_t, sparse_t)
        depth = np.clip(pred.float()[0, 0].cpu().numpy() * 256.0, 0, 65535).astype(np.uint16)
        Image.fromarray(depth).save(os.path.join(out_dir, name))
    return len(names)
