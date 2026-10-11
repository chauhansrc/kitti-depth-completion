"""Visualization of RGB / sparse input / prediction / GT / error."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD


def denormalize_rgb(rgb):
    """3xHxW normalized tensor -> HxWx3 float array in [0, 1]."""
    img = rgb.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def dilate(depth, k=5):
    """Grow sparse LiDAR points so they are visible in a plot. depth: HxW array."""
    t = torch.from_numpy(depth)[None, None]
    return F.max_pool2d(t, k, stride=1, padding=k // 2)[0, 0].numpy()


def colorize(depth, vmax, cmap="magma"):
    """HxW depth -> HxWx3 colors; pixels <= 0 are black."""
    colors = plt.get_cmap(cmap)(np.clip(depth / vmax, 0.0, 1.0))[..., :3]
    colors[depth <= 0] = 0.0
    return colors


def save_panel(path, rgb, sparse, pred, gt, title="", max_depth=80.0, max_error=5.0):
    """rgb: 3xHxW tensor, sparse/pred/gt: 1xHxW tensors in meters."""
    sparse, pred, gt = sparse[0].numpy(), pred[0].numpy(), gt[0].numpy()
    error = np.where(gt > 0, np.abs(pred - gt), 0.0)

    rows = [
        ("RGB", denormalize_rgb(rgb)),
        ("Sparse LiDAR input (dilated)", colorize(dilate(sparse), max_depth)),
        ("Prediction", colorize(pred, max_depth)),
        ("Ground truth (semi-dense)", colorize(gt, max_depth)),
        (f"|error| on GT pixels (0-{max_error:g} m)", colorize(error, max_error, cmap="hot")),
    ]
    h, w = gt.shape
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, len(rows) * 12 * h / w + 1))
    for ax, (label, img) in zip(axes, rows):
        ax.imshow(img)
        ax.set_title(label, fontsize=10, loc="left")
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
