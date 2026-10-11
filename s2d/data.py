"""KITTI depth completion data: file discovery, drive-level split, and the Dataset.

Expected layout (from data_depth_selection.zip):

    <root>/depth_selection/val_selection_cropped/
        image/               2011_09_26_drive_0002_sync_image_0000000005_image_02.png
        velodyne_raw/        2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.png
        groundtruth_depth/   2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png

Depth PNGs are uint16; depth in meters = value / 256, and 0 marks "no measurement".
"""

import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NAME_RE = re.compile(
    r"^(?P<drive>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_image_(?P<frame>\d{10})_image_(?P<cam>0[23])\.png$"
)


def resolve_selection_dir(root):
    """Accept the download dir, the depth_selection dir, or val_selection_cropped itself."""
    candidates = [
        os.path.join(root, "depth_selection", "val_selection_cropped"),
        os.path.join(root, "val_selection_cropped"),
        root,
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "image")) and os.path.isdir(os.path.join(c, "groundtruth_depth")):
            return c
    raise FileNotFoundError(
        f"Could not find val_selection_cropped (with image/ and groundtruth_depth/) under {root}"
    )


def find_samples(root):
    """Return a sorted list of dicts with image/sparse/gt paths and the drive name."""
    sel = resolve_selection_dir(root)
    samples = []
    for fname in sorted(os.listdir(os.path.join(sel, "image"))):
        m = NAME_RE.match(fname)
        if m is None:
            continue
        drive, frame, cam = m.group("drive"), m.group("frame"), m.group("cam")
        sample = {
            "name": fname,
            "drive": drive,
            "image": os.path.join(sel, "image", fname),
            "sparse": os.path.join(sel, "velodyne_raw", f"{drive}_velodyne_raw_{frame}_image_{cam}.png"),
            "gt": os.path.join(sel, "groundtruth_depth", f"{drive}_groundtruth_depth_{frame}_image_{cam}.png"),
        }
        if os.path.exists(sample["sparse"]) and os.path.exists(sample["gt"]):
            samples.append(sample)
    if not samples:
        raise RuntimeError(f"No complete samples found in {sel}")
    return samples


def split_samples(samples, val_fraction=0.2, seed=0, by_drive=True):
    """Split into train/val.

    Consecutive frames of one drive are nearly identical, so a random frame split
    leaks val scenes into training. By default whole drives go to val until
    roughly `val_fraction` of the frames are held out.
    """
    rng = random.Random(seed)
    drives = sorted({s["drive"] for s in samples})

    if not by_drive or len(drives) < 2:
        idx = list(range(len(samples)))
        rng.shuffle(idx)
        n_val = int(round(val_fraction * len(samples)))
        val_idx = set(idx[:n_val])
        train = [s for i, s in enumerate(samples) if i not in val_idx]
        val = [s for i, s in enumerate(samples) if i in val_idx]
        return train, val

    counts = {d: 0 for d in drives}
    for s in samples:
        counts[s["drive"]] += 1
    rng.shuffle(drives)
    target = val_fraction * len(samples)
    val_drives, n_val = set(), 0
    for d in drives:
        if n_val >= target:
            break
        val_drives.add(d)
        n_val += counts[d]
    train = [s for s in samples if s["drive"] not in val_drives]
    val = [s for s in samples if s["drive"] in val_drives]
    return train, val


def read_depth(path):
    depth = np.asarray(Image.open(path), dtype=np.float32) / 256.0
    return depth


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def color_jitter(rgb, strength):
    """Random brightness, contrast and saturation on an HxWx3 float image in [0, 1]."""
    rgb = rgb * random.uniform(1 - strength, 1 + strength)
    mean = rgb.mean()
    rgb = (rgb - mean) * random.uniform(1 - strength, 1 + strength) + mean
    gray = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    rgb = (rgb - gray) * random.uniform(1 - strength, 1 + strength) + gray
    return np.clip(rgb, 0.0, 1.0)


def crop_to_multiple(arrays, multiple=16):
    """Keep the bottom rows / left columns so H and W divide `multiple` (LiDAR covers the bottom)."""
    h, w = arrays[0].shape[:2]
    nh, nw = h - h % multiple, w - w % multiple
    return [a[h - nh:, :nw] for a in arrays]


class KittiDepthSelection(Dataset):
    """Returns normalized RGB (3xHxW), sparse LiDAR depth (1xHxW) and GT depth (1xHxW) in meters."""

    def __init__(self, samples, train, crop_size=(256, 1024), jitter=0.2):
        self.samples = samples
        self.train = train
        self.crop_h, self.crop_w = crop_size
        self.jitter = jitter
        if train and (self.crop_h % 16 or self.crop_w % 16):
            raise ValueError("crop size must be divisible by 16")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rgb = read_rgb(s["image"])
        sparse = read_depth(s["sparse"])
        gt = read_depth(s["gt"])

        if self.train:
            h, w = gt.shape
            ch, cw = min(self.crop_h, h - h % 16), min(self.crop_w, w - w % 16)
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            rgb = rgb[top:top + ch, left:left + cw]
            sparse = sparse[top:top + ch, left:left + cw]
            gt = gt[top:top + ch, left:left + cw]
            if random.random() < 0.5:
                rgb, sparse, gt = rgb[:, ::-1], sparse[:, ::-1], gt[:, ::-1]
            if self.jitter > 0:
                rgb = color_jitter(rgb, self.jitter)
        else:
            rgb, sparse, gt = crop_to_multiple([rgb, sparse, gt])

        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
            "sparse": torch.from_numpy(np.ascontiguousarray(sparse))[None],
            "gt": torch.from_numpy(np.ascontiguousarray(gt))[None],
            "index": idx,
            "name": s["name"],
        }
