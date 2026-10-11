"""Training and evaluation loops."""

import random

import numpy as np
import torch
from tqdm.auto import tqdm

from .metrics import DepthMetrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_loss(pred, gt, kind="l2"):
    """Loss over pixels that have ground truth (semi-dense KITTI GT is ~30% of pixels)."""
    mask = gt > 0
    if not mask.any():
        return pred.sum() * 0.0
    diff = pred[mask] - gt[mask]
    if kind == "l1":
        return diff.abs().mean()
    return (diff ** 2).mean()


def _to_device(batch, device):
    return (
        batch["rgb"].to(device, non_blocking=True),
        batch["sparse"].to(device, non_blocking=True),
        batch["gt"].to(device, non_blocking=True),
    )


def train_one_epoch(model, loader, optimizer, scaler, device, amp, loss_kind="l2", desc="train"):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        rgb, sparse, gt = _to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            pred = model(rgb, sparse)
        loss = masked_loss(pred.float(), gt, loss_kind)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{total / n:.3f}")
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp, on_batch=None):
    """Return metrics dict. `on_batch(batch, pred)` is called per batch (e.g. to save visualizations)."""
    model.eval()
    meter = DepthMetrics()
    for batch in tqdm(loader, desc="eval", leave=False):
        rgb, sparse, gt = _to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            pred = model(rgb, sparse)
        pred = pred.float()
        meter.update(pred, gt)
        if on_batch is not None:
            on_batch(batch, pred.cpu())
    return meter.compute()
