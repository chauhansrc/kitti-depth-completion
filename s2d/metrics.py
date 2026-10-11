"""KITTI depth completion metrics, averaged per image like the official devkit.

RMSE / MAE are in millimeters; iRMSE / iMAE are in 1/km. Only pixels with GT > 0 count.
"""

import torch


class DepthMetrics:
    NAMES = ("rmse", "mae", "irmse", "imae", "delta1")

    def __init__(self):
        self.sums = {k: 0.0 for k in self.NAMES}
        self.count = 0

    @torch.no_grad()
    def update(self, pred, gt):
        """pred, gt: Bx1xHxW tensors in meters."""
        for p, g in zip(pred, gt):
            mask = g > 0
            if not mask.any():
                continue
            p = p[mask].double()
            g = g[mask].double()
            err = p - g
            inv_err = 1.0 / p - 1.0 / g
            ratio = torch.maximum(p / g, g / p)
            self.sums["rmse"] += torch.sqrt((err ** 2).mean()).item() * 1000.0
            self.sums["mae"] += err.abs().mean().item() * 1000.0
            self.sums["irmse"] += torch.sqrt((inv_err ** 2).mean()).item() * 1000.0
            self.sums["imae"] += inv_err.abs().mean().item() * 1000.0
            self.sums["delta1"] += (ratio < 1.25).double().mean().item()
            self.count += 1

    def compute(self):
        n = max(self.count, 1)
        return {k: v / n for k, v in self.sums.items()}


def format_metrics(m):
    return (
        f"RMSE {m['rmse']:.1f} mm | MAE {m['mae']:.1f} mm | "
        f"iRMSE {m['irmse']:.2f} 1/km | iMAE {m['imae']:.2f} 1/km | d1 {m['delta1']:.4f}"
    )
