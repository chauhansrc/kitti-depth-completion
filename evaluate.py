"""Evaluate a checkpoint on its held-out validation drives and save visualizations.

Example:
    python evaluate.py --checkpoint runs/rgbd/best.pt --data_root /content/data
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from s2d.data import KittiDepthSelection, find_samples
from s2d.engine import evaluate
from s2d.metrics import format_metrics
from s2d.model import DepthCompletionNet
from s2d.vis import save_panel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="", help="Default: <checkpoint dir>/eval")
    p.add_argument("--all", action="store_true", help="Evaluate every frame, not just the val split")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_vis", type=int, default=8, help="Number of visualization panels to save")
    p.add_argument("--save_png", action="store_true",
                   help="Also save predictions as KITTI-format uint16 PNGs (depth * 256)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    model = DepthCompletionNet(train_args["modality"], train_args["layers"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Loaded {args.checkpoint} (epoch {ckpt['epoch'] + 1}, modality {train_args['modality']})")

    samples = find_samples(args.data_root)
    if not args.all:
        val_names = set(ckpt["val_names"])
        samples = [s for s in samples if s["name"] in val_names]
    print(f"Evaluating {len(samples)} frames")

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), "eval")
    vis_dir = os.path.join(out_dir, "vis")
    png_dir = os.path.join(out_dir, "pred_png")
    os.makedirs(vis_dir, exist_ok=True)
    if args.save_png:
        os.makedirs(png_dir, exist_ok=True)

    vis_indices = set(np.linspace(0, len(samples) - 1, num=min(args.num_vis, len(samples)), dtype=int).tolist())

    def on_batch(batch, pred):
        for i, idx in enumerate(batch["index"].tolist()):
            name = batch["name"][i]
            if idx in vis_indices:
                save_panel(os.path.join(vis_dir, name), batch["rgb"][i], batch["sparse"][i], pred[i],
                           batch["gt"][i], title=f"{train_args['modality']} | {name}")
            if args.save_png:
                depth = np.clip(pred[i, 0].numpy() * 256.0, 0, 65535).astype(np.uint16)
                Image.fromarray(depth).save(os.path.join(png_dir, name))

    loader = DataLoader(KittiDepthSelection(samples, train=False), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    metrics = evaluate(model, loader, device, amp, on_batch=on_batch)
    print(format_metrics(metrics))

    result = {"modality": train_args["modality"], "checkpoint": args.checkpoint,
              "epoch": ckpt["epoch"] + 1, "num_frames": len(samples), **metrics}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics and {len(vis_indices)} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
