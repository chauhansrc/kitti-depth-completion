"""Train the Sparse-to-Dense depth completion network on the KITTI depth selection subset.

Example (Colab, T4):
    python train.py --data_root /content/data --modality rgbd --out_dir runs/rgbd --time_budget_min 30
"""

import argparse
import csv
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from s2d.data import KittiDepthSelection, find_samples, split_samples
from s2d.engine import evaluate, seed_everything, train_one_epoch
from s2d.metrics import format_metrics
from s2d.model import MODALITIES, DepthCompletionNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="Folder containing depth_selection/")
    p.add_argument("--out_dir", default="runs/rgbd")
    p.add_argument("--modality", default="rgbd", choices=MODALITIES)
    p.add_argument("--layers", type=int, default=34, choices=[18, 34], help="ResNet encoder depth")
    p.add_argument("--no_pretrained", action="store_true", help="Do not load ImageNet weights")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--time_budget_min", type=float, default=0,
                   help="Stop before the next epoch would exceed this many minutes (0 = no limit)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--loss", default="l2", choices=["l2", "l1"])
    p.add_argument("--crop_h", type=int, default=256)
    p.add_argument("--crop_w", type=int, default=1024)
    p.add_argument("--jitter", type=float, default=0.2, help="Color jitter strength (0 disables)")

    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--random_split", action="store_true",
                   help="Split frames randomly instead of by drive (leaks near-duplicate frames into val)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--resume", default="", help="Path to a last.pt checkpoint")
    return p.parse_args()


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_rmse, args, val_names, metrics):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_rmse": best_rmse,
        "args": vars(args),
        "val_names": val_names,
        "metrics": metrics,
    }, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | AMP: {amp} | modality: {args.modality}")

    # Data
    samples = find_samples(args.data_root)
    train_samples, val_samples = split_samples(
        samples, args.val_fraction, args.seed, by_drive=not args.random_split)
    val_names = [s["name"] for s in val_samples]
    print(f"Samples: {len(train_samples)} train / {len(val_samples)} val "
          f"({len({s['drive'] for s in train_samples})} / {len({s['drive'] for s in val_samples})} drives)")
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({"train": [s["name"] for s in train_samples], "val": val_names}, f, indent=1)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_ds = KittiDepthSelection(train_samples, train=True, crop_size=(args.crop_h, args.crop_w),
                                   jitter=args.jitter)
    val_ds = KittiDepthSelection(val_samples, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model / optimization
    model = DepthCompletionNet(args.modality, args.layers, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f} M")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    start_epoch, best_rmse = 0, math.inf
    history = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_rmse = ckpt["epoch"] + 1, ckpt["best_rmse"]
        hist_path = os.path.join(args.out_dir, "history.csv")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = list(csv.DictReader(f))
        print(f"Resumed from epoch {ckpt['epoch']} (best RMSE {best_rmse:.1f} mm)")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        ep_start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, amp, args.loss,
                                     desc=f"epoch {epoch + 1}/{args.epochs}")
        scheduler.step()
        metrics = evaluate(model, val_loader, device, amp)
        ep_time = time.time() - ep_start

        is_best = metrics["rmse"] < best_rmse
        best_rmse = min(best_rmse, metrics["rmse"])
        save_checkpoint(os.path.join(args.out_dir, "last.pt"), model, optimizer, scheduler, scaler,
                        epoch, best_rmse, args, val_names, metrics)
        if is_best:
            save_checkpoint(os.path.join(args.out_dir, "best.pt"), model, optimizer, scheduler, scaler,
                            epoch, best_rmse, args, val_names, metrics)

        print(f"[{epoch + 1:3d}/{args.epochs}] loss {train_loss:.3f} | {format_metrics(metrics)} | "
              f"lr {lr:.1e} | {ep_time:.0f}s{' *best*' if is_best else ''}")

        history.append({"epoch": epoch + 1, "lr": lr, "train_loss": train_loss, "epoch_time_s": ep_time,
                        **metrics})
        with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[-1].keys()))
            writer.writeheader()
            writer.writerows(history)

        elapsed = time.time() - t0
        if args.time_budget_min and elapsed + ep_time > args.time_budget_min * 60:
            print(f"Stopping: time budget of {args.time_budget_min:g} min reached after {elapsed / 60:.1f} min")
            break

    print(f"Done. Best val RMSE {best_rmse:.1f} mm. Checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
