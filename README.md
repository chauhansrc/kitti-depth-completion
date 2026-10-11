# Sparse-to-Dense: camera + LiDAR depth completion on KITTI

Implementation of the supervised model from

> F. Ma, G. V. Cavalheiro, S. Karaman. *Self-Supervised Sparse-to-Dense: Self-Supervised Depth
> Completion from LiDAR and Monocular Camera.* ICRA 2019. [arXiv:1807.00275](https://arxiv.org/abs/1807.00275)

**Task:** given an RGB image and the sparse LiDAR depth projected into it (~5% of pixels), predict a
dense depth map.

**Model:** RGB and depth each get their own stem convolution. The concatenated features go through an
ImageNet-pretrained ResNet-34 encoder and a transposed-convolution decoder with skip connections.
The model is trained with masked L2 loss against KITTI's semi-dense ground truth.

## Data (~2 GB)

This project uses only `data_depth_selection.zip` from the
[KITTI depth completion benchmark](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).
Its `val_selection_cropped/` folder has 1000 frames (352×1216), each with an RGB image, sparse LiDAR
depth and ground truth.

These frames are split into train and validation **by drive**: whole driving sequences go to
validation, so near-duplicate consecutive frames can't leak between the two sets. The split is saved to
`runs/<name>/split.json`.

The full benchmark trains on ~86k frames, so absolute numbers here will be worse than the paper's
(~815 mm RMSE). The useful result is the **comparison between modalities**.

## Run on Google Colab

1. Push this repo to GitHub.
2. In Colab: File → Open notebook → GitHub tab → pick `colab_train.ipynb` from your repo. Choose a T4 GPU runtime.
3. Set `REPO_URL` in the setup cell. The code is cloned to `/content/kitti-code`, and results go to
   `MyDrive/kitti-runs` on Google Drive so they survive runtime resets.
4. Run the cells in order: download → train `rgbd` → optional `d` / `rgb` ablations → evaluate → compare.

After changing code locally, push and re-run the setup cell; it runs `git pull` on the existing clone.

## Command line

```bash
bash scripts/download_data.sh /content/data
python train.py --data_root /content/data --modality rgbd --out_dir runs/rgbd --time_budget_min 30
python evaluate.py --checkpoint runs/rgbd/best.pt --data_root /content/data
```

Useful flags for `train.py`:

| Flag | Default | Notes |
|---|---|---|
| `--modality` | `rgbd` | `rgbd` (fusion), `d` (LiDAR only), `rgb` (camera only) |
| `--layers` | 34 | 18 for a faster, smaller encoder |
| `--time_budget_min` | 0 | stops before an epoch would exceed the budget |
| `--batch_size` | 4 | lower it if you hit CUDA out-of-memory |
| `--crop_h`, `--crop_w` | 256, 1024 | random training crop; must divide 16 |
| `--loss` | `l2` | `l1` is often more robust |
| `--resume` | | continue from `runs/<name>/last.pt` |

## Outputs

```
runs/<name>/
  args.json, split.json
  history.csv          per-epoch loss and val metrics
  best.pt, last.pt     checkpoints (best by val RMSE)
  eval/metrics.json    from evaluate.py
  eval/vis/*.png       RGB | sparse input | prediction | GT | error
```

Metrics follow the KITTI devkit and are averaged per image:

- RMSE and MAE, in mm
- iRMSE and iMAE, in 1/km
- δ<1.25 (the fraction of pixels whose predicted depth is within a factor of 1.25 of the ground truth)

## Layout

```
s2d/data.py      file discovery, drive-level split, Dataset + augmentation
s2d/model.py     DepthCompletionNet
s2d/engine.py    train / eval loops, masked loss
s2d/metrics.py   KITTI metrics
s2d/vis.py       visualization panels
train.py         training entry point
evaluate.py      evaluation + visualizations + optional KITTI-format PNGs
```
