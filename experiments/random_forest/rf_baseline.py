#!/usr/bin/env python3
"""Per-pixel random forest baseline for SegTHOR, no CNN involved.

Reads the NIfTI volumes directly (real voxel spacing, full-resolution GT for metrics) and clips to its own
[-1000, 1000] HU window, wider than preprocess.py's, so the trachea's air stays distinct. Run from the repo root:

    python experiments/random_forest/rf_baseline.py --data data/segthor_part1/train --dest results/rf_baseline
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier

K = 4  # segthor_part1 GT only contains labels 0-3: the aorta is merged into class 1
CLASS_NAMES = ["background", "esophagus+aorta", "heart", "trachea"]
VAL = [1, 11, 15, 17, 19]  # same as slice_segthor.py --seed 0 --fold 0 --retains 5
TRAIN = [p for p in range(1, 21) if p not in VAL]

FEATURES = ["hu", "gauss_s1", "gauss_s2", "gauss_s4", "gauss_s8",
            "gradmag_s1", "gradmag_s2", "gradmag_s4", "log_s2", "log_s4", "local_std_s2",
            "x_body", "y_body", "z_rel", "depth_mm"]
N_APPEARANCE = 11  # first 11 features are appearance, the rest position

BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#e5e4df"
CLASS_COLORS = {1: "#1baf7a", 2: "#eda100", 3: "#4a3aa7"}


def load_patient(data: Path, pid: int) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """CT clipped to [-1000, 1000] HU and 2x2-averaged to 256x256, GT at full resolution, voxel spacing."""
    d = data / f"Patient_{pid:02d}"
    img = nib.load(d / f"Patient_{pid:02d}.nii.gz")
    ct = np.clip(img.get_fdata(dtype=np.float32), -1000, 1000)
    gt = np.asarray(nib.load(d / "GT.nii.gz").dataobj).astype(np.uint8)
    assert gt.max() < K, np.unique(gt)
    X, Y, Z = ct.shape
    ct = ct.reshape(X // 2, 2, Y // 2, 2, Z).mean(axis=(1, 3))
    return ct, gt, tuple(float(s) for s in img.header.get_zooms()[:3])


def body_masks(ct: np.ndarray) -> np.ndarray:
    """Per slice: HU > -500, holes filled (so lungs and trachea are inside), largest component."""
    masks = np.zeros(ct.shape, dtype=bool)
    for z in range(ct.shape[2]):
        m = ndi.binary_fill_holes(ct[:, :, z] > -500)
        lab, n = ndi.label(m)
        if n:
            masks[:, :, z] = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1
    return masks


def slice_features(hu: np.ndarray, body: np.ndarray, z_rel: float, px_mm: float) -> np.ndarray:
    """(H, W) HU slice -> (H, W, len(FEATURES)). Requires a non-empty body mask."""
    h = hu.astype(np.float64)
    xx, yy = np.indices(hu.shape, dtype=np.float64)
    cx, cy = ndi.center_of_mass(body)
    r = np.sqrt(body.sum() / np.pi)
    var = ndi.gaussian_filter(h ** 2, 2) - ndi.gaussian_filter(h, 2) ** 2
    feats = [h,
             *(ndi.gaussian_filter(h, s) for s in (1, 2, 4, 8)),
             *(ndi.gaussian_gradient_magnitude(h, s) for s in (1, 2, 4)),
             *(ndi.gaussian_laplace(h, s) for s in (2, 4)),
             np.sqrt(np.maximum(var, 0)),
             (xx - cx) / r,  # offset from body centre, normalised by body radius
             (yy - cy) / r,
             np.full_like(h, z_rel),
             ndi.distance_transform_edt(body) * px_mm]  # depth inside the body
    return np.stack(feats, axis=-1).astype(np.float32)


def sample_patient(data: Path, pid: int, rng: np.random.Generator, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels for up to per_class random body voxels of each class."""
    ct, gt, spacing = load_patient(data, pid)
    gt = gt[::2, ::2]
    body = body_masks(ct)
    Z = ct.shape[2]

    idx = np.concatenate([rng.choice(cand, size=min(per_class, cand.size), replace=False)
                          for cand in (np.flatnonzero((gt == k) & body) for k in range(K))])
    x, y, z = np.unravel_index(idx, gt.shape)
    rows = np.empty((idx.size, len(FEATURES)), dtype=np.float32)
    for zz in np.unique(z):
        sel = z == zz
        f = slice_features(ct[:, :, zz], body[:, :, zz], zz / (Z - 1), 2 * spacing[0])
        rows[sel] = f[x[sel], y[sel]]
    return rows, gt.ravel()[idx]


def predict_patient(model: RandomForestClassifier, ct: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """256x256xZ label volume. Voxels outside the body are background."""
    body = body_masks(ct)
    Z = ct.shape[2]
    pred = np.zeros(ct.shape, dtype=np.uint8)
    for z in range(Z):
        b = body[:, :, z]
        if not b.any():
            continue
        f = slice_features(ct[:, :, z], b, z / (Z - 1), 2 * spacing[0])
        pred[:, :, z][b] = model.predict(f[b])
    return pred


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def hd95(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> float:
    if not a.any() or not b.any():
        return float("nan")
    crop = tuple(slice(max(i.min() - 1, 0), i.max() + 2) for i in np.nonzero(a | b))
    a, b = a[crop], b[crop]
    sa, sb = a ^ ndi.binary_erosion(a), b ^ ndi.binary_erosion(b)
    da = ndi.distance_transform_edt(~sa, sampling=spacing)
    db = ndi.distance_transform_edt(~sb, sampling=spacing)
    return float(np.percentile(np.concatenate([db[sa], da[sb]]), 95))


def despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_metrics(results: dict, dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(1, K)
    for ax, key, label in zip(axes, ("dice", "hd95"), ("Dice", "HD95 (mm)")):
        vals = np.array([r[key] for r in results.values()])  # (patients, 3)
        ax.bar(x, np.nanmean(vals, 0), yerr=np.nanstd(vals, 0), width=0.6, color=BLUE, capsize=4)
        jitter = np.linspace(-0.15, 0.15, len(vals))
        for i, k in enumerate(x):
            ax.scatter(k + jitter, vals[:, i], color=INK, s=14, zorder=3)
        ax.set_xticks(x, CLASS_NAMES[1:])
        ax.set_ylabel(label)
        despine(ax)
    axes[0].set_ylim(0, 1)
    fig.suptitle("RF on validation patients: mean ± std, dots are patients")
    fig.tight_layout()
    fig.savefig(dest / "metrics.png", dpi=150)
    plt.close(fig)


def plot_importance(model: RandomForestClassifier, dest: Path) -> None:
    imp = model.feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(np.arange(len(order)), imp[order], color=[BLUE if i < N_APPEARANCE else ORANGE for i in order])
    ax.set_yticks(np.arange(len(order)), [FEATURES[i] for i in order])
    ax.set_xlabel("impurity importance")
    pos_share = imp[N_APPEARANCE:].sum()
    ax.legend(handles=[Patch(color=BLUE, label=f"appearance ({1 - pos_share:.0%})"),
                       Patch(color=ORANGE, label=f"position ({pos_share:.0%})")], frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(dest / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_confusion(confusion: np.ndarray, dest: Path) -> None:
    norm = confusion / confusion.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(K), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(K), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("Val confusion, row-normalised")
    fig.tight_layout()
    fig.savefig(dest / "confusion.png", dpi=150)
    plt.close(fig)


def plot_examples(examples: dict, dest: Path) -> None:
    fig, axes = plt.subplots(len(examples), 3, figsize=(10, 3.4 * len(examples)), squeeze=False)
    for row, (k, (_, title, hu, gt, pred)) in zip(axes, sorted(examples.items())):
        for ax, (name, lab) in zip(row, (("CT", None), ("GT", gt), ("RF", pred))):
            ax.imshow(hu.T, cmap="gray", vmin=-1000, vmax=600)  # .T: anterior up, patient right on the left
            if lab is not None:
                overlay = np.zeros((*lab.T.shape, 4))
                for c, color in CLASS_COLORS.items():
                    overlay[lab.T == c] = matplotlib.colors.to_rgba(color, 0.6)
                ax.imshow(overlay)
            ax.set_title(f"{name}: {title}" if name == "CT" else name, fontsize=9)
            ax.axis("off")
    fig.legend(handles=[Patch(color=c, label=CLASS_NAMES[k]) for k, c in CLASS_COLORS.items()],
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(dest / "examples.png", dpi=120)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    data, dest = Path(args.data), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    parts = []
    for pid in TRAIN:
        parts.append(sample_patient(data, pid, rng, args.per_class))
        print(f"sampled Patient_{pid:02d}: {len(parts[-1][1])} voxels ({time.time() - t0:.0f}s)")
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    del parts

    model = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=args.min_leaf,
                                   max_features="sqrt", n_jobs=-1, random_state=args.seed)
    model.fit(X, y)
    assert list(model.classes_) == list(range(K))
    print(f"trained on {X.shape} in {time.time() - t0:.0f}s total")
    del X, y

    results, examples = {}, {}
    confusion = np.zeros((K, K), dtype=np.int64)
    for pid in VAL:
        ct, gt, spacing = load_patient(data, pid)
        pred_small = predict_patient(model, ct, spacing)
        pred = pred_small.repeat(2, axis=0).repeat(2, axis=1)  # back to the original 512x512 grid
        gt_small = gt[::2, ::2]
        confusion += np.bincount(gt_small.ravel().astype(np.int64) * K + pred_small.ravel(),
                                 minlength=K * K).reshape(K, K)

        name = f"Patient_{pid:02d}"
        results[name] = {"dice": [dice(pred == k, gt == k) for k in range(1, K)],
                         "hd95": [hd95(pred == k, gt == k, spacing) for k in range(1, K)]}
        print(f"{name}: dice {np.round(results[name]['dice'], 3)} hd95 {np.round(results[name]['hd95'], 1)} "
              f"({time.time() - t0:.0f}s)")

        for k in range(1, K):  # keep the slice with the largest GT area per class for the example figure
            areas = (gt_small == k).sum(axis=(0, 1))
            z = int(areas.argmax())
            if k not in examples or areas[z] > examples[k][0]:
                examples[k] = (areas[z], f"{name} z={z}", ct[:, :, z], gt_small[:, :, z], pred_small[:, :, z])

    summary = {key: {CLASS_NAMES[k]: {"mean": float(np.nanmean([r[key][k - 1] for r in results.values()])),
                                      "std": float(np.nanstd([r[key][k - 1] for r in results.values()]))}
                     for k in range(1, K)} for key in ("dice", "hd95")}
    (dest / "metrics.json").write_text(json.dumps(
        {"per_patient": results, "summary": summary, "feature_importance": dict(
            zip(FEATURES, model.feature_importances_.tolist())), "args": vars(args)}, indent=2))
    for key, per_class in summary.items():
        print(key, {c: f"{v['mean']:.3f} ± {v['std']:.3f}" for c, v in per_class.items()})

    plot_metrics(results, dest)
    plot_importance(model, dest)
    plot_confusion(confusion, dest)
    plot_examples(examples, dest)
    print(f"saved to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-pixel random forest baseline")
    parser.add_argument("--data", type=str, default="data/segthor_part1/train")
    parser.add_argument("--dest", type=str, default="results/rf_baseline")
    parser.add_argument("--per-class", type=int, default=10000, help="voxels sampled per class per patient")
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
