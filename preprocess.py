#!/usr/bin/env python3

import json
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.io import imsave
from skimage.transform import resize

from utils import tqdm_


def crop(sl, x0, y0, w, h, fill):
    out = np.full((w, h), fill, dtype=sl.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + w, sl.shape[0]), min(y0 + h, sl.shape[1])
    out[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = sl[sx0:sx1, sy0:sy1]
    return out


def resample_volume(volume, source_spacing, target_spacing, order):
    factors = tuple(src / dst for src, dst in zip(source_spacing, target_spacing))
    return zoom(volume, factors, order=order)


def patient_ct_path(source_path, mode, id_):
    if mode == "test":
        return source_path / "test" / f"{id_}.nii.gz"
    return source_path / "train" / id_ / f"{id_}.nii.gz"


def patient_gt_path(source_path, id_):
    return source_path / "train" / id_ / "GT.nii.gz"


def slice_patient(id_, mode, dest_path, source_path, shape, hu_min, hu_max, mm_per_px, target_spacing):
    ct_nib = nib.load(str(patient_ct_path(source_path, mode, id_)))
    ct = np.asarray(ct_nib.dataobj).astype(np.float32)
    gt = None if mode == "test" else np.asarray(nib.load(str(patient_gt_path(source_path, id_))).dataobj)

    spacing = tuple(float(v) for v in ct_nib.header.get_zooms()[:3])
    if target_spacing is not None:
        ct = resample_volume(ct, spacing, target_spacing, order=1).astype(np.float32)
        if gt is not None:
            gt = resample_volume(gt, spacing, target_spacing, order=0).astype(gt.dtype)
        spacing = target_spacing

    dx, dy, _ = spacing
    X, Y, Z = ct.shape

    if mm_per_px > 0:
        w = int(round(shape[0] * mm_per_px / dx))
        h = int(round(shape[1] * mm_per_px / dy))
    else:
        w, h = X, Y
    x0, y0 = (X - w) // 2, (Y - h) // 2

    ct = 255 * (np.clip(ct, hu_min, hu_max) - hu_min) / (hu_max - hu_min)

    for z in range(Z):
        img = crop(ct[:, :, z], x0, y0, w, h, 0.0)
        img = resize(img, shape, order=1, mode="constant",
                     preserve_range=True, anti_aliasing=False).astype(np.uint8)

        outputs = [("img", img)]
        if gt is not None:
            lab = crop(gt[:, :, z], x0, y0, w, h, np.uint8(0))
            lab = resize(lab, shape, order=0, mode="constant",
                         preserve_range=True).astype(np.uint8) * 63
            outputs.append(("gt", lab))

        for sub, arr in outputs:
            out_dir = dest_path / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(out_dir / f"{id_}_{z:04d}.png"), arr)

    return {
        "source_shape": [int(v) for v in ct_nib.shape],
        "source_affine": ct_nib.affine.tolist(),
        "source_spacing": [float(v) for v in ct_nib.header.get_zooms()[:3]],
        "processed_spacing": list(spacing),
        "processed_shape": [int(v) for v in ct.shape],
        "crop_origin": [int(x0), int(y0)],
        "crop_shape": [int(w), int(h)],
        "slice_shape": [int(v) for v in shape],
    }


def make_split(source_path, retains, seed):
    ids = sorted(p.name for p in (source_path / "train").glob("Patient_*") if p.is_dir())
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    val = sorted(shuffled[:retains])
    train = sorted(i for i in ids if i not in val)
    return train, val


def get_test_ids(source_path):
    test_dir = source_path / "test"
    if not test_dir.exists():
        return []
    return sorted(Path(path.stem).stem for path in test_dir.glob("Patient_*.nii.gz"))


def main(args):
    source_path = Path(args.source_dir)
    dest_path = Path(args.dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    test_only = getattr(args, 'test_only', False)
    splits_file = dest_path / "splits.json"
    if test_only:
        train, val = [], []
    elif splits_file.exists():
        saved = json.loads(splits_file.read_text())
        train, val = saved["train"], saved["val"]
        print(f"reusing split from {splits_file}")
    else:
        train, val = make_split(source_path, args.retains, args.seed)
        splits_file.write_text(json.dumps(
            {"seed": args.seed, "train": train, "val": val}, indent=2))
        print(f"wrote split to {splits_file}")
    test = get_test_ids(source_path) if args.include_test or test_only else []
    if test_only and not test:
        raise ValueError(f'No test scans found in {source_path / "test"}')
    if not test_only:
        print(f"train {len(train)}, val {len(val)}: {', '.join(val)}")
    if test:
        print(f"test {len(test)}: {', '.join(test)}")

    target_spacing = tuple(args.target_spacing) if args.target_spacing else None
    metadata = {
        "source_dir": str(source_path),
        "shape": list(args.shape),
        "hu_min": args.hu_min,
        "hu_max": args.hu_max,
        "mm_per_px": args.mm_per_px,
        "target_spacing": list(target_spacing) if target_spacing else None,
        "patients": {},
    }

    split_items = [] if test_only else [("train", train), ("val", val)]
    if test:
        split_items.append(("test", test))

    for mode, ids in split_items:
        print(f"slicing {mode}")
        fn = partial(slice_patient,
                     mode=mode,
                     dest_path=dest_path / mode,
                     source_path=source_path,
                     shape=tuple(args.shape),
                     hu_min=args.hu_min,
                     hu_max=args.hu_max,
                     mm_per_px=args.mm_per_px,
                     target_spacing=target_spacing)
        if args.process == 1:
            patient_metadata = list(map(fn, tqdm_(ids)))
        else:
            patient_metadata = Pool(None if args.process == -1 else args.process).map(fn, tqdm_(ids))

        metadata["patients"][mode] = dict(zip(ids, patient_metadata))

    (dest_path / "preprocess.json").write_text(json.dumps(metadata, indent=2))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", type=str, default="data/segthor_train")
    p.add_argument("--dest_dir", type=str, default="data/SEGTHOR")
    p.add_argument("--shape", type=int, nargs=2, default=[256, 256])
    p.add_argument("--hu_min", type=float, default=-1000)
    p.add_argument("--hu_max", type=float, default=500)
    p.add_argument("--mm_per_px", type=float, default=0)
    p.add_argument("--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 2.5],
                   help="Target voxel spacing dx dy dz in mm before slicing. "
                        "CT uses linear interpolation; GT uses nearest-neighbor interpolation.")
    p.add_argument("--retains", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include_test", action="store_true",
                   help="Also preprocess source_dir/test/*.nii.gz into dest_dir/test/img when a test folder exists.")
    p.add_argument("--test_only", action="store_true",
                   help="Only preprocess source_dir/test/*.nii.gz; do not create training/validation splits.")
    p.add_argument("--process", "-p", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    main(get_args())
