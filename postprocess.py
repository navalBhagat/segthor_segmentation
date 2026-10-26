import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure, label


def remove_pieces(pred: np.ndarray, cls: int, min_voxels: float) -> None:
    components, n = label(pred == cls, structure=generate_binary_structure(3, 3))
    if n <= 1:
        return
    sizes: np.ndarray = np.bincount(components.ravel())
    sizes[0] = 0
    keep: np.ndarray = (sizes == sizes.max()) | (sizes >= min_voxels)
    keep[0] = False
    pred[(components != 0) & ~keep[components]] = 0


def close_gaps(pred: np.ndarray, cls: int, voxels: int) -> None:
    closed: np.ndarray = binary_closing(pred == cls, iterations=voxels)
    pred[closed & (pred == 0)] = cls


def fill_holes(pred: np.ndarray, cls: int) -> None:
    pred[binary_fill_holes(pred == cls) & (pred == 0)] = cls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--largest', action='store_true')
    parser.add_argument('--min_ml', type=float, default=None)
    parser.add_argument('--largest-classes', type=int, nargs='+',
                        help='Restrict --largest to these foreground labels; heart is 2 in SEGTHOR.')
    parser.add_argument('--min-classes', type=int, nargs='+',
                        help='Restrict --min_ml to these foreground labels; aorta is 4 in SEGTHOR.')
    parser.add_argument('--close', type=int, default=0,
                        help='Morphological closing size in voxels, fills small gaps.')
    parser.add_argument('--fill_holes', action='store_true')
    args = parser.parse_args()
    if args.largest_classes is not None:
        if not args.largest or any(cls <= 0 for cls in args.largest_classes):
            parser.error('--largest-classes requires --largest and positive foreground labels')
    if args.min_ml is not None and (not np.isfinite(args.min_ml) or args.min_ml <= 0):
        parser.error('--min_ml must be positive and finite')
    if args.min_classes is not None:
        if args.min_ml is None or any(cls <= 0 for cls in args.min_classes):
            parser.error('--min-classes requires --min_ml and positive foreground labels')

    args.dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.src.glob("Patient_*.nii.gz")):
        img = nib.load(path)
        pred: np.ndarray = np.asarray(img.dataobj, dtype=np.uint8)
        cleaned: np.ndarray = pred.copy()
        voxel_ml: float = float(np.prod(img.header.get_zooms()[:3])) / 1000
        for cls in np.unique(pred):
            if cls == 0:
                continue
            if args.largest and (args.largest_classes is None or cls in args.largest_classes):
                remove_pieces(cleaned, cls, np.inf)
            elif args.min_ml is not None and (args.min_classes is None or cls in args.min_classes):
                remove_pieces(cleaned, cls, args.min_ml / voxel_ml)
            if args.close > 0:
                close_gaps(cleaned, cls, args.close)
            if args.fill_holes:
                fill_holes(cleaned, cls)
        print(f"{path.name}: changed {int((pred != cleaned).sum())} voxels")
        nib.save(nib.Nifti1Image(cleaned, img.affine, img.header), args.dest / path.name)


if __name__ == "__main__":
    main()
