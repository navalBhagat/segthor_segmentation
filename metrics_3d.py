import argparse
import re
import typing
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt


def patient_id(stem: str) -> str:
    return stem.rsplit('_', 1)[0] if stem.startswith('Patient_') else stem


def mean_foreground_dice(dice_scores: np.ndarray) -> float:
    """Average all non-background classes, preserving NaN-aware 3D semantics."""
    return float(np.nanmean(dice_scores[..., 1:]))


class PatientDice:
    """Accumulate 2D slice predictions into equally weighted per-patient Dice.

    This mirrors stitched 3D validation for the in-epoch 2D validation loop:
    each patient contributes one Dice vector regardless of how many slices it
    contains, so longer scans do not dominate the summary score.
    """

    def __init__(self, classes: int, expected_stems):
        self.classes = classes
        self.expected = set(expected_stems)
        self.seen = set()
        self.counts = {}

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, stems) -> None:
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        for prediction, target, stem in zip(predictions, targets, stems):
            assert stem in self.expected and stem not in self.seen, stem
            self.seen.add(stem)
            counts = self.counts.setdefault(patient_id(stem), np.zeros((3, self.classes), dtype=np.int64))
            counts[0] += np.bincount(prediction.ravel(), minlength=self.classes)
            counts[1] += np.bincount(target.ravel(), minlength=self.classes)
            counts[2] += np.bincount(target[prediction == target], minlength=self.classes)

    def result(self) -> dict[str, typing.Any]:
        assert self.seen == self.expected, (len(self.seen), len(self.expected))
        patients = {}
        for patient, counts in sorted(self.counts.items()):
            denominator = counts[0] + counts[1]
            dice = np.divide(2 * counts[2], denominator, out=np.ones(self.classes), where=denominator != 0)
            patients[patient] = {'dice': dice.tolist(), 'counts_pred_gt_intersection': counts.tolist()}
        per_organ = np.array([p['dice'] for p in patients.values()]).mean(axis=0)
        return {'foreground_mean': mean_foreground_dice(per_organ),
                'per_organ': per_organ.tolist(),
                'patients': patients,
                'slices': len(self.seen)}


def crop_to_foreground_union(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # HD95 needs distance transforms; crop to the class foreground union to keep that computation small.
    foreground_union: np.ndarray = pred_mask | gt_mask
    foreground_coords: tuple[np.ndarray, ...] = np.nonzero(foreground_union)
    if len(foreground_coords[0]) == 0:
        return pred_mask, gt_mask

    lower_bounds: list[int] = []
    upper_bounds: list[int] = []
    for axis_coords, axis_size in zip(foreground_coords, pred_mask.shape):
        lower_bounds.append(max(int(axis_coords.min()) - 1, 0))
        upper_bounds.append(min(int(axis_coords.max()) + 2, axis_size))

    crop_slices: tuple[slice, ...] = tuple(slice(low, high) for low, high in zip(lower_bounds, upper_bounds))
    return pred_mask[crop_slices], gt_mask[crop_slices]


# 3D Dice per class on the full stitched volume.
def dice_3d(pred_np: np.ndarray, gt_np: np.ndarray, K: int) -> np.ndarray:
    dice_scores: np.ndarray = np.full((K,), np.nan, dtype=np.float64)
    for cls in range(K):
        pred_mask: np.ndarray = (pred_np == cls)
        gt_mask: np.ndarray = (gt_np == cls)

        pred_count: int = int(np.sum(pred_mask))
        gt_count: int = int(np.sum(gt_mask))

        # Absent classes are left as NaN so foreground summaries can ignore them explicitly.
        if pred_count == 0 and gt_count == 0:
            continue

        if pred_count == 0 or gt_count == 0:
            dice_scores[cls] = 0.0
        else:
            intersection: int = int(np.sum(pred_mask & gt_mask))
            dice_scores[cls] = (2.0 * intersection) / (pred_count + gt_count)

    return dice_scores


# 95th percentile Hausdorff distance in mm, computed on 3D object surfaces.
def hd95_3d(pred_np: np.ndarray, gt_np: np.ndarray, K: int, spacing: tuple[float, float, float]) -> np.ndarray:
    hd95_scores: np.ndarray = np.full((K,), np.nan, dtype=np.float64)
    for cls in range(1, K):
        pred_mask: np.ndarray = (pred_np == cls)
        gt_mask: np.ndarray = (gt_np == cls)

        if not pred_mask.any() or not gt_mask.any():
            continue

        pred_mask, gt_mask = crop_to_foreground_union(pred_mask, gt_mask)

        # Surface voxels are the mask boundary: mask minus its one-voxel erosion.
        pred_surface: np.ndarray = pred_mask ^ binary_erosion(pred_mask)
        gt_surface: np.ndarray = gt_mask ^ binary_erosion(gt_mask)

        # distance_transform_edt uses voxel spacing, so distances are physical millimeters.
        pred_dist: np.ndarray = np.asarray(distance_transform_edt(~pred_surface, sampling=spacing))
        gt_dist: np.ndarray = np.asarray(distance_transform_edt(~gt_surface, sampling=spacing))

        gt_to_pred_distances: np.ndarray = pred_dist[gt_surface]
        pred_to_gt_distances: np.ndarray = gt_dist[pred_surface]
        all_distances: np.ndarray = np.concatenate([pred_to_gt_distances, gt_to_pred_distances])

        hd95_scores[cls] = float(np.percentile(all_distances, 95))
    return hd95_scores


# Relative volume difference: positive means over-segmentation, negative means under-segmentation.
def rvd_3d(pred_np: np.ndarray, gt_np: np.ndarray, K: int) -> np.ndarray:
    rvd_scores: np.ndarray = np.full((K,), np.nan, dtype=np.float64)
    for cls in range(1, K):
        pred_volume: float = float(np.sum(pred_np == cls))
        gt_volume: float = float(np.sum(gt_np == cls))

        if gt_volume == 0:
            continue

        rvd_scores[cls] = (pred_volume - gt_volume) / gt_volume

    return rvd_scores


def args_parse() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--volumes_root', type=str, required=True, help='Path to the root directory containing the stitched volumes.')
    parser.add_argument('--gt_root', type=str, required=True, help='Path to the root directory containing the ground truth volumes.')
    parser.add_argument('--dest', type=str, required=True, help='Path to the destination directory where the metrics will be saved.')
    parser.add_argument('--val_patients', type=str, default=None, help='Comma-separated list of validation patient IDs (e.g., "1,2,3"). Defaults to all Patient_*.nii.gz in --volumes_root.')
    parser.add_argument('--K', type=int, default=5, help='Number of classes. Defaults to 5.')
    return parser.parse_args()


def parse_patient_number(path: Path) -> int:
    match = re.fullmatch(r"Patient_(\d+)", path.name.removesuffix(".nii.gz"))
    if match is None:
        raise ValueError(f"Expected Patient_XX.nii.gz, got {path.name}")
    return int(match.group(1))


def patient_volume(root: Path, patient: int) -> Path:
    volume_path = root / f"Patient_{patient:02d}.nii.gz"
    if volume_path.exists():
        return volume_path

    nested_path = root / f"Patient_{patient:02d}" / "GT.nii.gz"
    if nested_path.exists():
        return nested_path

    raise FileNotFoundError(f"Could not find Patient_{patient:02d}.nii.gz or Patient_{patient:02d}/GT.nii.gz in {root}")


def main() -> None:
    args: argparse.Namespace = args_parse()
    volumes_root: Path = Path(args.volumes_root) # volumes/segthor_part1/ce/Patient_<number 2 digits>.nii.gz
    gt_root: Path = Path(args.gt_root) # Either Patient_<number>.nii.gz files or Patient_<number>/GT.nii.gz folders
    dest_root: Path = Path(args.dest) # results/segthor_part1/ce
    if args.val_patients:
        val_patients: list[int] = [int(x) for x in args.val_patients.split(",")]
    else:
        val_patients = sorted(parse_patient_number(path) for path in volumes_root.glob("Patient_*.nii.gz"))
        if not val_patients:
            raise ValueError(f"No Patient_*.nii.gz volumes found in {volumes_root}")

    # Parse number of classes K
    K: int = args.K
    dest_root.mkdir(parents=True, exist_ok=True)

    val_pred_volumes: list[nib.Nifti1Image] = [
        typing.cast(nib.Nifti1Image, nib.load(patient_volume(volumes_root, patient)))
        for patient in val_patients
    ]
    val_gt_volumes: list[nib.Nifti1Image] = [
        typing.cast(nib.Nifti1Image, nib.load(patient_volume(gt_root, patient)))
        for patient in val_patients
    ]

    # Compute metrics for each patient
    results: dict[str, dict[str, np.ndarray]] = {
        "dice": {},
        "hd95": {},
        "rvd": {},
    }
    for (i, patient) in tqdm.tqdm(enumerate(val_patients), 'Evaluating 3D metrics for validation patients', total=len(val_patients)):
        pred_volume: np.ndarray = np.asarray(val_pred_volumes[i].dataobj, dtype=np.uint8)
        gt_volume: np.ndarray = np.asarray(val_gt_volumes[i].dataobj, dtype=np.uint8)
        voxel_spacing: tuple[float, float, float] = typing.cast(tuple[float, float, float], val_gt_volumes[i].header.get_zooms()[:3])

        assert pred_volume.shape == gt_volume.shape, f"Pred shape {pred_volume.shape} and GT shape {gt_volume.shape} do not match."

        print(f"\nComputing metrics for Patient {patient:02d}...")
        print("============================================================")
        print("Computing DICE score...")
        dice_score: np.ndarray = dice_3d(pred_volume, gt_volume, K=K)
        print("Dice: ", dice_score)
        print("============================================================")
        print("Computing HD95...")
        hd95_score: np.ndarray = hd95_3d(pred_volume, gt_volume, K=K, spacing=voxel_spacing)
        print("HD95: ", hd95_score)
        print("============================================================")
        print("Computing RVD...")
        rvd_score: np.ndarray = rvd_3d(pred_volume, gt_volume, K=K)
        print("RVD: ", rvd_score)
        print("============================================================")
        
        patient_number: str = f"Patient_{patient:02d}"
        results['dice'][patient_number] = dice_score
        results['hd95'][patient_number] = hd95_score
        results['rvd'][patient_number] = rvd_score

    np.savez(dest_root / "dice.npz", **typing.cast(typing.Any, results["dice"]))
    np.savez(dest_root / "hd95.npz", **typing.cast(typing.Any, results["hd95"]))
    np.savez(dest_root / "rvd.npz", **typing.cast(typing.Any, results["rvd"]))

if __name__ == "__main__":
    main()
