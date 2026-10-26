#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import json
import argparse
from pathlib import Path
from typing import Any, Pattern

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.io import imread
from skimage.transform import resize

from utils import tqdm_


def get_z(image: Path) -> int:
    return int(image.stem.split('_')[-1])


def load_preprocess_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}

    metadata = json.loads(path.read_text())
    patients: dict[str, dict[str, Any]] = {}
    for split_patients in metadata.get("patients", {}).values():
        patients.update(split_patients)
    return patients


def rescale_affine(affine: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    res = affine.copy()
    for axis, target_spacing in enumerate(spacing):
        direction = res[:3, axis]
        norm = np.linalg.norm(direction)
        if norm > 0:
            res[:3, axis] = direction / norm * target_spacing
        else:
            res[:3, axis] = 0
            res[axis, axis] = target_spacing
    return res


def source_geometry(source_path: str) -> tuple[tuple[int, int, int],
                                              tuple[float, float, float],
                                              np.ndarray,
                                              nib.nifti1.Nifti1Header]:
    orig_nib = nib.load(source_path)
    header = orig_nib.header.copy()
    shape = tuple(int(v) for v in np.asarray(orig_nib.dataobj).shape)
    spacing = tuple(float(v) for v in header.get_zooms()[:3])
    return shape, spacing, orig_nib.affine, header


def output_geometry(id_: str, source_pattern: str,
                    preprocess_metadata: dict[str, dict[str, Any]]) -> tuple[tuple[int, int, int],
                                                                             tuple[float, float, float],
                                                                             np.ndarray,
                                                                             nib.nifti1.Nifti1Header]:
    source_path = source_pattern.format(id_=id_)
    shape, spacing, affine, header = source_geometry(source_path)

    if id_ not in preprocess_metadata:
        return shape, spacing, affine, header

    patient_metadata = preprocess_metadata[id_]
    shape = tuple(int(v) for v in patient_metadata["processed_shape"])
    spacing = tuple(float(v) for v in patient_metadata["processed_spacing"])
    affine = rescale_affine(affine, spacing)
    header.set_data_shape(shape)
    header.set_zooms(spacing + tuple(header.get_zooms()[3:]))

    return shape, spacing, affine, header


def fit_shape(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if volume.shape == target_shape:
        return volume

    fitted = np.zeros(target_shape, dtype=volume.dtype)
    common_shape = tuple(min(src, dst) for src, dst in zip(volume.shape, target_shape))
    src_slices = tuple(slice(0, size) for size in common_shape)
    dst_slices = tuple(slice(0, size) for size in common_shape)
    fitted[dst_slices] = volume[src_slices]
    return fitted


def restore_original_geometry(volume: np.ndarray,
                              target_shape: tuple[int, int, int]) -> np.ndarray:
    if volume.shape == target_shape:
        return volume

    factors = tuple(dst / src for src, dst in zip(volume.shape, target_shape))
    restored = zoom(volume, factors, order=0)
    return fit_shape(restored.astype(volume.dtype), target_shape)


def restore_to_source_geometry(id_: str,
                               source_pattern: str,
                               preprocess_metadata: dict[str, dict[str, Any]],
                               volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, nib.nifti1.Nifti1Header]:
    source_path = source_pattern.format(id_=id_)
    source_shape, _, source_affine, source_header = source_geometry(source_path)

    if id_ not in preprocess_metadata:
        _, _, affine, header = output_geometry(id_, source_pattern, preprocess_metadata)
        return volume, affine, header

    restored = restore_original_geometry(volume, source_shape)
    source_header.set_data_shape(source_shape)
    return restored, source_affine, source_header


def label_scale_for_classes(K: int) -> float:
    return 63.0 if K == 5 else 255.0 / (K - 1)


def decode_segmentation(img_arr: np.ndarray, K: int, label_scale: float | None) -> np.ndarray:
    if img_arr.ndim == 3:
        img_arr = img_arr[..., 0]

    unique = set(np.unique(img_arr))
    if unique <= set(range(K)):
        decoded = img_arr.astype(np.int16)
    else:
        scale = label_scale if label_scale is not None else label_scale_for_classes(K)
        decoded = np.rint(img_arr.astype(np.float32) / scale).astype(np.int16)

    assert set(np.unique(decoded)) <= set(range(K)), (np.unique(img_arr), np.unique(decoded), K)
    return decoded


def paste_crop(volume_slice: np.ndarray, crop: np.ndarray, origin: tuple[int, int]) -> None:
    x0, y0 = origin
    w, h = crop.shape
    X, Y = volume_slice.shape

    dst_x0, dst_y0 = max(x0, 0), max(y0, 0)
    dst_x1, dst_y1 = min(x0 + w, X), min(y0 + h, Y)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return

    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
    volume_slice[dst_x0:dst_x1, dst_y0:dst_y1] = crop[src_x0:src_x1, src_y0:src_y1]


def merge_patient(id_: str, dest_folder: Path, images: list[Path], idxes: list[int],
                  K: int, source_pattern: str,
                  preprocess_metadata: dict[str, dict[str, Any]] | None = None,
                  label_scale: float | None = None) -> Path:
    metadata = preprocess_metadata or {}
    output_shape, _, affine, header = output_geometry(id_, source_pattern, metadata)
    X, Y, Z = output_shape
    assert Z == len(idxes), f"{id_}: expected {Z} slices, found {len(idxes)}"

    patient_metadata = metadata.get(id_, {})
    crop_shape = tuple(int(v) for v in patient_metadata.get("crop_shape", [X, Y]))
    crop_origin = tuple(int(v) for v in patient_metadata.get("crop_origin", [0, 0]))

    res_arr: np.ndarray = np.zeros(output_shape, dtype=np.int16)

    for idx in idxes:
        img: Path = images[idx]
        z = get_z(img)
        assert 0 <= z < Z, f"{id_}: slice index {z} outside stitched depth {Z}"

        img_arr = imread(img)
        assert img_arr.dtype == np.uint8, img_arr.dtype
        decoded = decode_segmentation(img_arr, K=K, label_scale=label_scale)
        resized: np.ndarray = resize(decoded, crop_shape,
                                     mode="constant",
                                     preserve_range=True,
                                     anti_aliasing=False,
                                     order=0).astype(np.int16)

        paste_crop(res_arr[:, :, z], resized, crop_origin)

    assert set(np.unique(res_arr)) <= set(range(K)), np.unique(res_arr)
    assert output_shape == res_arr.shape, (output_shape, res_arr.shape)

    res_arr, affine, header = restore_to_source_geometry(id_, source_pattern, metadata, res_arr)
    assert set(np.unique(res_arr)) <= set(range(K)), np.unique(res_arr)

    header.set_data_dtype(np.int16)
    new_nib = nib.nifti1.Nifti1Image(res_arr, affine=affine, header=header)
    dest_path = (dest_folder / id_).with_suffix(".nii.gz")
    nib.save(new_nib, dest_path)
    return dest_path


def group_patient_slices(data_folder: Path, grp_regex: str) -> tuple[list[Path], dict[str, list[int]]]:
    images: list[Path] = sorted(data_folder.glob("*.png"))
    if not images:
        raise ValueError(f"No PNG images found in {data_folder}")

    grouping_regex: Pattern = re.compile(grp_regex)
    matches = [grouping_regex.match(image.stem) for image in images]
    missing = [image.name for image, match in zip(images, matches) if match is None]
    if missing:
        raise ValueError(f"Some images did not match regex {grp_regex}: {missing[:5]}")

    idx_map: dict[str, list[int]] = {}
    for i, match in enumerate(matches):
        assert match is not None
        patient = match.group(1)
        idx_map.setdefault(patient, []).append(i)

    assert sum(len(idxes) for idxes in idx_map.values()) == len(images)
    return images, idx_map


def stitch_predictions(data_folder: Path, dest_folder: Path, source_scan_pattern: str,
                       grp_regex: str, num_classes: int,
                       preprocess_metadata_path: Path | None = None,
                       patient_ids: list[str] | None = None,
                       label_scale: float | None = None) -> list[Path]:
    images, idx_map = group_patient_slices(data_folder, grp_regex)
    selected_patients = sorted(idx_map) if patient_ids is None else patient_ids
    print(f"Found {len(selected_patients)} patients out of {len(images)} images ; regex: {grp_regex}")

    preprocess_metadata = load_preprocess_metadata(preprocess_metadata_path)
    dest_folder.mkdir(parents=True, exist_ok=True)

    return [merge_patient(p, dest_folder, images, idx_map[p], num_classes, source_scan_pattern,
                          preprocess_metadata=preprocess_metadata, label_scale=label_scale)
            for p in tqdm_(selected_patients)]


def main(args) -> None:
    stitch_predictions(data_folder=args.data_folder,
                       dest_folder=args.dest_folder,
                       source_scan_pattern=args.source_scan_pattern,
                       grp_regex=args.grp_regex,
                       num_classes=args.num_classes,
                       preprocess_metadata_path=args.preprocess_metadata,
                       label_scale=args.label_scale)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merging slices parameters')
    parser.add_argument('--data_folder', type=Path, required=True,
                        help="The folder containing the images to predict")
    parser.add_argument('--source_scan_pattern', type=str, required=True,
                        help="The pattern to get the original scan. This is used to get the correct metadata")
    parser.add_argument('--dest_folder', type=Path, required=True)
    parser.add_argument('--grp_regex', type=str, required=True)

    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--preprocess_metadata', type=Path, default=None,
                        help="Optional preprocess.json. When provided, predictions are first rebuilt in the "
                             "processed grid, then restored to the source NIfTI shape/header.")
    parser.add_argument('--label_scale', type=float, default=None,
                        help="Optional PNG label scale. Defaults to 63 for 5 classes, otherwise 255 / (K - 1).")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
