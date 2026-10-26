#!/usr/bin/env python3
"""Build a resumable full-FOV SegTHOR cache for the official 35/5 split.

The cache mirrors preprocess.py with HU [-1000, 500], 1 mm isotropic
resampling, and 256x256 planes. It bypasses PNG encoding only: labels are
stored as classes 0..4 instead of PNG values 0,63,126,189,252. Both arrays
are uint8 in [Z,X,Y] order (the repository calls the latter axes H,W).
Native NIfTI files remain available for organ crops and volume validation.

Run on a compute node, for example::

    python experiments/trachea_data.py --archive /path/to/train.zip \
        --dest /path/to/cache --workers 4 --max-seconds 1200

Completed cases survive a timeout and are reused on the next invocation.
No model results or existing Phase 7 provenance are implied by this cache.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize


REPO = Path(__file__).resolve().parents[1]
CLUSTER_ROOT = REPO.parent / "snellius-medicalai"
PATIENT_IDS = [f"Patient_{number:02d}" for number in range(1, 41)]
SETTINGS = {
    "hu_min": -1000.0,
    "hu_max": 500.0,
    "target_spacing": [1.0, 1.0, 1.0],
    "shape": [256, 256],
    "mm_per_px": 0,
    "array_axes": "Z,X,Y",
    "image_dtype": "uint8",
    "label_dtype": "uint8",
    "label_values": [0, 1, 2, 3, 4],
    "schema_version": 1,
}


def get_nibabel():
    try:
        import nibabel
    except ModuleNotFoundError:
        packages = CLUSTER_ROOT / "python-packages"
        if not (packages / "nibabel").is_dir():
            raise RuntimeError("nibabel is unavailable; load the project Python environment")
        sys.path.insert(0, str(packages))
        import nibabel
    return nibabel


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_splits() -> dict:
    shuffled = PATIENT_IDS.copy()
    random.Random(0).shuffle(shuffled)
    val = sorted(shuffled[:5])
    assert val == ["Patient_01", "Patient_13", "Patient_22", "Patient_28", "Patient_30"]
    return {"seed": 0, "train": sorted(set(PATIENT_IDS) - set(val)), "val": val}


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Cache wall-clock budget exhausted")


def load_manifest(dest: str | Path, require_complete: bool = True) -> dict:
    manifest = json.loads((Path(dest) / "manifest.json").read_text())
    if require_complete and manifest["status"] != "complete":
        raise ValueError(f"Cache is {manifest['status']}, not complete")
    return manifest


def extract_member(archive: zipfile.ZipFile, member: str, destination: Path,
                   deadline: float) -> dict:
    """Stream extraction and SHA256 together; ZIP's reader also validates CRC."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    digest = hashlib.sha256()
    with archive.open(member) as source, temporary.open("wb") as target:
        while True:
            check_deadline(deadline)
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            target.write(block)
    temporary.replace(destination)
    info = archive.getinfo(member)
    return {"member": member, "sha256": digest.hexdigest(),
            "bytes": info.file_size, "zip_crc32": info.CRC}


def create_arrays(ct_nib, gt_nib, image_path: Path, label_path: Path,
                  deadline: float, settings: dict = SETTINGS) -> dict:
    """Keep the production order: full 3D resampling, HU clipping, 2D resize."""
    ct = np.asarray(ct_nib.dataobj).astype(np.float32)
    gt = np.asarray(gt_nib.dataobj)
    spacing = tuple(float(value) for value in ct_nib.header.get_zooms()[:3])
    if ct.ndim != 3 or ct.shape != gt.shape:
        raise ValueError("Expected matching three-dimensional CT and GT volumes")
    if not np.allclose(ct_nib.affine, gt_nib.affine, rtol=0, atol=1e-4):
        raise ValueError("CT and GT affine mismatch")
    if not np.allclose(spacing, gt_nib.header.get_zooms()[:3], rtol=0, atol=1e-5):
        raise ValueError("CT and GT spacing mismatch")
    if not np.isin(gt, SETTINGS["label_values"]).all():
        raise ValueError("Expected GT labels 0..4")
    source_shape = list(ct.shape)
    factors = tuple(src / dst for src, dst in zip(spacing, settings["target_spacing"]))
    check_deadline(deadline)
    ct = zoom(ct, factors, order=1).astype(np.float32)
    check_deadline(deadline)
    gt = zoom(gt, factors, order=0).astype(gt.dtype)
    processed_shape = list(ct.shape)
    # Keep this expression identical to preprocess.py, including float32
    # rounding before uint8 truncation. Clipping before resampling is different.
    ct = 255 * (np.clip(ct, settings["hu_min"], settings["hu_max"]) - settings["hu_min"]) / (
        settings["hu_max"] - settings["hu_min"])
    shape = tuple(settings["shape"])
    cached_shape = (ct.shape[2], *shape)
    image_temporary = image_path.with_name(image_path.name + ".tmp")
    label_temporary = label_path.with_name(label_path.name + ".tmp")
    images = np.lib.format.open_memmap(image_temporary, mode="w+", dtype=np.uint8, shape=cached_shape)
    labels = np.lib.format.open_memmap(label_temporary, mode="w+", dtype=np.uint8, shape=cached_shape)
    foreground_z = []
    trachea_z = []
    for z in range(ct.shape[2]):
        check_deadline(deadline)
        images[z] = resize(ct[:, :, z], shape, order=1, mode="constant",
                           preserve_range=True, anti_aliasing=False).astype(np.uint8)
        labels[z] = resize(gt[:, :, z], shape, order=0, mode="constant",
                           preserve_range=True).astype(np.uint8)
        if np.any(labels[z]):
            foreground_z.append(z)
        if np.any(labels[z] == 3):
            trachea_z.append(z)
    images.flush()
    labels.flush()
    del images, labels, ct, gt
    image_temporary.replace(image_path)
    label_temporary.replace(label_path)
    return {
        "source_shape": source_shape,
        "source_spacing": list(spacing),
        "source_affine": ct_nib.affine.tolist(),
        "processed_spacing": list(settings["target_spacing"]),
        "processed_shape": processed_shape,
        "crop_origin": [0, 0],
        "crop_shape": processed_shape[:2],
        "slice_shape": list(shape),
        "cached_shape": list(cached_shape),
        "cache_axes": "Z,X,Y",
        "foreground_z": foreground_z,
        "trachea_z": trachea_z,
    }


def cache_patient(task: tuple) -> dict:
    archive_path, dest_string, patient, fingerprint, deadline = task
    dest = Path(dest_string)
    directory = dest / patient
    metadata_path = directory / "patient.json"
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError(f"{patient}: incompatible existing cache; use a new destination")
        paths = [saved[key] for key in ("image_path", "label_path", "native_ct_path", "native_gt_path")]
        if all((dest / path).is_file() for path in paths):
            for key in ("image_path", "label_path"):
                array = np.load(dest / saved[key], mmap_mode="r", allow_pickle=False)
                if list(array.shape) != saved["cached_shape"] or array.dtype != np.uint8:
                    raise ValueError(f"{patient}: invalid {key} header")
            return saved
    check_deadline(deadline)
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    native_ct = dest / "native" / patient / f"{patient}.nii.gz"
    native_gt = dest / "native" / patient / "GT.nii.gz"
    with zipfile.ZipFile(archive_path) as archive:
        ct_provenance = extract_member(archive, f"train/{patient}/{patient}.nii.gz", native_ct, deadline)
        gt_provenance = extract_member(archive, f"train/{patient}/GT.nii.gz", native_gt, deadline)
    nib = get_nibabel()
    metadata = create_arrays(nib.load(str(native_ct)), nib.load(str(native_gt)),
                             directory / "image.npy", directory / "label.npy", deadline)
    metadata.update({
        "patient": patient,
        "fingerprint": fingerprint,
        "image_path": str((directory / "image.npy").relative_to(dest)),
        "label_path": str((directory / "label.npy").relative_to(dest)),
        "native_ct_path": str(native_ct.relative_to(dest)),
        "native_gt_path": str(native_gt.relative_to(dest)),
        "source_ct": ct_provenance,
        "source_gt": gt_provenance,
        "seconds": time.monotonic() - started,
    })
    atomic_json(metadata_path, metadata)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=CLUSTER_ROOT / "datasets/segthor-official-train.zip")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--max-seconds", type=float, default=1200)
    args = parser.parse_args()
    if not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive")
    return args


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    deadline = started + args.max_seconds
    archive_path = args.archive.resolve()
    dest = args.dest.resolve()
    stat = archive_path.stat()
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for patient in PATIENT_IDS:
            for filename in (f"{patient}.nii.gz", "GT.nii.gz"):
                member = f"train/{patient}/{filename}"
                info = archive.getinfo(member)
                members[member] = {"bytes": info.file_size, "zip_crc32": info.CRC}
    splits = make_splits()
    source_sha256 = {"preprocess.py": file_sha256(REPO / "preprocess.py"),
                     "experiments/trachea_data.py": file_sha256(Path(__file__))}
    fingerprint = object_sha256({"settings": SETTINGS, "archive_members": members,
                                 "production_preprocess_sha256": source_sha256["preprocess.py"]})
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior["fingerprint"] != fingerprint:
            raise ValueError("Existing cache uses different inputs/settings; use a new destination")
    manifest = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Official-40 cache for matched evaluation; does not establish original ENet training provenance.",
        "fingerprint": fingerprint,
        "settings": SETTINGS,
        "archive": {"path": str(archive_path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "members": members,
                    "hash_note": "Compressed native CT/GT SHA256 recorded per patient during extraction; archive members also CRC-checked."},
        "splits": splits,
        "split_source": "preprocess.py seed=0 retains=5",
        "split_sha256": object_sha256(splits),
        "source_sha256": source_sha256,
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-image"]},
        "patient_ids": PATIENT_IDS,
        "patients": {},
        "workers": args.workers,
        "max_seconds": args.max_seconds,
    }

    def checkpoint():
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["completed_patients"] = sorted(manifest["patients"])
        atomic_json(manifest_path, manifest)

    checkpoint()
    tasks = [(str(archive_path), str(dest), patient, fingerprint, deadline) for patient in PATIENT_IDS]
    pool = multiprocessing.get_context("spawn").Pool(args.workers)
    completed = False
    try:
        iterator = pool.imap_unordered(cache_patient, tasks, chunksize=1)
        for _ in PATIENT_IDS:
            check_deadline(deadline)
            metadata = iterator.next(timeout=max(0.001, deadline - time.monotonic()))
            patient = metadata["patient"]
            manifest["patients"][patient] = metadata
            checkpoint()
            print(json.dumps({"patient": patient, "cached_shape": metadata["cached_shape"],
                              "completed": len(manifest["patients"]),
                              "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        manifest["status"] = "complete"
        completed = True
    except (TimeoutError, multiprocessing.TimeoutError) as exc:
        manifest["status"] = "budget_exhausted"
        manifest["stop_reason"] = str(exc) or "Cache wall-clock budget exhausted"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if completed:
            pool.close()
        else:
            pool.terminate()
        pool.join()
        checkpoint()
    print(json.dumps({"status": manifest["status"], "completed": len(manifest["patients"]),
                      "elapsed_seconds": manifest["elapsed_seconds"], "manifest": str(manifest_path)}), flush=True)
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
