#!/usr/bin/env python3
"""Cache deterministic coarse ENet predictions and full five-class logits.

Prediction caches bind checkpoint, preprocessing and inference source fingerprints.
Inference reads CT only; complete logits support all shared organ refiners."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ENet import ENet
from experiments.trachea_data import atomic_json, file_sha256, load_manifest, object_sha256


NORMALIZED_U8 = (np.arange(256) / 255).astype(np.float32)
INFERENCE = {"architecture": "ENet", "input_channels": 3, "classes": 5,
             "normalization": "cached_uint8 / 255", "neighbors": [-1, 0, 1],
             "neighbor_boundary": "clamp", "forward_dtype": "float32",
             "stored_logits_dtype": "float16", "array_axes": "Z,C,X,Y",
             "prediction_definition": "argmax_of_float32_logits_before_storage_quantization",
             "schema_version": 4}


def normalize_patient(value):
    if value.isdigit():
        return f"Patient_{int(value):02d}"
    return value


def select_patients(manifest, patients=None, split=None):
    available = set(manifest.get("patient_ids", manifest.get("patients", {})))
    if patients:
        selected = [normalize_patient(value) for value in patients]
    elif split == "all":
        selected = sorted(available)
    elif split in manifest["splits"]:
        selected = list(manifest["splits"][split])
    else:
        raise ValueError("Choose explicit --patients or an available --split")
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= available:
        raise ValueError("Patient selection must be nonempty, unique, and present in the data cache")
    return sorted(selected)


def context_batch(images_zxy, first, stop):
    """Three consecutive processed slices, edge replicated, matching training."""
    centers = np.arange(first, stop, dtype=np.intp)
    indices = np.clip(centers[:, None] + np.array([-1, 0, 1]), 0, len(images_zxy) - 1)
    return NORMALIZED_U8[images_zxy[indices]]


def load_model(checkpoint_path, device, data_fingerprint=None):
    # The checkpoint is an explicitly supplied, trusted local project artifact.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a raw state dictionary or model_state_dict checkpoint")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint does not contain a tensor model_state_dict")
    if checkpoint.get("epoch_complete") is False:
        raise ValueError("Use a completed-epoch checkpoint, not partial.pt")
    config = checkpoint.get("config", {})
    if data_fingerprint is not None and config.get("cache_fingerprint", data_fingerprint) != data_fingerprint:
        raise ValueError("Checkpoint preprocessing fingerprint differs from the supplied data cache")
    if state["conv0.weight"].shape[1] != 3:
        raise ValueError("Expected an ENet checkpoint with exactly three input slices")
    kernels = int(state["conv0.weight"].shape[0]) + 3
    projection_width = int(state["bottleneck1_0.block0.0.weight"].shape[0])
    if kernels % projection_width:
        raise ValueError("Cannot determine ENet projection factor from checkpoint")
    factor = kernels // projection_width
    if config.get("kernels", kernels) != kernels or config.get("factor", factor) != factor:
        raise ValueError("Checkpoint ENet config differs from its state tensor shapes")
    model = ENet(3, 5, kernels=kernels, factor=factor)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    provenance = {key: checkpoint[key] for key in ("epoch", "epochs_completed", "completed_epochs", "global_step")
                  if key in checkpoint and isinstance(checkpoint[key], (int, float, str))}
    provenance["architecture_config"] = {"kernels": kernels, "factor": factor, "input_slices": 3, "classes": 5}
    provenance["checkpoint_cache_fingerprint"] = config.get("cache_fingerprint")
    return model, provenance


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Prediction cache wall-clock budget exhausted")


def infer_arrays(model, images_zxy, device, batch_size, deadline):
    """Preserve ENet's FP32 labels while storing compact float16 refiner priors."""
    logits = np.empty((len(images_zxy), 5, *images_zxy.shape[1:]), dtype=np.float16)
    prediction = np.empty(images_zxy.shape, dtype=np.uint8)
    with torch.inference_mode():
        for first in range(0, len(images_zxy), batch_size):
            check_deadline(deadline)
            stop = min(first + batch_size, len(images_zxy))
            batch = torch.from_numpy(context_batch(images_zxy, first, stop)).to(device)
            scores = model(batch)
            expected = (stop - first, 5, *images_zxy.shape[1:])
            if tuple(scores.shape) != expected:
                raise ValueError(f"Unexpected model output {tuple(scores.shape)}; expected {expected}")
            values = scores.float().cpu().numpy()
            if not np.isfinite(values).all() or np.abs(values).max() > np.finfo(np.float16).max:
                raise ValueError("Logits are nonfinite or outside float16 storage range")
            # Quantizing first can turn close scores into ties and change ENet's
            # labels, so reproduce main.py/inference.py before compact storage.
            prediction[first:stop] = values.argmax(axis=1).astype(np.uint8)
            logits[first:stop] = values.astype(np.float16)
            del batch, scores, values
    return {"prediction": prediction, "logits": logits}


def _atomic_array(path, array):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def _valid_existing(directory, fingerprint, data_fingerprint):
    metadata_path = directory / "patient.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata["fingerprint"] != fingerprint or metadata["data_fingerprint"] != data_fingerprint:
        raise ValueError(f"Incompatible existing prediction cache: {directory}")
    required = ["prediction", "logits"]
    for name in required:
        if name not in metadata["outputs"]:
            return None
        item = metadata["outputs"][name]
        path = directory / item["file"]
        if not path.is_file():
            return None
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != item["shape"] or str(array.dtype) != item["dtype"]:
            raise ValueError(f"Invalid cached output header: {path}")
    return metadata


def cache_patient(patient, data_cache, destination, model, device, batch_size,
                  fingerprint, checkpoint_sha256, data_fingerprint, deadline):
    directory = destination / patient
    prior = _valid_existing(directory, fingerprint, data_fingerprint)
    if prior is not None:
        return prior, True
    started = time.monotonic()
    check_deadline(deadline)
    source = json.loads((data_cache / patient / "patient.json").read_text())
    if source["fingerprint"] != data_fingerprint:
        raise ValueError(f"Patient data fingerprint differs from manifest: {patient}")
    # Inference reads the cached CT array only; ground truth is not needed.
    images = np.load(data_cache / source["image_path"], mmap_mode="r", allow_pickle=False)
    if images.ndim != 3 or images.dtype != np.uint8 or list(images.shape) != source["cached_shape"]:
        raise ValueError(f"Invalid cached CT array: {patient}")
    arrays = infer_arrays(model, images, device, batch_size, deadline)
    check_deadline(deadline)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    # Metadata is the completion marker and is always replaced last.
    for name, array in arrays.items():
        check_deadline(deadline)
        path = directory / f"{name}.npy"
        _atomic_array(path, array)
        outputs[name] = {"file": path.name, "shape": list(array.shape), "dtype": str(array.dtype),
                         "bytes": path.stat().st_size}
    geometry_keys = ("source_shape", "source_spacing", "source_affine", "processed_shape",
                     "processed_spacing", "crop_shape", "crop_origin", "slice_shape", "cached_shape")
    metadata = {"patient": patient, "status": "complete", "fingerprint": fingerprint,
                "checkpoint_sha256": checkpoint_sha256, "data_fingerprint": data_fingerprint,
                "source_image_path": str((data_cache / source["image_path"]).resolve()),
                "source_ct_sha256": source.get("source_ct", {}).get("sha256"),
                "geometry": {key: source[key] for key in geometry_keys},
                "gt_read": False, "prediction_axes": "Z,X,Y", "logit_axes": "Z,C,X,Y",
                "inference": INFERENCE, "batch_size": batch_size, "outputs": outputs,
                "elapsed_seconds": time.monotonic() - started}
    atomic_json(directory / "patient.json", metadata)
    return metadata, False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--patients", nargs="+")
    selection.add_argument("--split", choices=("train", "val", "all"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("Batch size, threads and max seconds must be positive")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    deadline = started + args.max_seconds
    data_cache, destination = args.data_cache.resolve(), args.dest.resolve()
    manifest = load_manifest(data_cache)
    selected = select_patients(manifest, args.patients, args.split)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    source_hashes = {str(path.relative_to(REPO)): file_sha256(path) for path in
                     (Path(__file__).resolve(), REPO / "ENet.py")}
    fingerprint = object_sha256({"checkpoint_sha256": checkpoint_sha256,
                                 "data_fingerprint": manifest["fingerprint"],
                                 "batch_size": args.batch_size,
                                 "inference": INFERENCE, "source_sha256": source_hashes})
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "manifest.json"
    patients = {}
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        if prior["fingerprint"] != fingerprint:
            raise ValueError("Prediction destination has incompatible checkpoint/data/code; use a new destination")
        patients = prior.get("patients", {})
    report = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "fingerprint": fingerprint, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": checkpoint_sha256, "data_cache": str(data_cache),
              "data_fingerprint": manifest["fingerprint"], "source_sha256": source_hashes,
              "inference": INFERENCE, "requested_patients": selected,
              "patients": patients,
              "device": args.device, "batch_size": args.batch_size, "max_seconds": args.max_seconds,
              "gt_read": False, "completed_this_invocation": [], "reused_this_invocation": []}

    def checkpoint_report():
        report["elapsed_seconds"] = time.monotonic() - started
        report["available_patients"] = sorted(report["patients"])
        atomic_json(report_path, report)

    checkpoint_report()
    model = None
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; run on a GPU allocation")
        # FP32 forward with TF32 disabled makes logits portable across A100/H100.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        for patient in selected:
            check_deadline(deadline)
            existing = _valid_existing(destination / patient, fingerprint, manifest["fingerprint"])
            if existing is not None:
                metadata, reused = existing, True
            else:
                if model is None:
                    model, provenance = load_model(args.checkpoint, device, manifest["fingerprint"])
                    report["checkpoint_metadata"] = provenance
                metadata, reused = cache_patient(patient, data_cache, destination, model, device,
                                                args.batch_size, fingerprint, checkpoint_sha256,
                                                manifest["fingerprint"], deadline)
            report["patients"][patient] = metadata
            report["reused_this_invocation" if reused else "completed_this_invocation"].append(patient)
            checkpoint_report()
            print(json.dumps({"patient": patient, "reused": reused, "logits_saved": "logits" in metadata["outputs"],
                              "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
        report["status"] = "complete"
    except TimeoutError as exc:
        report["status"] = "budget_exhausted"
        report["stop_reason"] = str(exc)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        checkpoint_report()
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                      "manifest": str(report_path)}), flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
