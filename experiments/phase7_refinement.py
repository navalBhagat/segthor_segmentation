#!/usr/bin/env python3
"""Train the retained organ refiners on one frozen Phase 7 ENet seed.

Run ``--stage baseline`` first to reproduce the supplied ENet checkpoint's
validation result. Then run ``--stage all`` (or coarse, crops, train, predict,
evaluate separately). Each seed needs its own destination. Baseline and refined
predictions are scored both raw and with the same latest postprocessing policy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ORGANS = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
ORDER = (3, 1, 2)
VARIANTS = {3: "batch", 1: "batch", 2: "per_image"}
VALIDATION = [f"Patient_{i:02d}" for i in (1, 13, 22, 28, 30)]
LATEST_PP = {"min_ml": 1.0, "min_classes": [1, 3], "largest_classes": [4],
             "heart": "unchanged", "closing": False, "fill_holes": False}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lock_protocol(args):
    manifest = read_json(args.data_cache / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("The prepared data cache must be complete")
    expected = {"hu_min": -1000., "hu_max": 500., "target_spacing": [1., 1., 1.],
                "shape": [256, 256], "mm_per_px": 0}
    if any(manifest["settings"].get(k) != v for k, v in expected.items()):
        raise ValueError("Phase 7 requires full-FOV 256px, isotropic 1mm, HU [-1000,500] preprocessing")
    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    all_patients = {f"Patient_{i:02d}" for i in range(1, 41)}
    if sorted(val) != VALIDATION or set(train) != all_patients - set(val) or len(train) != 35:
        raise ValueError("Expected the reported Phase 7 official-40 35/5 patient split")
    sources = ["ENet.py", "postprocess.py", "metrics_3d.py", "experiments/phase7_refinement.py",
               "experiments/trachea_cache_predictions.py", "experiments/trachea_data.py",
               "experiments/trachea_evaluate.py", "experiments/organ_refine_data.py",
               "experiments/organ_refine_model.py", "experiments/organ_refine_train.py",
               "experiments/organ_refine_pipeline.py"]
    protocol = {"format_version": 1, "seed": args.seed, "epochs": args.epochs,
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
                "data_cache": str(args.data_cache), "data_fingerprint": manifest["fingerprint"],
                "train_patients": train, "validation_patients": val,
                "frozen_enet": True, "refinement_order": list(ORDER),
                "variants": {str(k): v for k, v in VARIANTS.items()},
                "batch_size": args.batch_size, "coarse_batch_size": args.coarse_batch_size,
                "learning_rate": 0.001, "widths": [8, 16, 32],
                "crop_size": 128, "augmentation": "deterministic_xy_flips",
                "checkpoint_selection": args.checkpoint_selection,
                "checkpoint_selection_metric": "native 3D validation Dice for the refined organ",
                "latest_postprocessing": LATEST_PP,
                "source_sha256": {name: sha256(ROOT / name) for name in sources}}
    args.dest.mkdir(parents=True, exist_ok=True)
    path = args.dest / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Destination protocol changed; use a fresh destination and frozen source")
    if not path.exists():
        write_json(path, protocol)
    return manifest, protocol


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Phase 7 refinement stage exceeded its time budget")
    return seconds


def run_script(args, deadline, script, *arguments, bounded=True):
    command = [sys.executable, "-u", str(ROOT / script), *map(str, arguments)]
    if bounded:
        command += ["--max-seconds", str(remaining(deadline))]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, timeout=remaining(deadline))


def coarse(args, deadline, split="all"):
    run_script(args, deadline, "experiments/trachea_cache_predictions.py",
               "--data-cache", args.data_cache, "--checkpoint", args.checkpoint,
               "--dest", args.dest / "coarse", "--split", split,
               "--batch-size", args.coarse_batch_size, "--device", args.device,
               "--threads", args.workers)


def crops(args, manifest, deadline):
    for class_id in selected_organs(args):
        for split in ("train", "val"):
            run_script(args, deadline, "experiments/organ_refine_data.py",
                       "--cache", args.data_cache, "--predictions", args.dest / "coarse",
                       "--dest", args.dest / "crops" / f"class-{class_id}" / split,
                       "--class-id", class_id, "--size", 128, "--workers", args.workers,
                       "--patients", *manifest["splits"][split],
                       *(["--inference"] if split == "val" else []))


def selected_organs(args):
    return (args.class_id,) if args.class_id is not None else ORDER


def train(args, manifest, deadline):
    for class_id in selected_organs(args):
        destination = args.dest / "models" / f"class-{class_id}"
        run_script(args, deadline, "experiments/organ_refine_train.py", "train",
                   "--crops", args.dest / "crops" / f"class-{class_id}" / "train",
                   "--dest", destination, "--class-id", class_id,
                   "--train-patients", *manifest["splits"]["train"],
                   "--validation-patients", *manifest["splits"]["val"],
                   "--variant", VARIANTS[class_id], "--epochs", args.epochs,
                   "--seed", args.seed, "--batch-size", args.batch_size,
                   "--learning-rate", 0.001, "--widths", 8, 16, 32,
                   "--workers", args.workers, "--device", args.device, "--augment",
                   "--checkpoint-selection", "best", "--validation-crops",
                   args.dest / "crops" / f"class-{class_id}" / "val",
                   "--data-cache", args.data_cache, "--coarse-predictions", args.dest / "coarse",
                   *(["--resume"] if (destination / "last.pt").exists() else []))


def require_final_checkpoint(args, class_id):
    directory = args.dest / "models" / f"class-{class_id}"
    status = read_json(directory / "status.json")
    if status.get("status") != "complete" or status.get("completed_epochs") != args.epochs:
        raise ValueError(f"Class {class_id} did not finish all {args.epochs} epochs")
    checkpoint = directory / "best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Class {class_id} has no best checkpoint")
    return checkpoint


def predict(args, manifest, deadline):
    for class_id in selected_organs(args):
        checkpoint = require_final_checkpoint(args, class_id)
        run_script(args, deadline, "experiments/organ_refine_train.py", "predict",
                   "--crops", args.dest / "crops" / f"class-{class_id}" / "val",
                   "--dest", args.dest / "predictions" / f"class-{class_id}",
                   "--checkpoint", checkpoint, "--patients", *manifest["splits"]["val"],
                   "--batch-size", args.batch_size, "--device", args.device)


def latest_postprocess(volume, spacing):
    import numpy as np
    from postprocess import remove_pieces
    result = volume.copy()
    voxel_ml = float(np.prod(spacing)) / 1000.
    for class_id in LATEST_PP["min_classes"]:
        remove_pieces(result, class_id, LATEST_PP["min_ml"] / voxel_ml)
    for class_id in LATEST_PP["largest_classes"]:
        remove_pieces(result, class_id, np.inf)
    return result


def exact_metrics(prediction, truth, spacing):
    import numpy as np
    from metrics_3d import dice_3d, hd95_3d, rvd_3d
    arrays = {"dice": dice_3d(prediction, truth, 5),
              "hd95_mm": hd95_3d(prediction, truth, 5, tuple(spacing)),
              "rvd": rvd_3d(prediction, truth, 5)}
    arrays["absolute_rvd"] = np.abs(arrays["rvd"])
    organs = {name: {metric: float(values[class_id]) if np.isfinite(values[class_id]) else None
                     for metric, values in arrays.items()} for class_id, name in ORGANS.items()}
    result = {"organs": organs}
    for metric, key in (("dice", "macro_organ_dice"), ("hd95_mm", "macro_hd95_mm"),
                        ("absolute_rvd", "macro_absolute_rvd")):
        values = [row[metric] for row in organs.values() if row[metric] is not None]
        result[key] = sum(values) / len(values) if values else None
    return result


def evaluate_patient(task):
    # All data-dependent imports occur in workers, so --help requires only Python.
    patient, metadata, cache, destination, output, stages, deadline, checkpoint_hash = task
    import numpy as np
    import torch
    torch.set_num_threads(1)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.organ_refine_pipeline import compose_patient, load_native, save_native, validate_coarse
    from experiments.trachea_evaluate import restore_hard_label_volume

    remaining(deadline)
    coarse_root = destination / "coarse"
    coarse_proof = read_json(coarse_root / patient / "patient.json")
    validate_coarse(coarse_proof, metadata)
    if coarse_proof["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Coarse prediction belongs to another ENet checkpoint")
    if stages:
        baseline, refined, composition = compose_patient(patient, metadata, coarse_root, stages, deadline)
        volumes = {"raw_enet": baseline, "raw_refined": refined}
    else:
        prediction = np.load(coarse_root / patient / "prediction.npy", allow_pickle=False)
        baseline = restore_hard_label_volume(prediction, metadata["processed_shape"], metadata["source_shape"])
        volumes, composition = {"raw_enet": baseline}, None
    # Composition is complete before targets are read.
    truth = load_native(cache / metadata["native_gt_path"], metadata)
    rows, artifacts = {}, {}
    for raw_name, raw_volume in volumes.items():
        for name, volume in ((raw_name, raw_volume),
                             (raw_name.replace("raw_", "latest_"), latest_postprocess(raw_volume, metadata["source_spacing"]))):
            remaining(deadline)
            path = output / "volumes" / name / f"{patient}.nii.gz"
            save_native(path, volume, metadata)
            rows[name] = exact_metrics(volume, truth, metadata["source_spacing"])
            artifacts[name] = {"path": str(path), "sha256": sha256(path)}
    result = {"patient": patient, "metrics": rows, "volumes": artifacts,
              "coarse_checkpoint_sha256": checkpoint_hash,
              "coarse_prediction_sha256": sha256(coarse_root / patient / "prediction.npy"),
              "composition": composition}
    write_json(output / "patients" / f"{patient}.json", result)
    return result


def summarize(patients):
    def mean(values):
        defined = [value for value in values if value is not None]
        return sum(defined) / len(defined) if defined else None
    output = {}
    first = next(iter(patients.values()))
    for candidate in first:
        rows = [patient[candidate] for patient in patients.values()]
        output[candidate] = {key: mean([row[key] for row in rows]) for key in
                             ("macro_organ_dice", "macro_hd95_mm", "macro_absolute_rvd")}
        output[candidate]["organs"] = {organ: {
            metric: mean([row["organs"][organ][metric] for row in rows])
            for metric in ("dice", "hd95_mm", "rvd", "absolute_rvd")} for organ in ORGANS.values()}
    return output


def evaluate(args, manifest, protocol, deadline, refined=True):
    output = args.dest / ("evaluation" if refined else "baseline")
    stages = []
    refiner_checkpoints = {}
    if refined:
        for class_id in ORDER:
            checkpoint = require_final_checkpoint(args, class_id)
            checkpoint_hash = sha256(checkpoint)
            prediction_root = args.dest / "predictions" / f"class-{class_id}"
            prediction = read_json(prediction_root / "prediction.json")
            if prediction.get("class_id") != class_id or prediction.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError(f"Class {class_id} cached prediction does not match selected checkpoint; rerun predict in a fresh destination")
            refiner_checkpoints[str(class_id)] = {"path": str(checkpoint), "sha256": checkpoint_hash}
            stages.append({"class_id": class_id, "kind": "logits",
                           "crops": str(args.dest / "crops" / f"class-{class_id}" / "val"),
                           "predictions": str(prediction_root)})
        write_json(args.dest / "stages.json", stages)
    report = {"status": "running", "seed": args.seed, "epochs": args.epochs,
              "protocol_sha256": sha256(args.dest / "protocol.json"),
              "checkpoint_sha256": protocol["checkpoint_sha256"],
              "checkpoint_selection": args.checkpoint_selection,
              "refiner_checkpoints": refiner_checkpoints,
              "metric_source": "metrics_3d.py", "postprocessing": LATEST_PP,
              "aggregation": "equal foreground organs within each patient, equal validation patients",
              "patients": {}, "summary": {}}
    write_json(output / "report.json", report)
    tasks = [(patient, manifest["patients"][patient], args.data_cache, args.dest,
              output, stages, deadline, protocol["checkpoint_sha256"])
             for patient in manifest["splits"]["val"]]
    try:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for result in pool.map(evaluate_patient, tasks):
                report["patients"][result["patient"]] = result["metrics"]
                report["summary"] = summarize(report["patients"])
                write_json(output / "report.json", report)
                print(json.dumps({"patient": result["patient"], "metrics": result["metrics"]}), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "report.json", report)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "coarse", "crops", "train", "predict", "evaluate", "all"), default="all")
    for name in ("data-cache", "checkpoint", "dest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026), required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--checkpoint-selection", choices=("best",), default="best",
                        help="Select best native 3D validation organ Dice after the full training schedule")
    parser.add_argument("--class-id", type=int, choices=ORDER,
                        help="Run only this organ for crops, train or predict stages")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coarse-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-seconds", type=float, default=14000.)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.coarse_batch_size, args.workers) < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("Epochs, batch sizes, workers and time budget must be positive")
    if args.class_id is not None and args.stage not in ("crops", "train", "predict"):
        parser.error("--class-id is only valid for crops, train and predict")
    for name in ("data_cache", "checkpoint", "dest"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main():
    args = parse_args()
    deadline = time.monotonic() + args.max_seconds
    manifest, protocol = lock_protocol(args)
    stage = args.stage
    if stage in ("baseline", "all"):
        coarse(args, deadline, split="val")
        evaluate(args, manifest, protocol, deadline, refined=False)
    if stage in ("coarse", "all"):
        coarse(args, deadline)
    if stage in ("crops", "all"):
        crops(args, manifest, deadline)
    if stage in ("train", "all"):
        train(args, manifest, deadline)
    if stage in ("predict", "all"):
        predict(args, manifest, deadline)
    if stage in ("evaluate", "all"):
        evaluate(args, manifest, protocol, deadline)
    print(json.dumps({"status": "complete", "stage": stage, "seed": args.seed, "destination": str(args.dest)}), flush=True)


if __name__ == "__main__":
    main()
