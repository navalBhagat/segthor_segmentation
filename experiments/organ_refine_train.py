#!/usr/bin/env python3
"""Complete-epoch organ refinement and class-bound, target-free prediction."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from organ_refine_model import VARIANTS, PRIOR_CLIP, Refiner, canonical_variant, model_logits, refiner_loss


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_hashes():
    return {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("organ_refine_model.py"))}


def checkpoint_save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Time budget exhausted; only complete epochs/patients are published")


class Crops(Dataset):
    """Read class-bound crops; inference never opens target or validity arrays."""
    def __init__(self, root, patients, class_id, variant="batch", training=False, seed=42, augment=False):
        canonical_variant(variant, class_id)
        if not patients or len(set(patients)) != len(patients):
            raise ValueError("Explicit unique patients are required")
        if any(Path(patient).name != patient or patient in (".", "..") for patient in patients):
            raise ValueError("Patient IDs must be directory names")
        self.root, self.patients, self.class_id = Path(root), list(patients), class_id
        self.training, self.seed, self.augment, self.epoch = training, seed, augment, 0
        self.arrays, self.geometry, self.indices = {}, {}, []
        self.keys = ["ct", "base"] + (["target", "valid"] if training else [])
        for patient in self.patients:
            directory = self.root / patient
            geometry = json.loads((directory / "geometry.json").read_text())
            if geometry.get("class_id") != class_id:
                raise ValueError(f"Crop class_id does not match requested organ: {patient}")
            if not geometry.get("fingerprint"):
                raise ValueError(f"Missing crop fingerprint: {patient}")
            if training and (geometry.get("contains_ground_truth") is not True or geometry.get("target_is_placeholder", False)):
                raise ValueError(f"Training requires genuine targets: {patient}")
            shape = tuple(geometry["array_shape"])
            if len(shape) != 3 or min(shape) < 1 or min(shape[1:]) < 4:
                raise ValueError(f"Expected nonempty Z,H,W crop with H,W >= 4: {patient}")
            if geometry["source_shape"][2] != shape[0]:
                raise ValueError(f"Crop must preserve every native axial slice: {patient}")
            spacing = geometry["source_spacing"]
            if len(spacing) != 3 or not all(math.isfinite(v) and v > 0 for v in spacing):
                raise ValueError(f"Positive finite native spacing is required: {patient}")
            arrays = {key: np.load(directory / (key + ".npy"), mmap_mode="r", allow_pickle=False) for key in self.keys}
            for key, array in arrays.items():
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite or mismatched {key} array: {patient}")
                if key in ("target", "valid") and not np.isin(array, [0, 1]).all():
                    raise ValueError(f"{key} must be binary: {patient}")
                if key == "ct" and (array.min() < 0 or array.max() > 1):
                    raise ValueError(f"CT must be normalized to [0,1]: {patient}")
            self.arrays[patient], self.geometry[patient] = arrays, geometry
            self.indices.extend((patient, z) for z in range(shape[0]))

    def fingerprints(self):
        return {patient: {"class_id": self.class_id,
                          "geometry": sha256(self.root / patient / "geometry.json"),
                          "crop_fingerprint": self.geometry[patient]["fingerprint"],
                          "arrays": {key: sha256(self.root / patient / (key + ".npy")) for key in self.keys}}
                for patient in self.patients}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        patient, z = self.indices[index]
        arrays = self.arrays[patient]
        neighbors = np.clip([z - 1, z, z + 1], 0, len(arrays["ct"]) - 1)
        image = torch.from_numpy(np.asarray(arrays["ct"][neighbors], dtype=np.float32))
        base = torch.from_numpy(np.array(arrays["base"][z], dtype=np.float32, copy=True))[None]
        result = {"image": image, "base": base, "patient": patient, "z": z}
        if self.training:
            for key in ("target", "valid"):
                result[key] = torch.from_numpy(np.array(arrays[key][z], dtype=np.float32, copy=True))[None]
        if self.augment:
            generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003 + index)
            axes = [axis for axis in (-2, -1) if torch.rand((), generator=generator) < 0.5]
            if axes:
                for key in ("image", "base", "target", "valid"):
                    if key in result:
                        result[key] = torch.flip(result[key], axes)
        return result


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def configure(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def native_organ_dice(prediction, truth, class_id):
    """One Dice ratio over the full native volume, including outside the crop."""
    if prediction.ndim != 3 or prediction.shape != truth.shape:
        raise ValueError("Expected matching full native XYZ prediction and ground truth")
    predicted, target = prediction == class_id, truth == class_id
    pred_count, gt_count = int(predicted.sum()), int(target.sum())
    intersection = int(np.count_nonzero(predicted & target))
    denominator = pred_count + gt_count
    return {"dice": 2. * intersection / denominator if denominator else None,
            "predicted_voxels": pred_count, "ground_truth_voxels": gt_count,
            "intersection_voxels": intersection}


class NativeDiceValidation:
    """Validate GT-free crop predictions after restoring the complete native CT.

    Each patient's organ contributes one full-volume Dice. Native ground truth
    is used only for scoring; it cannot affect crop placement or model inputs.
    """
    metric = "native_volume_organ_dice"

    def __init__(self, args, training):
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from experiments.organ_refine_pipeline import check_geometry, validate_coarse
        from experiments.trachea_data import load_manifest

        if not args.validation_patients or any(value is None for value in
                (args.validation_crops, args.data_cache, args.coarse_predictions)):
            raise ValueError("Best checkpoint selection requires validation patients/crops, data cache and coarse predictions")
        self.args = args
        self.dataset = Crops(args.validation_crops, args.validation_patients,
                             args.class_id, args.variant, training=False)
        manifest = load_manifest(args.data_cache)
        if (not set(training.patients) <= set(manifest["splits"]["train"])
                or not set(self.dataset.patients) <= set(manifest["splits"]["val"])):
            raise ValueError("Training and validation patients must follow the data cache's disjoint split")
        self.metadata = {patient: manifest["patients"][patient] for patient in self.dataset.patients}
        checkpoint_hashes, bindings = set(), {}
        for dataset in (training, self.dataset):
            for patient in dataset.patients:
                geometry, metadata = dataset.geometry[patient], manifest["patients"][patient]
                proof_path = args.coarse_predictions / patient / "patient.json"
                proof = json.loads(proof_path.read_text())
                check_geometry(geometry, metadata)
                validate_coarse(proof, metadata)
                source = geometry["source"]
                if (geometry.get("patient") != patient or proof.get("patient") != patient
                        or source.get("class_id") != args.class_id
                        or source.get("data") != metadata["fingerprint"]
                        or source.get("coarse") != proof["fingerprint"]
                        or geometry.get("coarse_checkpoint_sha256") != proof["checkpoint_sha256"]):
                    raise ValueError(f"Crop and coarse/data provenance differ: {patient}")
                checkpoint_hashes.add(proof["checkpoint_sha256"])
                if dataset is self.dataset:
                    arrays = {name: sha256(args.coarse_predictions / patient / f"{name}.npy")
                              for name in ("prediction", "logits")}
                    if any(source.get(name + "_sha256") != value for name, value in arrays.items()):
                        raise ValueError(f"Validation coarse arrays changed after crop preparation: {patient}")
                    bindings[patient] = {"coarse_metadata_sha256": sha256(proof_path),
                                         "coarse_arrays_sha256": arrays,
                                         "native_gt_sha256": sha256(args.data_cache / metadata["native_gt_path"])}
        if len(checkpoint_hashes) != 1:
            raise ValueError("Training and validation crops must use the same frozen ENet checkpoint")
        self.provenance = {"metric": self.metric, "class_id": args.class_id,
            "aggregation": "one full native-volume Dice per patient, then equal-patient mean",
            "selection": "maximum validation score; keep earliest epoch on exact ties",
            "postprocessing": "none", "crop_input_ground_truth_read": False,
            "data_cache": str(args.data_cache.resolve()), "data_fingerprint": manifest["fingerprint"],
            "manifest_sha256": sha256(args.data_cache / "manifest.json"),
            "coarse_predictions": str(args.coarse_predictions.resolve()),
            "coarse_checkpoint_sha256": next(iter(checkpoint_hashes)),
            "crop_root": str(args.validation_crops.resolve()), "crops": self.dataset.fingerprints(),
            "patients": bindings,
            "source_sha256": {name: sha256(root / "experiments" / name) for name in
                ("organ_refine_data.py", "organ_refine_pipeline.py", "trachea_data.py", "trachea_evaluate.py")}}
        self.baselines, self.truths = {}, {}
        self.indices = {patient: [i for i, (name, _) in enumerate(self.dataset.indices) if name == patient]
                        for patient in self.dataset.patients}

    @torch.no_grad()
    def evaluate(self, model, device, batch_size, deadline):
        from experiments.organ_refine_data import compose_crop_logits
        from experiments.organ_refine_pipeline import load_native
        from experiments.trachea_evaluate import restore_hard_label_volume

        previous_mode = model.training
        model.eval()
        patients = {}
        try:
            for patient in self.dataset.patients:
                check_deadline(deadline)
                geometry, metadata = self.dataset.geometry[patient], self.metadata[patient]
                logits = np.empty(geometry["array_shape"], dtype=np.float32)
                indices = self.indices[patient]
                for start in range(0, len(indices), batch_size):
                    check_deadline(deadline)
                    batch = [self.dataset[index] for index in indices[start:start + batch_size]]
                    images = torch.stack([item["image"] for item in batch]).to(device)
                    base = torch.stack([item["base"] for item in batch]).to(device)
                    scores = model_logits(model, images, base)[:, 0].cpu().numpy()
                    if not np.isfinite(scores).all():
                        raise RuntimeError(f"Nonfinite validation logits: {patient}")
                    for item, score in zip(batch, scores):
                        logits[item["z"]] = score
                if patient not in self.baselines:
                    coarse = np.load(self.args.coarse_predictions / patient / "prediction.npy",
                                     mmap_mode="r", allow_pickle=False)
                    self.baselines[patient] = restore_hard_label_volume(
                        coarse, metadata["processed_shape"], metadata["source_shape"])
                refined = compose_crop_logits(self.baselines[patient], logits,
                                               geometry["bbox_native_xy"], self.args.class_id)
                # First target access happens only after full-volume composition.
                if patient not in self.truths:
                    self.truths[patient] = load_native(self.args.data_cache / metadata["native_gt_path"], metadata)
                patients[patient] = native_organ_dice(refined, self.truths[patient], self.args.class_id)
        finally:
            model.train(previous_mode)
        scores = [row["dice"] for row in patients.values() if row["dice"] is not None]
        if not scores:
            raise ValueError("Validation organ Dice is undefined for every patient")
        return {"metric": self.metric, "class_id": self.args.class_id,
                "score": float(np.mean(scores)), "patients": patients,
                "defined_patients": len(scores), "aggregation": "equal native-volume patients"}


def save_epoch_checkpoints(destination, state, validation=None, previous_selection=None):
    """Commit the last epoch and retain the best native-volume validation state."""
    selected = validation is not None and (previous_selection is None or validation["score"] > previous_selection["score"])
    selection = ({"epoch": state["completed_epochs"], **validation} if selected else previous_selection)
    if validation is not None:
        state["selection"] = selection
    checkpoint_save(destination / "last.pt", state)
    if selected:
        checkpoint_save(destination / "best.pt", state)
    return selection


def restore_best_checkpoint(destination, state):
    """Repair an interrupted best-file publication using its committed last epoch."""
    selection = state.get("selection")
    if not selection:
        raise ValueError("Resume of best selection requires recorded native validation scores")
    best_path = destination / "best.pt"
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if (best.get("config") == state["config"] and best.get("selection") == selection
                and best.get("completed_epochs") == selection["epoch"]):
            return selection
    if selection["epoch"] == state["completed_epochs"]:
        checkpoint_save(best_path, state)
        return selection
    raise ValueError("The best native-validation checkpoint is missing or inconsistent with last.pt")


def training_config(args, dataset, validation):
    config = {"format_version": 2, "class_id": args.class_id, "variant": args.variant,
            "widths": list(args.widths), "input_channels": 3, "seed": args.seed,
            "train_patients": list(args.train_patients), "validation_patients": list(args.validation_patients),
            "crop_root": str(args.crops.resolve()), "data": dataset.fingerprints(),
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "optimizer": "Adam", "gradient_clip": 5.0,
            "augmentation": "deterministic_xy_flips" if args.augment else "none",
            "base_loss": "0.5 masked BCE + 0.5 foreground soft Dice",
            "dice_reduction": "image" if args.variant == "per_image" else "batch",
            "context": "adjacent native slices [-1,0,1]; clamp at scan edges",
            "residual_prior": {"source": f"central base.npy: L{args.class_id} - max(other class logits)",
                               "clip": [-PRIOR_CLIP, PRIOR_CLIP], "correction": "unbounded CT-only logit",
                               "zero_head_initialization": "fresh training only"},
            "training_native_slices": len(dataset), "updates_per_epoch": math.ceil(len(dataset) / args.batch_size),
            "epoch_definition": "every training native axial crop once, no drop_last",
            "selection": "best mean per-patient full native-volume organ Dice; earliest epoch on ties",
            "threshold": "combined binary logit > 0", "amp": False, "tf32": False,
            "device_type": torch.device(args.device).type, "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__), "source_sha256": source_hashes()}
    config["validation"] = validation.provenance
    return config


def train(args):
    if args.variant not in VARIANTS:
        raise ValueError("Training requires a named shared variant")
    canonical_variant(args.variant, args.class_id)
    if set(args.train_patients) & set(args.validation_patients):
        raise ValueError("Training and validation patients overlap")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Positive epochs/batch size and nonnegative workers required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Positive finite learning rate required")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("Positive finite time budget required")
    started = time.monotonic()
    deadline = started + args.max_seconds
    configure(args.seed)
    dataset = Crops(args.crops, args.train_patients, args.class_id, args.variant,
                    training=True, seed=args.seed, augment=args.augment)
    validation = NativeDiceValidation(args, dataset)
    config = training_config(args, dataset, validation)
    device = torch.device(args.device)
    model = Refiner(args.widths).to(device)
    if not args.resume:
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history, completed, updates, selection = [], 0, 0, None
    if args.resume:
        state = torch.load(args.dest / "last.pt", map_location="cpu", weights_only=False)
        if state["config"] != config:
            changed = sorted(key for key in set(config) | set(state["config"]) if config.get(key) != state["config"].get(key))
            raise ValueError(f"Resume configuration changed: {', '.join(changed)}")
        completed, updates, history = state["completed_epochs"], state["updates"], state["history"]
        if args.epochs < completed:
            raise ValueError("Requested epochs precede the checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        selection = restore_best_checkpoint(args.dest, state)
    elif args.dest.exists():
        raise FileExistsError(f"Choose a fresh destination or use --resume: {args.dest}")
    else:
        args.dest.mkdir(parents=True)
    write_json(args.dest / "run.json", config)
    write_json(args.dest / "history.json", history)
    write_json(args.dest / "invocation.json", {"requested_epochs": args.epochs, "resume": args.resume,
        "workers": args.workers, "max_seconds": args.max_seconds, "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    def write_status(status, **extra):
        value = {"status": status, "completed_epochs": completed, "requested_epochs": args.epochs,
                 "checkpoint_selection": args.checkpoint_selection, **extra}
        if selection is not None:
            value.update(selection=selection, best_epoch=selection["epoch"], best_score=selection["score"])
        write_json(args.dest / "status.json", value)

    write_status("running")
    try:
        for epoch in range(completed + 1, args.epochs + 1):
            check_deadline(deadline)
            dataset.epoch = epoch
            generator = torch.Generator().manual_seed(args.seed + epoch * 1009)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                                num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=False)
            model.train()
            sums = {key: 0.0 for key in ("total", "bce", "dice")}
            samples = 0
            epoch_start = time.monotonic()
            for batch in loader:
                check_deadline(deadline)
                image, target, valid, base = [batch[key].to(device, non_blocking=True)
                     for key in ("image", "target", "valid", "base")]
                optimizer.zero_grad(set_to_none=True)
                logits = model_logits(model, image, base)
                loss, components = refiner_loss(logits, target, valid, args.variant)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite refiner loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite refiner gradients")
                optimizer.step()
                count = len(image)
                for key, value in {"total": loss, **components}.items():
                    sums[key] += float(value.detach()) * count
                samples += count
                updates += 1
            if samples != len(dataset):
                raise RuntimeError("Incomplete native-slice epoch")
            row = {"epoch": epoch, "updates": updates, "training_samples": samples,
                   "loss": {key: value / samples for key, value in sums.items()}, "seconds": time.monotonic() - epoch_start}
            validation_started = time.monotonic()
            validation_result = validation.evaluate(model, device, args.batch_size, deadline)
            row["validation"] = validation_result
            row["validation_dice"] = validation_result["score"]
            row["validation_seconds"] = time.monotonic() - validation_started
            history.append(row)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "completed_epochs": epoch, "updates": updates, "history": history,
                     "config": config, "rng": rng_state()}
            selection = save_epoch_checkpoints(args.dest, state, validation_result, selection)
            completed = epoch
            write_json(args.dest / "history.json", history)
            write_status("running")
            print(json.dumps(row), flush=True)
    except Exception as error:
        write_status("budget_exhausted" if isinstance(error, TimeoutError) else "failed",
                     error=str(error), elapsed_seconds=time.monotonic() - started)
        raise
    write_status("complete", elapsed_seconds=time.monotonic() - started)
    return 0


@torch.no_grad()
def predict(args):
    if args.batch_size < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("Positive batch size and finite time budget are required")
    deadline = time.monotonic() + args.max_seconds
    configure(0)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    class_id = config["class_id"]
    variant = canonical_variant(config["variant"], class_id)
    if config.get("input_channels") != 3 or state["completed_epochs"] < 1:
        raise ValueError("Prediction requires a complete three-CT-channel refiner checkpoint")
    dataset = Crops(args.crops, args.patients, class_id, variant, training=False)
    source = {"class_id": class_id, "checkpoint_sha256": sha256(args.checkpoint),
              "completed_epochs": state["completed_epochs"], "variant": variant,
              "widths": config["widths"], "amp": False, "tf32": False,
              "crop_root": str(args.crops.resolve()), "data": dataset.fingerprints(), "training_config": config,
              "training_config_sha256": config_sha256(config),
              "ground_truth_read": False, "target_files_read": False,
              "output": "full combined binary logits: clipped coarse margin + unbounded correction, Z,H,W float32",
              "source_sha256": source_hashes()}
    if args.dest.exists():
        metadata = args.dest / "prediction.json"
        if not metadata.exists() or json.loads(metadata.read_text()) != source:
            raise ValueError("Prediction destination belongs to different model/data/configuration")
    else:
        args.dest.mkdir(parents=True)
        write_json(args.dest / "prediction.json", source)
    device = torch.device(args.device)
    model = Refiner(config["widths"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    for patient in dataset.patients:
        check_deadline(deadline)
        destination = args.dest / patient
        destination.mkdir(exist_ok=True)
        completed = destination / "patient.json"
        binding = {"patient": patient, "class_id": class_id,
                   "array_shape": list(dataset.geometry[patient]["array_shape"]),
                   "crop_fingerprint": dataset.geometry[patient]["fingerprint"],
                   "checkpoint_sha256": source["checkpoint_sha256"], "ground_truth_read": False,
                   "target_files_read": False, "completed_epochs": state["completed_epochs"],
                   "training_config_sha256": source["training_config_sha256"]}
        if completed.exists():
            metadata = json.loads(completed.read_text())
            output = destination / "logits.npy"
            if all(metadata.get(key) == value for key, value in binding.items()) and output.exists() and metadata.get("logits_sha256") == sha256(output):
                continue
            raise ValueError(f"Incomplete or changed published patient: {patient}")
        temporary = destination / "logits.npy.tmp"
        shape = tuple(dataset.geometry[patient]["array_shape"])
        indices = [index for index, (name, _) in enumerate(dataset.indices) if name == patient]
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
        try:
            for start in range(0, len(indices), args.batch_size):
                check_deadline(deadline)
                batch = [dataset[index] for index in indices[start:start + args.batch_size]]
                images = torch.stack([item["image"] for item in batch]).to(device)
                base = torch.stack([item["base"] for item in batch]).to(device)
                logits = model_logits(model, images, base)[:, 0].cpu().numpy()
                if not np.isfinite(logits).all():
                    raise RuntimeError(f"Nonfinite prediction: {patient}")
                for index, item in enumerate(batch):
                    values[item["z"]] = logits[index]
            values.flush()
            del values
            temporary.replace(destination / "logits.npy")
            write_json(completed, {**binding, "logits_sha256": sha256(destination / "logits.npy")})
        except Exception:
            if "values" in locals():
                del values
            temporary.unlink(missing_ok=True)
            raise
    return 0


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    training, inference = [commands.add_parser(name) for name in ("train", "predict")]
    for command in (training, inference):
        command.add_argument("--crops", type=Path, required=True)
        command.add_argument("--dest", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--device", default="cuda")
        command.add_argument("--max-seconds", type=float, default=3600)
    training.add_argument("--class-id", type=int, choices=(1, 2, 3, 4), required=True)
    training.add_argument("--train-patients", nargs="+", required=True)
    training.add_argument("--validation-patients", nargs="+", required=True)
    training.add_argument("--checkpoint-selection", choices=("best",), default="best")
    training.add_argument("--validation-crops", type=Path, required=True)
    training.add_argument("--data-cache", type=Path, required=True)
    training.add_argument("--coarse-predictions", type=Path, required=True)
    training.add_argument("--variant", choices=VARIANTS, default="batch")
    training.add_argument("--widths", nargs=3, type=int, default=[8, 16, 32])
    training.add_argument("--epochs", type=int, default=25)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--workers", type=int, default=2)
    training.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--resume", action="store_true")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--patients", nargs="+", required=True)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    sys.exit(train(args) if args.command == "train" else predict(args))
