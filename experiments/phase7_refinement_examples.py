#!/usr/bin/env python3
"""Render verified 3-slice ENet/refinement predictions without editing their masks.

All five validation patients use the same prespecified positive-slice Q25/Q75
views. Patient 13, native slice 115, is also retained for comparison with the
earlier presentation. Ground truth selects display views only, never model
crops. CT display uses bilinear interpolation; segmentation contours use the
stored labels. Run only after all three evaluation reports are complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np


ORGANS = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
PATIENTS = [f"Patient_{i:02d}" for i in (1, 13, 22, 28, 30)]
SEEDS = (42, 123, 2026)
WINDOW = (-160., 240.)
GT, PRED = "#70E4CF", "#FFB84A"
QUANTILES = (.25, .75)
ILLUSTRATION_PATIENT, ILLUSTRATION_Z = "Patient_13", 115


def read_json(path):
    return json.loads(Path(path).read_text())


def signature(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def native_volume(path, metadata, labels=False):
    image = nib.load(str(path))
    if (list(image.shape) != metadata["source_shape"]
            or not np.allclose(image.affine, metadata["source_affine"], atol=1e-4, rtol=0)
            or not np.allclose(image.header.get_zooms()[:3], metadata["source_spacing"], atol=1e-6, rtol=0)):
        raise ValueError(f"Native image geometry differs: {path}")
    values = np.asarray(image.dataobj)
    if not np.isfinite(values).all() or (labels and not np.isin(values, range(5)).all()):
        raise ValueError(f"Invalid CT/label values: {path}")
    return values.astype(np.uint8 if labels else np.float32)


def load_case(patient, cache, seed_root, report, manifest):
    metadata = manifest["patients"][patient]
    paths = {"ct": cache / metadata["native_ct_path"], "reference": cache / metadata["native_gt_path"],
             "raw_enet": seed_root / "evaluation/volumes/raw_enet" / f"{patient}.nii.gz",
             "final": seed_root / "evaluation/volumes/raw_refined" / f"{patient}.nii.gz",
             "crop_geometry": seed_root / "crops/class-1/val" / patient / "geometry.json"}
    inputs = {key: signature(path) for key, path in paths.items()}
    proof_path = seed_root / "evaluation/patients" / f"{patient}.json"
    proof = read_json(proof_path)
    if proof["metrics"] != report["patients"][patient]:
        raise ValueError(f"Patient proof differs from evaluation report: {patient}")
    for source, candidate in (("raw_enet", "raw_enet"), ("final", "raw_refined")):
        if inputs[source]["sha256"] != proof["volumes"][candidate]["sha256"]:
            raise ValueError(f"Displayed predictions differ from evaluated volumes: {patient}/{candidate}")
    for source, metadata_key in (("ct", "source_ct"), ("reference", "source_gt")):
        if inputs[source]["sha256"] != metadata[metadata_key]["sha256"]:
            raise ValueError(f"Cached native source changed: {patient}/{source}")
    geometry = read_json(paths["crop_geometry"])
    if (geometry["placement"].get("uses_ground_truth", geometry["placement"].get("uses_gt")) is not False
            or geometry["coarse_checkpoint_sha256"] != report["checkpoint_sha256"]):
        raise ValueError(f"Expected a prediction-only crop from this 3-slice ENet: {patient}")
    stage = next(item for item in proof["composition"]["stages"] if item["class_id"] == 1)
    if not any(item["sha256"] == inputs["crop_geometry"]["sha256"] for item in stage["files"]):
        raise ValueError(f"Displayed crop differs from the composed model crop: {patient}")
    volumes = {key: native_volume(paths[key], metadata, labels=key != "ct")
               for key in ("ct", "reference", "raw_enet", "final")}
    if not np.array_equal(volumes["raw_enet"] == 4, volumes["final"] == 4):
        raise ValueError(f"The retained refinement pipeline changed aorta labels: {patient}")
    return {"patient": patient, "metadata": metadata, "volumes": volumes, "geometry": geometry,
            "metrics": report["patients"][patient], "inputs": inputs, "patient_report": signature(proof_path)}


def organ_plan(truth, spacing, class_id, padding_mm=20.):
    mask = truth == class_id
    positive = np.flatnonzero(mask.any(axis=(0, 1)))
    if not len(positive):
        raise ValueError(f"No annotated {ORGANS[class_id]} for fixed display plan")
    bounds = []
    for axis in range(2):
        occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        padding = int(np.ceil(padding_mm / spacing[axis]))
        bounds.append([max(0, int(occupied[0]) - padding), min(truth.shape[axis], int(occupied[-1]) + 1 + padding)])
    return {"class_id": class_id, "bbox_native_xy": bounds,
            "slices": [{"quantile": quantile, "native_z": int(np.quantile(positive, quantile, method="nearest"))}
                       for quantile in QUANTILES]}


def contour(axis, mask, spacing, color, style):
    if not mask.any():
        return
    padded = np.pad(mask.T, 1)
    axis.contour((np.arange(mask.shape[0] + 2) - .5) * spacing[0],
                 (np.arange(mask.shape[1] + 2) - .5) * spacing[1], padded,
                 levels=[.5], colors=[color], linestyles=style, linewidths=1.25)


def panel(axis, case, z, bounds, class_id=1, prediction=None, reference=True, crop_box=False):
    (x0, x1), (y0, y1) = bounds
    spacing = case["metadata"]["source_spacing"]
    volumes = case["volumes"]
    image = volumes["ct"][x0:x1, y0:y1, z]
    extent = (0, image.shape[0] * spacing[0], 0, image.shape[1] * spacing[1])
    axis.imshow(image.T, cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1], origin="lower", extent=extent,
                interpolation="bilinear", interpolation_stage="data")
    if reference:
        contour(axis, volumes["reference"][x0:x1, y0:y1, z] == class_id, spacing, GT, "solid")
    if prediction is not None:
        contour(axis, volumes[prediction][x0:x1, y0:y1, z] == class_id, spacing, PRED, "--")
    if crop_box:
        (bx0, bx1), (by0, by1) = case["geometry"]["bbox_native_xy"]
        axis.add_patch(Rectangle(((bx0-x0)*spacing[0], (by0-y0)*spacing[1]),
                                (bx1-bx0)*spacing[0], (by1-by0)*spacing[1],
                                facecolor="none", edgecolor="white", linewidth=1.2))
    axis.set_xlim(extent[:2])
    axis.set_ylim(extent[2:])
    axis.set_aspect("equal")
    axis.set_axis_off()


def legend(figure, crop=False, y=.974):
    items = [Line2D([0], [0], color=GT, label="Reference (solid)"),
             Line2D([0], [0], color=PRED, linestyle="--", label="Prediction (dashed)")]
    if crop:
        items.append(Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="white", label="Actual model crop (white)"))
    figure.legend(handles=items, loc="upper center", bbox_to_anchor=(.5, y), ncol=len(items), frameon=False, fontsize=9)


def comparison_page(case, destination, pdf, dpi, seed):
    plans = [organ_plan(case["volumes"]["reference"], case["metadata"]["source_spacing"], class_id)
             for class_id in ORGANS]
    figure, axes = plt.subplots(8, 2, figsize=(10, 24), gridspec_kw={"hspace": .17, "wspace": .05})
    figure.subplots_adjust(top=.946, bottom=.025, left=.025, right=.975)
    figure.suptitle(f"{case['patient']} · seed {seed} · 3-slice ENet and organ refinement", fontsize=14, y=.995)
    legend(figure, y=.987)
    for index, plan in enumerate(plans):
        organ = ORGANS[plan["class_id"]]
        for offset, selection in enumerate(plan["slices"]):
            z = selection["native_z"]
            for column, (key, metric_key, label) in enumerate((("raw_enet", "raw_enet", "3-slice ENet"),
                                                             ("final", "raw_refined", "3-slice ENet + refinement"))):
                axis = axes[index * 2 + offset, column]
                panel(axis, case, z, plan["bbox_native_xy"], plan["class_id"], prediction=key)
                score = case["metrics"][metric_key]["organs"][organ]["dice"]
                axis.set_title(f"{label} · {organ} · z={z} · Q{selection['quantile']:.0%}\nFull-volume Dice {score:.3f}", fontsize=9)
    figure.text(.5, .009, "Raw predictions; no cleanup. Same native CT, HU window [−160,240], bilinear display, and crop in both columns.\n"
                "Display bounds: full annotated organ + 20 mm. Slice ranks Q25/Q75 fixed before inspecting refined predictions.",
                ha="center", fontsize=8)
    path = destination / f"comparison_{case['patient']}.png"
    figure.savefig(path, dpi=dpi, facecolor="white")
    pdf.savefig(figure, dpi=dpi, facecolor="white")
    plt.close(figure)
    return {"plans": plans, "png": signature(path)}


def crop_coverage(case):
    target = case["volumes"]["reference"] == 1
    (x0, x1), (y0, y1) = case["geometry"]["bbox_native_xy"]
    totals = target.sum(axis=(0, 1))
    inside = target[x0:x1, y0:y1].sum(axis=(0, 1))
    positive = np.flatnonzero(totals)
    coverage = inside[positive] / totals[positive]
    worst = int(positive[np.argmin(coverage)])
    return {"target_voxels": int(totals.sum()), "inside_crop_voxels": int(inside.sum()),
            "coverage": float(inside.sum() / totals.sum()), "worst_native_z": worst,
            "worst_slice_coverage": float(coverage.min()), "missed_target_voxels": int((totals-inside).sum())}


def crop_page(case, destination, pdf, dpi, seed):
    plan = organ_plan(case["volumes"]["reference"], case["metadata"]["source_spacing"], 1)
    coverage = crop_coverage(case)
    selections = [(f"Q{item['quantile']:.0%}", item["native_z"]) for item in plan["slices"]]
    selections.append(("Lowest crop coverage (first tie)", coverage["worst_native_z"]))
    # Include the complete model crop and the reference-derived context in both detail panels.
    bounds = [[min(a[0], b[0]), max(a[1], b[1])]
              for a, b in zip(plan["bbox_native_xy"], case["geometry"]["bbox_native_xy"])]
    full = [[0, case["metadata"]["source_shape"][0]], [0, case["metadata"]["source_shape"][1]]]
    figure, axes = plt.subplots(3, 3, figsize=(12, 12), gridspec_kw={"wspace": .08, "hspace": .18})
    figure.subplots_adjust(top=.9, bottom=.05, left=.025, right=.975)
    figure.suptitle(f"{case['patient']} · seed {seed} · esophagus crop coverage audit\n"
                   f"Actual model crop covers {coverage['coverage']:.2%} of reference voxels; "
                   f"{coverage['missed_target_voxels']} reference voxels lie outside", fontsize=13, y=.985)
    legend(figure, crop=True, y=.95)
    for row, (label, z) in enumerate(selections):
        for column, (title, display, prediction) in enumerate((("Native CT and model crop", full, None),
                ("3-slice ENet", bounds, "raw_enet"), ("3-slice ENet + refinement", bounds, "final"))):
            panel(axes[row, column], case, z, display, prediction=prediction, crop_box=True)
            axes[row, column].set_title(f"{title}\n{label} · z={z}", fontsize=9)
    figure.text(.5, .015, "Every validation patient is shown. The final row audits the weakest crop coverage, not the best refinement outcome.\n"
                "Reference masks select display views and measure coverage only. Model crops were derived from frozen 3-slice ENet predictions.\n"
                "The historical filename is retained for links; this audit does not presume that a crop failed.", ha="center", fontsize=8)
    path = destination / f"crop_{case['patient']}.png"
    figure.savefig(path, dpi=dpi, facecolor="white")
    pdf.savefig(figure, dpi=dpi, facecolor="white")
    plt.close(figure)
    return {**coverage, "slices": [{"rule": label, "native_z": z} for label, z in selections],
            "model_bbox_native_xy": case["geometry"]["bbox_native_xy"], "display_bbox_native_xy": bounds,
            "png": signature(path)}


def illustration(case, destination, dpi):
    z = ILLUSTRATION_Z
    if z >= case["metadata"]["source_shape"][2]:
        raise ValueError("The fixed Patient 13 illustration is outside the native volume")
    union = np.logical_or.reduce([case["volumes"][name][:, :, z] == 1 for name in ("reference", "raw_enet", "final")])
    coords = np.nonzero(union)
    if not len(coords[0]):
        raise ValueError("No esophagus contour at the preselected illustrative slice")
    width = max(64, max(int(axis.max()-axis.min()+1) for axis in coords) + 30)
    detail = []
    for axis, limit in zip(coords, case["metadata"]["source_shape"]):
        size = min(width, limit)
        start = max(0, min(int((int(axis.min())+int(axis.max())+1-size)/2), limit-size))
        detail.append([start, start+size])
    full = [[0, size] for size in case["metadata"]["source_shape"][:2]]
    crop = case["geometry"]["bbox_native_xy"]
    specs = [("locate", full, "raw_enet", False, True), ("zoom", crop, None, False, False),
             ("correct", crop, "final", False, False), ("detail-enet", detail, "raw_enet", True, False),
             ("detail-refined", detail, "final", True, False)]
    assets = destination / "slide_assets"
    assets.mkdir()
    files = {}
    for name, bounds, prediction, reference, box in specs:
        figure = plt.figure(figsize=(4, 4), facecolor="black")
        axis = figure.add_axes([0, 0, 1, 1])
        panel(axis, case, z, bounds, prediction=prediction, reference=reference, crop_box=box)
        path = assets / f"{name}.png"
        figure.savefig(path, dpi=max(240, dpi), facecolor="black")
        plt.close(figure)
        files[name] = signature(path)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4.7))
    for axis, key, title in zip(axes, ("raw_enet", "final"), ("3-slice ENet", "3-slice ENet + refinement")):
        panel(axis, case, z, detail, prediction=key)
        axis.set_title(title)
    figure.suptitle("Patient 13 · native z=115 · fixed illustrative example", fontsize=12)
    legend(figure, y=.08)
    figure.subplots_adjust(top=.87, bottom=.13, wspace=.025)
    for suffix in ("png", "pdf"):
        path = destination / f"patient13_z115.{suffix}"
        figure.savefig(path, dpi=max(180, dpi), facecolor="white")
        files[suffix] = signature(path)
    plt.close(figure)
    return {"patient": case["patient"], "native_z_zero_based": z,
            "selection": "Same patient and native slice as the previous presentation; retained before new results were inspected. Illustrative, not a cohort estimate.",
            "model_crop_native_xy": crop, "display_zoom_native_xy": detail,
            "display_zoom_rule": "union of reference and both stored predictions + 30 pixels, minimum 64 pixels",
            "inputs": case["inputs"], "assets": files}


def aggregate_scores(reports):
    result = {}
    for organ in (*ORGANS.values(), "mean"):
        result[organ] = {}
        for alias, key in (("common", "raw_enet"), ("final", "raw_refined")):
            means = [report["summary"][key]["macro_organ_dice"] if organ == "mean" else
                     report["summary"][key]["organs"][organ]["dice"] for report in reports.values()]
            result[organ][alias] = {"mean": float(np.mean(means)), "sd": float(np.std(means, ddof=1))}
    return result


def main(args):
    if args.dest.exists() and any(args.dest.iterdir()):
        raise ValueError("Use a new empty output directory; preserve historical figures separately")
    manifest = read_json(args.data_cache / "manifest.json")
    if manifest["status"] != "complete" or sorted(manifest["splits"]["val"]) != PATIENTS:
        raise ValueError("Expected the completed official-40 cache with the five reported validation patients")
    reports = {}
    for seed in SEEDS:
        seed_root = args.study / f"seed-{seed}"
        report = read_json(seed_root / "evaluation/report.json")
        if (report["status"] != "complete" or report["seed"] != seed or sorted(report["patients"]) != PATIENTS
                or signature(seed_root / "protocol.json")["sha256"] != report["protocol_sha256"]
                or read_json(seed_root / "protocol.json")["data_fingerprint"] != manifest["fingerprint"]):
            raise ValueError(f"Incomplete or mismatched seed {seed} evaluation")
        reports[seed] = report
    args.dest.mkdir(parents=True, exist_ok=True)
    output = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "renderer": signature(__file__), "seed": args.seed, "patients": {},
              "evaluation_reports": {str(seed): signature(args.study / f"seed-{seed}/evaluation/report.json") for seed in SEEDS},
              "purpose": "Qualitative display and crop audit; no training, prediction modification or model selection",
              "ct_window_hu": list(WINDOW), "ct_display_interpolation": "bilinear in data space",
              "mask_display": "contours extracted directly from unchanged native label masks",
              "reference_used_for_model_crops": False, "reference_used_for_display": True,
              "slice_selection": "Q25/Q75 of native positive reference slices, nearest quantile; same rule for all patients and organs"}
    write_json(args.dest / "manifest.json", output)
    try:
        with PdfPages(args.dest / "final_pipeline_comparison.pdf") as comparison_pdf, PdfPages(args.dest / "crop_failure_examples.pdf") as crop_pdf:
            for patient in PATIENTS:
                case = load_case(patient, args.data_cache, args.study / f"seed-{args.seed}", reports[args.seed], manifest)
                output["patients"][patient] = {"inputs": case["inputs"], "patient_report": case["patient_report"],
                    "comparison": comparison_page(case, args.dest, comparison_pdf, args.dpi, args.seed),
                    "crop_coverage": crop_page(case, args.dest, crop_pdf, args.dpi, args.seed)}
                if patient == ILLUSTRATION_PATIENT:
                    output["illustration"] = illustration(case, args.dest, args.dpi)
                write_json(args.dest / "manifest.json", output)
                print(json.dumps({"patient": patient, "rendered": True}), flush=True)
        output["scores"] = aggregate_scores(reports)
        output["pdfs"] = {name: signature(args.dest / name) for name in ("final_pipeline_comparison.pdf", "crop_failure_examples.pdf")}
        output["status"] = "complete"
    except BaseException as error:
        output.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.dest / "manifest.json", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "data-cache", "dest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--dpi", type=int, default=140)
    options = parser.parse_args()
    if options.dpi < 40:
        parser.error("Use --dpi >= 40")
    main(options)
