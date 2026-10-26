"""Render the repository's patient-level 3D plots for the matched comparison."""
from __future__ import annotations

import hashlib
import importlib.util
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"esophagus": 1, "heart": 2, "trachea": 3, "aorta": 4}
METRIC_FILES = {"dice": "dice", "hd95": "hd95_mm", "rvd": "rvd"}


def export_metrics(snapshot, stage, destination):
    """Export exact observations, keeping repeated patients distinct by seed.

    plot.py pivots by run/patient/class. Including the seed in each patient key
    prevents that pivot from averaging the three seeds before plotting. Class 0
    is unavailable in the snapshot and is ignored by plot.py, so it remains NaN.
    """
    protocol = snapshot["protocol"]
    observations = list(product(protocol["seeds"], protocol["validation_patients"]))
    expected = {(seed, patient, organ) for seed, patient in observations for organ in LABELS}
    rows = [row for row in snapshot["patient_metrics"] if row["stage"] == stage]
    observed = [(row["seed"], row["patient"], row["organ"]) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(f"{stage}: require one row per seed, patient and organ")
    arrays = {metric: {f"s{seed}:{patient}": np.full(5, np.nan)
                       for seed, patient in observations} for metric in METRIC_FILES}
    for row in rows:
        signed_rvd = row.get("rvd")
        if signed_rvd is None or not np.isfinite(signed_rvd):
            raise ValueError("Signed RVD must come from the original evaluation reports")
        if not np.isclose(abs(signed_rvd), row["absolute_rvd"], rtol=0, atol=1e-12):
            raise ValueError("Signed and absolute RVD disagree")
        key, label = f"s{row['seed']}:{row['patient']}", LABELS[row["organ"]]
        for metric, field in METRIC_FILES.items():
            value = row[field]
            if value is None or not np.isfinite(value):
                raise ValueError(f"{stage}: expected finite {field} for {key}")
            arrays[metric][key][label] = value
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric, values in arrays.items():
        path = destination / f"{metric}.npz"
        np.savez_compressed(path, **values)
        paths.append(path)
    return paths


def render_diagnostics(snapshot, destination):
    """Reuse plot.py's Dice/HD95 scatter and signed-RVD distribution unchanged."""
    spec = importlib.util.spec_from_file_location("phase7_repository_plot", ROOT / "plot.py")
    plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotting)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seeds = snapshot["protocol"]["seeds"]
    patients = snapshot["protocol"]["validation_patients"]
    caption = (f"{len(seeds) * len(patients)} seed–patient observations: "
               f"{len(seeds)} seeds × the same {len(patients)} validation patients\n"
               "Repeated patients are not independent; signed RVD is a ratio, not a percentage.")
    manifest = {}
    with TemporaryDirectory(prefix="phase7-diagnostic-metrics-") as temporary:
        for stage, label in snapshot["stages"].items():
            paths = export_metrics(snapshot, stage, Path(temporary) / stage)
            plotting.run(SimpleNamespace(metric_file=paths, dest=None, title=label, headless=True))
            figure = plotting.plt.gcf()
            figure.axes[-1].set_ylabel("Signed RVD (volume ratio)")
            figure.text(.5, .01, caption, ha="center", fontsize=9)
            figure.tight_layout(rect=(0, .105, 1, .97))
            name = f"patient_3d_metrics_{stage}"
            manifest[name] = {}
            for extension in ("png", "pdf"):
                path = destination / f"{name}.{extension}"
                figure.savefig(path, dpi=260, bbox_inches="tight")
                manifest[name][extension] = {
                    "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            plotting.plt.close(figure)
    return manifest
