#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

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

import argparse
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection


METRIC_LABELS = {
    "dice": "Dice",
    "hd95": "HD95 (mm)",
    "rvd": "RVD",
}


def nanmean(values: np.ndarray, axis=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def run(args: argparse.Namespace) -> None:
    metric_files: list[Path] = args.metric_file
    suffixes = {p.suffix for p in metric_files}

    if suffixes == {".npy"}:
        plot_training_metrics(metric_files, args)
    elif suffixes == {".npz"}:
        plot_3d_metrics(metric_files, args)
    else:
        raise ValueError("Use one .npy training metric file, or one or more .npz 3D metric files.")


def plot_training_metric(metric_file: Path, args: argparse.Namespace) -> None:
    plot_training_metric_frame(training_metric_frame(metric_file, run_id=metric_file.parent.name or metric_file.stem),
                               title=args.title or default_plot_title([metric_file]),
                               sparse_3d=is_sparse_3d_training_metric([metric_file]),
                               args=args)


def training_metric_frame(metric_file: Path, run_id: str) -> pd.DataFrame:
    metrics: np.ndarray = np.asarray(np.load(metric_file))
    rows = []

    match metrics.ndim:
        case 1:
            E = metrics.shape[0]
            rows.extend({"epoch": e, "curve": "Metric", "run": run_id, "value": float(metrics[e])}
                        for e in range(E))
        case 2:
            E, K = metrics.shape
            if K <= 5:
                for k in range(1, K):
                    rows.extend({"epoch": e, "curve": f"Class {k}", "run": run_id, "value": float(metrics[e, k])}
                                for e in range(E))

                if K > 1:
                    foreground = nanmean(metrics[:, 1:], axis=1)
                    rows.extend({"epoch": e, "curve": "Foreground mean", "run": run_id, "value": float(foreground[e])}
                                for e in range(E))
            else:
                y = nanmean(metrics, axis=1)
                rows.extend({"epoch": e, "curve": "Metric", "run": run_id, "value": float(y[e])}
                            for e in range(E))
        case 3:
            E, _, K = metrics.shape
            for k in range(1, K):
                y = nanmean(metrics[:, :, k], axis=1)
                rows.extend({"epoch": e, "curve": f"Class {k}", "run": run_id, "value": float(y[e])}
                            for e in range(E))

            foreground = nanmean(metrics[:, :, 1:], axis=(1, 2))
            rows.extend({"epoch": e, "curve": "Foreground mean", "run": run_id, "value": float(foreground[e])}
                        for e in range(E))
        case _:
            raise ValueError(f"Unsupported metric shape for {metric_file}: {metrics.shape}")

    return pd.DataFrame(rows)


def plot_training_metrics(metric_files: list[Path], args: argparse.Namespace) -> None:
    if len(metric_files) == 1:
        plot_training_metric(metric_files[0], args)
        return

    df = pd.concat(
        [training_metric_frame(path, run_id=path.parent.name or path.stem) for path in metric_files],
        ignore_index=True,
    )

    plot_training_metric_frame(df,
                               title=args.title or default_plot_title(metric_files),
                               sparse_3d=is_sparse_3d_training_metric(metric_files),
                               args=args)


def is_sparse_3d_training_metric(metric_files: list[Path]) -> bool:
    return all(path.stem.startswith("dice_3d") for path in metric_files)


def interpolate_sparse_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    dense_rows = []
    max_epoch = int(df["epoch"].max())

    for (run_id, curve), group in df.groupby(["run", "curve"], sort=False):
        series = (
            group.sort_values("epoch")
            .set_index("epoch")["value"]
            .reindex(range(max_epoch + 1))
        )
        if series.notna().sum() > 1:
            series = series.interpolate(method="linear", limit_area="inside")

        dense_rows.extend(
            {"epoch": int(epoch), "curve": curve, "run": run_id, "value": float(value)}
            for epoch, value in series.dropna().items()
        )

    return pd.DataFrame(dense_rows)


def plot_training_metric_frame(df: pd.DataFrame, title: str, sparse_3d: bool, args: argparse.Namespace) -> None:
    if sparse_3d:
        df = interpolate_sparse_training_frame(df)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    sns.lineplot(
        data=df,
        x="epoch",
        y="value",
        hue="curve",
        style="curve",
        estimator="mean",
        errorbar=None if sparse_3d else "sd",
        linewidth=1.7,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, args)


def load_3d_metric(metric_file: Path) -> pd.DataFrame:
    metric = metric_file.stem.lower()
    run = run_label(metric_file)
    with np.load(metric_file) as data:
        rows = [
            {"patient": patient, "class": int(cls), "metric": metric, "run": run, "value": float(value)}
            for patient in data.files
            for cls, value in enumerate(data[patient])
        ]
    df = pd.DataFrame(rows)
    return df[df["class"] != 0]


def run_label(metric_file: Path) -> str:
    parent = metric_file.parent
    if parent.name in {"metrics_3d", "best_epoch"}:
        parent = parent.parent

    label = parent.name or metric_file.stem
    return re.sub(r"_s\d+.*$", "", label)


def common_label(labels: list[str]) -> str:
    if not labels:
        return "Metrics"

    prefix = os.path.commonprefix(labels).rstrip("_- ")
    return prefix or labels[0]


def metric_label_from_paths(metric_files: list[Path]) -> str:
    stems = {path.stem.lower() for path in metric_files}
    if stems == {"dice_3d_val"}:
        return "3D validation Dice"
    if stems == {"dice_val"}:
        return "2D validation Dice"
    if stems == {"dice_3d_val_fg"}:
        return "3D validation foreground Dice"
    if stems == {"dice"}:
        return "best-epoch 3D Dice"
    if stems == {"hd95"}:
        return "best-epoch HD95"
    if stems == {"rvd"}:
        return "best-epoch RVD"
    if {"dice", "hd95", "rvd"}.issubset(stems):
        return "best-epoch 3D metrics"
    if {"dice", "hd95"}.issubset(stems):
        return "best-epoch Dice vs HD95"

    return "Validation metric"


def default_plot_title(metric_files: list[Path]) -> str:
    baseline = common_label([run_label(path) for path in metric_files])
    metric = metric_label_from_paths(metric_files)
    if len(metric_files) > 1:
        return f"{baseline}: {metric} across seeds"
    return f"{baseline}: {metric}"


def plot_3d_metrics(metric_files: list[Path], args: argparse.Namespace) -> None:
    df = pd.concat([load_3d_metric(p) for p in metric_files], ignore_index=True)
    df = df.dropna(subset=["value"])
    metrics = set(df["metric"])

    if len(metrics) == 1:
        plot_3d_strip(df, title=args.title or default_plot_title(metric_files), args=args)
        return

    has_scatter = {"dice", "hd95"}.issubset(metrics)
    has_rvd = "rvd" in metrics
    if not has_scatter and not has_rvd:
        raise ValueError("Multiple .npz files need dice+hd95 and/or rvd to make a combined 3D metrics plot.")

    ncols = int(has_scatter) + int(has_rvd)
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 4.8), squeeze=False)
    axes_flat = axes.ravel()
    ax_i = 0

    if has_scatter:
        plot_dice_hd95_scatter(df, axes_flat[ax_i])
        ax_i += 1
    if has_rvd:
        plot_rvd_distribution(df[df["metric"] == "rvd"], axes_flat[ax_i])

    fig.suptitle(args.title or default_plot_title(metric_files))
    fig.tight_layout()
    save_or_show(fig, args)


def plot_3d_strip(df: pd.DataFrame, title: str, args: argparse.Namespace) -> None:
    metric = str(df["metric"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if metric == "rvd":
        plot_rvd_distribution(df, ax)
    else:
        plot_metric_distribution(df, ax)
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric.upper()))
        if metric == "hd95":
            ax.set_yscale("log")
    fig.tight_layout()
    save_or_show(fig, args)


def plot_dice_hd95_scatter(df: pd.DataFrame, ax) -> None:
    wide = df[df["metric"].isin(["dice", "hd95"])].pivot_table(
        index=["run", "patient", "class"], columns="metric", values="value"
    ).reset_index().dropna(subset=["dice", "hd95"])

    if wide.empty:
        ax.text(0.5, 0.5, "No paired Dice/HD95 values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    positive_hd95 = wide.loc[wide["hd95"] > 0, "hd95"]
    if positive_hd95.empty:
        hd95_floor = 1e-3
    else:
        hd95_floor = max(float(positive_hd95.min()) * 0.5, 1e-3)
    wide["hd95_plot"] = wide["hd95"].clip(lower=hd95_floor)

    palette = dict(zip(sorted(wide["class"].unique()), sns.color_palette("muted", n_colors=wide["class"].nunique())))
    sns.scatterplot(
        data=wide,
        x="dice",
        y="hd95_plot",
        hue="class",
        palette=palette,
        s=42,
        alpha=0.58,
        linewidth=0,
        ax=ax,
    )

    medians = wide.groupby("class", as_index=False)[["dice", "hd95_plot"]].median()
    sns.scatterplot(
        data=medians,
        x="dice",
        y="hd95_plot",
        hue="class",
        palette=palette,
        marker="D",
        s=120,
        edgecolor="#222222",
        linewidth=0.9,
        legend=False,
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Dice")
    ax.set_ylabel("HD95 (mm, log scale)")
    ax.set_title("Dice vs HD95")
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.grid(True, which="minor", axis="y", alpha=0.12)
    ax.legend(title="Class", frameon=False, loc="best")


def plot_metric_distribution(df: pd.DataFrame, ax) -> None:
    sns.violinplot(
        data=df,
        x="class",
        y="value",
        hue="class",
        palette="tab10",
        inner=None,
        cut=0,
        density_norm="width",
        width=0.55,
        linewidth=1.6,
        ax=ax,
    )
    for violin in ax.findobj(PolyCollection):
        violin.set_alpha(0.35)
    remove_legend(ax)
    sns.boxplot(
        data=df,
        x="class",
        y="value",
        width=0.11,
        showfliers=False,
        color="#e6e6e6",
        boxprops={"facecolor": "#e6e6e6", "edgecolor": "#777777", "alpha": 0.7},
        whiskerprops={"color": "#777777", "linewidth": 1},
        capprops={"color": "#777777", "linewidth": 1},
        medianprops={"color": "#555555", "linewidth": 1.4},
        ax=ax,
    )
    remove_legend(ax)
    ax.grid(True, axis="y", alpha=0.25)


def plot_rvd_distribution(df: pd.DataFrame, ax) -> None:
    plot_metric_distribution(df, ax)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Class")
    ax.set_ylabel("RVD")
    ax.set_title("Relative Volume Difference")


def remove_legend(ax) -> None:
    legend = ax.get_legend()
    if legend:
        legend.remove()


def save_or_show(fig, args: argparse.Namespace) -> None:
    if args.dest:
        fig.savefig(args.dest, dpi=150)

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--metric_file', type=Path, nargs="+", required=True, metavar="METRIC_MODE",
                        help="One .npy training metric file, or one or more .npz 3D metric files.")
    parser.add_argument('--dest', type=Path, metavar="METRIC_MODE.png",
                        help="Optional: save the plot to a .png file")
    parser.add_argument('--title', type=str, default=None,
                        help="Optional plot title. Defaults to a title inferred from metric paths.")
    parser.add_argument("--headless", action="store_true",
                        help="Does not display the plot and save it directly (implies --dest to be provided.")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())
