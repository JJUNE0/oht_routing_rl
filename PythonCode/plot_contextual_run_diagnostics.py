"""Create azuv9qhx-style diagnostic plots for contextual TD7 W&B exports.

The script reads ``results/wandb/<run_id>/history.csv`` and writes four PNGs
(critic Q mean, TAT, b_rl mean, and queued jobs) plus the detected episode
boundaries to ``results/<run_id>_analysis``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_IDS = (
    "0lwv8bat",
    "nccazyej",
    "26hxsuet",
    "zuw9nno8",
    "f8u5by9y",
    "z8vaw2af",
    "whzsal63",
    "4uyhq9jg",
    "jrm4nh2j",
)
ROLLING_WINDOW = 31
REQUIRED_COLUMNS = (
    "_step",
    "env/step",
    "env/episode",
    "critic/q1_mean",
    "critic/q2_mean",
    "b_rl/mean",
)
METRIC_COLUMN_CANDIDATES = {
    "tat": ("env/tat", "global/tat"),
    "queued": ("env/queued", "global/queued"),
}
COLORS = {
    "q_mean": "#8f63c7",
    "queued": "#1976d2",
    "tat": "#ff7f0e",
    "b_rl": "#00a6b2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_ids",
        nargs="*",
        default=list(DEFAULT_RUN_IDS),
        help="W&B run IDs (default: the nine runs in the analysis report).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "wandb",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    return parser.parse_args()


def available_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    # Either W&B's explicit step or the environment step is sufficient.
    missing = [column for column in missing if column not in {"_step", "env/step"}]
    for label, candidates in METRIC_COLUMN_CANDIDATES.items():
        if not any(column in header for column in candidates):
            missing.append(f"{label} ({' or '.join(candidates)})")
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    selected = [column for column in REQUIRED_COLUMNS if column in header]
    for candidates in METRIC_COLUMN_CANDIDATES.values():
        selected.extend(column for column in candidates if column in header)
    return selected


def metric_column(data: pd.DataFrame, label: str) -> str:
    for column in METRIC_COLUMN_CANDIDATES[label]:
        if column in data:
            return column
    raise KeyError(f"no {label} metric column is available")


def load_history(path: Path) -> tuple[pd.DataFrame, str]:
    columns = available_columns(path)
    data = pd.read_csv(path, usecols=columns, low_memory=False)
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    step_column = "_step" if "_step" in data else "env/step"
    data = data.loc[data[step_column].notna()].sort_values(step_column)
    return data, step_column


def episode_boundaries(data: pd.DataFrame, step_column: str) -> pd.DataFrame:
    frame = data.loc[
        data["env/episode"].notna(), [step_column, "env/episode"]
    ].copy()
    frame["env/episode"] = frame["env/episode"].astype(int)
    boundaries = (
        frame.groupby("env/episode", sort=True)[step_column]
        .min()
        .rename("_step")
        .reset_index()
        .rename(columns={"env/episode": "episode"})
    )
    return boundaries[["_step", "episode"]]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.facecolor": "#f5f7fa",
            "axes.facecolor": "#f3f6fa",
            "axes.edgecolor": "#aab5c2",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.color": "#4d5966",
            "ytick.color": "#4d5966",
            "grid.color": "#cbd3dc",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.8,
        }
    )


def rolling_median(values: pd.Series) -> pd.Series:
    return values.rolling(
        ROLLING_WINDOW,
        center=True,
        min_periods=max(5, ROLLING_WINDOW // 4),
    ).median()


def add_episode_guides(
    ax: plt.Axes,
    boundaries: pd.DataFrame,
    x_min: float,
    x_max: float,
) -> None:
    records = list(boundaries.itertuples(index=False, name=None))
    if not records:
        return
    span = max(x_max - x_min, 1.0)
    for index, record in enumerate(records):
        start = float(record[0])
        episode = int(record[1])
        if index > 0:
            ax.axvline(
                start,
                color="#ff6b6b",
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                zorder=1,
            )
        end = float(records[index + 1][0]) if index + 1 < len(records) else x_max
        width = max(0.0, end - start)
        if width / span >= 0.085:
            ax.text(
                start + width / 2,
                0.975,
                f"Episode {episode}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="#35404a",
            )
        else:
            ax.text(
                start + width * 0.5,
                0.98,
                f"E{episode}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                rotation=90,
                fontsize=7.5,
                fontweight="bold",
                color="#35404a",
            )


def finalize_axes(ax: plt.Axes, x_min: float, x_max: float) -> None:
    ax.set_xlim(x_min, x_max)
    ax.grid(True, axis="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("W&B step", fontsize=12)
    ax.tick_params(labelsize=10)


def save_q_plot(
    run_id: str,
    data: pd.DataFrame,
    step_column: str,
    boundaries: pd.DataFrame,
    output: Path,
) -> None:
    frame = data[
        [step_column, "critic/q1_mean", "critic/q2_mean"]
    ].copy()
    frame["q_mean"] = (frame["critic/q1_mean"] + frame["critic/q2_mean"]) / 2.0
    x = frame[step_column]
    x_min, x_max = float(data[step_column].min()), float(data[step_column].max())

    fig, ax = plt.subplots(figsize=(20.48, 7.04), dpi=100)
    fig.suptitle(
        f"Critic Q mean diagnostics - {run_id}",
        fontsize=22,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.925,
        "Q mean = (Q1 mean + Q2 mean) / 2 | original Q scale | "
        "31-log rolling median | red dashed lines: episode boundaries",
        ha="center",
        fontsize=11,
        color="#536170",
    )
    valid = frame["q_mean"].notna()
    ax.plot(
        x[valid],
        rolling_median(frame.loc[valid, "q_mean"]),
        color=COLORS["q_mean"],
        linewidth=2.0,
        label="Q mean",
        zorder=3,
    )
    add_episode_guides(ax, boundaries, x_min, x_max)
    finalize_axes(ax, x_min, x_max)
    ax.set_ylabel("Q value", fontsize=12)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=1,
        frameon=True,
        framealpha=0.95,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.17)
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def add_exact_extreme_ticks(
    ax: plt.Axes,
    minimum: float,
    maximum: float,
) -> None:
    lower, upper = ax.get_ylim()
    ticks = [tick for tick in ax.get_yticks() if lower <= tick <= upper]
    # Remove nearly overlapping automatic ticks so the exact values stay legible.
    tolerance = max((upper - lower) * 0.018, 1e-9)
    ticks = [
        tick
        for tick in ticks
        if abs(tick - minimum) > tolerance and abs(tick - maximum) > tolerance
    ]
    ticks.extend([minimum, maximum])
    ticks = sorted(set(round(float(tick), 10) for tick in ticks))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:g}" for tick in ticks])
    for tick, label in zip(ticks, ax.get_yticklabels()):
        if abs(tick - minimum) < 1e-8:
            label.set_color("#2ca02c")
            label.set_fontweight("bold")
        elif abs(tick - maximum) < 1e-8:
            label.set_color("#d62728")
            label.set_fontweight("bold")


def episode_end_tat_points(
    data: pd.DataFrame,
    step_column: str,
    tat_metric: str,
) -> pd.DataFrame:
    """Return the final positive TAT before each observed episode transition.

    The highest episode ID is excluded because there is no following episode
    boundary proving that it ended within the downloaded history.
    """
    frame = data.loc[
        data["env/episode"].notna()
        & data[tat_metric].notna()
        & (data[tat_metric] > 0),
        [step_column, "env/episode", tat_metric],
    ].copy()
    if frame.empty:
        return frame
    frame["env/episode"] = frame["env/episode"].astype(int)
    last_episode = int(frame["env/episode"].max())
    completed = frame.loc[frame["env/episode"] < last_episode]
    return completed.groupby("env/episode", sort=True, as_index=False).tail(1)


def save_single_metric_plot(
    run_id: str,
    data: pd.DataFrame,
    step_column: str,
    boundaries: pd.DataFrame,
    metric: str,
    metric_label: str,
    title: str,
    color: str,
    output: Path,
    *,
    fixed_limits: tuple[float, float] | None = None,
    show_tat_extremes: bool = False,
) -> None:
    x = data[step_column]
    x_min, x_max = float(x.min()), float(x.max())
    values = data[metric].copy()
    valid = values.notna()
    if show_tat_extremes:
        # TAT=0 is the reset/no-completed-job sentinel, not a measured TAT.
        valid &= values > 0
    finite_values = values.loc[valid]
    if finite_values.empty:
        raise ValueError(f"{run_id} has no finite values for {metric}")

    fig, ax = plt.subplots(figsize=(20.48, 7.04), dpi=100)
    fig.suptitle(
        f"{title} - {run_id}",
        fontsize=22,
        fontweight="bold",
        y=0.975,
    )
    subtitle = "Original metric scale | raw values (faint) | 31-log rolling median"
    if show_tat_extremes:
        subtitle += " | dotted lines: episode-end TAT minimum / maximum"
    subtitle += " | red dashed lines: episode boundaries"
    fig.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        fontsize=11,
        color="#536170",
    )
    ax.plot(
        x[valid],
        finite_values,
        color=color,
        linewidth=0.65,
        alpha=0.13,
        zorder=2,
    )
    smoothed_values = rolling_median(finite_values)
    ax.plot(
        x[valid],
        smoothed_values,
        color=color,
        linewidth=2.1,
        label=f"{metric_label} (31-log median)",
        zorder=3,
    )
    add_episode_guides(ax, boundaries, x_min, x_max)
    finalize_axes(ax, x_min, x_max)
    ax.set_ylabel(metric_label, fontsize=12)

    if fixed_limits is not None:
        ax.set_ylim(*fixed_limits)
    elif metric in METRIC_COLUMN_CANDIDATES["queued"]:
        ax.set_ylim(0.0, max(1.0, float(finite_values.max()) * 1.06))

    if show_tat_extremes:
        endpoints = episode_end_tat_points(data, step_column, metric)
        if endpoints.empty:
            raise ValueError(f"{run_id} has no completed episode-end TAT")
        minimum = float(endpoints[metric].min())
        maximum = float(endpoints[metric].max())
        ax.scatter(
            endpoints[step_column],
            endpoints[metric],
            s=34,
            color="#4b5563",
            edgecolor="white",
            linewidth=0.8,
            label="Episode-end TAT",
            zorder=5,
        )
        displayed_values = smoothed_values.dropna()
        visible_minimum = min(float(displayed_values.min()), minimum)
        visible_maximum = max(float(displayed_values.max()), maximum)
        padding = max((visible_maximum - visible_minimum) * 0.08, 1.0)
        ax.set_ylim(visible_minimum - padding, visible_maximum + padding)
        ax.axhline(
            minimum,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.8,
            label=f"Episode-end minimum = {minimum:g}",
            zorder=1,
        )
        ax.axhline(
            maximum,
            color="#d62728",
            linestyle=":",
            linewidth=1.8,
            label=f"Episode-end maximum = {maximum:g}",
            zorder=1,
        )
        add_exact_extreme_ticks(ax, minimum, maximum)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=4 if show_tat_extremes else 1,
        frameon=True,
        framealpha=0.95,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.062, right=0.995, top=0.86, bottom=0.17)
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_run(run_id: str, input_root: Path, output_root: Path) -> None:
    history_path = input_root / run_id / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    output_dir = output_root / f"{run_id}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    data, step_column = load_history(history_path)
    boundaries = episode_boundaries(data, step_column)
    boundaries.to_csv(
        output_dir / f"{run_id}_episode_boundaries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_q_plot(
        run_id,
        data,
        step_column,
        boundaries,
        output_dir / f"{run_id}_q_mean_labeled.png",
    )
    save_single_metric_plot(
        run_id,
        data,
        step_column,
        boundaries,
        metric_column(data, "tat"),
        "TAT",
        "TAT diagnostics",
        COLORS["tat"],
        output_dir / f"{run_id}_tat_labeled.png",
        show_tat_extremes=True,
    )
    save_single_metric_plot(
        run_id,
        data,
        step_column,
        boundaries,
        "b_rl/mean",
        "b_rl mean",
        "b_rl action diagnostics",
        COLORS["b_rl"],
        output_dir / f"{run_id}_b_rl_mean_labeled.png",
        fixed_limits=(-0.02, 1.02),
    )
    save_single_metric_plot(
        run_id,
        data,
        step_column,
        boundaries,
        metric_column(data, "queued"),
        "Queued jobs",
        "Queued-job diagnostics",
        COLORS["queued"],
        output_dir / f"{run_id}_queued_labeled.png",
    )
    print(
        f"{run_id}: {len(data):,} rows, {len(boundaries)} episodes -> {output_dir}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    configure_style()
    for run_id in args.run_ids:
        plot_run(run_id, args.input_root, args.output_root)


if __name__ == "__main__":
    main()
