"""Compare per-rail TD7 with two region-token TD7 implementations.

The input CSV files are produced by
``oht_routing/utils/download_wandb_run.py``. Use
``--refresh`` to download the selected W&B metrics again before plotting.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "figures" / "per_region_rl_diagnosis"
RUNS = {
    "ozl3xyft": {
        "label": "Per-rail TD7 (ozl3xyft)",
        "short": "Per-rail TD7",
        "color": "#009E73",
        "project": "bjy6614-postech/oht-routing-rl-td7",
        "kind": "per_rail",
    },
    "atndxx19": {
        "label": "Region TD7 v1: concat attention (atndxx19)",
        "short": "Region TD7 v1",
        "color": "#D55E00",
        "project": "bjy6614-postech/oht-routing-rl-td7-region",
        "kind": "region",
    },
    "w5l4ytzi": {
        "label": "Region TD7 v2: bounded attention + normalized actor/critic (w5l4ytzi)",
        "short": "Region TD7 v2",
        "color": "#0072B2",
        "project": "bjy6614-postech/oht-routing-rl-td7-region",
        "kind": "region",
    },
}
METRICS = (
    "_step",
    "step",
    "episode/index",
    "global/tat",
    "episode/final_tat",
    "loss/critic",
    "critic/q_std",
    "critic/q_one_token_delta",
    "critic/q_one_region_delta",
    "region/size_min",
    "region/size_mean",
    "region/size_max",
    "region/size_std",
    "reward/region_internal_std_mean",
    "reward/region_local_std_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without extension.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download the selected metrics from W&B before plotting.",
    )
    return parser.parse_args()


def configure_font() -> None:
    candidates = ("Malgun Gothic", "Noto Sans CJK KR", "AppleGothic", "DejaVu Sans")
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def refresh_history(run_id: str) -> None:
    output_dir = PROJECT_ROOT / "results" / "wandb" / run_id
    run_meta = RUNS[run_id]
    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "PythonCode"
            / "oht_routing"
            / "utils"
            / "download_wandb_run.py"
        ),
        "--run-path",
        f"{run_meta['project']}/{run_id}",
        "--output-dir",
        str(output_dir),
    ]
    # The older per-rail run exposes some sparse metrics only when all history
    # keys are requested. Region runs can use the smaller selected schema.
    if run_meta["kind"] == "region":
        command.extend(["--keys", *METRICS])
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def load_history(run_id: str, refresh: bool) -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "results" / "wandb" / run_id / "history.csv"
    if refresh or not csv_path.exists():
        refresh_history(run_id)
    data = pd.read_csv(csv_path)
    if "global/tat" not in data and "system/tat" in data:
        data["global/tat"] = data["system/tat"]
    for key in METRICS:
        if key in data:
            data[key] = pd.to_numeric(data[key], errors="coerce")
    return data


def rolling_line(
    ax: plt.Axes,
    data: pd.DataFrame,
    key: str,
    color: str,
    label: str,
    window: int = 35,
) -> None:
    frame = data.loc[data["step"].notna() & data[key].notna(), ["step", key]]
    x = frame["step"].to_numpy() / 1000.0
    y = frame[key]
    ax.plot(x, y, color=color, alpha=0.10, linewidth=0.7)
    ax.plot(
        x,
        y.rolling(window, center=True, min_periods=max(5, window // 4)).median(),
        color=color,
        linewidth=2.1,
        label=label,
    )


def finite_ratio(data: pd.DataFrame) -> pd.Series:
    token = data["critic/q_one_token_delta"]
    region = data["critic/q_one_region_delta"]
    ratio = region / token
    return ratio.where((token > 1e-6) & np.isfinite(ratio))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def build_summary(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for run_id, data in histories.items():
        finals = data.loc[data["episode/final_tat"] > 0, "episode/final_tat"]
        is_region = RUNS[run_id]["kind"] == "region"
        ratio = finite_ratio(data).dropna() if is_region else pd.Series(dtype=float)
        rows.append(
            {
                "run_id": run_id,
                "run_label": RUNS[run_id]["label"],
                "learning_unit": RUNS[run_id]["kind"],
                "last_env_step": int(data["step"].max()),
                "episodes_completed": int(finals.size),
                "final_tat_median": finals.median(),
                "final_tat_min": finals.min(),
                "final_tat_max": finals.max(),
                "critic_loss_median": data["loss/critic"].median(),
                "critic_loss_p95": data["loss/critic"].quantile(0.95),
                "q_std_median": data["critic/q_std"].median() if is_region else np.nan,
                "q_std_p95": (
                    data["critic/q_std"].quantile(0.95) if is_region else np.nan
                ),
                "q_std_max": data["critic/q_std"].max() if is_region else np.nan,
                "q_sensitivity_ratio_median": ratio.median(),
                "q_sensitivity_ratio_p25": ratio.quantile(0.25),
                "q_sensitivity_ratio_p75": ratio.quantile(0.75),
                "reward_internal_std_median": (
                    data["reward/region_internal_std_mean"].median()
                    if is_region
                    else np.nan
                ),
                "reward_local_std_median": (
                    data["reward/region_local_std_mean"].median()
                    if is_region
                    else np.nan
                ),
                "region_size_min": (
                    data["region/size_min"].median() if is_region else np.nan
                ),
                "region_size_mean": (
                    data["region/size_mean"].median() if is_region else np.nan
                ),
                "region_size_max": (
                    data["region/size_max"].median() if is_region else np.nan
                ),
                "region_size_std": (
                    data["region/size_std"].median() if is_region else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    configure_font()
    histories = {
        run_id: load_history(run_id, args.refresh)
        for run_id in RUNS
    }
    summary = build_summary(histories)

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 13.2))
    fig.patch.set_facecolor("#FAFAF8")
    for ax in axes.flat:
        ax.set_facecolor("#FAFAF8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.65)

    # A. Observable episode result for all three runs.
    ax = axes[0, 0]
    for run_id, meta in RUNS.items():
        data = histories[run_id]
        points = data.loc[
            data["episode/final_tat"] > 0,
            ["_step", "episode/final_tat"],
        ]
        ax.plot(
            points["_step"] / 1000.0,
            points["episode/final_tat"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=meta["color"],
            label=meta["label"],
        )
        median = points["episode/final_tat"].median()
        ax.text(
            0.98,
            0.93 - 0.075 * list(RUNS).index(run_id),
            f"{meta['short']}: median {median:.1f} (n={len(points)})",
            color=meta["color"],
            transform=ax.transAxes,
            fontsize=8.5,
            ha="right",
        )
    ax.set_title("Observed result: episode-final TAT by training step")
    ax.set_xlabel("Environment step (×1,000)")
    ax.set_ylabel("TAT (lower is better)")
    ax.set_xlim(0, 850)
    add_panel_label(ax, "A")

    # B. TAT trajectories for all three runs.
    ax = axes[0, 1]
    for run_id, meta in RUNS.items():
        rolling_line(
            ax,
            histories[run_id],
            "global/tat",
            meta["color"],
            meta["short"],
            window=55,
        )
    ax.set_title("Observed result: TAT during the first 260k steps")
    ax.set_xlabel("Environment step (×1,000)")
    ax.set_ylabel("TAT (rolling median)")
    ax.set_xlim(0, 260)
    ax.legend(frameon=False, fontsize=8)
    add_panel_label(ax, "B")

    # C. Q dispersion comparison within the region-token implementations.
    ax = axes[1, 0]
    for run_id in ("atndxx19", "w5l4ytzi"):
        meta = RUNS[run_id]
        rolling_line(
            ax,
            histories[run_id],
            "critic/q_std",
            meta["color"],
            meta["short"],
        )
    v1_max = histories["atndxx19"]["critic/q_std"].max()
    v2_max = histories["w5l4ytzi"]["critic/q_std"].max()
    ax.text(
        0.98,
        0.94,
        f"max Q std: {v1_max:.1f} → {v2_max:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "pad": 4},
    )
    ax.set_title("Region runs only: critic Q dispersion")
    ax.set_xlabel("Environment step (×1,000)")
    ax.set_ylabel("critic/q_std")
    ax.legend(frameon=False, fontsize=8)
    add_panel_label(ax, "C")

    # D. Region-level action changes dominate one-token action changes.
    ax = axes[1, 1]
    for run_id in ("atndxx19", "w5l4ytzi"):
        meta = RUNS[run_id]
        data = histories[run_id]
        ratio = finite_ratio(data)
        frame = pd.DataFrame({"step": data["step"], "ratio": ratio}).dropna()
        smooth = frame["ratio"].rolling(41, center=True, min_periods=10).median()
        ax.plot(
            frame["step"] / 1000.0,
            smooth,
            color=meta["color"],
            linewidth=2,
            label=f"{meta['short']} (median {ratio.median():.1f}×)",
        )
    ax.axhline(50, color="#444444", linestyle="--", linewidth=1.1, label="50× reference")
    ax.set_title("Region runs only: weak one-rail Q sensitivity")
    ax.set_xlabel("Environment step (×1,000)")
    ax.set_ylabel("ΔQ(all region actions) / ΔQ(one rail)")
    ax.set_ylim(1, 150)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    add_panel_label(ax, "D")

    # E. Non-zero within-region variation is discarded by scalar averaging.
    ax = axes[2, 0]
    positions = [0.82, 1.18, 1.82, 2.18]
    values = []
    colors = []
    region_run_ids = ("atndxx19", "w5l4ytzi")
    for key in ("reward/region_internal_std_mean", "reward/region_local_std_mean"):
        for run_id in region_run_ids:
            meta = RUNS[run_id]
            values.append(histories[run_id][key].dropna().to_numpy())
            colors.append(meta["color"])
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.5},
        whiskerprops={"color": "#777777"},
        capprops={"color": "#777777"},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(color)
    ax.set_xticks([1, 2], ["Rail reward", "Local reward"])
    ax.set_ylabel("Mean within-region std")
    ax.set_title("Problem: rail-specific reward variation exists,\nbut replay stores only the region mean")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=RUNS[r]["color"], lw=7, alpha=0.65)
            for r in region_run_ids
        ],
        labels=[RUNS[r]["short"] for r in region_run_ids],
        frameon=False,
        fontsize=9,
    )
    add_panel_label(ax, "E")

    # F. Equal region sampling induces 1/N effective rail weighting.
    ax = axes[2, 1]
    size_stats = summary.loc[summary["run_id"] == "w5l4ytzi"].iloc[0]
    sizes = np.arange(1, int(size_stats["region_size_max"]) + 1)
    weights = 1.0 / sizes
    ax.plot(sizes, weights, color="#6A3D9A", linewidth=2.5)
    marks = [
        ("min", size_stats["region_size_min"]),
        ("mean", size_stats["region_size_mean"]),
        ("max", size_stats["region_size_max"]),
    ]
    annotation_offsets = {"min": (12, -48), "mean": (8, -34), "max": (8, 8)}
    for label, size in marks:
        weight = 1.0 / size
        ax.scatter(size, weight, s=55, color="#6A3D9A", zorder=3)
        ax.annotate(
            f"{label}: N={size:.1f}\nweight={weight:.3f}",
            (size, weight),
            xytext=annotation_offsets[label],
            textcoords="offset points",
            fontsize=9,
            va="top" if label != "max" else "bottom",
        )
    ax.set_title("Disadvantage: equal region sampling overweights small regions")
    ax.set_xlabel("Rails in region (N)")
    ax.set_ylabel("Effective weight per rail ∝ 1/N")
    ax.set_yscale("log")
    ax.text(
        0.97,
        0.88,
        "A rail in N=1 receives\n50× the weight of N=50",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#6A3D9A",
        fontweight="bold",
    )
    add_panel_label(ax, "F")

    fig.suptitle(
        "Per-rail vs region-token TD7",
        fontsize=17,
        fontweight="bold",
        y=0.996,
    )
    fig.text(
        0.5,
        0.967,
        "Per-rail TD7 stores one transition, reward, and scalar Q target per rail. "
        "Region TD7 v1/v2 group up to 50 rail tokens, average their rewards into one transition, "
        "and supervise mean token-Q; v2 uses bounded attention and normalized actor/critic paths.\n"
        "All runs use TD7 with b_rl costs, 10k warm-up, and a 10k–40k action curriculum. "
        "The per-rail run uses an earlier reward implementation, so performance differences are descriptive, not a controlled ablation.",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#333333",
        linespacing=1.35,
    )
    fig.text(
        0.5,
        0.012,
        "Source: W&B runs ozl3xyft, atndxx19, and w5l4ytzi. "
        "Lines are centered rolling medians; faint traces are raw values.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.025, 0.035, 0.99, 0.91), h_pad=3.0, w_pad=2.6)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    summary.to_csv(output.with_name(output.name + "_summary.csv"), index=False)
    print(f"Saved {output.with_suffix('.png')}")
    print(f"Saved {output.with_suffix('.pdf')}")
    print(f"Saved {output.with_name(output.name + '_summary.csv')}")


if __name__ == "__main__":
    main()
