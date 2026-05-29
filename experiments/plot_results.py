"""Publication-quality chart generator for Avatar ablation experiments.

Reads CSV files from experiments/results/ and writes PNG charts to
experiments/charts/. All 6 ablation conditions are overlaid using a
colorblind-safe palette.

Usage:
    python -m experiments.plot_results
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
CHARTS_DIR = Path(__file__).parent / "charts"

# Colorblind-safe palette (ColorBrewer Dark2)
COLORS: dict[str, str] = {
    "full_avatar":          "#1b9e77",
    "no_cop":               "#d95f02",
    "no_senses":            "#7570b3",
    "no_dreams":            "#e7298a",
    "no_bohmian_q":         "#66a61e",
    "transformer_baseline": "#a6761d",
}

LABELS: dict[str, str] = {
    "full_avatar":          "Full Avatar v4.0",
    "no_cop":               "No COP (if/elif emotions)",
    "no_senses":            "No Senses (zero FNO)",
    "no_dreams":            "No Dreams (continuous)",
    "no_bohmian_q":         "No Bohmian Q",
    "transformer_baseline": "Transformer Baseline",
}

# Ordered list so the legend is deterministic
CONDITIONS: list[str] = list(COLORS.keys())

# Numeric CSV fields (everything except emotion, query, discovery)
NUMERIC_FIELDS = {
    "tick", "r_mean", "fe_delta", "chi", "tau", "K",
    "unity", "intensity", "prediction_error", "topic_diversity",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_csv(name: str) -> dict[str, list]:
    """Read a results CSV into a dict of lists.

    Numeric fields are converted to float; missing values become NaN.
    String fields (emotion, query, discovery) remain strings.

    Args:
        name: experiment name, e.g. "full_avatar"

    Returns:
        dict mapping column name -> list of values
    """
    path = RESULTS_DIR / f"{name}.csv"
    data: dict[str, list] = {}
    if not path.exists():
        return data

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                if key in NUMERIC_FIELDS:
                    try:
                        data[key].append(float(val))
                    except (ValueError, TypeError):
                        data[key].append(float("nan"))
                else:
                    data[key].append(val)
    return data


def rolling_mean(values: list | np.ndarray, window: int) -> np.ndarray:
    """Compute a centered rolling mean via numpy convolution.

    Edges are handled with 'same' mode, which pads with zeros; we correct
    for the truncated kernel at the edges so the mean is unbiased.

    Args:
        values: 1-D sequence of floats
        window: rolling window size

    Returns:
        numpy array of the same length as *values*
    """
    arr = np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    # Use 'same' convolution then correct the edges
    smoothed = np.convolve(arr, kernel, mode="same")
    # Build correction factors for edge bins
    correction = np.convolve(np.ones_like(arr), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(correction > 0, smoothed / correction * (correction * window / window), smoothed)
    # Simpler, correct approach: just use pandas-style via cumsum trick
    cs = np.nancumsum(np.where(np.isnan(arr), 0, arr))
    cs = np.concatenate([[0], cs])
    counts = np.nancumsum(~np.isnan(arr)).astype(float)
    counts = np.concatenate([[0], counts])
    half = window // 2
    result = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        s = cs[hi] - cs[lo]
        c = counts[hi] - counts[lo]
        result[i] = s / c if c > 0 else float("nan")
    return result


def _legend_patches() -> list[mpatches.Patch]:
    """Return a legend patch list for all conditions."""
    return [
        mpatches.Patch(color=COLORS[c], label=LABELS[c])
        for c in CONDITIONS
        if COLORS.get(c)
    ]


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to CHARTS_DIR/<name>.png at 200 DPI."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHARTS_DIR / f"{name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


def _load_all() -> dict[str, dict[str, list]]:
    """Load CSVs for all conditions; silently skip missing files."""
    return {c: load_csv(c) for c in CONDITIONS}


# ---------------------------------------------------------------------------
# Chart 1 — r_mean trajectories
# ---------------------------------------------------------------------------

def plot_r_trajectories() -> None:
    """Plot r_mean over ticks for all 6 conditions, 10-tick rolling average."""
    fig, ax = plt.subplots(figsize=(10, 5))
    all_data = _load_all()
    plotted = False

    for cond in CONDITIONS:
        d = all_data[cond]
        if not d or "tick" not in d or "r_mean" not in d:
            continue
        ticks = np.asarray(d["tick"])
        r = rolling_mean(d["r_mean"], window=10)
        ax.plot(ticks, r, color=COLORS[cond], label=LABELS[cond], linewidth=1.6)
        plotted = True

    ax.set_xlabel("Tick")
    ax.set_ylabel("r_mean (10-tick rolling avg)")
    ax.set_title("Kuramoto Order Parameter r — All Conditions")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(handles=_legend_patches(), fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, "r_trajectories")


# ---------------------------------------------------------------------------
# Chart 2 — cumulative discoveries
# ---------------------------------------------------------------------------

def plot_cumulative_discoveries() -> None:
    """Plot running count of discovery=True rows per condition."""
    fig, ax = plt.subplots(figsize=(10, 5))
    all_data = _load_all()
    plotted = False

    for cond in CONDITIONS:
        d = all_data[cond]
        if not d or "tick" not in d or "discovery" not in d:
            continue
        ticks = np.asarray(d["tick"])
        # A row counts as a discovery when the discovery field is non-empty
        is_discovery = np.asarray([1 if str(v).strip() else 0 for v in d["discovery"]])
        cumulative = np.cumsum(is_discovery)
        ax.plot(ticks, cumulative, color=COLORS[cond], label=LABELS[cond], linewidth=1.6)
        plotted = True

    ax.set_xlabel("Tick")
    ax.set_ylabel("Cumulative Discoveries")
    ax.set_title("Cumulative Discoveries Over Time")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(handles=_legend_patches(), fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, "cumulative_discoveries")


# ---------------------------------------------------------------------------
# Chart 3 — free-energy reduction (prediction_error)
# ---------------------------------------------------------------------------

def plot_fe_reduction() -> None:
    """Plot prediction_error over ticks, 20-tick rolling avg, log-scale y-axis."""
    fig, ax = plt.subplots(figsize=(10, 5))
    all_data = _load_all()
    plotted = False

    for cond in CONDITIONS:
        d = all_data[cond]
        if not d or "tick" not in d or "prediction_error" not in d:
            continue
        ticks = np.asarray(d["tick"])
        pe = np.asarray(d["prediction_error"], dtype=float)
        # Clip to positive for log scale
        pe = np.where(pe <= 0, np.nan, pe)
        smoothed = rolling_mean(pe, window=20)
        smoothed = np.where(smoothed <= 0, np.nan, smoothed)
        ax.plot(ticks, smoothed, color=COLORS[cond], label=LABELS[cond], linewidth=1.6)
        plotted = True

    ax.set_xlabel("Tick")
    ax.set_ylabel("Prediction Error (20-tick rolling avg, log scale)")
    ax.set_title("Free-Energy Reduction — Prediction Error Over Time")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    if plotted:
        ax.legend(handles=_legend_patches(), fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, "fe_reduction")


# ---------------------------------------------------------------------------
# Chart 4 — topic diversity
# ---------------------------------------------------------------------------

def plot_topic_diversity() -> None:
    """Plot topic_diversity (unique query count) over ticks for all conditions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    all_data = _load_all()
    plotted = False

    for cond in CONDITIONS:
        d = all_data[cond]
        if not d or "tick" not in d or "topic_diversity" not in d:
            continue
        ticks = np.asarray(d["tick"])
        td = np.asarray(d["topic_diversity"], dtype=float)
        ax.plot(ticks, td, color=COLORS[cond], label=LABELS[cond], linewidth=1.6)
        plotted = True

    ax.set_xlabel("Tick")
    ax.set_ylabel("Unique Queries (topic diversity)")
    ax.set_title("Topic Diversity Over Time")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(handles=_legend_patches(), fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, "topic_diversity")


# ---------------------------------------------------------------------------
# Chart 5 — COP phase portrait (r vs chi)
# ---------------------------------------------------------------------------

def plot_cop_phase_portrait() -> None:
    """Scatter r vs chi for full_avatar and no_cop, colour-coded by emotion."""
    all_data = _load_all()
    cond_a = "full_avatar"
    cond_b = "no_cop"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Colour map for emotions (reuse a categorical palette)
    emotion_palette = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]

    for ax, cond in zip(axes, [cond_a, cond_b]):
        d = all_data[cond]
        if not d or "r_mean" not in d or "chi" not in d or "emotion" not in d:
            ax.set_title(f"{LABELS.get(cond, cond)} (no data)")
            continue

        r = np.asarray(d["r_mean"], dtype=float)
        chi = np.asarray(d["chi"], dtype=float)
        emotions = d["emotion"]

        # Build integer colour indices from unique emotion strings
        unique_emotions = sorted(set(e for e in emotions if e))
        emo_to_idx = {e: i for i, e in enumerate(unique_emotions)}
        c_idx = np.asarray([emo_to_idx.get(e, 0) for e in emotions])
        colours = [emotion_palette[i % len(emotion_palette)] for i in c_idx]

        sc = ax.scatter(r, chi, c=colours, alpha=0.5, s=12, linewidths=0)
        ax.set_xlabel("r_mean (order parameter)")
        ax.set_ylabel("chi (susceptibility)")
        ax.set_title(LABELS.get(cond, cond))
        ax.grid(alpha=0.3)

        # Mini legend for emotions
        legend_patches = [
            mpatches.Patch(
                color=emotion_palette[i % len(emotion_palette)],
                label=emo,
            )
            for i, emo in enumerate(unique_emotions)
        ]
        ax.legend(handles=legend_patches, fontsize=7, loc="upper right",
                  title="emotion", title_fontsize=7)

    fig.suptitle("COP Phase Portrait: r vs chi (colour = emotion)", fontsize=11)
    fig.tight_layout()
    _save(fig, "cop_phase_portrait")


# ---------------------------------------------------------------------------
# Chart 6 — K trajectory
# ---------------------------------------------------------------------------

def plot_k_trajectory() -> None:
    """Plot coupling K over ticks for full_avatar vs no_cop."""
    fig, ax = plt.subplots(figsize=(10, 5))
    all_data = _load_all()
    plotted = False

    for cond in ["full_avatar", "no_cop"]:
        d = all_data[cond]
        if not d or "tick" not in d or "K" not in d:
            continue
        ticks = np.asarray(d["tick"])
        K = np.asarray(d["K"], dtype=float)
        ax.plot(ticks, K, color=COLORS[cond], label=LABELS[cond], linewidth=1.6)
        plotted = True

    # Annotate the K clamp bounds
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="K clamp [0.05, 2.0]")
    ax.axhline(2.00, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Tick")
    ax.set_ylabel("Coupling K")
    ax.set_title("SOC Coupling K Trajectory: Full Avatar vs No COP")
    ax.grid(alpha=0.3)
    if plotted:
        handles = [
            mpatches.Patch(color=COLORS["full_avatar"], label=LABELS["full_avatar"]),
            mpatches.Patch(color=COLORS["no_cop"],      label=LABELS["no_cop"]),
        ]
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=0.8,
                               label="K clamp [0.05, 2.0]"))
        ax.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, "k_trajectory")


# ---------------------------------------------------------------------------
# Chart 7 — emotion distribution
# ---------------------------------------------------------------------------

def plot_emotion_distribution() -> None:
    """Grouped bar chart: fraction of ticks per emotion per condition."""
    all_data = _load_all()

    # Collect all unique emotion labels across conditions
    all_emotions: set[str] = set()
    for cond in CONDITIONS:
        d = all_data[cond]
        if "emotion" in d:
            all_emotions.update(e for e in d["emotion"] if e)
    emotion_list = sorted(all_emotions)

    if not emotion_list:
        print("  plot_emotion_distribution: no emotion data found, skipping")
        return

    # Build matrix: conditions x emotions
    n_cond = len(CONDITIONS)
    n_emo = len(emotion_list)
    fractions = np.zeros((n_cond, n_emo))

    for ci, cond in enumerate(CONDITIONS):
        d = all_data[cond]
        if not d or "emotion" not in d:
            continue
        total = len(d["emotion"])
        if total == 0:
            continue
        for ei, emo in enumerate(emotion_list):
            fractions[ci, ei] = sum(1 for e in d["emotion"] if e == emo) / total

    # Plot grouped bars
    x = np.arange(n_emo)
    bar_width = 0.12
    offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * bar_width

    fig, ax = plt.subplots(figsize=(max(10, n_emo * 1.2), 5))

    for ci, cond in enumerate(CONDITIONS):
        ax.bar(
            x + offsets[ci],
            fractions[ci],
            width=bar_width,
            color=COLORS[cond],
            label=LABELS[cond],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(emotion_list, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of Ticks")
    ax.set_title("Emotion Distribution Per Condition")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=_legend_patches(), fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "emotion_distribution")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all() -> None:
    """Generate all 7 publication-quality comparison charts.

    Reads from experiments/results/*.csv and writes PNGs to
    experiments/charts/. Missing CSV files are silently skipped so the
    function is safe to call even before experiments have run.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading CSVs from : {RESULTS_DIR}")
    print(f"Writing charts to : {CHARTS_DIR}")
    print()

    steps = [
        ("r_trajectories",        plot_r_trajectories),
        ("cumulative_discoveries", plot_cumulative_discoveries),
        ("fe_reduction",           plot_fe_reduction),
        ("topic_diversity",        plot_topic_diversity),
        ("cop_phase_portrait",     plot_cop_phase_portrait),
        ("k_trajectory",           plot_k_trajectory),
        ("emotion_distribution",   plot_emotion_distribution),
    ]

    for label, fn in steps:
        print(f"[{label}]")
        try:
            fn()
        except Exception as exc:
            print(f"  WARNING: {label} failed — {exc}")
        print()

    print("generate_all() complete.")


if __name__ == "__main__":
    generate_all()
