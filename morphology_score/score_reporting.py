"""Create compact summaries and reference-style morphology score figures."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd


SCORE_COLUMN = "morphology_complexity_score"
GROUP_COLORS = [
    "#8C8C8C",
    "#7BAFD4",
    "#1F4E79",
    "#59A14F",
    "#F28E2B",
    "#B07AA1",
    "#76B7B2",
    "#E15759",
    "#EDC948",
    "#4E79A7",
]


def apply_style() -> None:
    """Apply compact publication-style defaults with editable SVG text."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def mix_with_white(color: str, fraction: float) -> tuple[float, float, float]:
    """Return an opaque pale companion color."""
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb * (1.0 - fraction) + fraction)


def save_svg(fig: plt.Figure, path: Path) -> None:
    """Save an SVG with editable text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def resolve_group_order(
    score_table: pd.DataFrame, requested_order: Sequence[str] | None
) -> list[str]:
    """Validate an optional display order against the observed group labels."""
    observed = sorted(score_table["group_label"].dropna().astype(str).unique())
    if not observed:
        raise ValueError("No group labels are available for reporting")
    if requested_order is None:
        return observed

    requested = [str(label) for label in requested_order]
    if len(requested) != len(set(requested)):
        raise ValueError("Group order contains duplicate labels")

    missing = sorted(set(observed) - set(requested))
    unknown = sorted(set(requested) - set(observed))
    if missing or unknown:
        raise ValueError(
            "Group order must contain every observed label exactly once. "
            f"Missing={missing}, unknown={unknown}"
        )
    return requested


def group_color_map(group_order: Sequence[str]) -> dict[str, object]:
    """Assign stable colors while supporting more than ten groups."""
    if len(group_order) <= len(GROUP_COLORS):
        colors: Sequence[object] = GROUP_COLORS
    else:
        colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(group_order)))
    return {group: colors[index] for index, group in enumerate(group_order)}


def build_group_summary(
    score_table: pd.DataFrame, group_order: Sequence[str]
) -> pd.DataFrame:
    """Summarize scores and reference-trajectory QC by group."""
    rows = []
    for group in group_order:
        subset = score_table[score_table["group_label"] == group]
        scores = subset[SCORE_COLUMN].dropna()
        residuals = subset["trajectory_residual"].dropna()
        rows.append(
            {
                "group_label": group,
                "n_cells": int(len(scores)),
                "score_mean": float(scores.mean()) if len(scores) else np.nan,
                "score_sd": (float(scores.std(ddof=1)) if len(scores) > 1 else np.nan),
                "score_q25": (float(scores.quantile(0.25)) if len(scores) else np.nan),
                "score_median": (float(scores.median()) if len(scores) else np.nan),
                "score_q75": (float(scores.quantile(0.75)) if len(scores) else np.nan),
                "trajectory_residual_median": (
                    float(residuals.median()) if len(residuals) else np.nan
                ),
                "high_residual_fraction": (
                    float(subset["high_trajectory_residual"].mean())
                    if len(subset)
                    else np.nan
                ),
                "lower_endpoint_fraction": (
                    float(subset["lower_endpoint_saturation"].mean())
                    if len(subset)
                    else np.nan
                ),
                "upper_endpoint_fraction": (
                    float(subset["upper_endpoint_saturation"].mean())
                    if len(subset)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def fixed_bandwidth_density(
    values: np.ndarray, grid: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Estimate an area-normalized Gaussian density with fixed bandwidth."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros_like(grid)

    standardized = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * standardized * standardized).mean(axis=1) / (
        bandwidth * np.sqrt(2.0 * np.pi)
    )
    area = np.trapz(density, grid)
    return density / area if area > 0 else density


def plot_score_density(
    score_table: pd.DataFrame,
    group_order: Sequence[str],
    output_path: Path,
    bandwidth: float,
    grid_points: int,
) -> None:
    """Plot reference-style score-density ridgelines for all groups."""
    grid = np.linspace(0.0, 1.0, grid_points)
    densities = {
        group: fixed_bandwidth_density(
            score_table.loc[score_table["group_label"] == group, SCORE_COLUMN].to_numpy(
                float
            ),
            grid,
            bandwidth,
        )
        for group in group_order
    }
    global_max = max(max(density.max(), 1e-12) for density in densities.values())
    colors = group_color_map(group_order)
    figure_height = max(2.45, 0.58 * len(group_order) + 0.7)
    fig, ax = plt.subplots(figsize=(3.55, figure_height))

    for index, group in enumerate(group_order):
        baseline = len(group_order) - 1 - index
        scaled_density = 0.78 * densities[group] / global_max
        ax.fill_between(
            grid,
            baseline,
            baseline + scaled_density,
            color=mix_with_white(colors[group], 0.34),
            alpha=1.0,
            linewidth=0,
        )
        ax.plot(
            grid,
            baseline + scaled_density,
            color=colors[group],
            linewidth=0.9,
        )
        ax.axhline(baseline, color="#D8D8D8", linewidth=0.45, zorder=0)
        median = score_table.loc[
            score_table["group_label"] == group, SCORE_COLUMN
        ].median()
        if np.isfinite(median):
            ax.plot(
                [median, median],
                [baseline, baseline + 0.18],
                color="#222222",
                linewidth=0.7,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.1, len(group_order) - 0.05)
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(list(group_order)[::-1])
    ax.set_xlabel("Fixed-reference morphology score")
    ax.set_ylabel("")
    ax.set_title("RamiGlyph morphology distribution", loc="left", fontweight="bold")
    ax.text(
        0.99,
        1.01,
        f"fixed bandwidth={bandwidth:g}; area normalized",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#666666",
    )
    save_svg(fig, output_path)


def plot_trajectory_residuals(
    score_table: pd.DataFrame,
    group_order: Sequence[str],
    output_path: Path,
    threshold: float,
) -> None:
    """Plot group-wise projection residuals and the fixed reference P99."""
    residuals = [
        score_table.loc[score_table["group_label"] == group, "trajectory_residual"]
        .dropna()
        .to_numpy(float)
        for group in group_order
    ]
    colors = group_color_map(group_order)
    figure_width = max(3.8, 0.62 * len(group_order) + 1.8)
    fig, ax = plt.subplots(figsize=(figure_width, 2.55))
    box = ax.boxplot(
        residuals,
        positions=np.arange(len(residuals)),
        widths=0.58,
        patch_artist=True,
        showfliers=False,
    )
    for patch, group in zip(box["boxes"], group_order):
        patch.set_facecolor(mix_with_white(colors[group], 0.28))
        patch.set_alpha(1.0)
        patch.set_linewidth(0.6)
    for item in box["whiskers"] + box["caps"] + box["medians"]:
        item.set_color("#3A3A3A")
        item.set_linewidth(0.7)

    ax.axhline(
        threshold,
        color="#B2182B",
        linestyle="--",
        linewidth=0.8,
        label="Reference P99",
    )
    ax.set_xticks(np.arange(len(group_order)))
    if len(group_order) > 4:
        ax.set_xticklabels(group_order, rotation=30, ha="right")
    else:
        ax.set_xticklabels(group_order)
    ax.set_ylabel("Distance to reference trajectory")
    ax.set_title("External-projection residuals", loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    save_svg(fig, output_path)


def write_core_outputs(
    score_table: pd.DataFrame,
    output_dir: Path,
    requested_group_order: Sequence[str] | None,
    density_bandwidth: float,
    density_grid_points: int,
) -> pd.DataFrame:
    """Write the compact summary and two core SVG figures."""
    apply_style()
    group_order = resolve_group_order(score_table, requested_group_order)
    summary = build_group_summary(score_table, group_order)
    summary.to_csv(output_dir / "group_summary.csv", index=False)

    plot_score_density(
        score_table,
        group_order,
        output_dir / "score_density_all_groups.svg",
        bandwidth=density_bandwidth,
        grid_points=density_grid_points,
    )
    threshold = float(score_table["reference_high_residual_threshold"].iloc[0])
    plot_trajectory_residuals(
        score_table,
        group_order,
        output_dir / "trajectory_residual_by_group.svg",
        threshold=threshold,
    )
    return summary
