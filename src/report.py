"""
Reporting and chart generation for the Donor Readiness Index.

Merges capacity and alignment scores, ranks countries by gap, and produces:
  - outputs/dri_output.csv            — ranked per-country summary
  - outputs/charts/chart1_gap_ranking.png
  - outputs/charts/chart2_giving_rate.png
  - outputs/charts/chart3_capacity_vs_giving_rate.png
  - outputs/charts/chart4_alignment_vs_gap.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
CHARTS = OUTPUTS / "charts"

CAPACITY_SCORES_PATH = DATA_PROCESSED / "capacity_scores.csv"
ALIGNMENT_SCORES_PATH = DATA_PROCESSED / "alignment_scores.csv"
DRI_OUTPUT_PATH = OUTPUTS / "dri_output.csv"

DPI = 150
INCOME_COLORS = {
    "HIC": "#2196F3",   # blue
    "UMC": "#FF9800",   # orange
    "LMC": "#4CAF50",   # green
    "LIC": "#9C27B0",   # purple
}
DEFAULT_COLOR = "#607D8B"


def _ensure_output_dirs():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Merge and rank
# ---------------------------------------------------------------------------

def build_dri_output(
    capacity: pd.DataFrame | None = None,
    alignment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge capacity and alignment scores; sort by gap_usd descending.
    Writes outputs/dri_output.csv and returns the DataFrame.
    """
    _ensure_output_dirs()

    if capacity is None:
        capacity = pd.read_csv(CAPACITY_SCORES_PATH)
    if alignment is None:
        alignment = pd.read_csv(ALIGNMENT_SCORES_PATH)

    # Merge on iso3; keep all capacity rows
    align_cols = [c for c in alignment.columns if c not in {"country_name"}]
    merged = capacity.merge(alignment[align_cols], on="iso3", how="left")

    # Sort by gap descending (largest gap = most underperforming)
    merged = merged.sort_values("gap_usd", ascending=False, na_position="last")
    merged = merged.reset_index(drop=True)

    merged.to_csv(DRI_OUTPUT_PATH, index=False)
    logger.info("DRI output written to %s (%d countries)", DRI_OUTPUT_PATH, len(merged))
    return merged


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _income_color(income_group: str | None) -> str:
    return INCOME_COLORS.get(str(income_group), DEFAULT_COLOR)


def _millions(x, _):
    return f"${x/1e6:.0f}M"


def _billions(x, _):
    if abs(x) >= 1e9:
        return f"${x/1e9:.1f}B"
    return f"${x/1e6:.0f}M"


# ---------------------------------------------------------------------------
# Chart 1: Gap ranking bar chart
# ---------------------------------------------------------------------------

def chart1_gap_ranking(dri: pd.DataFrame, top_n: int = 30) -> None:
    """Horizontal bar chart — top N countries by gap_usd."""
    valid = dri[dri["gap_usd"].notna()].copy()
    if len(valid) < top_n:
        logger.info("Chart 1: only %d countries with valid gap (requested %d)", len(valid), top_n)
    plot_df = valid.head(top_n).sort_values("gap_usd", ascending=True)

    colors = [_income_color(g) for g in plot_df["income_group"]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))
    bars = ax.barh(plot_df["iso3"], plot_df["gap_usd"], color=colors, edgecolor="white", linewidth=0.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_billions))
    ax.set_xlabel("Contribution Gap (Target − Actual)", fontsize=11)
    ax.set_title(f"Donor Readiness Index — Top {len(plot_df)} Countries by IDA Contribution Gap", fontsize=13, pad=12)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="y", labelsize=9)

    # Income group legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for label, color in INCOME_COLORS.items()
        if label in plot_df["income_group"].values
    ]
    if handles:
        ax.legend(handles=handles, title="Income Group", loc="lower right", fontsize=8)

    plt.tight_layout()
    path = CHARTS / "chart1_gap_ranking.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart 1 saved to %s", path)


# ---------------------------------------------------------------------------
# Chart 2: Giving rate bar chart
# ---------------------------------------------------------------------------

def chart2_giving_rate(dri: pd.DataFrame) -> None:
    """Horizontal bar chart of giving rate for all countries, sorted ascending."""
    valid = dri[dri["giving_rate"].notna()].sort_values("giving_rate", ascending=True).copy()

    colors = ["#4CAF50" if r >= 1.0 else "#F44336" for r in valid["giving_rate"]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(valid) * 0.28)))
    ax.barh(valid["iso3"], valid["giving_rate"], color=colors, edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--", label="Benchmark (1.0)")
    ax.set_xlabel("Giving Rate (Actual / Adjusted Target)", fontsize=11)
    ax.set_title("Donor Readiness Index — Giving Rate by Country", fontsize=13, pad=12)
    ax.tick_params(axis="y", labelsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="At or above benchmark"),
        Patch(facecolor="#F44336", label="Below benchmark"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = CHARTS / "chart2_giving_rate.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart 2 saved to %s", path)


# ---------------------------------------------------------------------------
# Chart 3: Capacity vs. giving rate scatter
# ---------------------------------------------------------------------------

def chart3_capacity_vs_giving_rate(dri: pd.DataFrame) -> None:
    """Scatter: adjusted_target_usd (x) vs. giving_rate (y), ISO3 labels."""
    valid = dri[dri["adjusted_target_usd"].notna() & dri["giving_rate"].notna()].copy()

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = [_income_color(g) for g in valid["income_group"]]
    ax.scatter(valid["adjusted_target_usd"], valid["giving_rate"], c=colors, s=60, alpha=0.75, zorder=3)

    for _, row in valid.iterrows():
        ax.annotate(
            row["iso3"],
            (row["adjusted_target_usd"], row["giving_rate"]),
            fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points",
        )

    # Reference lines
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="Giving rate = 1.0")
    median_cap = valid["adjusted_target_usd"].median()
    ax.axvline(median_cap, color="gray", linewidth=0.8, linestyle=":", label="Median capacity target")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_billions))
    ax.set_xlabel("Adjusted Capacity Target (USD)", fontsize=11)
    ax.set_ylabel("Giving Rate (Actual / Target)", fontsize=11)
    ax.set_title("Capacity vs. Giving Rate", fontsize=13, pad=12)
    ax.legend(fontsize=9)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)
        for label, color in INCOME_COLORS.items()
        if label in valid["income_group"].values
    ]
    if handles:
        ax.legend(handles=handles, title="Income Group", loc="upper right", fontsize=8)

    plt.tight_layout()
    path = CHARTS / "chart3_capacity_vs_giving_rate.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart 3 saved to %s", path)


# ---------------------------------------------------------------------------
# Chart 5: All countries by gap (full ranked list)
# ---------------------------------------------------------------------------

def chart5_all_countries_gap(dri: pd.DataFrame) -> None:
    """Horizontal bar chart — all countries ranked by gap_usd."""
    valid = dri[dri["gap_usd"].notna()].sort_values("gap_usd", ascending=True).copy()

    colors = ["#E53935" if g < 0 else _income_color(ig)
              for g, ig in zip(valid["gap_usd"], valid["income_group"])]

    fig, ax = plt.subplots(figsize=(12, max(8, len(valid) * 0.22)))
    ax.barh(valid["country_name"], valid["gap_usd"], color=colors, edgecolor="none", height=0.8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_billions))
    ax.set_xlabel("Contribution Gap (Target − Actual)", fontsize=11)
    ax.set_title("Donor Readiness Index — All Countries by IDA Contribution Gap", fontsize=13, pad=12)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="y", labelsize=7)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=color, label=label)
        for label, color in INCOME_COLORS.items()
        if label in valid["income_group"].values
    ] + [Patch(facecolor="#E53935", label="Over-contributor")]
    ax.legend(handles=handles, title="Income Group", loc="lower right", fontsize=8)

    plt.tight_layout()
    path = CHARTS / "chart5_all_countries_gap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart 5 saved to %s", path)


# ---------------------------------------------------------------------------
# Chart 4: Alignment vs. gap scatter
# ---------------------------------------------------------------------------

def chart4_alignment_vs_gap(dri: pd.DataFrame) -> None:
    """Scatter: alignment_score (x) vs. gap_usd (y), point size ∝ GDP."""
    valid = dri[dri["alignment_score"].notna() & dri["gap_usd"].notna()].copy()
    if valid.empty:
        logger.warning("Chart 4: no countries with both alignment score and gap — skipping")
        return

    # Normalize GDP to point size (30–300)
    gdp = valid["gdp_usd"].fillna(valid["gdp_usd"].median())
    sizes = 30 + (gdp / gdp.max()) * 270

    colors = [_income_color(g) for g in valid["income_group"]]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(valid["alignment_score"], valid["gap_usd"], s=sizes, c=colors, alpha=0.70, zorder=3)

    for _, row in valid.iterrows():
        ax.annotate(
            row["iso3"],
            (row["alignment_score"], row["gap_usd"]),
            fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points",
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_billions))
    ax.set_xlabel("Strategic Alignment Score (0–100)", fontsize=11)
    ax.set_ylabel("Contribution Gap (Target − Actual, USD)", fontsize=11)
    ax.set_title("Strategic Alignment vs. Contribution Gap", fontsize=13, pad=12)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)
        for label, color in INCOME_COLORS.items()
        if label in valid["income_group"].values
    ]
    if handles:
        ax.legend(handles=handles, title="Income Group", loc="upper right", fontsize=8)

    plt.tight_layout()
    path = CHARTS / "chart4_alignment_vs_gap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart 4 saved to %s", path)


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------

def generate_report(
    capacity: pd.DataFrame | None = None,
    alignment: pd.DataFrame | None = None,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Build the DRI output CSV and generate all four charts.

    Returns the merged DRI DataFrame.
    """
    _ensure_output_dirs()
    dri = build_dri_output(capacity, alignment)
    chart1_gap_ranking(dri, top_n=top_n)
    chart2_giving_rate(dri)
    chart3_capacity_vs_giving_rate(dri)
    chart4_alignment_vs_gap(dri)
    chart5_all_countries_gap(dri)
    logger.info("All charts generated in %s", CHARTS)
    return dri
