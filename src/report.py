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
# World map: interactive choropleth
# ---------------------------------------------------------------------------

def _load_country_interior_points() -> dict[str, tuple[float, float]]:
    """
    Return {iso3: (lat, lon)} using shapely representative_point() — guaranteed inside polygon.

    Downloads Natural Earth 110m countries on first call and caches to data/cache/.
    """
    import json
    import geopandas as gpd

    cache_path = ROOT / "data" / "cache" / "country_interior_points.json"
    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        return {k: tuple(v) for k, v in raw.items()}

    logger.info("Downloading Natural Earth 110m countries for label placement (cached after this run)...")
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)

    points: dict[str, tuple[float, float]] = {}
    for _, row in world.iterrows():
        iso3 = row.get("ADM0_A3") or row.get("ISO_A3")
        if iso3 and iso3 != "-99":
            pt = row.geometry.representative_point()
            points[iso3] = (pt.y, pt.x)  # (lat, lon)

    with open(cache_path, "w") as f:
        json.dump(points, f)
    logger.info("Interior points cached to %s (%d countries)", cache_path, len(points))
    return points


def generate_world_map(dri: pd.DataFrame) -> None:
    """Interactive choropleth HTML map — gap_usd shaded red (over-contributor) to green (large gap)."""
    import plotly.graph_objects as go

    valid = dri[dri["gap_usd"].notna()].copy()

    def _fmt_usd(v):
        if pd.isna(v):
            return "N/A"
        if abs(v) >= 1e9:
            return f"${v/1e9:.2f}B"
        return f"${v/1e6:.2f}M"

    def _fmt_pct(v):
        return "N/A" if pd.isna(v) else f"{v:.1%}"

    def _fmt_score(v):
        return "N/A" if pd.isna(v) else f"{v:.1f}"

    valid["_gap_fmt"] = valid["gap_usd"].apply(_fmt_usd)
    valid["_rate_fmt"] = valid["giving_rate"].apply(_fmt_pct)
    valid["_align_fmt"] = valid["alignment_score"].apply(_fmt_score)
    valid["_target_fmt"] = valid["adjusted_target_usd"].apply(_fmt_usd)

    pos_gaps = valid.loc[valid["gap_usd"] > 0, "gap_usd"]
    zmax = float(pos_gaps.quantile(0.95)) if len(pos_gaps) else 1e9
    zmin = float(valid["gap_usd"].min())

    fig = go.Figure(go.Choropleth(
        locations=valid["iso3"],
        z=valid["gap_usd"],
        locationmode="ISO-3",
        colorscale="RdYlGn",
        zmin=zmin,
        zmax=zmax,
        zmid=0,
        customdata=valid[["country_name", "_gap_fmt", "_rate_fmt", "_align_fmt", "_target_fmt"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Gap: %{customdata[1]}<br>"
            "Giving Rate: %{customdata[2]}<br>"
            "Alignment Score: %{customdata[3]}<br>"
            "Capacity Target: %{customdata[4]}"
            "<extra></extra>"
        ),
        colorbar=dict(
            title="Contribution Gap",
            tickformat="$,.0f",
            len=0.75,
        ),
        marker_line_color="white",
        marker_line_width=0.5,
    ))

    # Build label positions using representative_point() — always inside polygon
    interior = _load_country_interior_points()
    label_rows = valid[valid["iso3"].isin(interior)].copy()
    label_rows["_lat"] = label_rows["iso3"].map(lambda c: interior[c][0])
    label_rows["_lon"] = label_rows["iso3"].map(lambda c: interior[c][1])

    fig.add_trace(go.Scattergeo(
        lat=label_rows["_lat"],
        lon=label_rows["_lon"],
        text=label_rows["country_name"],
        mode="text",
        textfont=dict(size=7, color="black"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="Donor Readiness Index — IDA Contribution Gap by Country", font=dict(size=16)),
        geo=dict(
            showland=True,
            landcolor="lightgray",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="white",
            projection_type="natural earth",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # Warn on unmatched ISO-3 codes
    for iso3 in valid["iso3"]:
        if iso3 not in interior:
            logger.warning("ISO-3 code '%s' not found in Natural Earth geometry — label omitted", iso3)

    path = CHARTS / "chart5_world_map.html"
    fig.write_html(str(path), include_plotlyjs=True)
    logger.info("World map saved to %s", path)


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
    generate_world_map(dri)
    logger.info("All charts generated in %s", CHARTS)
    return dri
