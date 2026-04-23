"""
Reporting and chart generation for the Donor Readiness Index.

Ranks countries by contribution gap and produces:
  - outputs/dri_output.csv            — ranked per-country summary
  - outputs/charts/chart1_gap_ranking.png
  - outputs/charts/chart2_giving_rate.png
  - outputs/charts/chart3_capacity_vs_giving_rate.png
  - outputs/charts/chart5_all_countries_gap.png
  - outputs/charts/chart5_world_map.html
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
CHARTS = OUTPUTS / "charts"

CAPACITY_SCORES_PATH = DATA_PROCESSED / "capacity_scores.csv"
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

def build_dri_output(capacity: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Sort capacity scores by contribution gap descending.

    Writes outputs/dri_output.csv and returns the DataFrame.
    """
    _ensure_output_dirs()

    if capacity is None:
        capacity = pd.read_csv(CAPACITY_SCORES_PATH)

    gap_col = "gap_usd_signed" if "gap_usd_signed" in capacity.columns else "gap_usd"
    merged = capacity.sort_values(gap_col, ascending=False, na_position="last").reset_index(drop=True)
    merged.insert(0, "rank", range(1, len(merged) + 1))

    merged.to_csv(DRI_OUTPUT_PATH, index=False)
    logger.info("DRI output written to %s (%d countries)", DRI_OUTPUT_PATH, len(merged))
    return merged


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _income_color(income_group: str | None) -> str:
    return INCOME_COLORS.get(str(income_group), DEFAULT_COLOR)


def _billions(x, _):
    if abs(x) >= 1e9:
        return f"${x/1e9:.1f}B"
    return f"${x/1e6:.0f}M"


def _fmt_usd_label(v):
    if pd.isna(v):
        return "N/A"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    return f"${v/1e6:.2f}M"


def _gap_col(df: pd.DataFrame) -> str:
    return "gap_usd_signed" if "gap_usd_signed" in df.columns else "gap_usd"


def _rate_col(df: pd.DataFrame) -> str:
    return "giving_rate_raw" if "giving_rate_raw" in df.columns else "giving_rate"


# ---------------------------------------------------------------------------
# Chart 1: Gap ranking bar chart
# ---------------------------------------------------------------------------

def chart1_gap_ranking(dri: pd.DataFrame, top_n: int = 30) -> None:
    """Horizontal bar chart — top N countries by gap_usd_signed, with 90% CI error bars."""
    gap_col = _gap_col(dri)
    valid = dri[dri[gap_col].notna()].copy()
    if len(valid) < top_n:
        logger.info("Chart 1: only %d countries with valid gap (requested %d)", len(valid), top_n)
    plot_df = valid.sort_values(gap_col, ascending=False).head(top_n).sort_values(gap_col, ascending=True)

    colors = [_income_color(g) for g in plot_df["income_group"]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))
    bars = ax.barh(plot_df["iso3"], plot_df[gap_col], color=colors, edgecolor="white", linewidth=0.5)

    # Error bars from 90% CI columns (clipped at ±200% of point estimate)
    has_ci = "gap_usd_lower" in plot_df.columns and "gap_usd_upper" in plot_df.columns
    clipped_any = False
    if has_ci and plot_df["gap_usd_lower"].notna().any():
        xerr_lo = []
        xerr_hi = []
        for _, r in plot_df.iterrows():
            gap = r[gap_col]
            lo = r.get("gap_usd_lower")
            hi = r.get("gap_usd_upper")
            if pd.isna(lo) or pd.isna(hi) or pd.isna(gap):
                xerr_lo.append(0)
                xerr_hi.append(0)
                continue
            cap = abs(gap) * 2.0
            err_lo = min(abs(gap - lo), cap)
            err_hi = min(abs(hi - gap), cap)
            if abs(gap - lo) > cap or abs(hi - gap) > cap:
                clipped_any = True
            xerr_lo.append(err_lo)
            xerr_hi.append(err_hi)

        ax.barh(
            plot_df["iso3"],
            [0] * len(plot_df),
            xerr=[xerr_lo, xerr_hi],
            error_kw={"ecolor": "gray", "capsize": 3, "linewidth": 0.8, "alpha": 0.6},
        )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_billions))
    ax.set_xlabel("Contribution Gap (Target − Actual)", fontsize=11)
    title = f"Donor Readiness Index — Top {len(plot_df)} Countries by IDA Contribution Gap"
    ax.set_title(title, fontsize=13, pad=12)
    if has_ci and clipped_any:
        ax.set_xlabel(ax.get_xlabel() + "\n(error bars clipped at ±200% of estimate)", fontsize=9)
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
    rate_col = _rate_col(dri)
    valid = dri[dri[rate_col].notna()].sort_values(rate_col, ascending=True).copy()

    colors = ["#4CAF50" if r >= 1.0 else "#F44336" for r in valid[rate_col]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(valid) * 0.28)))
    ax.barh(valid["iso3"], valid[rate_col], color=colors, edgecolor="white", linewidth=0.3)
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
    rate_col = _rate_col(dri)
    valid = dri[dri["adjusted_target_usd"].notna() & dri[rate_col].notna()].copy()

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = [_income_color(g) for g in valid["income_group"]]
    ax.scatter(valid["adjusted_target_usd"], valid[rate_col], c=colors, s=60, alpha=0.75, zorder=3)

    for _, row in valid.iterrows():
        ax.annotate(
            row["iso3"],
            (row["adjusted_target_usd"], row[rate_col]),
            fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points",
        )

    # Reference lines — add to ax before the single legend call
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="Giving rate = 1.0")
    median_cap = valid["adjusted_target_usd"].median()
    ax.axvline(median_cap, color="gray", linewidth=0.8, linestyle=":", label="Median capacity target")

    # Income group markers (added as artists so get_legend_handles_labels picks them up)
    for label, color in INCOME_COLORS.items():
        if label in valid["income_group"].values:
            ax.plot(
                [], [], marker="o", color="w", markerfacecolor=color,
                markersize=8, label=label,
            )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_billions))
    ax.set_xlabel("Adjusted Capacity Target (USD)", fontsize=11)
    ax.set_ylabel("Giving Rate (Actual / Target)", fontsize=11)
    ax.set_title("Capacity vs. Giving Rate", fontsize=13, pad=12)

    # Single consolidated legend call — collects all handles added above
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    path = CHARTS / "chart3_capacity_vs_giving_rate.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart 3 saved to %s", path)


# ---------------------------------------------------------------------------
# Chart 5: All countries by gap (full ranked list)
# ---------------------------------------------------------------------------

def chart5_all_countries_gap(dri: pd.DataFrame) -> None:
    """Horizontal bar chart — all countries ranked by gap_usd_signed."""
    gap_col = _gap_col(dri)
    valid = dri[dri[gap_col].notna()].sort_values(gap_col, ascending=True).copy()

    colors = ["#E53935" if g < 0 else _income_color(ig)
              for g, ig in zip(valid[gap_col], valid["income_group"])]

    fig, ax = plt.subplots(figsize=(12, max(8, len(valid) * 0.22)))
    ax.barh(valid["country_name"], valid[gap_col], color=colors, edgecolor="none", height=0.8)
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
    """Interactive choropleth HTML map — gap shaded red (over-contributor) to green (large gap).

    Tooltip shows gap estimate with 90% CI range where available.
    """
    import plotly.graph_objects as go

    gap_col = _gap_col(dri)
    valid = dri[dri[gap_col].notna()].copy()

    if "p_donate" not in valid.columns:
        raise ValueError("p_donate column required to compute probability-weighted gap for the world map")
    valid["expected_gap_usd"] = valid[gap_col] * valid["p_donate"].fillna(0.0)

    def _fmt_pct(v):
        return "N/A" if pd.isna(v) else f"{v:.1%}"

    valid["_expected_fmt"] = valid["expected_gap_usd"].apply(_fmt_usd_label)
    valid["_pdonate_fmt"] = valid["p_donate"].apply(_fmt_pct)
    valid["_rate_fmt"] = valid["giving_rate"].apply(_fmt_pct) if "giving_rate" in valid.columns else "N/A"
    valid["_target_fmt"] = valid["adjusted_target_usd"].apply(_fmt_usd_label) if "adjusted_target_usd" in valid.columns else "N/A"

    # Probability-weighted 90% CI
    has_ci = "gap_usd_lower" in valid.columns and "gap_usd_upper" in valid.columns
    if has_ci:
        p = valid["p_donate"].fillna(0.0)
        def _expected_ci_str(row, p_val):
            lo, hi = row.get("gap_usd_lower"), row.get("gap_usd_upper")
            if pd.isna(lo) or pd.isna(hi):
                return ""
            return f" [{_fmt_usd_label(lo * p_val)} – {_fmt_usd_label(hi * p_val)} 90% CI]"
        valid["_expected_with_ci"] = [
            valid["_expected_fmt"].iloc[i] + _expected_ci_str(valid.iloc[i], p.iloc[i])
            for i in range(len(valid))
        ]
    else:
        valid["_expected_with_ci"] = valid["_expected_fmt"]

    # Symlog transform
    def _symlog(x):
        return np.sign(x) * np.log1p(np.abs(x) / 1e6)

    valid["_z"] = valid["expected_gap_usd"].apply(_symlog)
    zmax = float(valid["_z"].quantile(0.98))
    zmin = float(valid["_z"].quantile(0.02))

    tick_dollars = [-1e9, -1e8, -1e7, -1e6, 0, 1e6, 1e7, 1e8, 1e9]
    tick_vals = [_symlog(v) for v in tick_dollars]
    tick_text = ["-$1B", "-$100M", "-$10M", "-$1M", "$0", "$1M", "$10M", "$100M", "$1B"]

    fig = go.Figure(go.Choropleth(
        locations=valid["iso3"],
        z=valid["_z"],
        locationmode="ISO-3",
        colorscale="RdYlGn",
        zmin=zmin,
        zmax=zmax,
        zmid=0,
        customdata=valid[["country_name", "_expected_with_ci", "_pdonate_fmt", "_rate_fmt", "_target_fmt"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Expected Gap: %{customdata[1]}<br>"
            "P(donate): %{customdata[2]}<br>"
            "Giving Rate: %{customdata[3]}<br>"
            "Capacity Target: %{customdata[4]}"
            "<extra></extra>"
        ),
        colorbar=dict(
            title="Expected Gap<br>(gap × P(donate))",
            tickvals=tick_vals,
            ticktext=tick_text,
            len=0.75,
        ),
        marker_line_color="white",
        marker_line_width=0.5,
    ))

    interior = _load_country_interior_points()
    top_iso3 = (
        valid.nlargest(30, "_z")["iso3"].tolist()
        + valid.nsmallest(5, "_z")["iso3"].tolist()
    )
    label_rows = valid[valid["iso3"].isin(interior) & valid["iso3"].isin(top_iso3)].copy()
    label_rows["_lat"] = label_rows["iso3"].map(lambda c: interior[c][0])
    label_rows["_lon"] = label_rows["iso3"].map(lambda c: interior[c][1])

    fig.add_trace(go.Scattergeo(
        lat=label_rows["_lat"],
        lon=label_rows["_lon"],
        text=label_rows["country_name"],
        mode="text",
        textfont=dict(size=8, color="black"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="Donor Readiness Index — Probability-Weighted IDA Contribution Gap", font=dict(size=16)),
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
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Build the DRI output CSV and generate all charts.

    Returns the DRI DataFrame.
    """
    _ensure_output_dirs()
    dri = build_dri_output(capacity)
    chart1_gap_ranking(dri, top_n=top_n)
    chart2_giving_rate(dri)
    chart3_capacity_vs_giving_rate(dri)
    chart5_all_countries_gap(dri)
    generate_world_map(dri)
    logger.info("All charts generated in %s", CHARTS)
    return dri
