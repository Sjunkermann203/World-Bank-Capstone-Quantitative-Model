"""
Capacity scoring for the Donor Readiness Index.

Computes:
  - GDP-weighted peer-group median IDA/GDP benchmark ratio (per income tier)
  - Capacity-based target contribution per country
  - Fiscal modifier (±20% max, linear)
  - Adjusted target contribution
  - Signed contribution gap (positive = shortfall, negative = over-contribution)
  - Giving rate (actual / adjusted_target); raw (uncapped) and capped at 1.0
  - Donor segment, including "Exceeded Target" for over-contributors
  - PPP-adjusted gap metrics

Reads:  data/processed/master.csv
Writes: data/processed/capacity_scores.csv
        data/processed/run_metadata.json (benchmark ratios + donor set)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"

MASTER_PATH = DATA_PROCESSED / "master.csv"
CAPACITY_SCORES_PATH = DATA_PROCESSED / "capacity_scores.csv"
RUN_METADATA_PATH = DATA_PROCESSED / "run_metadata.json"

# Fiscal modifier parameters
FISCAL_MODIFIER_CAP = 0.20      # ±20% cap
FISCAL_SCALE_FACTOR = 0.04      # 5% fiscal balance → 20% modifier  (0.20 / 5.0)

# Minimum donors per peer group to use group-level benchmark
MIN_PEER_GROUP_DONORS = 3


# ---------------------------------------------------------------------------
# Weighted median utility
# ---------------------------------------------------------------------------

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the weighted median of `values` using `weights` as weights.

    Handles edge cases: single value, zero-weight rows, and degenerate inputs.
    Returns the weighted 50th percentile.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Drop NaN values or zero/negative weights
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return float(values[0])

    # Sort by value
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    # Cumulative weight fractions
    cum_weights = np.cumsum(weights)
    total = cum_weights[-1]

    # Find index where cumulative weight first reaches 50th percentile
    idx = np.searchsorted(cum_weights, total * 0.5)
    idx = min(idx, len(values) - 1)
    return float(values[idx])


# ---------------------------------------------------------------------------
# Peer-group benchmark calculation
# ---------------------------------------------------------------------------

def compute_peer_benchmarks(master: pd.DataFrame) -> tuple[dict[str, float], float, list]:
    """
    Compute GDP-weighted median IDA/GDP contribution rate per income-tier peer group.

    Uses PPP-adjusted GDP (gdp_ppp) where available; falls back to nominal GDP.
    Returns (peer_benchmarks, global_benchmark, donor_iso3_list) where:
      - peer_benchmarks: {peer_group_label: benchmark_rate}
      - global_benchmark: GDP-weighted median across all donors (fallback)
      - donor_iso3_list: list of ISO3 codes used as the donor set
    """
    donors = master[master["is_current_donor"] == 1].copy()

    donors["actual_contribution_usd"] = donors["ida21_contribution_usd"].combine_first(
        donors["ida20_contribution_usd"]
    )

    # Use PPP GDP where available, else nominal
    donors["_gdp_for_benchmark"] = np.where(
        donors.get("ppp_data_available", pd.Series(False, index=donors.index)).astype(bool)
        & donors["gdp_ppp"].notna()
        & (donors["gdp_ppp"] > 0),
        donors["gdp_ppp"],
        donors["gdp_usd"],
    )

    donors = donors[
        donors["_gdp_for_benchmark"].notna()
        & (donors["_gdp_for_benchmark"] > 0)
        & donors["actual_contribution_usd"].notna()
    ].copy()

    donors["ida_gdp_ratio"] = donors["actual_contribution_usd"] / donors["_gdp_for_benchmark"]

    # Global GDP-weighted median (fallback)
    global_benchmark = weighted_median(
        donors["ida_gdp_ratio"].values,
        donors["_gdp_for_benchmark"].values,
    )
    logger.info("Global GDP-weighted IDA/GDP median (all donors): %.6f", global_benchmark)

    # Per-peer-group benchmarks
    peer_benchmarks: dict[str, float] = {}
    if "peer_group" not in donors.columns:
        logger.warning("peer_group column missing from master — using global benchmark for all countries")
        return peer_benchmarks, global_benchmark, donors["iso3"].tolist()

    for group, grp_df in donors.groupby("peer_group"):
        n = len(grp_df)
        if n < MIN_PEER_GROUP_DONORS:
            logger.warning(
                "Peer group '%s' has only %d donors (need %d) — using global benchmark as fallback",
                group, n, MIN_PEER_GROUP_DONORS,
            )
            peer_benchmarks[group] = global_benchmark
        else:
            rate = weighted_median(
                grp_df["ida_gdp_ratio"].values,
                grp_df["_gdp_for_benchmark"].values,
            )
            peer_benchmarks[group] = rate
            logger.info("Peer group '%s' (%d donors): benchmark rate=%.6f", group, n, rate)

    return peer_benchmarks, global_benchmark, donors["iso3"].tolist()


# ---------------------------------------------------------------------------
# Fiscal modifier
# ---------------------------------------------------------------------------

def compute_fiscal_modifier(fiscal_balance_pct: float | None) -> float:
    """
    Map fiscal balance (% of GDP) to a linear modifier in [-0.20, +0.20].

    0% balance → 0 modifier.
    ±5% balance → ±0.20 modifier (capped).
    Null balance → 0 modifier.
    """
    if fiscal_balance_pct is None or np.isnan(fiscal_balance_pct):
        return 0.0
    modifier = fiscal_balance_pct * FISCAL_SCALE_FACTOR
    return float(np.clip(modifier, -FISCAL_MODIFIER_CAP, FISCAL_MODIFIER_CAP))


# ---------------------------------------------------------------------------
# Segment assignment
# ---------------------------------------------------------------------------

def assign_segment(giving_rate_raw: float | None, is_current_donor: bool) -> str:
    """
    Assign donor segment. 'Exceeded Target' takes precedence for over-contributors.
    """
    if giving_rate_raw is None or np.isnan(giving_rate_raw):
        return "Unknown"
    if giving_rate_raw > 1.0:
        return "Exceeded Target"
    if is_current_donor:
        if giving_rate_raw >= 0.80:
            return "Reliable Donor"
        return "Under-Contributing Donor"
    return "Non-Donor"


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_capacity(master: pd.DataFrame | None = None, fiscal_modifier: bool = True) -> pd.DataFrame:
    """
    Compute capacity scores for all countries in master.csv.

    Parameters
    ----------
    master : DataFrame, optional
        If None, loads from data/processed/master.csv.
    fiscal_modifier : bool
        If False, the fiscal balance modifier is set to 0 for all countries.

    Returns
    -------
    DataFrame with capacity scoring columns, written to capacity_scores.csv.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if master is None:
        master = pd.read_csv(MASTER_PATH)

    peer_benchmarks, global_benchmark, donor_list = compute_peer_benchmarks(master)

    # Save run metadata
    metadata = {
        "benchmark_methodology": "GDP-weighted peer-group median IDA/GDP ratio",
        "global_benchmark_ida_gdp_ratio": global_benchmark,
        "peer_benchmarks": peer_benchmarks,
        "donor_set": donor_list,
    }
    with open(RUN_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Run metadata written to %s", RUN_METADATA_PATH)

    # Score every country
    results = []
    for _, row in master.iterrows():
        iso3 = row["iso3"]
        gdp_usd = row.get("gdp_usd")
        gdp_ppp = row.get("gdp_ppp")
        ppp_available = bool(row.get("ppp_data_available", False))
        peer_group = row.get("peer_group", "Unclassified")

        # Choose benchmark: peer-group first, then global fallback
        benchmark_ratio = peer_benchmarks.get(peer_group, global_benchmark)

        # Use PPP GDP for target where available, else nominal
        gdp_for_target = gdp_ppp if (ppp_available and gdp_ppp and gdp_ppp > 0) else gdp_usd

        # Target contribution
        if pd.isna(gdp_for_target) or gdp_for_target == 0:
            logger.warning("Null/zero GDP for %s — skipping capacity score", iso3)
            target_usd = None
            fiscal_mod = None
            adjusted_target_usd = None
        else:
            if not ppp_available and not pd.isna(gdp_usd) and gdp_usd > 0:
                logger.debug("%s: PPP GDP unavailable — using nominal GDP for target", iso3)
            target_usd = float(gdp_for_target) * benchmark_ratio
            fiscal_mod = (
                compute_fiscal_modifier(row.get("fiscal_balance_pct_gdp"))
                if fiscal_modifier else 0.0
            )
            adjusted_target_usd = target_usd * (1.0 + fiscal_mod)

        # Actual contribution: prefer IDA21, fall back to IDA20, default 0
        ida21 = row.get("ida21_contribution_usd")
        ida20 = row.get("ida20_contribution_usd")
        if pd.notna(ida21):
            actual = float(ida21)
        elif pd.notna(ida20):
            actual = float(ida20)
            logger.info("%s: using IDA20 as actual (no IDA21 record)", iso3)
        else:
            actual = 0.0

        # Gap and giving rates
        if adjusted_target_usd is not None and adjusted_target_usd > 0:
            giving_rate_raw = actual / adjusted_target_usd
            giving_rate = min(giving_rate_raw, 1.0)
            gap_usd_signed = adjusted_target_usd - actual  # positive = shortfall
        else:
            giving_rate_raw = None
            giving_rate = None
            gap_usd_signed = None

        # PPP gap percentage
        if gap_usd_signed is not None and ppp_available and gdp_ppp and gdp_ppp > 0:
            gap_pct_ppp_gdp = gap_usd_signed / gdp_ppp * 100.0
        else:
            gap_pct_ppp_gdp = None

        # Segment
        is_donor = row.get("is_current_donor", 0) == 1
        segment = assign_segment(giving_rate_raw, is_donor)

        results.append({
            "iso3": iso3,
            "country_name": row.get("country_name"),
            "income_group": row.get("income_group"),
            "peer_group": peer_group,
            "ppp_data_available": ppp_available,
            "gdp_usd": gdp_usd,
            "gdp_ppp": gdp_ppp,
            "benchmark_ida_gdp_ratio": benchmark_ratio,
            "target_usd": target_usd,
            "fiscal_modifier": fiscal_mod,
            "adjusted_target_usd": adjusted_target_usd,
            "actual_contribution_usd": actual,
            "gap_usd_signed": gap_usd_signed,
            "gap_pct_ppp_gdp": gap_pct_ppp_gdp,
            "giving_rate_raw": giving_rate_raw,
            "giving_rate": giving_rate,
            "donor_segment": segment,
            # CI columns: not available in rule-based scorer
            "gap_usd_lower": None,
            "gap_usd_upper": None,
        })

    scores = pd.DataFrame(results)
    scores.to_csv(CAPACITY_SCORES_PATH, index=False)
    logger.info(
        "Capacity scores written to %s (%d countries, %d with valid gap)",
        CAPACITY_SCORES_PATH,
        len(scores),
        scores["gap_usd_signed"].notna().sum(),
    )
    return scores
