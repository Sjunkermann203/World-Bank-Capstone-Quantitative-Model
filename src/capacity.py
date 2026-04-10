"""
Capacity scoring for the Donor Readiness Index.

Computes:
  - Income-tier-adjusted benchmark IDA/GDP ratio (from current donors)
  - Capacity-based target contribution per country
  - Fiscal modifier (±20% max, linear)
  - Adjusted target contribution
  - Contribution gap (adjusted_target − actual)
  - Giving rate (actual / adjusted_target)

Reads:  data/processed/master.csv
Writes: data/processed/capacity_scores.csv
        data/processed/run_metadata.json (tier medians + donor set)
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
MIN_DONORS_PER_TIER = 3         # fall back to global median if below this


# ---------------------------------------------------------------------------
# Tier median calculation
# ---------------------------------------------------------------------------

def compute_tier_medians(master: pd.DataFrame) -> dict[str, float]:
    """
    Compute the median IDA/GDP ratio for each income tier using current donors.

    For tiers with enough donors (≥ MIN_DONORS_PER_TIER), uses the observed
    median IDA/GDP ratio for that tier's donors.

    For tiers with no (or too few) donors, derives a scaled benchmark from the
    HIC median proportional to that tier's median GDP per capita vs. HIC median
    GDP per capita. This ensures LIC/LMC/UMC countries are benchmarked relative
    to their income level rather than holding them to an HIC standard.

    Returns a dict mapping income_group → median_ida_gdp_ratio, plus "_global".
    """
    donors = master[master["is_current_donor"] == 1].copy()

    # Use IDA21 contribution as the primary actual; fall back to IDA20
    donors["actual_contribution_usd"] = donors["ida21_contribution_usd"].combine_first(
        donors["ida20_contribution_usd"]
    )

    # Compute IDA/GDP ratio for donors with valid data
    donors = donors[donors["gdp_usd"].notna() & (donors["gdp_usd"] > 0)]
    donors = donors[donors["actual_contribution_usd"].notna()]
    donors["ida_gdp_ratio"] = donors["actual_contribution_usd"] / donors["gdp_usd"]

    # Global median (used as ultimate fallback and for scaling reference)
    global_median = float(donors["ida_gdp_ratio"].median())
    logger.info("Global IDA/GDP median (all donors): %.6f", global_median)

    # Median GDP per capita per tier across ALL countries (not just donors),
    # used to scale benchmarks for tiers with no donors.
    gdp_pc = (
        master[master["gdp_per_capita_usd"].notna()]
        .groupby("income_group")["gdp_per_capita_usd"]
        .median()
    )

    tier_medians: dict[str, float] = {"_global": global_median}
    donor_set: dict[str, list] = {"_global": donors["iso3"].tolist()}

    # Donor-based medians for tiers that have enough donors
    for group, group_df in donors.groupby("income_group"):
        n = len(group_df)
        if n >= MIN_DONORS_PER_TIER:
            median = float(group_df["ida_gdp_ratio"].median())
            tier_medians[group] = median
            logger.info("IDA/GDP median for %s: %.6f (%d donors)", group, median, n)
        donor_set[group] = group_df["iso3"].tolist()

    # For every income group in the universe, fill any missing tier median
    # using GDP-per-capita scaling from the HIC benchmark.
    hic_median = tier_medians.get("HIC", global_median)
    hic_gdp_pc = gdp_pc.get("HIC")

    for group in master["income_group"].dropna().unique():
        if group in tier_medians:
            continue  # already set from donor data
        tier_gdp_pc = gdp_pc.get(group)
        if hic_gdp_pc and tier_gdp_pc and hic_gdp_pc > 0:
            scale = tier_gdp_pc / hic_gdp_pc
            tier_medians[group] = hic_median * scale
            logger.info(
                "IDA/GDP median for %s: %.6f (scaled from HIC; GDP/cap ratio=%.3f)",
                group, tier_medians[group], scale,
            )
        else:
            tier_medians[group] = global_median
            logger.warning(
                "IDA/GDP median for %s: using global median fallback (no GDP/cap data)",
                group,
            )

    return tier_medians, donor_set


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
        If False, the fiscal balance modifier is set to 0 for all countries,
        so adjusted_target_usd == target_usd. Useful for comparing with/without.

    Returns
    -------
    DataFrame with capacity scoring columns, written to capacity_scores.csv.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if master is None:
        master = pd.read_csv(MASTER_PATH)

    tier_medians, donor_set = compute_tier_medians(master)

    # Save run metadata
    metadata = {
        "tier_medians": {k: v for k, v in tier_medians.items()},
        "donor_sets": donor_set,
    }
    with open(RUN_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Run metadata written to %s", RUN_METADATA_PATH)

    # Score every country
    results = []
    for _, row in master.iterrows():
        iso3 = row["iso3"]
        income_group = row.get("income_group")
        gdp_usd = row.get("gdp_usd")

        # Target contribution
        if pd.isna(gdp_usd) or gdp_usd == 0:
            logger.warning("Null/zero GDP for %s — skipping capacity score", iso3)
            target_usd = None
            fiscal_mod = None
            adjusted_target_usd = None
        else:
            # Pick tier median; fall back to global if tier not in table
            benchmark_ratio = tier_medians.get(income_group, tier_medians["_global"])
            target_usd = float(gdp_usd) * benchmark_ratio

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

        # Gap and giving rate
        if adjusted_target_usd is not None and adjusted_target_usd > 0:
            gap_usd = adjusted_target_usd - actual
            giving_rate = actual / adjusted_target_usd
        else:
            gap_usd = None
            giving_rate = None

        tier_median_used = tier_medians.get(income_group, tier_medians["_global"])

        results.append({
            "iso3": iso3,
            "country_name": row.get("country_name"),
            "income_group": income_group,
            "gdp_usd": gdp_usd,
            "tier_median_ida_gdp_ratio": tier_median_used,
            "target_usd": target_usd,
            "fiscal_modifier": fiscal_mod,
            "adjusted_target_usd": adjusted_target_usd,
            "actual_contribution_usd": actual,
            "gap_usd": gap_usd,
            "giving_rate": giving_rate,
        })

    scores = pd.DataFrame(results)
    scores.to_csv(CAPACITY_SCORES_PATH, index=False)
    logger.info(
        "Capacity scores written to %s (%d countries, %d with valid gap)",
        CAPACITY_SCORES_PATH,
        len(scores),
        scores["gap_usd"].notna().sum(),
    )
    return scores
