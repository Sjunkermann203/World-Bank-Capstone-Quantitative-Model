"""
Strategic alignment scoring for the Donor Readiness Index.

Computes three component scores (each 0-100) and their equally-weighted composite:
  1. UNGA voting alignment with a reference group
  2. World Bank vote share
  3. IFC active portfolio presence (binary)

Reads:  data/processed/master.csv (for country universe)
        data/raw/ifc_presence.csv
        data/raw/unga_votes.csv       (if available; see note below)
        data/raw/wb_vote_shares.csv   (if available; see note below)

Writes: data/processed/alignment_scores.csv

Data notes
----------
UNGA voting dataset: Use the UN General Assembly Voting Data maintained by
  Erik Voeten et al. (Harvard Dataverse). Download as CSV and place at
  data/raw/unga_votes.csv with columns: iso3, year, agreement_score
  where agreement_score is the share of votes aligned with the reference group (0-1).

World Bank vote share: Download from https://www.worldbank.org/en/about/leadership/votesandsubscriptions
  Place at data/raw/wb_vote_shares.csv with columns: iso3, vote_share_pct
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

MASTER_PATH = DATA_PROCESSED / "master.csv"
IFC_PRESENCE_PATH = DATA_RAW / "ifc_presence.csv"
UNGA_VOTES_PATH = DATA_RAW / "unga_votes.csv"
WB_VOTE_SHARES_PATH = DATA_RAW / "wb_vote_shares.csv"
ALIGNMENT_SCORES_PATH = DATA_PROCESSED / "alignment_scores.csv"

MIN_RESOLUTIONS = 10   # minimum UNGA resolutions to compute alignment score


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def score_unga_alignment(master: pd.DataFrame) -> pd.DataFrame:
    """
    Score UNGA voting alignment for each country.

    Returns a DataFrame with columns: iso3, unga_alignment_score (0-100 or null).
    Assigns null if the country has fewer than MIN_RESOLUTIONS resolutions.
    """
    if not UNGA_VOTES_PATH.exists():
        logger.warning(
            "UNGA votes file not found at %s — UNGA alignment scores will be null for all countries. "
            "Download from Harvard Dataverse and place at data/raw/unga_votes.csv.",
            UNGA_VOTES_PATH,
        )
        return master[["iso3"]].assign(unga_alignment_score=np.nan)

    votes = pd.read_csv(UNGA_VOTES_PATH, comment="#")
    votes.columns = votes.columns.str.strip().str.lower()
    votes["iso3"] = votes["iso3"].str.strip().str.upper()

    results = []
    for iso3 in master["iso3"].tolist():
        country_votes = votes[votes["iso3"] == iso3]
        if len(country_votes) < MIN_RESOLUTIONS:
            logger.warning(
                "%s: only %d UNGA resolutions (need %d) — alignment score set to null",
                iso3, len(country_votes), MIN_RESOLUTIONS,
            )
            results.append({"iso3": iso3, "unga_alignment_score": np.nan})
        else:
            # agreement_score is expected to be in [0, 1]; scale to [0, 100]
            avg_agreement = country_votes["agreement_score"].mean()
            score = float(avg_agreement * 100)
            results.append({"iso3": iso3, "unga_alignment_score": score})

    return pd.DataFrame(results)


def score_wb_vote_share(master: pd.DataFrame) -> pd.DataFrame:
    """
    Score World Bank vote share, normalized to [0, 100].

    Returns a DataFrame with columns: iso3, wb_vote_share_score (0-100).
    Assigns 0 for countries not in the WB vote share file.
    """
    if not WB_VOTE_SHARES_PATH.exists():
        logger.warning(
            "WB vote shares file not found at %s — WB vote share scores will be 0 for all countries. "
            "Download from World Bank and place at data/raw/wb_vote_shares.csv.",
            WB_VOTE_SHARES_PATH,
        )
        return master[["iso3"]].assign(wb_vote_share_score=0.0)

    wb = pd.read_csv(WB_VOTE_SHARES_PATH, comment="#")
    wb.columns = wb.columns.str.strip().str.lower()
    wb["iso3"] = wb["iso3"].str.strip().str.upper()

    if "vote_share_pct" not in wb.columns:
        logger.warning("WB vote shares file missing 'vote_share_pct' column — scores set to 0")
        return master[["iso3"]].assign(wb_vote_share_score=0.0)

    max_share = wb["vote_share_pct"].max()
    if max_share == 0:
        logger.warning("Max WB vote share is 0 — scores set to 0")
        return master[["iso3"]].assign(wb_vote_share_score=0.0)

    wb["wb_vote_share_score"] = (wb["vote_share_pct"] / max_share * 100).clip(0, 100)

    result = master[["iso3"]].merge(wb[["iso3", "wb_vote_share_score"]], on="iso3", how="left")
    result["wb_vote_share_score"] = result["wb_vote_share_score"].fillna(0.0)
    return result[["iso3", "wb_vote_share_score"]]


def score_ifc_presence(master: pd.DataFrame) -> pd.DataFrame:
    """
    Score IFC active portfolio presence as binary 0 or 100.

    Returns a DataFrame with columns: iso3, ifc_presence_score.
    """
    ifc = pd.read_csv(IFC_PRESENCE_PATH, comment="#")
    ifc.columns = ifc.columns.str.strip().str.lower()
    ifc["iso3"] = ifc["iso3"].str.strip().str.upper()
    ifc["ifc_presence_score"] = ifc["active_portfolio"].fillna(0).astype(int) * 100

    result = master[["iso3"]].merge(ifc[["iso3", "ifc_presence_score"]], on="iso3", how="left")
    result["ifc_presence_score"] = result["ifc_presence_score"].fillna(0.0)
    return result[["iso3", "ifc_presence_score"]]


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_composite(row: pd.Series) -> tuple[float | None, list[str]]:
    """
    Compute the equally-weighted composite alignment score from available components.

    Returns (composite_score, list_of_excluded_components).
    """
    components = {
        "unga_alignment_score": row.get("unga_alignment_score"),
        "wb_vote_share_score": row.get("wb_vote_share_score"),
        "ifc_presence_score": row.get("ifc_presence_score"),
    }

    available = {k: v for k, v in components.items() if pd.notna(v)}
    excluded = [k for k in components if pd.isna(components[k])]

    if not available:
        return None, list(components.keys())

    composite = sum(available.values()) / len(available)
    return float(composite), excluded


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_alignment(master: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute strategic alignment scores for all countries.

    Parameters
    ----------
    master : DataFrame, optional
        If None, loads from data/processed/master.csv.

    Returns
    -------
    DataFrame written to data/processed/alignment_scores.csv.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if master is None:
        master = pd.read_csv(MASTER_PATH)

    unga = score_unga_alignment(master)
    wb = score_wb_vote_share(master)
    ifc = score_ifc_presence(master)

    scores = master[["iso3", "country_name"]].copy()
    scores = scores.merge(unga, on="iso3", how="left")
    scores = scores.merge(wb, on="iso3", how="left")
    scores = scores.merge(ifc, on="iso3", how="left")

    composites = []
    excluded_log = []
    for _, row in scores.iterrows():
        composite, excluded = compute_composite(row)
        composites.append(composite)
        if excluded:
            excluded_log.append({"iso3": row["iso3"], "excluded_components": excluded})

    scores["alignment_score"] = composites
    scores["excluded_components"] = [
        ",".join(e["excluded_components"]) if e else ""
        for e in (
            next((x for x in excluded_log if x["iso3"] == row["iso3"]), {"excluded_components": []})
            for _, row in scores.iterrows()
        )
    ]

    scores.to_csv(ALIGNMENT_SCORES_PATH, index=False)
    logger.info(
        "Alignment scores written to %s (%d countries, %d with valid composite)",
        ALIGNMENT_SCORES_PATH,
        len(scores),
        scores["alignment_score"].notna().sum(),
    )
    return scores
