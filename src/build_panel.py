"""
Build the Heckman panel dataset from real IDA contribution data.

Parses the wide-format IDA replenishment CSV, joins historical WDI covariates,
and produces the long-format country × round panel expected by src/heckman.py.

Reads:
  data/raw/IDA_Contributions.xlsx - IDAReplenishments (1-21).csv
  data/country_map.csv
  data/cache/wdi_historical.csv  (auto-created; delete to refresh)

Writes:
  data/raw/heckman_panel.csv

Usage:
  uv run python src/build_panel.py              # use WDI cache if present
  uv run python src/build_panel.py --refresh    # force re-fetch from World Bank API
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"

CONTRIBUTIONS_PATH = DATA_RAW / "IDA_Contributions.xlsx - IDAReplenishments (1-21).csv"
COUNTRY_MAP_PATH = ROOT / "data" / "country_map.csv"
WDI_HIST_CACHE = DATA_CACHE / "wdi_historical.csv"
PANEL_OUT = DATA_RAW / "heckman_panel.csv"

# ---------------------------------------------------------------------------
# IDA round metadata
# ---------------------------------------------------------------------------

# Year the replenishment was negotiated / signed (used to look up covariates)
IDA_ROUND_YEAR: dict[str, int] = {
    "IDA1": 1963, "IDA2": 1968, "IDA3": 1970, "IDA4": 1973, "IDA5": 1977,
    "IDA6": 1980, "IDA7": 1984, "IDA8": 1987, "IDA9": 1990, "IDA10": 1993,
    "IDA11": 1996, "IDA12": 1999, "IDA13": 2002, "IDA14": 2005, "IDA15": 2007,
    "IDA16": 2010, "IDA17": 2013, "IDA18": 2016, "IDA19": 2019, "IDA20": 2021,
    "IDA21": 2024,
}

# Rounds whose amounts are in SDR millions (IDA6–IDA18); all others are USD millions
SDR_ROUNDS = {f"IDA{i}" for i in range(6, 19)}

# Approximate period-average SDR/USD exchange rates for each SDR-denominated round
SDR_TO_USD: dict[str, float] = {
    "IDA6": 1.292,
    "IDA7": 1.094,
    "IDA8": 1.418,
    "IDA9": 1.370,
    "IDA10": 1.379,
    "IDA11": 1.453,
    "IDA12": 1.372,
    "IDA13": 1.356,
    "IDA14": 1.513,
    "IDA15": 1.622,
    "IDA16": 1.547,
    "IDA17": 1.547,
    "IDA18": 1.394,
}

ALL_ROUNDS = list(IDA_ROUND_YEAR.keys())

# ---------------------------------------------------------------------------
# Country name → ISO3 lookup (names as they appear in the contributions CSV)
# ---------------------------------------------------------------------------

CONTRIB_NAME_TO_ISO3: dict[str, str] = {
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bulgaria": "BGR",
    "Croatia": "HRV",
    "Cyprus": "CYP",
    "Czechia": "CZE",
    "Denmark": "DNK",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Hungary": "HUN",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Malta": "MLT",
    "Netherlands": "NLD",
    "Poland": "POL",
    "Portugal": "PRT",
    "Romania": "ROU",
    "Slovak Republic": "SVK",
    "Slovenia": "SVN",
    "Spain": "ESP",
    "Sweden": "SWE",
    "UK": "GBR",
    "Canada": "CAN",
    "China": "CHN",
    "Japan": "JPN",
    "Korea": "KOR",
    "Norway": "NOR",
    "Saudi Arabia": "SAU",
    "Switzerland": "CHE",
    "United States": "USA",
    # Australia and New Zealand appear in the hand-curated ida_contributions.csv
    # but not in the historical replenishment sheet; they're included below as
    # non-donors for all historical rounds.
    "Australia": "AUS",
    "New Zealand": "NZL",
    "Kuwait": "KWT",
    "Singapore": "SGP",
}

# Rows that are subtotals/aggregates — skip these
SKIP_ROWS = {
    "Total EU MS", "EU (+ UK)", "Total Replenishment", "",
    "EU MS Total", "Grand Total",
}

# ---------------------------------------------------------------------------
# Static country-level covariates
# ---------------------------------------------------------------------------

# Year each country joined the OECD DAC (0 = never / unknown)
DAC_JOIN_YEAR: dict[str, int] = {
    "USA": 1961, "GBR": 1961, "FRA": 1961, "DEU": 1961, "JPN": 1961,
    "CAN": 1961, "ITA": 1961, "BEL": 1961, "NLD": 1961, "DNK": 1962,
    "NOR": 1962, "SWE": 1965, "AUT": 1965, "AUS": 1966, "CHE": 1968,
    "NZL": 1973, "FIN": 1975, "IRL": 1985, "PRT": 1991, "ESP": 1991,
    "LUX": 1992, "GRC": 1999, "KOR": 2010, "ISL": 2013, "CZE": 2013,
    "SVK": 2013, "HUN": 2016, "POL": 2013, "SVN": 2013, "LVA": 2022,
    "LTU": 2022,
}

# 1 if country is an EU member or close US/Western ally (broad definition)
US_EU_ALLY: dict[str, int] = {iso3: 1 for iso3 in [
    # EU members
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
    # Close US allies (NATO non-EU + others)
    "USA", "CAN", "GBR", "NOR", "ISL", "TUR", "AUS", "NZL", "JPN", "KOR",
    "ISR", "CHE",
]}

# Approximate long-run sovereign credit ratings (S&P scale) used as proxy
# for all rounds — ordinal encode in heckman.py (AAA=20 … D=1)
SOVEREIGN_RATING: dict[str, str] = {
    "AUS": "AAA", "AUT": "AA+", "BEL": "AA", "BGR": "BBB", "CAN": "AAA",
    "CHE": "AAA", "CHN": "A+", "CYP": "BBB+", "CZE": "AA-", "DEU": "AAA",
    "DNK": "AAA", "ESP": "A", "EST": "AA-", "FIN": "AA+", "FRA": "AA-",
    "GBR": "AA", "GRC": "BBB-", "HRV": "BBB+", "HUN": "BBB", "IRL": "AA",
    "ITA": "BBB", "JPN": "A+", "KOR": "AA", "KWT": "AA-", "LTU": "A",
    "LUX": "AAA", "LVA": "A-", "MLT": "A+", "NLD": "AAA", "NOR": "AAA",
    "NZL": "AA+", "POL": "A-", "PRT": "BBB+", "ROU": "BBB-", "SAU": "A",
    "SGP": "AAA", "SVK": "A+", "SVN": "AA-", "SWE": "AAA", "USA": "AA+",
    # Non-donor HICs (approximate)
    "ARE": "AA", "BHR": "B+", "CHL": "A", "HKG": "AA+", "HRV": "BBB+",
    "ISL": "A", "ISR": "A+", "OMN": "BB+", "QAT": "AA-", "SGP": "AAA",
    "TTO": "BBB-", "URY": "BBB",
}

# Approximate UN voting alignment with the US (0–1 scale, higher = more aligned)
# Based on recent UNGA voting record patterns — used as a static proxy
UN_VOTING_ALIGN: dict[str, float] = {
    "USA": 1.00,
    "ISR": 0.85, "GBR": 0.62, "FRA": 0.55, "DEU": 0.54, "CAN": 0.60,
    "AUS": 0.65, "NZL": 0.62, "JPN": 0.60, "KOR": 0.58, "NOR": 0.50,
    "DNK": 0.53, "SWE": 0.50, "FIN": 0.51, "NLD": 0.53, "BEL": 0.52,
    "ITA": 0.54, "ESP": 0.52, "PRT": 0.52, "GRC": 0.48, "IRL": 0.50,
    "AUT": 0.50, "CHE": 0.47, "LUX": 0.52, "POL": 0.55, "CZE": 0.54,
    "SVK": 0.52, "HUN": 0.50, "HRV": 0.52, "SVN": 0.52, "EST": 0.55,
    "LVA": 0.55, "LTU": 0.55, "BGR": 0.51, "ROU": 0.53, "MLT": 0.48,
    "CYP": 0.42, "SGP": 0.42, "KWT": 0.40, "SAU": 0.38, "CHN": 0.25,
    "ARE": 0.38, "QAT": 0.38, "BHR": 0.40,
}

# Approximate IDA vote share (%) — based on recent IDA replenishment data
# Used as a lag proxy for all rounds (simplified)
IDA_VOTE_SHARE: dict[str, float] = {
    "USA": 10.53, "JPN": 8.00, "GBR": 6.00, "DEU": 5.50, "FRA": 5.00,
    "CAN": 3.50, "ITA": 3.50, "NLD": 3.00, "SWE": 2.80, "NOR": 2.00,
    "AUS": 2.50, "CHE": 2.50, "DEN": 1.80, "BEL": 2.00, "CHN": 3.50,
    "DNK": 1.80, "KOR": 1.80, "FIN": 1.20, "AUT": 1.20, "ESP": 1.50,
    "SAU": 2.00, "SGP": 0.50, "IRL": 0.70, "NZL": 0.50, "PRT": 0.50,
    "LUX": 0.40, "KWT": 0.60, "POL": 0.30, "HUN": 0.20, "SVK": 0.10,
    "CZE": 0.20, "HRV": 0.05, "CYP": 0.05, "LVA": 0.05, "LTU": 0.05,
    "EST": 0.05, "BGR": 0.05, "ROU": 0.05, "SVN": 0.05, "MLT": 0.02,
    "GRC": 0.10,
}

# Approximate trade exposure to IDA-eligible countries (% of total trade)
# Rough proxy based on geography and economic structure
TRADE_EXPOSURE_IDA: dict[str, float] = {
    "CHN": 0.45, "JPN": 0.30, "KOR": 0.28, "SGP": 0.35, "AUS": 0.25,
    "IND": 0.30, "GBR": 0.20, "FRA": 0.18, "DEU": 0.16, "NLD": 0.18,
    "BEL": 0.17, "ITA": 0.19, "ESP": 0.18, "PRT": 0.22, "NOR": 0.15,
    "SWE": 0.14, "DNK": 0.13, "FIN": 0.12, "AUT": 0.12, "CHE": 0.14,
    "USA": 0.18, "CAN": 0.15, "SAU": 0.28, "KWT": 0.25, "ARE": 0.35,
    "QAT": 0.30, "NZL": 0.20,
}

# ---------------------------------------------------------------------------
# Parse IDA contributions CSV
# ---------------------------------------------------------------------------

def parse_contributions() -> pd.DataFrame:
    """
    Parse the wide-format IDA replenishment CSV into a long-format DataFrame.

    Returns columns: country_iso3, replenishment_round, donation_usd_millions
    Only rows where an amount is present are returned (i.e., actual donations).
    """
    raw = pd.read_csv(CONTRIBUTIONS_PATH, header=None, low_memory=False)

    # Row 0: round names like "IDA21 (2024)", "IDA20 (2021)", ...
    # Row 1: column type labels ("Share", "Amount (US$, millions)", ...)
    # Row 2+: country data

    # Build column labels from rows 0 and 1
    round_row = raw.iloc[0].tolist()
    type_row = raw.iloc[1].tolist()

    # Walk columns to identify (round, col_type) pairs
    current_round = None
    col_meta: list[tuple[str | None, str]] = []  # (round_label, "share" | "amount" | "other")
    for i, (rval, tval) in enumerate(zip(round_row, type_row)):
        rval = str(rval).strip() if pd.notna(rval) else ""
        tval = str(tval).strip() if pd.notna(tval) else ""

        if rval.startswith("IDA"):
            # e.g. "IDA21 (2024)" → "IDA21"
            current_round = rval.split()[0].strip()
        elif rval and not rval.startswith("IDA"):
            current_round = None  # aggregate/summary columns at the end

        if "Amount" in tval:
            col_meta.append((current_round, "amount"))
        elif "Share" in tval:
            col_meta.append((current_round, "share"))
        else:
            col_meta.append((None, "other"))

    # Data starts at row 2 (0-indexed)
    data_rows = raw.iloc[2:].reset_index(drop=True)

    records = []
    for _, row in data_rows.iterrows():
        country_raw = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if not country_raw or country_raw in SKIP_ROWS:
            continue
        iso3 = CONTRIB_NAME_TO_ISO3.get(country_raw)
        if iso3 is None:
            logger.debug("Skipping unrecognized country name: %r", country_raw)
            continue

        for col_idx, (rnd, ctype) in enumerate(col_meta):
            if ctype != "amount" or rnd is None:
                continue
            if rnd not in IDA_ROUND_YEAR:
                continue

            raw_val = row.iloc[col_idx]
            if pd.isna(raw_val):
                continue
            # Clean string amounts ("  6.50 ", "1,345.77")
            val_str = str(raw_val).strip().replace(",", "").replace(" ", "")
            if not val_str:
                continue
            try:
                amount = float(val_str)
            except ValueError:
                continue
            if amount <= 0:
                continue

            # Convert SDR → USD for the relevant rounds
            if rnd in SDR_ROUNDS:
                amount = amount * SDR_TO_USD[rnd]

            records.append({
                "country_iso3": iso3,
                "replenishment_round": rnd,
                "donation_usd_millions": round(amount, 4),
            })

    df = pd.DataFrame(records)
    logger.info("Parsed %d contribution records from %d countries across %d rounds",
                len(df),
                df["country_iso3"].nunique() if len(df) else 0,
                df["replenishment_round"].nunique() if len(df) else 0)
    return df


# ---------------------------------------------------------------------------
# Build country × round universe
# ---------------------------------------------------------------------------

def build_universe(contributions: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full (country, round) panel universe.

    Includes:
    - All contributor ISO3 codes across all rounds they appear in
    - All HIC countries from the country_map that are not already donors
      (as non-donor observations for every round)
    """
    country_map = pd.read_csv(COUNTRY_MAP_PATH, comment="#")
    country_map["iso3"] = country_map["iso3"].str.strip().str.upper()

    # Donor countries (those appearing in the contributions CSV)
    donor_iso3 = set(contributions["country_iso3"].unique())

    # Potential non-donor universe: all HICs not in the donor list,
    # plus any UMC countries that have been donors historically
    non_donor_pool = country_map[
        (country_map["income_group"] == "HIC") &
        (~country_map["iso3"].isin(donor_iso3))
    ]["iso3"].tolist()

    all_iso3 = sorted(donor_iso3) + sorted(non_donor_pool)
    rows = []
    for iso3 in all_iso3:
        for rnd in ALL_ROUNDS:
            rows.append({"country_iso3": iso3, "replenishment_round": rnd})
    universe = pd.DataFrame(rows)

    # Merge in actual contributions
    universe = universe.merge(contributions, on=["country_iso3", "replenishment_round"], how="left")
    universe["donate_dummy"] = (universe["donation_usd_millions"].notna() &
                                (universe["donation_usd_millions"] > 0)).astype(int)
    universe["donation_usd"] = universe["donation_usd_millions"].fillna(0.0) * 1e6

    logger.info("Universe: %d country × round observations (%d countries, %d rounds)",
                len(universe), universe["country_iso3"].nunique(), universe["replenishment_round"].nunique())
    return universe


# ---------------------------------------------------------------------------
# World Bank historical WDI fetch
# ---------------------------------------------------------------------------

_WB_API_BASE = "https://api.worldbank.org/v2"
_HIST_INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_usd",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "NE.TRD.GNFS.ZS": "trade_openness",
    "GC.REV.XGRT.GD.ZS": "govt_revenue_pct_gdp",
    "GC.XPN.TOTL.GD.ZS": "govt_expenditure_pct_gdp",
    "GB.XPD.RSDV.GD.ZS": "rd_pct_gdp",  # optional; drop if unavailable
    "GOV_WGI_GE.EST": "gov_effectiveness",  # WGI governance; available 1996+
}
_HIST_BATCH = 10
_HIST_TIMEOUT = 180
_HIST_DATE_RANGE = "1960:2024"


def _fetch_indicator_all_years(indicator_code: str, iso3_list: list[str]) -> pd.DataFrame:
    """
    Fetch all available years for one indicator across all countries.
    Returns DataFrame with columns: iso3, year, value.
    """
    all_records: list[dict] = []
    batches = [iso3_list[i:i + _HIST_BATCH] for i in range(0, len(iso3_list), _HIST_BATCH)]
    for batch in batches:
        codes = ";".join(batch)
        url = f"{_WB_API_BASE}/countries/{codes}/indicators/{indicator_code}"
        params = {"format": "json", "per_page": 10000, "date": _HIST_DATE_RANGE}
        try:
            resp = requests.get(url, params=params, timeout=_HIST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            records = payload[1] if (isinstance(payload, list) and len(payload) > 1) else []
            for rec in (records or []):
                iso3 = rec.get("countryiso3code", "")
                year = rec.get("date")
                value = rec.get("value")
                if iso3 and year and value is not None:
                    try:
                        all_records.append({
                            "iso3": iso3, "year": int(year), "value": float(value)
                        })
                    except (TypeError, ValueError):
                        pass
        except Exception as exc:
            logger.warning("WDI hist fetch failed for %s batch %s: %s", indicator_code, batch, exc)

    return pd.DataFrame(all_records) if all_records else pd.DataFrame(columns=["iso3", "year", "value"])


def fetch_wdi_historical(iso3_list: list[str], refresh: bool = False) -> pd.DataFrame:
    """
    Fetch (or load from cache) all historical WDI data for the given countries.

    Returns a wide DataFrame with columns:
      iso3, year, gdp_usd, gdp_per_capita_usd, trade_openness,
      govt_revenue_pct_gdp, govt_expenditure_pct_gdp
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)

    if WDI_HIST_CACHE.exists() and not refresh:
        cached = pd.read_csv(WDI_HIST_CACHE)
        missing_fields = [f for f in _HIST_INDICATORS.values() if f not in cached.columns]
        if not missing_fields:
            logger.info("Loading historical WDI data from cache: %s", WDI_HIST_CACHE)
            return cached
        # Fetch only missing indicators and merge into existing cache
        logger.info("Cache missing columns %s — fetching only those indicators", missing_fields)
        missing_codes = {k: v for k, v in _HIST_INDICATORS.items() if v in missing_fields}
        for code, field in missing_codes.items():
            logger.info("  Fetching %s → %s ...", code, field)
            long = _fetch_indicator_all_years(code, iso3_list)
            if long.empty:
                logger.warning("  No data returned for %s", code)
                continue
            long = long.rename(columns={"value": field})
            cached = cached.merge(long, on=["iso3", "year"], how="outer")
        cached.to_csv(WDI_HIST_CACHE, index=False)
        logger.info("Updated cache saved to %s (%d rows)", WDI_HIST_CACHE, len(cached))
        return cached

    logger.info("Fetching historical WDI data for %d countries (this may take a few minutes)...",
                len(iso3_list))

    wide: pd.DataFrame | None = None
    for code, field in _HIST_INDICATORS.items():
        logger.info("  Fetching %s → %s ...", code, field)
        long = _fetch_indicator_all_years(code, iso3_list)
        if long.empty:
            logger.warning("  No data returned for %s", code)
            continue
        long = long.rename(columns={"value": field})
        if wide is None:
            wide = long
        else:
            wide = wide.merge(long, on=["iso3", "year"], how="outer")

    if wide is None or wide.empty:
        logger.error("No WDI historical data fetched — panel covariates will be null")
        wide = pd.DataFrame({"iso3": iso3_list})
        wide["year"] = 2000

    wide.to_csv(WDI_HIST_CACHE, index=False)
    logger.info("Historical WDI data cached to %s (%d rows)", WDI_HIST_CACHE, len(wide))
    return wide


def lookup_wdi_for_round(wdi: pd.DataFrame, iso3: str, year: int,
                          field: str, window: int = 3) -> float | None:
    """
    Find the closest available WDI value for a country-year within ±window years.
    Returns None if no data found.
    """
    subset = wdi[(wdi["iso3"] == iso3) & (wdi["year"].between(year - window, year + window))]
    if subset.empty or field not in subset.columns:
        return None
    subset = subset.dropna(subset=[field])
    if subset.empty:
        return None
    # Pick the year closest to the target
    idx = (subset["year"] - year).abs().idxmin()
    return float(subset.loc[idx, field])


# ---------------------------------------------------------------------------
# Derive covariates
# ---------------------------------------------------------------------------

def derive_donation_lag(universe: pd.DataFrame) -> dict[tuple[str, str], float]:
    """
    For each (country, round), compute log of donation in the previous round.
    Returns a dict keyed by (country_iso3, replenishment_round).
    """
    lag: dict[tuple[str, str], float] = {}
    for iso3 in universe["country_iso3"].unique():
        country_rows = universe[universe["country_iso3"] == iso3].copy()
        country_rows = country_rows.set_index("replenishment_round").reindex(ALL_ROUNDS)
        prev_donation = 0.0
        for rnd in ALL_ROUNDS:
            lag[(iso3, rnd)] = np.log(prev_donation + 1)
            row = country_rows.loc[rnd] if rnd in country_rows.index else None
            if row is not None and pd.notna(row.get("donation_usd")) and row.get("donation_usd", 0) > 0:
                prev_donation = float(row["donation_usd"])
            else:
                prev_donation = 0.0
    return lag


def derive_peer_donor(universe: pd.DataFrame) -> dict[tuple[str, str], int]:
    """
    For each (country, round), flag 1 if >50% of established donors contributed
    that round (indicating strong peer pressure in that round).
    """
    donor_counts = universe.groupby("replenishment_round")["donate_dummy"].sum()
    total_donors_ever = universe[universe["donate_dummy"] == 1]["country_iso3"].nunique()
    threshold = total_donors_ever * 0.5

    result: dict[tuple[str, str], int] = {}
    for iso3 in universe["country_iso3"].unique():
        for rnd in ALL_ROUNDS:
            count = donor_counts.get(rnd, 0)
            result[(iso3, rnd)] = 1 if count >= threshold else 0
    return result


# ---------------------------------------------------------------------------
# Assemble panel
# ---------------------------------------------------------------------------

def build_panel(refresh: bool = False) -> pd.DataFrame:
    """End-to-end panel construction."""
    # 1. Parse contributions
    contributions = parse_contributions()

    # 2. Build country × round universe
    universe = build_universe(contributions)
    all_iso3 = universe["country_iso3"].unique().tolist()

    # 3. Fetch historical WDI
    wdi = fetch_wdi_historical(all_iso3, refresh=refresh)

    # 4. Precompute derived panel variables
    lag_map = derive_donation_lag(universe)
    peer_map = derive_peer_donor(universe)

    # 5. Build rows
    rows = []
    for _, obs in universe.iterrows():
        iso3: str = obs["country_iso3"]
        rnd: str = obs["replenishment_round"]
        year: int = IDA_ROUND_YEAR[rnd]

        gdp_usd = lookup_wdi_for_round(wdi, iso3, year, "gdp_usd")
        gdp_pc = lookup_wdi_for_round(wdi, iso3, year, "gdp_per_capita_usd")
        trade = lookup_wdi_for_round(wdi, iso3, year, "trade_openness")
        rev = lookup_wdi_for_round(wdi, iso3, year, "govt_revenue_pct_gdp")
        exp_ = lookup_wdi_for_round(wdi, iso3, year, "govt_expenditure_pct_gdp")

        log_gdp_pc = np.log(gdp_pc) if gdp_pc and gdp_pc > 0 else np.nan
        log_gdp_lvl = np.log(gdp_usd) if gdp_usd and gdp_usd > 0 else np.nan
        fiscal_bal = (rev - exp_) if (rev is not None and exp_ is not None) else np.nan

        # Governance effectiveness: WGI only from 1996 (IDA11+).
        # For pre-1996 rounds, backfill with the country's earliest available WGI value.
        gov_eff_raw = lookup_wdi_for_round(wdi, iso3, year, "gov_effectiveness", window=4)
        if gov_eff_raw is None and "gov_effectiveness" in wdi.columns:
            country_wgi = wdi[(wdi["iso3"] == iso3) & wdi["gov_effectiveness"].notna()]
            if not country_wgi.empty:
                gov_eff_raw = float(country_wgi.sort_values("year").iloc[0]["gov_effectiveness"])
        gov_eff = gov_eff_raw if gov_eff_raw is not None else np.nan

        dac_year = DAC_JOIN_YEAR.get(iso3, 9999)
        dac_member = 1 if year >= dac_year else 0

        row = {
            "country_iso3": iso3,
            "replenishment_round": rnd,
            "donate_dummy": int(obs["donate_dummy"]),
            "donation_usd": float(obs["donation_usd"]),
            "log_gdp_per_capita": round(log_gdp_pc, 6) if not np.isnan(log_gdp_pc) else np.nan,
            "dac_member": dac_member,
            "un_voting_align": UN_VOTING_ALIGN.get(iso3, 0.45),
            "trade_openness": round(trade, 4) if trade is not None else np.nan,
            "gov_effectiveness": round(gov_eff, 6),
            "peer_donor": peer_map[(iso3, rnd)],
            "log_gdp_level": round(log_gdp_lvl, 6) if not np.isnan(log_gdp_lvl) else np.nan,
            "fiscal_balance_pct_gdp": round(fiscal_bal, 4) if not np.isnan(fiscal_bal) else np.nan,
            "ida_vote_share_lag": IDA_VOTE_SHARE.get(iso3, 0.05),
            "trade_exposure_ida": TRADE_EXPOSURE_IDA.get(iso3, 0.15),
            "log_donation_lag": round(lag_map[(iso3, rnd)], 6),
            "us_eu_ally": US_EU_ALLY.get(iso3, 0),
            "sovereign_credit_rating": SOVEREIGN_RATING.get(iso3, "BBB"),
        }
        rows.append(row)

    panel = pd.DataFrame(rows)

    # Sanity check: no duplicate keys
    dupes = panel[panel.duplicated(subset=["country_iso3", "replenishment_round"])]
    if not dupes.empty:
        raise ValueError(f"Duplicate panel keys found: {dupes[['country_iso3','replenishment_round']].head()}")

    donor_obs = panel["donate_dummy"].sum()
    logger.info(
        "Panel complete: %d rows, %d donor observations (%.1f%%)",
        len(panel), donor_obs, 100 * donor_obs / len(panel)
    )
    return panel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build heckman_panel.csv from real IDA data")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch WDI historical data even if cache exists")
    args = parser.parse_args()

    panel = build_panel(refresh=args.refresh)
    panel.to_csv(PANEL_OUT, index=False)
    logger.info("Panel written to %s (%d rows)", PANEL_OUT, len(panel))

    # Summary
    print("\n--- Panel summary ---")
    print(f"Countries:  {panel['country_iso3'].nunique()}")
    print(f"Rounds:     {panel['replenishment_round'].nunique()}")
    print(f"Total obs:  {len(panel)}")
    print(f"Donors:     {panel['donate_dummy'].sum()} obs ({panel['donate_dummy'].mean()*100:.1f}%)")
    print(f"WDI GDP null rate:    {panel['log_gdp_level'].isna().mean()*100:.1f}%")
    print(f"WDI trade null rate:  {panel['trade_openness'].isna().mean()*100:.1f}%")
    print()
    print("Top donors by total contributions (USD billions):")
    top = (
        panel[panel["donate_dummy"] == 1]
        .groupby("country_iso3")["donation_usd"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    for iso3, total in top.items():
        print(f"  {iso3}: ${total/1e9:.2f}B")


if __name__ == "__main__":
    main()
