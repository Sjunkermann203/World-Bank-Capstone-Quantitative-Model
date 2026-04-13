"""
Data ingestion pipeline for the Donor Readiness Index.

Fetches and normalizes inputs from:
  - World Bank WDI API (GDP, GDP per capita, fiscal balance)
  - IMF WEO file (government debt % GDP)
  - IDA contributions CSV (IDA20/IDA21 actuals)
  - Country mapping table (ISO3 canonical identities)

Produces: data/processed/master.csv
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import requests
import pandas as pd
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
DATA_PROCESSED = ROOT / "data" / "processed"

COUNTRY_MAP_PATH = ROOT / "data" / "country_map.csv"
IDA_CONTRIBUTIONS_PATH = DATA_RAW / "ida_contributions.csv"
IFC_PRESENCE_PATH = DATA_RAW / "ifc_presence.csv"
IMF_WEO_PATH = DATA_RAW / "imf_weo.csv"
HECKMAN_PANEL_PATH = DATA_RAW / "heckman_panel.csv"
WDI_CACHE_PATH = DATA_CACHE / "wdi.csv"
MASTER_PATH = DATA_PROCESSED / "master.csv"

# WDI indicator codes
WDI_INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_usd",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    # GC.BAL.CASH.GD.ZS (cash surplus/deficit) has no WB coverage; derive from revenue - expenditure
    "GC.REV.XGRT.GD.ZS": "govt_revenue_pct_gdp",
    "GC.XPN.TOTL.GD.ZS": "govt_expenditure_pct_gdp",
    "NE.TRD.GNFS.ZS": "trade_openness",
    "GOV_WGI_GE.EST": "gov_effectiveness",
}

# Expected IMF WEO columns
IMF_REQUIRED_COLUMNS = {"iso3", "govt_debt_pct_gdp"}


class SchemaValidationError(ValueError):
    """Raised when a source data file does not match the expected schema."""


# ---------------------------------------------------------------------------
# Country map
# ---------------------------------------------------------------------------

def load_country_map() -> pd.DataFrame:
    """Load the canonical country mapping table keyed on ISO3."""
    df = pd.read_csv(COUNTRY_MAP_PATH, comment="#")
    df["iso3"] = df["iso3"].str.strip().str.upper()
    return df


# ---------------------------------------------------------------------------
# WDI fetch helpers
# ---------------------------------------------------------------------------

_WB_API_BASE = "https://api.worldbank.org/v2"
_WDI_BATCH_SIZE = 3    # countries per request — API is slow; small batches stay under timeout
_WDI_TIMEOUT = 120     # seconds per request — single country can take 10-15s
_WDI_RETRIES = 2       # attempts per batch before skipping
_WDI_DATE_RANGE = "2018:2024"  # fetch recent years; we take the most recent non-null value


def _fetch_wdi_batch(indicator_code: str, field_name: str, batch: list[str]) -> dict[str, float | None]:
    """Fetch one indicator for a batch of countries. Returns {iso3: value}."""
    codes = ";".join(batch)
    url = f"{_WB_API_BASE}/countries/{codes}/indicators/{indicator_code}"
    params = {"format": "json", "per_page": 500, "date": _WDI_DATE_RANGE}
    for attempt in range(1, _WDI_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=_WDI_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            records = payload[1] if (isinstance(payload, list) and len(payload) > 1) else []
            # Records are ordered newest-first; take the first non-null value per country
            result: dict[str, float | None] = {}
            for rec in (records or []):
                iso3 = rec.get("countryiso3code", "")
                if iso3 not in result and rec.get("value") is not None:
                    try:
                        result[iso3] = float(rec["value"])
                    except (TypeError, ValueError):
                        pass
            return result
        except Exception as exc:
            if attempt < _WDI_RETRIES:
                logger.warning(
                    "WDI fetch attempt %d/%d failed for %s batch %s: %s — retrying",
                    attempt, _WDI_RETRIES, indicator_code, batch, exc,
                )
            else:
                logger.warning(
                    "WDI fetch failed for %s batch %s after %d attempts: %s — these countries will be null",
                    indicator_code, batch, _WDI_RETRIES, exc,
                )
                return {}


def _fetch_wdi_via_requests(indicators: dict[str, str], iso3_list: list[str]) -> pd.DataFrame:
    """
    Fetch WDI indicators using the World Bank REST API directly.

    The WB API is slow (~10-15s per country). Uses small batches with long
    timeouts and retries. Returns a DataFrame with columns: iso3, <field_name>, ...
    """
    rows: dict[str, dict] = {iso3: {} for iso3 in iso3_list}
    batches = [iso3_list[i:i + _WDI_BATCH_SIZE] for i in range(0, len(iso3_list), _WDI_BATCH_SIZE)]
    n_batches = len(batches)
    for indicator_code, field_name in indicators.items():
        for i, batch in enumerate(batches, 1):
            logger.info(
                "WDI fetch: %s — batch %d/%d (%s)", indicator_code, i, n_batches, batch
            )
            values = _fetch_wdi_batch(indicator_code, field_name, batch)
            for iso3, value in values.items():
                if iso3 in rows:
                    rows[iso3][field_name] = value

    result = pd.DataFrame([{"iso3": k, **v} for k, v in rows.items()])
    for field_name in indicators.values():
        if field_name not in result.columns:
            result[field_name] = None
    return result


# ---------------------------------------------------------------------------
# WDI fetch
# ---------------------------------------------------------------------------

def fetch_wdi(iso3_list: list[str], refresh: bool = False) -> pd.DataFrame:
    """
    Retrieve WDI indicators for the given ISO3 list.

    Uses cache at data/cache/wdi.csv unless refresh=True.
    Returns a DataFrame indexed by iso3.
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)

    if WDI_CACHE_PATH.exists() and not refresh:
        logger.info("Loading WDI data from cache: %s", WDI_CACHE_PATH)
        return pd.read_csv(WDI_CACHE_PATH)

    logger.info("Fetching WDI indicators from World Bank API (batch request for %d countries)...", len(iso3_list))
    try:
        result = _fetch_wdi_via_requests(WDI_INDICATORS, iso3_list)
    except Exception:
        logger.error("Direct REST API fetch failed — returning all-null dataset", exc_info=True)
        result = pd.DataFrame({"iso3": iso3_list})
        for col in WDI_INDICATORS.values():
            result[col] = None
        result.to_csv(WDI_CACHE_PATH, index=False)
        return result

    result = result[["iso3"] + [c for c in WDI_INDICATORS.values() if c in result.columns]]

    # Warn on per-indicator nulls
    for col in WDI_INDICATORS.values():
        null_count = result[col].isna().sum() if col in result.columns else len(result)
        if null_count > 0:
            logger.warning("WDI indicator '%s' is null for %d countries", col, null_count)

    result.to_csv(WDI_CACHE_PATH, index=False)
    logger.info("WDI data cached to %s (%d rows)", WDI_CACHE_PATH, len(result))
    return result


# ---------------------------------------------------------------------------
# IMF WEO loader
# ---------------------------------------------------------------------------

def load_imf_weo() -> pd.DataFrame:
    """Load government gross debt (% GDP) from the IMF WEO file."""
    df = pd.read_csv(IMF_WEO_PATH, comment="#")
    df.columns = df.columns.str.strip().str.lower()

    missing = IMF_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"IMF WEO file is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    df["iso3"] = df["iso3"].str.strip().str.upper()
    return df[["iso3", "govt_debt_pct_gdp"]].copy()


# ---------------------------------------------------------------------------
# IDA contributions loader
# ---------------------------------------------------------------------------

def load_ida_contributions(valid_iso3: set[str]) -> pd.DataFrame:
    """
    Load IDA20 and IDA21 contribution records.

    Skips rows whose ISO3 code is not in valid_iso3 (with a warning).
    Returns a wide DataFrame with columns: iso3, ida20_contribution_usd, ida21_contribution_usd.
    """
    df = pd.read_csv(IDA_CONTRIBUTIONS_PATH, comment="#")
    df.columns = df.columns.str.strip().str.lower()
    df["iso3"] = df["iso3"].str.strip().str.upper()

    # Warn and drop rows with unresolved ISO3
    unknown = df[~df["iso3"].isin(valid_iso3)]
    for _, row in unknown.iterrows():
        logger.warning(
            "IDA contributions: ISO3 '%s' not in country universe — skipping", row["iso3"]
        )
    df = df[df["iso3"].isin(valid_iso3)].copy()

    # Convert to wide format: one row per country with IDA20 and IDA21 columns
    df["contribution_usd"] = pd.to_numeric(df["contribution_usd_millions"], errors="coerce") * 1e6

    ida20 = df[df["cycle"].str.upper() == "IDA20"][["iso3", "contribution_usd"]].rename(
        columns={"contribution_usd": "ida20_contribution_usd"}
    )
    ida21 = df[df["cycle"].str.upper() == "IDA21"][["iso3", "contribution_usd"]].rename(
        columns={"contribution_usd": "ida21_contribution_usd"}
    )

    wide = ida20.merge(ida21, on="iso3", how="outer")
    return wide


# ---------------------------------------------------------------------------
# Panel IDA21 actuals
# ---------------------------------------------------------------------------

def load_panel_ida21_actuals() -> pd.DataFrame:
    """
    Extract IDA21 actual contributions from heckman_panel.csv.

    The panel records final replenishment totals, which are more complete and
    accurate than the pledge figures in ida_contributions.csv. Returns a
    DataFrame with columns: iso3, ida21_contribution_usd.

    Returns an empty DataFrame if the panel file is not present.
    """
    if not HECKMAN_PANEL_PATH.exists():
        logger.warning(
            "heckman_panel.csv not found at %s — IDA21 actuals will rely solely on ida_contributions.csv",
            HECKMAN_PANEL_PATH,
        )
        return pd.DataFrame(columns=["iso3", "ida21_contribution_usd"])

    panel = pd.read_csv(HECKMAN_PANEL_PATH, comment="#")
    panel.columns = panel.columns.str.strip().str.lower()
    panel["country_iso3"] = panel["country_iso3"].str.strip().str.upper()

    ida21 = panel[
        (panel["replenishment_round"].str.upper() == "IDA21") &
        (panel["donate_dummy"] == 1) &
        panel["donation_usd"].notna()
    ][["country_iso3", "donation_usd"]].copy()

    ida21 = ida21.rename(columns={"country_iso3": "iso3", "donation_usd": "ida21_contribution_usd"})
    logger.info("Panel IDA21 actuals loaded: %d donor countries", len(ida21))
    return ida21


# ---------------------------------------------------------------------------
# Country identity resolution
# ---------------------------------------------------------------------------

def resolve_countries(country_map: pd.DataFrame) -> pd.DataFrame:
    """
    Return the canonical country list with iso3 as the join key.
    Logs a warning for any country_map entries with null iso3.
    """
    null_iso3 = country_map[country_map["iso3"].isna()]
    for _, row in null_iso3.iterrows():
        logger.warning("Country map: null iso3 for '%s' — skipping", row.get("country_name", "?"))

    return country_map[country_map["iso3"].notna()].copy()


# ---------------------------------------------------------------------------
# Master dataset assembly
# ---------------------------------------------------------------------------

def build_master(refresh: bool = False) -> pd.DataFrame:
    """
    Assemble the normalized per-country master dataset.

    Joins WDI, IMF WEO, and IDA contributions on ISO3.
    Writes data/processed/master.csv.
    Returns the master DataFrame.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    country_map = load_country_map()
    canonical = resolve_countries(country_map)
    valid_iso3 = set(canonical["iso3"].tolist())

    iso3_list = canonical["iso3"].tolist()

    # Load each source
    wdi = fetch_wdi(iso3_list, refresh=refresh)
    imf = load_imf_weo()
    ida = load_ida_contributions(valid_iso3)
    panel_ida21 = load_panel_ida21_actuals()

    # Merge panel IDA21 actuals into ida, overriding ida_contributions.csv where available.
    # Panel figures are final replenishment totals; CSV figures are earlier pledges.
    if not panel_ida21.empty:
        ida = ida.merge(panel_ida21, on="iso3", how="outer", suffixes=("_csv", "_panel"))
        # Panel takes priority; fall back to CSV; then to null
        panel_col = ida["ida21_contribution_usd_panel"] if "ida21_contribution_usd_panel" in ida.columns else pd.Series(dtype=float)
        csv_col   = ida["ida21_contribution_usd_csv"]   if "ida21_contribution_usd_csv"   in ida.columns else pd.Series(dtype=float)
        ida["ida21_contribution_usd"] = panel_col.combine_first(csv_col)
        n_panel   = panel_col.notna().sum()
        n_csv_only = (panel_col.isna() & csv_col.notna()).sum()
        logger.info(
            "IDA21 actuals: %d from panel (final totals), %d from CSV only (pledges)",
            n_panel, n_csv_only,
        )
        drop_cols = [c for c in ida.columns if c.endswith("_panel") or c.endswith("_csv")]
        ida = ida.drop(columns=drop_cols)

    # Join everything on ISO3
    master = canonical[["iso3", "country_name", "income_group", "is_current_donor"]].copy()
    master = master.merge(wdi, on="iso3", how="left")
    master = master.merge(imf, on="iso3", how="left")
    master = master.merge(ida, on="iso3", how="left")

    # Derive fiscal balance (% GDP) = revenue - expenditure
    if "govt_revenue_pct_gdp" in master.columns and "govt_expenditure_pct_gdp" in master.columns:
        master["fiscal_balance_pct_gdp"] = (
            master["govt_revenue_pct_gdp"] - master["govt_expenditure_pct_gdp"]
        )

    # Warn for any WDI indicator null
    for col in WDI_INDICATORS.values():
        if col in master.columns:
            n = master[col].isna().sum()
            if n > 0:
                null_countries = master.loc[master[col].isna(), "iso3"].tolist()
                logger.warning("Null '%s' for %d countries: %s", col, n, null_countries)

    # Exclude countries missing ALL economic data
    econ_cols = list(WDI_INDICATORS.values()) + ["govt_debt_pct_gdp"]
    econ_cols = [c for c in econ_cols if c in master.columns]
    all_null = master[econ_cols].isna().all(axis=1)
    excluded = master.loc[all_null, "iso3"].tolist()
    if excluded:
        logger.warning("Excluding %d countries with no economic data: %s", len(excluded), excluded)
    master = master[~all_null].copy()

    # Ensure required output columns exist (fill nulls for missing IDA records)
    for col in ["ida20_contribution_usd", "ida21_contribution_usd"]:
        if col not in master.columns:
            master[col] = None

    output_cols = [
        "iso3", "country_name", "income_group", "is_current_donor",
        "gdp_usd", "gdp_per_capita_usd", "fiscal_balance_pct_gdp",
        "govt_debt_pct_gdp", "trade_openness", "gov_effectiveness",
        "ida20_contribution_usd", "ida21_contribution_usd",
    ]
    output_cols = [c for c in output_cols if c in master.columns]
    master = master[output_cols]

    master.to_csv(MASTER_PATH, index=False)
    logger.info("Master dataset written to %s (%d countries)", MASTER_PATH, len(master))
    return master
