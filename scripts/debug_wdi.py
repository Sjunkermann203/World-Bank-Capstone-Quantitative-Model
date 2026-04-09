"""
Standalone WDI API debug script.

Runs each part of the WDI fetch logic in isolation so you can see exactly
where it breaks without having to run the full pipeline.

Usage:
    uv run python scripts/debug_wdi.py
    uv run python scripts/debug_wdi.py --full    # also test all 83 countries
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import requests

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

SEP = "=" * 70
MINI_COUNTRIES = ["AUS", "CAN", "DEU", "FRA", "NLD"]
MINI_INDICATORS = {"NY.GDP.MKTP.CD": "gdp_usd"}
ALL_INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_usd",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "GC.BAL.CASH.GD.ZS": "fiscal_balance_pct_gdp",
}
WB_BASE = "https://api.worldbank.org/v2"

results: dict[str, str] = {}


def sep(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ---------------------------------------------------------------------------
# Section 1: Direct REST API check (no wbdata)
# ---------------------------------------------------------------------------

def test_direct_rest() -> bool:
    sep("SECTION 1: Direct REST API (no wbdata)")
    url = f"{WB_BASE}/countries/AUS;CAN/indicators/NY.GDP.MKTP.CD"
    params = {"format": "json", "per_page": 10, "date": "2020:2024"}
    print(f"GET {url}")
    print(f"params: {params} (timeout=120s)")
    try:
        r = requests.get(url, params=params, timeout=120)
        print(f"HTTP status: {r.status_code}")
        data = r.json()
        print(f"Response[0] (metadata): {data[0]}")
        print(f"Response[1] (first 2 records):")
        for rec in (data[1] or [])[:2]:
            print(f"  {rec}")
        results["direct_rest"] = "PASS"
        return True
    except Exception:
        print("FAILED:")
        traceback.print_exc()
        results["direct_rest"] = "FAIL"
        return False


# ---------------------------------------------------------------------------
# Section 2: Minimal wbdata call
# ---------------------------------------------------------------------------

def test_minimal_wbdata() -> bool:
    sep("SECTION 2: Minimal wbdata call (AUS + CAN, 1 indicator)")
    try:
        import wbdata
        print(f"wbdata version: {getattr(wbdata, '__version__', 'unknown')}")
        print(f"Calling wbdata.get_dataframe({MINI_INDICATORS}, country=['AUS','CAN'], parse_dates=False) ...")
        raw = wbdata.get_dataframe(MINI_INDICATORS, country=["AUS", "CAN"], parse_dates=False)
        print(f"\ntype(raw): {type(raw)}")
        print(f"raw.shape: {raw.shape}")
        print(f"type(raw.index): {type(raw.index)}")
        if hasattr(raw.index, "names"):
            print(f"raw.index.names: {raw.index.names}")
        print(f"raw.columns: {list(raw.columns)}")
        print(f"raw.dtypes:\n{raw.dtypes}")
        print(f"\nraw.head():\n{raw.head()}")

        print("\n--- After reset_index() ---")
        flat = raw.reset_index()
        print(f"flat.columns: {list(flat.columns)}")
        print(f"flat.head():\n{flat.head()}")
        results["minimal_wbdata"] = "PASS"
        return True
    except Exception:
        print("FAILED:")
        traceback.print_exc()
        results["minimal_wbdata"] = "FAIL"
        return False


# ---------------------------------------------------------------------------
# Section 3: Column detection stepthrough
# ---------------------------------------------------------------------------

def test_column_detection() -> bool:
    sep("SECTION 3: Column detection logic (5 countries, 1 indicator)")
    try:
        import wbdata
        raw = wbdata.get_dataframe(MINI_INDICATORS, country=MINI_COUNTRIES, parse_dates=False)
        flat = raw.reset_index()

        indicator_cols = list(MINI_INDICATORS.values())
        non_indicator = [c for c in flat.columns if c not in indicator_cols]
        date_col = next((c for c in non_indicator if c.lower() in {"date", "year", "time", "period"}), None)
        country_col = next(
            (c for c in non_indicator if c.lower() in {"country", "economy", "iso3", "countryiso3code", "id"}),
            None,
        )
        if country_col is None and non_indicator:
            country_col = next((c for c in non_indicator if c != date_col), None) or non_indicator[0]

        print(f"All columns after reset_index: {list(flat.columns)}")
        print(f"indicator_cols: {indicator_cols}")
        print(f"non_indicator: {non_indicator}")
        print(f"date_col detected: {date_col!r}")
        print(f"country_col detected: {country_col!r}")

        if country_col:
            print(f"\nSample country values: {flat[country_col].unique().tolist()[:5]}")
        results["column_detection"] = "PASS"
        return True
    except Exception:
        print("FAILED:")
        traceback.print_exc()
        results["column_detection"] = "FAIL"
        return False


# ---------------------------------------------------------------------------
# Section 4: Full fetch_wdi stepthrough (3 countries, all 3 indicators)
# ---------------------------------------------------------------------------

def test_fetch_wdi_stepthrough() -> bool:
    sep("SECTION 4: fetch_wdi stepthrough (DEU, FRA, NLD — 3 indicators)")
    countries = ["DEU", "FRA", "NLD"]
    try:
        import wbdata
        from src.ingest import load_country_map

        # Step 1: country map
        print("Step 1: load_country_map() ...")
        country_map = load_country_map()
        wdi_name_to_iso3: dict[str, str] = {}
        for _, r in country_map.iterrows():
            iso3 = r["iso3"]
            wdi_name = str(r.get("wdi_name", "") or "").strip()
            cname = str(r.get("country_name", "") or "").strip()
            if wdi_name:
                wdi_name_to_iso3[wdi_name.lower()] = iso3
            if cname:
                wdi_name_to_iso3[cname.lower()] = iso3
            wdi_name_to_iso3[iso3.lower()] = iso3
        print(f"  wdi_name_to_iso3 sample (first 5): { {k: v for k, v in list(wdi_name_to_iso3.items())[:5]} }")

        # Step 2: API call
        print("\nStep 2: wbdata.get_dataframe() ...")
        raw = wbdata.get_dataframe(ALL_INDICATORS, country=countries, parse_dates=False)
        print(f"  raw.shape: {raw.shape}, raw.index.names: {getattr(raw.index, 'names', 'N/A')}")
        print(f"  raw.head(3):\n{raw.head(3)}")

        # Step 3: reset_index
        print("\nStep 3: reset_index() ...")
        raw = raw.reset_index()
        print(f"  columns: {list(raw.columns)}")

        # Step 4: column detection
        print("\nStep 4: column detection ...")
        indicator_cols = list(ALL_INDICATORS.values())
        non_indicator = [c for c in raw.columns if c not in indicator_cols]
        date_col = next((c for c in non_indicator if c.lower() in {"date", "year", "time", "period"}), None)
        country_col = next(
            (c for c in non_indicator if c.lower() in {"country", "economy", "iso3", "countryiso3code", "id"}),
            None,
        )
        if country_col is None and non_indicator:
            country_col = next((c for c in non_indicator if c != date_col), None) or non_indicator[0]
        print(f"  date_col={date_col!r}, country_col={country_col!r}")

        # Step 5: sort
        if date_col:
            raw = raw.sort_values(date_col, ascending=False)
            print(f"\nStep 5: sorted by '{date_col}' desc. Head:\n{raw.head(3)}")

        # Step 6: groupby loop
        print("\nStep 6: groupby loop ...")
        rows = []
        for country_val, grp in raw.groupby(country_col):
            row: dict = {}
            for col in indicator_cols:
                if col in grp.columns:
                    valid = grp[col].dropna()
                    row[col] = float(valid.iloc[0]) if not valid.empty else None
                else:
                    row[col] = None
            key = str(country_val).strip().lower()
            matched = wdi_name_to_iso3.get(key, str(country_val))
            row["iso3"] = matched
            print(f"  country_val={country_val!r} -> iso3={matched!r} | {row}")
            rows.append(row)

        # Step 7: final DataFrame
        import pandas as pd
        result = pd.DataFrame(rows)
        print(f"\nStep 7: final result:\n{result}")
        results["fetch_stepthrough"] = "PASS"
        return True
    except Exception:
        print("FAILED:")
        traceback.print_exc()
        results["fetch_stepthrough"] = "FAIL"
        return False


# ---------------------------------------------------------------------------
# Section 5: Full 83-country batch (--full only)
# ---------------------------------------------------------------------------

def test_full_batch() -> bool:
    sep("SECTION 5: Full batch (all countries from country_map.csv)")
    try:
        import wbdata
        from src.ingest import load_country_map
        iso3_list = load_country_map()["iso3"].dropna().tolist()
        print(f"Country count: {len(iso3_list)}")
        print("Calling wbdata.get_dataframe() for all countries + 3 indicators ...")
        t0 = time.time()
        raw = wbdata.get_dataframe(ALL_INDICATORS, country=iso3_list, parse_dates=False)
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.1f}s")
        print(f"raw.shape: {raw.shape}")
        print(f"Non-null counts:\n{raw.notna().sum()}")
        results["full_batch"] = "PASS"
        return True
    except Exception:
        print("FAILED:")
        traceback.print_exc()
        results["full_batch"] = "FAIL"
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Debug WDI API integration")
    parser.add_argument("--full", action="store_true", help="Also run full 83-country batch test")
    args = parser.parse_args()

    test_direct_rest()
    test_minimal_wbdata()
    test_column_detection()
    test_fetch_wdi_stepthrough()
    if args.full:
        test_full_batch()

    sep("SUMMARY")
    for section, status in results.items():
        marker = "✓" if status == "PASS" else "✗"
        print(f"  {marker} {section}: {status}")
    print()


if __name__ == "__main__":
    main()
