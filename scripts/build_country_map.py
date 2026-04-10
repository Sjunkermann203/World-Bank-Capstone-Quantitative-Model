"""
Build (or rebuild) data/country_map.csv from the World Bank country API.

Fetches all sovereign economies recognised by the World Bank, maps income
levels to the pipeline's tier codes (HIC/UMC/LMC/LIC), and preserves
is_current_donor flags from any existing country_map.csv.

Also extends data/raw/ifc_presence.csv with new countries defaulting to 0.

Usage:
    uv run python scripts/build_country_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import pandas as pd

ROOT = Path(__file__).parent.parent
COUNTRY_MAP_PATH = ROOT / "data" / "country_map.csv"
IFC_PRESENCE_PATH = ROOT / "data" / "raw" / "ifc_presence.csv"

WB_INCOME_MAP = {
    "High income":          "HIC",
    "Upper middle income":  "UMC",
    "Lower middle income":  "LMC",
    "Low income":           "LIC",
}

# Manual classifications for countries the WB currently marks "Not classified"
MANUAL_INCOME_OVERRIDES: dict[str, str] = {
    "ETH": "LIC",  # Ethiopia — LIC per most recent available WB data
    "VEN": "UMC",  # Venezuela — UMC historically; WB suspended classification
}

HEADER = """\
# Canonical country mapping table for the Donor Readiness Index
# Key: iso3 (ISO 3166-1 alpha-3)
# income_group: HIC (High Income), UMC (Upper-Middle Income), LMC (Lower-Middle Income), LIC (Low Income)
# wdi_name, imf_name, ida_name: source-specific name variants (blank = same as country_name)
"""


def fetch_wb_countries() -> list[dict]:
    """Fetch all sovereign economies from the World Bank API."""
    url = "https://api.worldbank.org/v2/countries"
    params = {"format": "json", "per_page": 400}
    print("Fetching World Bank country list...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    all_entries = payload[1]
    print(f"  Total entries returned: {len(all_entries)}")

    # Filter to sovereign countries — exclude regional aggregates (region.id == "NA")
    countries = [c for c in all_entries if c.get("region", {}).get("id") != "NA"]
    print(f"  Sovereign economies after filtering aggregates: {len(countries)}")
    return countries


def load_existing_map() -> pd.DataFrame | None:
    """Load the existing country_map.csv, or return None if it doesn't exist."""
    if not COUNTRY_MAP_PATH.exists():
        return None
    return pd.read_csv(COUNTRY_MAP_PATH, comment="#")


def load_existing_donors() -> set[str]:
    """Return ISO3 codes of current donors from the existing country_map.csv."""
    df = load_existing_map()
    if df is None:
        return set()
    return set(df.loc[df["is_current_donor"] == 1, "iso3"].str.upper())


def load_existing_overrides() -> dict[str, dict]:
    """
    Return per-country overrides from the existing country_map.csv.
    Preserves hand-curated wdi_name / imf_name / ida_name and income_group
    (used as fallback for countries the WB marks 'Not classified').
    """
    df = load_existing_map()
    if df is None:
        return {}
    overrides = {}
    for _, row in df.iterrows():
        iso3 = str(row["iso3"]).strip().upper()
        overrides[iso3] = {
            "wdi_name":     str(row.get("wdi_name", "")     or "").strip(),
            "imf_name":     str(row.get("imf_name", "")     or "").strip(),
            "ida_name":     str(row.get("ida_name", "")     or "").strip(),
            "income_group": str(row.get("income_group", "") or "").strip(),
        }
    return overrides


def build_country_map(wb_countries: list[dict], donors: set[str], overrides: dict) -> pd.DataFrame:
    """Construct the country_map DataFrame from WB data."""
    rows = []
    unclassified_used = []
    unclassified_skipped = []

    for c in wb_countries:
        iso3 = c.get("id", "").strip().upper()
        name = c.get("name", "").strip()
        income_label = c.get("incomeLevel", {}).get("value", "")
        income_group = WB_INCOME_MAP.get(income_label)

        if not iso3:
            continue

        if not income_group:
            # 1. Try manual override table
            if iso3 in MANUAL_INCOME_OVERRIDES:
                income_group = MANUAL_INCOME_OVERRIDES[iso3]
                unclassified_used.append((iso3, name, income_label, income_group))
            # 2. Fall back to existing classification from country_map
            elif overrides.get(iso3, {}).get("income_group", "") in WB_INCOME_MAP.values():
                income_group = overrides[iso3]["income_group"]
                unclassified_used.append((iso3, name, income_label, income_group))
            else:
                unclassified_skipped.append((iso3, name, income_label))
                continue

        o = overrides.get(iso3, {})
        rows.append({
            "iso3":             iso3,
            "country_name":     name,
            "income_group":     income_group,
            "wdi_name":         o.get("wdi_name", ""),
            "imf_name":         o.get("imf_name", ""),
            "ida_name":         o.get("ida_name", ""),
            "is_current_donor": 1 if iso3 in donors else 0,
        })

    if unclassified_used:
        print(f"  Used existing classification for {len(unclassified_used)} WB-unclassified countries:")
        for iso3, name, label, grp in unclassified_used:
            print(f"    {iso3:4}  {name[:40]:<40}  → {grp} (WB says '{label}')")

    if unclassified_skipped:
        print(f"  Skipped {len(unclassified_skipped)} with no classification (WB + existing):")
        for iso3, name, label in unclassified_skipped:
            print(f"    {iso3:4}  {name[:40]:<40}  income='{label}'")

    df = pd.DataFrame(rows)
    df = df.sort_values(["income_group", "iso3"]).reset_index(drop=True)
    return df


def extend_ifc_presence(new_iso3_set: set[str]) -> None:
    """Add new countries to ifc_presence.csv with active_portfolio=0."""
    if not IFC_PRESENCE_PATH.exists():
        print("  ifc_presence.csv not found — skipping IFC extension")
        return

    existing = pd.read_csv(IFC_PRESENCE_PATH, comment="#")
    existing["iso3"] = existing["iso3"].str.strip().str.upper()
    existing_iso3 = set(existing["iso3"])

    new_countries = sorted(new_iso3_set - existing_iso3)
    if not new_countries:
        print("  ifc_presence.csv already covers all countries — no changes needed")
        return

    print(f"  Adding {len(new_countries)} new countries to ifc_presence.csv (active_portfolio=0)")
    new_rows = pd.DataFrame({
        "iso3":             new_countries,
        "country_name":     "",
        "active_portfolio": 0,
    })
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.sort_values("iso3").reset_index(drop=True)

    # Preserve header comments
    header_lines = []
    with open(IFC_PRESENCE_PATH) as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.rstrip())
            else:
                break

    with open(IFC_PRESENCE_PATH, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        combined.to_csv(f, index=False)

    print(f"  ifc_presence.csv updated: {len(combined)} total countries")


def main() -> None:
    wb_countries = fetch_wb_countries()
    donors = load_existing_donors()
    overrides = load_existing_overrides()

    print(f"\nPreserving {len(donors)} current donors: {sorted(donors)}")

    df = build_country_map(wb_countries, donors, overrides)

    print(f"\nCountry map summary ({len(df)} countries):")
    print(df["income_group"].value_counts().sort_index().to_string())
    print(f"Current donors preserved: {df['is_current_donor'].sum()}")

    # Write country_map.csv with header comment
    with open(COUNTRY_MAP_PATH, "w") as f:
        f.write(HEADER)
        df.to_csv(f, index=False)
    print(f"\nWritten: {COUNTRY_MAP_PATH}")

    # Extend ifc_presence.csv
    print("\nExtending ifc_presence.csv...")
    extend_ifc_presence(set(df["iso3"]))

    print("\nDone. Next steps:")
    print("  1. Delete data/cache/wdi.csv to force a full WDI refresh")
    print("  2. Run: uv run python main.py --dry-run --refresh")


if __name__ == "__main__":
    main()
