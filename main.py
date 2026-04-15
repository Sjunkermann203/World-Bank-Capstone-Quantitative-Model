#!/usr/bin/env python3
"""
Donor Readiness Index — Pipeline Runner

Runs the full DRI pipeline in order:
  1. ingest   — fetch/load source data → data/processed/master.csv
  2. capacity — score capacity targets, gaps, giving rates → data/processed/capacity_scores.csv
  3. report   — rank and produce charts → outputs/

Usage
-----
  python main.py                    # full pipeline
  python main.py --refresh          # re-fetch WDI (bypass cache) and run full pipeline
  python main.py --refresh-wdi      # re-fetch WDI only, then stop
  python main.py --top-n 20         # show top 20 countries in Chart 1
  python main.py --dry-run          # ingest only, print summary
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import build_master
from report import generate_report


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Donor Readiness Index pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch WDI data from the World Bank API, bypassing the local cache.",
    )
    parser.add_argument(
        "--refresh-wdi",
        action="store_true",
        help="Re-fetch WDI data only, update the cache, then stop.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        metavar="N",
        help="Number of countries to show in Chart 1 (gap ranking). Default: 30.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run ingestion only and print a master dataset summary. "
            "Does not run scoring or generate output files."
        ),
    )
    parser.add_argument(
        "--no-fiscal-modifier",
        action="store_true",
        help="Disable the fiscal balance modifier (rule-based scorer only; ignored by Heckman).",
    )
    parser.add_argument(
        "--skip-heckman",
        action="store_true",
        help=(
            "Fall back to the rule-based capacity scorer instead of the Heckman model. "
            "Useful when data/raw/heckman_panel.csv is not present."
        ),
    )
    return parser.parse_args()


def print_dry_run_summary(master):
    """Print row count and null counts per column to stdout."""
    print("\n" + "=" * 60)
    print(f"  Master dataset summary ({len(master)} countries)")
    print("=" * 60)
    print(f"  {'Column':<35} {'Nulls':>6}  {'%':>6}")
    print("  " + "-" * 52)
    for col in master.columns:
        n_null = master[col].isna().sum()
        pct = n_null / len(master) * 100 if len(master) > 0 else 0
        print(f"  {col:<35} {n_null:>6}  {pct:>5.1f}%")
    print("=" * 60 + "\n")


def main():
    print('main')
    configure_logging()
    args = parse_args()
    logger = logging.getLogger("main")

    # ── WDI-only refresh ─────────────────────────────────────────────────────
    if args.refresh_wdi:
        from ingest import fetch_wdi, load_country_map, resolve_countries
        country_map = load_country_map()
        iso3_list = resolve_countries(country_map)["iso3"].tolist()
        logger.info("Refreshing WDI cache for %d countries...", len(iso3_list))
        fetch_wdi(iso3_list, refresh=True)
        logger.info("WDI cache updated.")
        return

    # ── Stage 1: Ingest ──────────────────────────────────────────────────────
    logger.info("Stage 1/4 — Data ingestion (refresh=%s)", args.refresh)
    master = build_master(refresh=args.refresh)
    logger.info("Ingestion complete: %d countries in master dataset", len(master))

    if args.dry_run:
        print_dry_run_summary(master)
        logger.info("Dry-run mode: stopping after ingestion.")
        return

    # ── Stage 2: Capacity scoring ─────────────────────────────────────────────
    _panel_path = Path(__file__).parent / "data" / "raw" / "heckman_panel.csv"
    _use_heckman = not args.skip_heckman and _panel_path.exists()

    if _use_heckman:
        logger.info("Stage 2/3 — Heckman capacity scoring")
        from heckman import score_capacity
    else:
        if not args.skip_heckman:
            logger.warning(
                "heckman_panel.csv not found at %s — falling back to rule-based capacity scorer. "
                "Pass --skip-heckman to suppress this warning.",
                _panel_path,
            )
        else:
            logger.info("Stage 2/3 — Rule-based capacity scoring (--skip-heckman)")
        from capacity import score_capacity

    capacity = score_capacity(master, fiscal_modifier=not args.no_fiscal_modifier)
    logger.info(
        "Capacity scoring complete: %d countries scored, %d with valid gap",
        len(capacity),
        capacity["gap_usd"].notna().sum(),
    )

    # ── Stage 3: Report ───────────────────────────────────────────────────────
    logger.info("Stage 3/3 — Generating report and charts (top_n=%d)", args.top_n)
    dri = generate_report(capacity, top_n=args.top_n)
    logger.info("Pipeline complete. %d countries in final DRI output.", len(dri))

    print("\n" + "=" * 60)
    print("  Donor Readiness Index — Pipeline Complete")
    print("=" * 60)
    print(f"  Countries scored:  {len(dri)}")
    print(f"  With valid gap:    {dri['gap_usd'].notna().sum()}")
    print()
    print("  Output files:")
    print("    outputs/dri_output.csv")
    print("    outputs/charts/chart1_gap_ranking.png")
    print("    outputs/charts/chart2_giving_rate.png")
    print("    outputs/charts/chart3_capacity_vs_giving_rate.png")
    print("    outputs/charts/chart5_all_countries_gap.png")
    print("    outputs/charts/chart5_world_map.html")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
