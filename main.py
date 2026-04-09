#!/usr/bin/env python3
"""
Donor Readiness Index — Pipeline Runner

Runs the full DRI pipeline in order:
  1. ingest   — fetch/load source data → data/processed/master.csv
  2. capacity — score capacity targets, gaps, giving rates → data/processed/capacity_scores.csv
  3. align    — score strategic alignment → data/processed/alignment_scores.csv
  4. report   — merge, rank, and produce charts → outputs/

Usage
-----
  python main.py                    # full pipeline
  python main.py --refresh          # re-fetch WDI (bypass cache)
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
from capacity import score_capacity
from alignment import score_alignment
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

    # ── Stage 1: Ingest ──────────────────────────────────────────────────────
    logger.info("Stage 1/4 — Data ingestion (refresh=%s)", args.refresh)
    master = build_master(refresh=args.refresh)
    logger.info("Ingestion complete: %d countries in master dataset", len(master))

    if args.dry_run:
        print_dry_run_summary(master)
        logger.info("Dry-run mode: stopping after ingestion.")
        return

    # ── Stage 2: Capacity scoring ─────────────────────────────────────────────
    logger.info("Stage 2/4 — Capacity scoring")
    capacity = score_capacity(master)
    logger.info(
        "Capacity scoring complete: %d countries scored, %d with valid gap",
        len(capacity),
        capacity["gap_usd"].notna().sum(),
    )

    # ── Stage 3: Alignment scoring ────────────────────────────────────────────
    logger.info("Stage 3/4 — Strategic alignment scoring")
    alignment = score_alignment(master)
    logger.info(
        "Alignment scoring complete: %d countries scored, %d with composite score",
        len(alignment),
        alignment["alignment_score"].notna().sum(),
    )

    # ── Stage 4: Report ───────────────────────────────────────────────────────
    logger.info("Stage 4/4 — Generating report and charts (top_n=%d)", args.top_n)
    dri = generate_report(capacity, alignment, top_n=args.top_n)
    logger.info("Pipeline complete. %d countries in final DRI output.", len(dri))

    print("\n" + "=" * 60)
    print("  Donor Readiness Index — Pipeline Complete")
    print("=" * 60)
    print(f"  Countries scored:  {len(dri)}")
    print(f"  With valid gap:    {dri['gap_usd'].notna().sum()}")
    print(f"  With alignment:    {dri['alignment_score'].notna().sum()}")
    print()
    print("  Output files:")
    print("    outputs/dri_output.csv")
    print("    outputs/charts/chart1_gap_ranking.png")
    print("    outputs/charts/chart2_giving_rate.png")
    print("    outputs/charts/chart3_capacity_vs_giving_rate.png")
    print("    outputs/charts/chart4_alignment_vs_gap.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
