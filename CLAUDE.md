# QuantMod — Donor Readiness Index

## Setup

```bash
uv sync          # install dependencies from uv.lock
```

## Running the pipeline

```bash
# Full pipeline
uv run python main.py

# Re-fetch World Bank data (bypass local cache)
uv run python main.py --refresh

# Show top N countries in Chart 1 (default: 30)
uv run python main.py --top-n 20

# Ingest only — print master dataset summary, no outputs written
uv run python main.py --dry-run
```

## Managing dependencies

```bash
uv add <package>        # add a dependency
uv remove <package>     # remove a dependency
uv sync                 # sync environment to match pyproject.toml / uv.lock
uv lock                 # regenerate uv.lock without installing
```

## Project structure

```
main.py          # pipeline runner (entry point)
src/
  ingest.py      # Stage 1: fetch/load source data → data/processed/master.csv
  capacity.py    # Stage 2: score capacity targets, gaps, giving rates
  alignment.py   # Stage 3: score strategic alignment
  report.py      # Stage 4: merge, rank, produce charts
data/            # input data and processed intermediates
outputs/         # pipeline outputs (CSV + charts)
```

## Output files

```
outputs/dri_output.csv
outputs/charts/chart1_gap_ranking.png
outputs/charts/chart2_giving_rate.png
outputs/charts/chart3_capacity_vs_giving_rate.png
outputs/charts/chart4_alignment_vs_gap.png
```
