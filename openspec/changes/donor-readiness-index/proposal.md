## Why

IDA replenishment campaigns lack a systematic, data-driven method to identify which sovereign nations have the fiscal and economic capacity to contribute but are currently giving below their benchmark. The Donor Readiness Index fills that gap, enabling evidence-based outreach targeting ~80 countries before and during IDA replenishment cycles.

## What Changes

- Introduce a new quantitative model (`donor-readiness-index`) that scores and ranks sovereign nations by their IDA contribution gap
- Implement a capacity-based target contribution calculation tiered by income group, using the median IDA/GDP ratio of existing donors as the benchmark
- Apply a fiscal modifier (±20% max) derived from fiscal balance as % of GDP
- Compute per-country giving rate (actual ÷ target) and contribution gap (target − actual)
- Score strategic alignment separately using UNGA voting patterns, World Bank vote share, and IFC presence
- Produce four standard charts: gap ranking, giving rate, capacity vs. giving rate scatter, alignment vs. gap scatter
- Ingest data from World Bank WDI, IMF WEO, and IDA replenishment records (IDA20 and IDA21)

## Capabilities

### New Capabilities

- `data-ingestion`: Fetch and normalize inputs from World Bank WDI API, IMF WEO, and IDA replenishment contribution files; produce a clean per-country dataset
- `capacity-scoring`: Calculate income-tier-adjusted target contributions, apply fiscal modifier, compute capacity score and gap
- `strategic-alignment-scoring`: Score each country on UNGA voting alignment, WB vote share, and IFC presence; combine into a composite alignment score
- `reporting-and-charts`: Rank countries by gap size and generate the four standard output charts and a per-country summary table

### Modified Capabilities

<!-- None — this is a net-new model with no existing specs to modify -->

## Impact

- **New code**: Python package (or scripts) for data ingestion, scoring, and visualization
- **Dependencies**: World Bank `wbdata` or direct WDI API, IMF WEO dataset (CSV/SDMX), `pandas`, `numpy`, `matplotlib`/`seaborn`
- **Data**: Requires IDA21 and IDA20 contribution records (static files or scrape from IDA website)
- **No breaking changes** to any existing system — this is a standalone analytical model
