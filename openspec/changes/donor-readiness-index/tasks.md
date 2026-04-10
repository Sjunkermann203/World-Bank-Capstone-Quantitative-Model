## 1. Project Setup

- [x] 1.1 Create project directory structure: `src/`, `data/raw/`, `data/cache/`, `data/processed/`, `outputs/charts/`
- [x] 1.2 Create `requirements.txt` with dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `requests`, `wbdata` (or `world_bank_data`), `openpyxl`
- [x] 1.3 Create `data/country_map.csv` with ISO 3166-1 alpha-3 codes mapping WDI, IMF, and IDA country name variants to canonical ISO3 and income group
- [x] 1.4 Create `data/raw/ida_contributions.csv` with IDA20 and IDA21 actual contribution amounts per country (hand-curated from IDA replenishment records)
- [x] 1.5 Create `data/raw/ifc_presence.csv` with binary IFC active portfolio flag per country
- [x] 1.6 Create `data/raw/imf_weo.csv` with government gross debt (% GDP) per country from IMF WEO download

## 2. Data Ingestion Pipeline

- [x] 2.1 Implement `src/ingest.py` — WDI fetch function: retrieve GDP, GDP per capita, and fiscal balance for all target countries; write to `data/cache/wdi.csv`; skip API call if cache exists and `--refresh` not set. Uses direct REST API with batched requests (3 countries/batch, 120s timeout) — wbdata replaced due to memory issues.
- [x] 2.2 Implement IMF WEO loader in `src/ingest.py`: read `data/raw/imf_weo.csv`, validate expected columns, raise `SchemaValidationError` with column names if schema mismatch
- [x] 2.3 Implement IDA contributions loader in `src/ingest.py`: read `data/raw/ida_contributions.csv`, warn and skip rows with unresolved country codes
- [x] 2.4 Implement country identity resolution in `src/ingest.py`: load `data/country_map.csv`, join all sources on ISO3 code, log warnings for unresolved country names
- [x] 2.5 Implement master dataset assembly: join WDI, IMF, and IDA data on ISO3; write `data/processed/master.csv` with all required columns; exclude countries missing all economic data

## 3. Capacity Scoring

- [x] 3.1 Implement `src/capacity.py` — tier median calculation: compute median IDA/GDP ratio per income group from current donors in `master.csv`; fall back to global median if fewer than 3 donors in a tier; write tier medians to `data/processed/run_metadata.json`
- [x] 3.2 Implement target contribution calculation: `target_usd = gdp_usd × tier_median_ida_gdp_ratio`; set null for countries with missing GDP
- [x] 3.3 Implement fiscal modifier: linear scale from fiscal balance (% GDP) to modifier in [−0.20, +0.20]; default to 0 if fiscal balance is null; compute `adjusted_target_usd`
- [x] 3.4 Implement gap and giving rate calculation: `gap_usd = adjusted_target_usd − actual_contribution_usd`; `giving_rate = actual_contribution_usd / adjusted_target_usd`; set `actual_contribution_usd = 0` for countries with no IDA record
- [x] 3.5 Write `data/processed/capacity_scores.csv` with all required columns

## 4. Strategic Alignment Scoring

- [x] 4.1 Implement `src/alignment.py` — UNGA voting alignment scorer: load UNGA voting dataset, compute alignment score vs. reference group, normalize to [0, 100]; assign null if fewer than 10 resolutions
- [x] 4.2 Implement WB vote share scorer: load WB voting power data, normalize each country's share to [0, 100] relative to dataset maximum; assign 0 for missing countries
- [x] 4.3 Implement IFC presence scorer: read `data/raw/ifc_presence.csv`, return 100 for active portfolio, 0 otherwise
- [x] 4.4 Implement composite alignment score: equally-weighted average of available components; exclude null components from average; record which components were excluded
- [x] 4.5 Write `data/processed/alignment_scores.csv` with all required columns

## 5. Reporting and Charts

- [x] 5.1 Implement `src/report.py` — merge capacity and alignment scores on ISO3; sort by `gap_usd` descending; write `outputs/dri_output.csv`
- [x] 5.2 Implement Chart 1: horizontal bar chart of top 30 countries by `gap_usd`, color-coded by income group; save to `outputs/charts/chart1_gap_ranking.png` at 150 DPI
- [x] 5.3 Implement Chart 2: horizontal bar chart of `giving_rate` for all countries, sorted ascending; vertical reference line at 1.0; countries above/below visually distinguished; save to `outputs/charts/chart2_giving_rate.png` at 150 DPI
- [x] 5.4 Implement Chart 3: scatter plot of adjusted capacity target (x) vs. giving rate (y); ISO3 labels; reference lines at `giving_rate = 1.0` and median capacity; save to `outputs/charts/chart3_capacity_vs_giving_rate.png` at 150 DPI
- [x] 5.5 Implement Chart 4: scatter plot of `alignment_score` (x) vs. `gap_usd` (y); ISO3 labels; point size proportional to GDP; exclude countries missing alignment score; save to `outputs/charts/chart4_alignment_vs_gap.png` at 150 DPI
- [x] 5.6 Ensure `outputs/` and `outputs/charts/` directories are created automatically before any file writes

## 6. Pipeline Runner

- [x] 6.1 Implement `main.py` pipeline runner: parse CLI args (`--refresh` flag, optional `--top-n` for chart count); call stages in order: ingest → capacity → alignment → report; print stage completion messages
- [x] 6.2 Add a `--dry-run` flag that runs ingestion only and prints the master dataset summary (row count, null counts per column) without writing scored output
- [x] 6.3 Add `--no-fiscal-modifier` flag to disable fiscal balance adjustment for comparison runs; passes `fiscal_modifier=False` to `score_capacity()`

## 7. Validation and Testing

- [x] 7.1 Create `data/raw/sample_countries.txt` listing 10 test countries spanning both income tiers and a mix of current donors and non-donors
- [x] 7.2 Run the full pipeline against sample countries; verify `dri_output.csv` has expected columns and no unexpected nulls in scored fields
- [x] 7.3 Manually verify gap and giving rate for 2–3 known countries against hand-calculated benchmarks (USA: $3.1B with modifier / $4.9B without; NOR: surplus correctly reduces gap)
- [x] 7.4 Verify all four chart files are created and open correctly; check that reference lines and labels appear as expected
- [x] 7.5 Confirm `run_metadata.json` logs tier medians and the donor set used

## 8. Bug Fixes and Data Quality

- [x] 8.1 Replace wbdata library with direct World Bank REST API calls — wbdata consumed 1.8GB RAM without returning; new implementation uses 3-country batches with 120s timeouts and retry logic
- [x] 8.2 Fix `fiscal_balance_pct_gdp` — `GC.BAL.CASH.GD.ZS` returns all-null; derive from `GC.REV.XGRT.GD.ZS` (revenue % GDP) minus `GC.XPN.TOTL.GD.ZS` (expenditure % GDP); coverage improved from 0% to 86%
- [x] 8.3 Create `scripts/debug_wdi.py` for isolated WDI API testing without running the full pipeline

## 9. Remaining Data Gaps

- [ ] 9.1 Source UNGA votes data — download from Harvard Dataverse, place at `data/raw/unga_votes.csv`; UNGA alignment scores currently null for all countries
- [ ] 9.2 Source World Bank vote shares data — download from World Bank, place at `data/raw/wb_vote_shares.csv`; WB vote share scores currently 0 for all countries
