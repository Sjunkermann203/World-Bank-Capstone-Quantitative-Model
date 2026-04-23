## 1. Configuration and Infrastructure

- [x] 1.1 Create `config.yaml` at project root with `dri_weights: {alpha: 0.5, beta: 0.3, gamma: 0.2}`
- [x] 1.2 Add a `load_config()` helper (in `src/config.py` or top of `report.py`) that reads `config.yaml`, validates weights sum to 1.0, and returns defaults if file absent
- [x] 1.3 Add `peer_group` and `ppp_data_available` to the master dataframe column contract (update any schema validation or column-order definitions)

## 2. Ingest — PPP Data Fetch

- [x] 2.1 Add WDI fetch for `NY.GDP.MKTP.PP.KD` (PPP GDP constant 2017 intl $) in `src/ingest.py`, storing result as `gdp_ppp`
- [x] 2.2 Add WDI fetch for `NY.GNP.PCAP.PP.CD` (GNI per capita PPP) in `src/ingest.py`, storing result as `gni_per_capita_ppp`
- [x] 2.3 Add `ppp_data_available` boolean column based on non-null `gdp_ppp`
- [x] 2.4 Fetch World Bank income-level classification from `wbgapi.economy` metadata and store as `peer_group` column in master dataframe

## 3. Capacity — Peer-Group Benchmark

- [x] 3.1 Implement `weighted_median(values, weights)` utility function using numpy (handles edge cases: single value, zero-weight rows)
- [x] 3.2 Implement `compute_peer_benchmarks(df)` in `src/capacity.py` that groups current donors by `peer_group`, computes GDP-weighted median contribution rate per group
- [x] 3.3 Add fallback to global GDP-weighted median when a peer group has fewer than 3 donors (log warning)
- [x] 3.4 Apply peer-group benchmark (using `gdp_ppp` where available, nominal GDP as fallback) to compute `benchmark_ratio` per country; log warning when falling back to nominal

## 4. Capacity — Giving Rate Normalization and Signed Gap

- [x] 4.1 Rename existing `giving_rate` computation output to `giving_rate_raw` before any capping
- [x] 4.2 Add `giving_rate = min(giving_rate_raw, 1.0)` capped column
- [x] 4.3 Update segment assignment logic: add "Exceeded Target" as the first-evaluated segment rule (when `giving_rate_raw > 1.0`); ensure existing thresholds operate on capped `giving_rate`
- [x] 4.4 Compute `gap_usd_signed = benchmark_contribution_usd - actual_contribution_usd` (positive = shortfall, negative = over-contribution)
- [x] 4.5 Compute `gap_pct_ppp_gdp = gap_usd_signed / gdp_ppp * 100` (NaN where `ppp_data_available = False`)

## 5. Capacity — Confidence Intervals

- [x] 5.1 Extract `se_predicted` (HC3 robust SE on predicted log-contribution coefficient) from the Heckman second-stage OLS fitted model object
- [x] 5.2 Compute `gap_usd_lower = gap_usd_signed - 1.645 * se_predicted * gdp` and `gap_usd_upper = gap_usd_signed + 1.645 * se_predicted * gdp`
- [x] 5.3 Handle missing Heckman fit gracefully: set CI columns to NaN and log warning if model cannot be fit

## 6. Scoring — Composite DRI Score

- [x] 6.1 Add `compute_dri_score(df, alpha, beta, gamma)` function in `report.py` (or new `src/score.py`)
- [x] 6.2 Implement min-max normalization for `gap_usd_signed` (inverted: larger shortfall → score closer to 1), `alignment_score`, and `p_donate`; handle degenerate case (all identical → 0.5)
- [x] 6.3 Compute `DRI_score = alpha * norm_gap + beta * norm_alignment + gamma * norm_p_donate`
- [x] 6.4 Compute `gap_usd_expected = gap_usd_signed * p_donate`
- [x] 6.5 Replace `gap_usd`-based sort with `DRI_score` descending; update `rank` column accordingly
- [x] 6.6 Wire `load_config()` into the scoring call so weights are read from `config.yaml`

## 7. Charts — Bug Fix and Enhancements

- [x] 7.1 Fix Chart 3: remove intermediate `ax.legend()` call; consolidate to single call using `handles, labels = ax.get_legend_handles_labels()` after all series are plotted
- [x] 7.2 Add error bars to Chart 1 (gap bar chart) using `gap_usd_lower`/`gap_usd_upper`; clip at ±200% of point estimate; add subtitle note when clipping occurs
- [x] 7.3 Update Chart 5 (choropleth) tooltip to display CI range: "Gap: $X.XB [$Y.YB – $Z.ZB 90% CI]"; gracefully omit CI range when columns are NaN

## 8. Output Verification

- [x] 8.1 Run full pipeline (`uv run python main.py`) and verify `dri_output.csv` contains all new columns: `peer_group`, `giving_rate_raw`, `gap_usd_signed`, `gap_pct_ppp_gdp`, `gap_usd_expected`, `gap_usd_lower`, `gap_usd_upper`, `DRI_score`, `ppp_data_available`
- [x] 8.2 Verify no countries are incorrectly labeled "Reliable Donor" when `giving_rate_raw > 1.0`
- [x] 8.3 Verify Chart 3 legend shows all series labels
- [x] 8.4 Verify Chart 1 error bars are visible and chart renders without errors
- [x] 8.5 Run `uv run python main.py --dry-run` and confirm no errors or NaN-leakage warnings in the printed summary
