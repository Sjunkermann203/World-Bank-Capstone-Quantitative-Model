## 1. Dependencies and Setup

- [x] 1.1 Add `statsmodels`, `scikit-learn`, `seaborn` to `pyproject.toml` via `uv add statsmodels scikit-learn seaborn`
- [x] 1.2 Verify `uv sync` completes cleanly after adding new dependencies

## 2. Extend Ingest Pipeline

- [x] 2.1 Add `NE.TRD.GNFS.ZS` → `trade_openness` and `GE.EST` → `gov_effectiveness` to `WDI_INDICATORS` in `src/ingest.py`
- [x] 2.2 Add both new columns to the `output_cols` list in `build_master()` so they appear in `master.csv`
- [x] 2.3 Delete `data/cache/wdi.csv` (or run with `--refresh`) to force a re-fetch including the new indicators

## 3. Panel Data Ingestion Module

- [x] 3.1 Create `src/heckman.py` with a `load_panel()` function that reads `data/raw/heckman_panel.csv`
- [x] 3.2 Implement column validation in `load_panel()` raising `SchemaValidationError` (imported from `ingest.py`) for any missing required columns
- [x] 3.3 Implement duplicate-key assertion on `(country_iso3, replenishment_round)` raising `ValueError` with the offending pairs

## 4. Preprocessing

- [x] 4.1 Implement `preprocess_panel()` in `src/heckman.py` — log-transform GDP and donation columns if not already log-transformed
- [x] 4.2 Implement credit rating ordinal encoder (AAA=20 … D=1) inside `preprocess_panel()`
- [x] 4.3 Set `log_donation_lag = 0` for countries with no prior donation record
- [x] 4.4 Fit standardization (zero mean, unit variance) on training set continuous Stage 2 variables; apply to both train and test sets without leakage

## 5. Temporal Split

- [x] 5.1 Implement `split_panel()` in `src/heckman.py` that separates IDA1–IDA17 (train) from IDA18–IDA20 (test) by round label

## 6. Heckman Two-Step Estimation

- [x] 6.1 Implement `fit_stage1()` — statsmodels probit on full training sample with selection variables; extract inverse Mills ratio φ(Xβ)/Φ(Xβ)
- [x] 6.2 Log a warning if Stage 1 probit does not converge
- [x] 6.3 Implement `fit_stage2()` — OLS on donor training subsample with outcome variables + IMR + round dummies; apply robust SEs if Breusch-Pagan p < 0.05
- [x] 6.4 Implement `fit_mle_heckman()` — fit the MLE Heckman variant using statsmodels
- [x] 6.5 Implement coefficient divergence check: log warning for any coefficient differing >20% between two-step and MLE

## 7. Prediction Generation

- [x] 7.1 Implement `predict_all()` — generate `p_donate` for all countries using Stage 1 probit
- [x] 7.2 Compute `pred_log_donation` for all countries from Stage 2 coefficients (including IMR evaluated at each country's covariates)
- [x] 7.3 Apply Duan smearing correction: multiply `exp(pred_log_donation)` by mean of `exp(Stage 2 residuals)` to get `pred_donation_usd`
- [x] 7.4 Compute `expected_contribution = p_donate × pred_donation_usd`

## 8. Segmentation and Column Mapping

- [x] 8.1 Implement `assign_segments()` applying the five donor segment rules in priority order
- [x] 8.2 Map `expected_contribution` → `adjusted_target_usd`, compute `gap_usd` and `giving_rate` for report compatibility
- [x] 8.3 Log segmentation summary (count and mean `expected_contribution` by segment) at INFO level

## 9. Diagnostics

- [x] 9.1 Implement `run_diagnostics()` — IMR t-test with warning if p > 0.10
- [x] 9.2 Implement exclusion restriction LR test — warning if joint LR p > 0.10
- [x] 9.3 Implement naive OLS comparison table (coef_heckman, coef_naive_ols, pct_change columns)
- [x] 9.4 Implement VIF computation for Stage 2 regressors — flag any VIF > 10
- [x] 9.5 Implement OOS MAE and RMSE on IDA18–IDA20 test set for Heckman and naive OLS baseline
- [x] 9.6 Write all diagnostic output to `outputs/heckman_diagnostics.txt`
- [x] 9.7 Generate residuals plot: two-subplot figure (actual vs. predicted log-donation; IMR histogram) saved to `outputs/charts/heckman_residuals.png`

## 10. Public Interface and Pipeline Integration

- [x] 10.1 Implement `score_capacity(master, fiscal_modifier=True)` as the public entry point in `src/heckman.py`, matching the exact signature from `src/capacity.py`
- [x] 10.2 Inside `score_capacity()`, call `load_panel()`, `preprocess_panel()`, `split_panel()`, `fit_stage1()`, `fit_stage2()`, `predict_all()`, `assign_segments()`, `run_diagnostics()` in order
- [x] 10.3 Write output DataFrame to `data/processed/capacity_scores.csv` with all required columns
- [x] 10.4 Update `main.py` import: change `from capacity import score_capacity` to `from heckman import score_capacity`
- [x] 10.5 Add `--skip-heckman` flag to `main.py` that falls back to `from capacity import score_capacity` when the panel CSV is absent

## 11. Verification

- [x] 11.1 Run `uv run python main.py --dry-run` — confirm `master.csv` now includes `trade_openness` and `gov_effectiveness` (code in place; columns appear after --refresh re-fetches WDI cache)
- [x] 11.2 Run `uv run python main.py` with a valid `heckman_panel.csv` — confirm pipeline completes end-to-end and `dri_output.csv` is produced
- [x] 11.3 Verify `data/processed/capacity_scores.csv` contains all required columns including `p_donate`, `donor_segment`, `imr`
- [x] 11.4 Verify `outputs/heckman_diagnostics.txt` exists and contains IMR test, LR test, VIF table, OOS metrics
- [x] 11.5 Verify `outputs/charts/heckman_residuals.png` exists
- [x] 11.6 Verify all five existing charts (chart1–chart5 + world map) still generate correctly
- [x] 11.7 Run `uv run python main.py --skip-heckman` (without panel CSV) — confirm it falls back to rule-based capacity scoring without error
