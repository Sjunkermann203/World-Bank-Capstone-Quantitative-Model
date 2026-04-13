## Context

`capacity.py` currently scores countries using a rule-based gap model: multiply GDP by a global median IDA/GDP benchmark ratio, apply a fiscal modifier, subtract actuals. This produces `gap_usd`, `giving_rate`, and `adjusted_target_usd` consumed by `report.py`. The rule-based approach cannot produce estimates for non-donors and does not correct for selection bias.

The Heckman two-stage model replaces this logic in Stage 2 of the pipeline. It produces the same output columns (`gap_usd`, `giving_rate`, `adjusted_target_usd`) so `report.py` requires no structural changes, while adding new Heckman-specific columns (`p_donate`, `donor_segment`, `imr`).

The model requires a country×round panel (IDA1–IDA20) plus several new variables not currently fetched by `ingest.py`. These are provided via a new raw data file (`data/raw/heckman_panel.csv`) and extended `ingest.py` fetches for WGI governance effectiveness and trade openness.

## Goals / Non-Goals

**Goals:**
- Replace `src/capacity.py` with `src/heckman.py` implementing the two-stage Heckman estimator
- Maintain output column compatibility with `report.py` (`iso3`, `country_name`, `income_group`, `gdp_usd`, `gap_usd`, `giving_rate`, `adjusted_target_usd`, `actual_contribution_usd`)
- Add new output columns: `p_donate`, `pred_donation_usd`, `expected_contribution`, `donor_segment`, `imr`
- Extend `ingest.py` to fetch `trade_openness` and `gov_effectiveness` from WDI/WGI
- Accept `data/raw/heckman_panel.csv` as the primary panel input
- Run diagnostics suite and write to `outputs/heckman_diagnostics.txt`
- Integrate into `main.py` with no changes to the function call signature at the stage boundary

**Non-Goals:**
- Replacing `report.py` charts (charts remain gap/giving-rate based, now Heckman-derived)
- Automating fetches for UN voting alignment, IDA vote share, bilateral trade, credit ratings — these remain external inputs in the panel CSV
- GUI or interactive model-selection interface

## Decisions

### Replace capacity.py, keep function interface
`main.py` calls `score_capacity(master, fiscal_modifier=...)`. The new `heckman.py` exports this same function name and signature. `main.py` import changes from `from capacity import score_capacity` to `from heckman import score_capacity`. No other changes to `main.py` are required.

**Alternative considered:** rename to `score_heckman()` and update all call sites. Rejected — unnecessary churn; the function still "scores capacity" in spirit.

### Column mapping: Heckman outputs → existing DRI columns
- `expected_contribution` (= p_donate × pred_donation_usd) maps to `adjusted_target_usd` — the model's best estimate of what the country should contribute
- `expected_contribution − actual_contribution_usd` maps to `gap_usd`
- `actual_contribution_usd / expected_contribution` maps to `giving_rate`

This preserves all downstream chart logic in `report.py` unchanged.

### Panel CSV as external input
Variables like UN voting alignment, IDA vote share by round, bilateral trade exposure, and sovereign credit ratings require data assembly outside this codebase. `data/raw/heckman_panel.csv` is the externally prepared panel. `heckman.py` validates required columns at ingestion and raises `SchemaValidationError` (reused from `ingest.py`) for missing fields.

### Temporal split: IDA1–IDA17 train, IDA18–IDA20 test
The most-recent round in the panel is IDA20; IDA21 actuals are in `master.csv` but IDA21 is not yet a completed replenishment round with a full set of panel covariates. Training on IDA1–IDA17 leaves three complete rounds for OOS evaluation.

### No fixed effects
Country fixed effects would prevent predictions for non-donors (no estimated fixed effect) and absorb the between-country variation that identifies the cross-sectional predictors. Round dummies are included in Stage 2 as a lighter time control.

### log_donation_lag = 0 for new donors
A `first_time_donor` binary would be collinear with `log_donation_lag = 0`. Setting the lag to zero is the simpler and conventional approach.

### Diagnostics written to text file, not printed
`outputs/heckman_diagnostics.txt` preserves the pipeline's clean stdout convention. The diagnostics include: IMR t-test, exclusion restriction LR test, naive OLS comparison, Breusch-Pagan, VIF table, OOS MAE/RMSE vs. naive OLS baseline.

## Risks / Trade-offs

- **Exclusion restriction validity** — Cannot be formally tested. Mitigation: document assumption; naive OLS comparison shows sensitivity.
- **Panel CSV not present** — Pipeline will raise `SchemaValidationError` clearly rather than silently. Mitigation: add a `--skip-heckman` flag to `main.py` that falls back to the original rule-based capacity scoring for runs without panel data.
- **Small Stage-2 sample** — ~40–60 donors per round × 17 training rounds. Coefficients on rare dummies (e.g., `us_eu_ally`) may be imprecise. Mitigation: VIF and confidence interval reporting in diagnostics.
- **Log-normality assumption** — Verify with residual plots added to `outputs/charts/heckman_residuals.png`.
- **Retransformation bias** — `exp(E[log y]) ≠ E[y]`. Apply Duan smearing correction after exponentiation.

## Migration Plan

1. Add `statsmodels`, `scikit-learn`, `seaborn` to `pyproject.toml` via `uv add`
2. Extend `ingest.py` WDI indicators to include `trade_openness` and `gov_effectiveness`
3. Write `src/heckman.py` exporting `score_capacity(master, ...)` 
4. Update `main.py` import: `from heckman import score_capacity`
5. Remove `--no-fiscal-modifier` flag from `main.py` (no longer applicable) or silently ignore it
6. Provide `data/raw/heckman_panel.csv` with required columns before running

## Open Questions

- Should `--skip-heckman` fall back to the old rule-based scorer (keep `capacity.py` alongside), or should the old scorer be deleted entirely?
- What base year should constant-USD deflation use? (Assume 2015 USD until confirmed.)
- Should attrited donors (graduated developing countries that stopped donating) be a separate `donor_segment` = `"Attrited Donor"`, or folded into existing segments?
