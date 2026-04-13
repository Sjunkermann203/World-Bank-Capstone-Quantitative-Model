## Why

The existing DRI (Donor Readiness Index) scores countries using a rule-based gap model but cannot generate contribution estimates for non-donors or correct for the self-selection bias inherent in observed donation data. A Heckman selection model addresses this by jointly modeling the decision to donate and the donation amount, enabling unconditional expected-contribution forecasts for all UN member states.

## What Changes

- Add a new `heckman_model` pipeline module implementing two-stage Heckman estimation
- Add a data ingestion layer supporting a country×round panel (IDA1–IDA20, ~19 rounds)
- Add preprocessing for new input variables: WGI governance effectiveness, OECD DAC membership, UN voting alignment, sovereign credit ratings, IDA vote share, trade exposure to IDA-eligible countries, and peer-donor flags
- Add temporal train/test split logic (IDA1–IDA17 train, IDA18–IDA20 test)
- Add segmentation logic assigning donor segments based on p_donate and predicted vs. actual gap
- Add a diagnostics module running post-estimation checks (IMR significance, exclusion restrictions, Breusch-Pagan, VIF, OOS accuracy)
- Add outputs: ranked scoring CSV, segmentation summary, coefficient tables, diagnostics report, residual plots

## Capabilities

### New Capabilities

- `panel-data-ingestion`: Load and validate country×round panel data; enforce no-duplicate constraint on (country_iso3, replenishment_round)
- `heckman-estimation`: Two-stage Heckman model — Stage 1 probit (selection), Stage 2 OLS with IMR correction; MLE variant for comparison
- `donor-segmentation`: Assign donor_segment labels based on p_donate thresholds and predicted-vs-actual contribution gap
- `model-diagnostics`: Post-estimation checks: IMR t-test, exclusion restriction LR test, naive OLS comparison, Breusch-Pagan, VIF, OOS MAE/RMSE
- `prediction-output`: Generate ranked country scoring table with columns: country_iso3, country_name, replenishment_round, p_donate, pred_donation_usd, expected_contribution, donor_segment, imr

### Modified Capabilities

## Impact

- New directory `src/heckman/` with modules: `ingest.py`, `preprocess.py`, `model.py`, `segment.py`, `diagnostics.py`, `output.py`
- New entry point `heckman_main.py` (standalone pipeline, does not alter existing `main.py`)
- New dependencies: `statsmodels`, `scipy`, `scikit-learn`, `seaborn` (add via `uv add`)
- Input data requirements: panel CSV with IDA contributions by round in constant USD, bilateral trade data, IDA eligibility list, OECD DAC membership list, UN voting alignment scores, World Bank WGI scores, sovereign credit ratings
- Outputs written to `outputs/heckman/`
