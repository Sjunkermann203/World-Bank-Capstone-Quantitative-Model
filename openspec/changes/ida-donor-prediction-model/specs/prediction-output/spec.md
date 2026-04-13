## ADDED Requirements

### Requirement: Per-country prediction generation
The system SHALL generate predictions for all countries including non-donors using the most recent replenishment round covariates. For each country the following SHALL be computed:
- `p_donate`: predicted probability from Stage 1 probit
- `pred_log_donation`: predicted log-donation from Stage 2 OLS (using country's covariates, evaluated at the IMR for that country)
- `pred_donation_usd`: `exp(pred_log_donation)` with Duan smearing correction applied
- `expected_contribution`: `p_donate × pred_donation_usd`

#### Scenario: Non-donor gets a prediction
- **WHEN** a country has `donate_dummy == 0` for all rounds in the training set
- **THEN** `p_donate`, `pred_donation_usd`, and `expected_contribution` are still computed and non-null

#### Scenario: Duan smearing applied
- **WHEN** `pred_donation_usd` is computed from `pred_log_donation`
- **THEN** the smearing correction factor (mean of `exp(Stage 2 residuals)`) is multiplied before exponentiation output

### Requirement: Ranked output DataFrame
The system SHALL produce a DataFrame with the following columns and write it to `data/processed/capacity_scores.csv` (preserving the path used by `report.py`): `iso3`, `country_name`, `income_group`, `gdp_usd`, `actual_contribution_usd`, `adjusted_target_usd`, `gap_usd`, `giving_rate`, `p_donate`, `pred_donation_usd`, `expected_contribution`, `donor_segment`, `imr`.

#### Scenario: Output written to expected path
- **WHEN** `score_capacity()` completes
- **THEN** `data/processed/capacity_scores.csv` exists and contains all required columns

#### Scenario: Output sorted by gap descending
- **WHEN** the output DataFrame is returned
- **THEN** rows are sorted by `gap_usd` descending with nulls last, matching the convention in `report.py`'s `build_dri_output()`

### Requirement: Segmentation summary log
The system SHALL log a segmentation summary at INFO level showing count and mean expected contribution by `donor_segment` after scoring completes.

#### Scenario: Summary logged
- **WHEN** `score_capacity()` completes
- **THEN** the log contains a per-segment count and mean `expected_contribution`
