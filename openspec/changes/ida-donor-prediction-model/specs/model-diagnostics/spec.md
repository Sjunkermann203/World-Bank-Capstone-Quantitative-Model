## ADDED Requirements

### Requirement: IMR significance test
The system SHALL test whether the inverse Mills ratio coefficient in Stage 2 is statistically significant. If the p-value exceeds 0.10, a warning SHALL be written to the diagnostics report: "IMR p-value=<x>: selection bias may be negligible."

#### Scenario: IMR is significant
- **WHEN** IMR p-value <= 0.10
- **THEN** no warning is written for this check

#### Scenario: IMR is not significant
- **WHEN** IMR p-value > 0.10
- **THEN** the diagnostics report includes the warning message with the actual p-value

### Requirement: Exclusion restriction strength test
The system SHALL compute a likelihood ratio test on the jointly excluded variables (`un_voting_align`, `peer_donor`, `dac_member`) in Stage 1. If the excluded variables are jointly insignificant (LR test p-value > 0.10), a warning SHALL be written: "Exclusion restrictions may be weak: LR p-value=<x>."

#### Scenario: Strong exclusion restrictions
- **WHEN** the LR test p-value <= 0.10
- **THEN** no warning is written for this check

#### Scenario: Weak exclusion restrictions
- **WHEN** the LR test p-value > 0.10
- **THEN** the diagnostics report includes the warning message

### Requirement: Naive OLS comparison
The system SHALL re-estimate Stage 2 without the IMR as a regressor and log both coefficient tables side-by-side. Large coefficient shifts confirm that selection bias correction was material.

#### Scenario: OLS comparison is written
- **WHEN** diagnostics run
- **THEN** the diagnostics report contains a table with columns: variable, coef_heckman, coef_naive_ols, pct_change

### Requirement: Heteroskedasticity test
The system SHALL run a Breusch-Pagan test on Stage 2 residuals. If the p-value is below 0.05, heteroskedasticity-robust standard errors SHALL be used for Stage 2 and a note SHALL be written to the diagnostics report.

#### Scenario: Homoskedastic residuals
- **WHEN** Breusch-Pagan p-value >= 0.05
- **THEN** standard OLS standard errors are used

#### Scenario: Heteroskedastic residuals detected
- **WHEN** Breusch-Pagan p-value < 0.05
- **THEN** robust standard errors are applied to Stage 2 and the diagnostics report notes "Robust SEs applied (BP p=<x>)"

### Requirement: Multicollinearity check
The system SHALL compute VIF for all Stage 2 regressors. Any variable with VIF > 10 SHALL be flagged in the diagnostics report: "High VIF for <variable>: <vif>."

#### Scenario: No multicollinearity
- **WHEN** all VIFs are <= 10
- **THEN** no VIF flag is written

#### Scenario: High VIF detected
- **WHEN** one or more variables have VIF > 10
- **THEN** each is listed in the diagnostics report

### Requirement: Out-of-sample accuracy reporting
The system SHALL compute MAE and RMSE on the IDA18–IDA20 holdout set for both the Heckman model and a naive OLS baseline (Stage 2 without IMR, no selection correction). Both sets of metrics SHALL be written to the diagnostics report.

#### Scenario: OOS metrics written
- **WHEN** diagnostics run after estimation
- **THEN** the diagnostics report contains a table with rows for Heckman and naive OLS, columns for MAE and RMSE on the test set

### Requirement: Diagnostics output file
The system SHALL write all diagnostic results to `outputs/heckman_diagnostics.txt` and SHALL write a residuals plot to `outputs/charts/heckman_residuals.png` showing actual vs. predicted log-donation and IMR distribution.

#### Scenario: Diagnostics file created
- **WHEN** the pipeline completes estimation
- **THEN** `outputs/heckman_diagnostics.txt` exists and contains all flagged warnings and metric tables

#### Scenario: Residuals chart created
- **WHEN** the pipeline completes estimation
- **THEN** `outputs/charts/heckman_residuals.png` exists with two subplots: actual vs. predicted log-donation, and IMR distribution histogram
