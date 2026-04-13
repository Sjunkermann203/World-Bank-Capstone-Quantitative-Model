## ADDED Requirements

### Requirement: Temporal train/test split
The system SHALL split the panel by replenishment round: rounds IDA1–IDA17 form the training set; rounds IDA18–IDA20 form the held-out test set. Splitting SHALL be done by round label, not randomly.

#### Scenario: Split produces non-overlapping sets
- **WHEN** the panel is split
- **THEN** no country×round observation appears in both train and test sets

#### Scenario: Test set contains IDA18, IDA19, IDA20
- **WHEN** the split is applied
- **THEN** the test set contains only observations from rounds IDA18, IDA19, and IDA20

### Requirement: Preprocessing pipeline
The system SHALL apply the following preprocessing steps before estimation:
- Log-transform `gdp_level`, `gdp_per_capita`, and `donation_usd` (already log-transformed in panel for donors; verify)
- Encode `sovereign_credit_rating` as integers (AAA=20, AA+=19, … D=1)
- Set `log_donation_lag` to 0 for countries with no prior donation record
- Standardize all continuous Stage 2 regressors to zero mean and unit variance using training-set statistics
- Standardization parameters SHALL be fit on the training set and applied to both train and test sets

#### Scenario: Credit rating encoding
- **WHEN** `sovereign_credit_rating` contains the string "AAA"
- **THEN** it is encoded as 20

#### Scenario: No leakage from test set in standardization
- **WHEN** continuous variables are standardized
- **THEN** mean and variance are computed only from the training set observations

### Requirement: Two-step Heckman estimation
The system SHALL estimate a Heckman two-step model using `statsmodels`:
- Stage 1: probit on the full training sample with selection variables `log_gdp_per_capita`, `dac_member`, `un_voting_align`, `trade_openness`, `gov_effectiveness`, `peer_donor`; extract inverse Mills ratio (IMR = φ(Xβ)/Φ(Xβ)) for each observation
- Stage 2: OLS on the donor subsample only with outcome variables `log_gdp_level`, `fiscal_balance_pct_gdp`, `ida_vote_share_lag`, `trade_exposure_ida`, `log_donation_lag`, `us_eu_ally`, `sovereign_credit_rating`, plus the IMR as an additional regressor and round dummies
- Exclusion restrictions: `un_voting_align`, `peer_donor`, `dac_member` appear in Stage 1 only

#### Scenario: Stage 1 probit converges
- **WHEN** the probit is fit on the training set
- **THEN** `result.mle_retvals['converged']` is True; if False a warning is logged

#### Scenario: Stage 2 OLS includes IMR
- **WHEN** the OLS model is constructed for Stage 2
- **THEN** the inverse Mills ratio is included as a regressor alongside the outcome-equation variables

### Requirement: MLE Heckman variant for comparison
The system SHALL also fit `statsmodels.duration.hazard_regression` or the equivalent MLE Heckman model and compare coefficients with the two-step estimates. If any coefficient diverges by more than 20% in relative terms, a warning SHALL be logged naming the variable.

#### Scenario: Coefficients agree
- **WHEN** two-step and MLE coefficients are within 20% of each other for all variables
- **THEN** no divergence warning is logged

#### Scenario: Coefficient divergence detected
- **WHEN** a coefficient differs by more than 20% between two-step and MLE
- **THEN** a warning is logged: "Coefficient divergence >20% for <variable>: two-step=<x>, MLE=<y>"
