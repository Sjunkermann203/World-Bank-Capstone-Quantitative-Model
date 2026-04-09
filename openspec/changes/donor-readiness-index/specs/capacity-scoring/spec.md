## ADDED Requirements

### Requirement: Compute income-tier-adjusted benchmark IDA/GDP ratio
The system SHALL compute the median IDA/GDP ratio separately for each World Bank income group (High Income, Upper-Middle Income) using the IDA21 actual contributions of confirmed current donors. The resulting tier medians SHALL be logged and included in a run-metadata file (`data/processed/run_metadata.json`) for auditability.

#### Scenario: Tier medians calculated from current donors
- **WHEN** the scoring stage runs with a valid master dataset containing current donors
- **THEN** the system computes a separate median IDA/GDP ratio for each income tier present among current donors and stores them in run metadata

#### Scenario: Fewer than 3 donors in a tier
- **WHEN** an income tier has fewer than 3 current donors with valid IDA/GDP data
- **THEN** the system SHALL log a warning and fall back to the global median across all current donors for countries in that tier

### Requirement: Calculate capacity-based target contribution
The system SHALL calculate a target IDA contribution for each country in the universe as: `target_usd = gdp_usd × tier_median_ida_gdp_ratio`. Countries with null GDP SHALL be excluded from scoring.

#### Scenario: Target computed for a candidate country
- **WHEN** a country has valid GDP data and an applicable tier median
- **THEN** `target_usd` is set to `gdp_usd × tier_median_ida_gdp_ratio`

#### Scenario: Country GDP is null
- **WHEN** a country's GDP is null or zero
- **THEN** the country SHALL receive a null `target_usd` and be excluded from gap rankings

### Requirement: Apply fiscal modifier to target contribution
The system SHALL apply a fiscal modifier to each country's target contribution based on its fiscal balance as % of GDP. The modifier SHALL be linear, ranging from −20% to +20%, capped at those bounds. A fiscal balance of 0% maps to a modifier of 0. The adjusted target SHALL be: `adjusted_target_usd = target_usd × (1 + fiscal_modifier)`.

#### Scenario: Positive fiscal balance increases target
- **WHEN** a country has a fiscal surplus (fiscal balance > 0%)
- **THEN** the fiscal modifier is positive (up to +0.20) and `adjusted_target_usd > target_usd`

#### Scenario: Negative fiscal balance decreases target
- **WHEN** a country has a fiscal deficit (fiscal balance < 0%)
- **THEN** the fiscal modifier is negative (down to −0.20) and `adjusted_target_usd < target_usd`

#### Scenario: Fiscal modifier is capped at bounds
- **WHEN** a country's fiscal balance would produce a modifier outside [−0.20, +0.20]
- **THEN** the modifier is clamped to the nearest bound

#### Scenario: Fiscal balance data is missing
- **WHEN** fiscal balance data is null for a country
- **THEN** the fiscal modifier defaults to 0 (no adjustment) and a warning is logged

### Requirement: Compute contribution gap and giving rate
The system SHALL compute for each country:
- `gap_usd = adjusted_target_usd − actual_contribution_usd` (positive = underperforming, negative = overperforming)
- `giving_rate = actual_contribution_usd / adjusted_target_usd`

Where `actual_contribution_usd` is the IDA21 amount (falling back to IDA20 if IDA21 is unavailable). Countries with no actual contribution record SHALL have `actual_contribution_usd = 0`.

#### Scenario: Gap computed for underperforming country
- **WHEN** a country's actual contribution is less than its adjusted target
- **THEN** `gap_usd` is positive and `giving_rate < 1.0`

#### Scenario: Gap computed for overperforming country
- **WHEN** a country's actual contribution exceeds its adjusted target
- **THEN** `gap_usd` is negative and `giving_rate > 1.0`

#### Scenario: Country has no IDA contribution record
- **WHEN** a country has no entry in the IDA contributions file
- **THEN** `actual_contribution_usd = 0`, `gap_usd = adjusted_target_usd`, `giving_rate = 0`

### Requirement: Output capacity scores dataset
The system SHALL write `data/processed/capacity_scores.csv` containing per-country fields: `iso3`, `country_name`, `income_group`, `gdp_usd`, `tier_median_ida_gdp_ratio`, `target_usd`, `fiscal_modifier`, `adjusted_target_usd`, `actual_contribution_usd`, `gap_usd`, `giving_rate`.

#### Scenario: Capacity scores file produced
- **WHEN** the capacity scoring stage completes successfully
- **THEN** `data/processed/capacity_scores.csv` is written with one row per scored country and all required columns present
