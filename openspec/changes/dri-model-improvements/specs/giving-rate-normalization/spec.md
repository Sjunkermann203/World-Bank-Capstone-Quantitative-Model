## ADDED Requirements

### Requirement: giving_rate_raw preserves uncapped value
The pipeline SHALL compute `giving_rate_raw` as the raw ratio of actual IDA21 contribution to the IDA20-derived target, with no upper bound applied. This column SHALL be present in all intermediate dataframes and in `dri_output.csv`.

#### Scenario: Over-contributing country
- **WHEN** a country's actual IDA21 contribution exceeds its IDA20-derived target
- **THEN** `giving_rate_raw` is greater than 1.0 for that country

#### Scenario: Under-contributing country
- **WHEN** a country's actual IDA21 contribution is less than its IDA20-derived target
- **THEN** `giving_rate_raw` is between 0.0 and 1.0

### Requirement: giving_rate capped at 1.0 for segment assignment
For the purpose of donor segment assignment only, the pipeline SHALL use `giving_rate = min(giving_rate_raw, 1.0)`. The existing `giving_rate` column SHALL reflect this capped value. Segment thresholds SHALL be applied against the capped value.

#### Scenario: Segment assignment for over-contributor
- **WHEN** `giving_rate_raw > 1.0`
- **THEN** the country is assigned segment "Exceeded Target" regardless of other thresholds, and `giving_rate = 1.0`

#### Scenario: Segment assignment for on-track donor
- **WHEN** `giving_rate_raw <= 1.0` and giving_rate meets the "Reliable Donor" threshold
- **THEN** the country is assigned "Reliable Donor"

### Requirement: Exceeded Target segment
The pipeline SHALL define an explicit donor segment "Exceeded Target" assigned to all countries where `giving_rate_raw > 1.0`. This segment SHALL take precedence over all other segment rules.

#### Scenario: Exceeded Target segment in output
- **WHEN** the pipeline runs and a country has `giving_rate_raw > 1.0`
- **THEN** its `segment` column contains exactly "Exceeded Target"

### Requirement: Signed gap column
The pipeline SHALL compute `gap_usd_signed` as a signed monetary gap: positive values indicate a shortfall (country contributes less than benchmark target), negative values indicate over-contribution. The formula SHALL be: `gap_usd_signed = benchmark_contribution_usd - actual_contribution_usd`.

#### Scenario: Shortfall country
- **WHEN** a country contributes less than its peer-group benchmark target
- **THEN** `gap_usd_signed > 0`

#### Scenario: Over-contributing country
- **WHEN** a country contributes more than its peer-group benchmark target
- **THEN** `gap_usd_signed < 0`

#### Scenario: Exact contributor
- **WHEN** a country contributes exactly its benchmark target
- **THEN** `gap_usd_signed == 0`
