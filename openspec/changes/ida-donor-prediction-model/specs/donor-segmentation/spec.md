## ADDED Requirements

### Requirement: Donor segment assignment
The system SHALL assign a `donor_segment` label to every country using the following rules in priority order:
1. If `is_current_donor == 1` AND `(expected_contribution − actual_contribution_usd) / expected_contribution > 0.20`: label `"Under-Contributing Donor"`
2. If `is_current_donor == 1` AND the gap is within 20%: label `"Reliable Donor"`
3. If `is_current_donor == 0` AND `p_donate >= 0.50`: label `"High-Potential Prospect"`
4. If `is_current_donor == 0` AND `0.20 <= p_donate < 0.50`: label `"Emerging Prospect"`
5. Otherwise: label `"Low Probability"`

#### Scenario: Under-contributing donor
- **WHEN** a country is a current donor and its expected contribution exceeds actual by more than 20%
- **THEN** `donor_segment` is `"Under-Contributing Donor"`

#### Scenario: Reliable donor
- **WHEN** a country is a current donor and its predicted gap is within 20%
- **THEN** `donor_segment` is `"Reliable Donor"`

#### Scenario: High-potential non-donor
- **WHEN** a country is not a current donor and `p_donate >= 0.50`
- **THEN** `donor_segment` is `"High-Potential Prospect"`

#### Scenario: Emerging prospect
- **WHEN** a country is not a current donor and `0.20 <= p_donate < 0.50`
- **THEN** `donor_segment` is `"Emerging Prospect"`

#### Scenario: Low probability non-donor
- **WHEN** a country is not a current donor and `p_donate < 0.20`
- **THEN** `donor_segment` is `"Low Probability"`

### Requirement: Column mapping for report compatibility
The system SHALL populate the following columns in the output DataFrame so that `report.py` operates without changes:
- `adjusted_target_usd` = `expected_contribution`
- `gap_usd` = `expected_contribution − actual_contribution_usd`
- `giving_rate` = `actual_contribution_usd / expected_contribution` (null when `expected_contribution` is zero or null)

#### Scenario: Non-donor giving rate
- **WHEN** a country has no actual contribution (`actual_contribution_usd == 0`) and a valid `expected_contribution`
- **THEN** `giving_rate` is 0.0 and `gap_usd` equals `expected_contribution`

#### Scenario: Null expected contribution
- **WHEN** `expected_contribution` is null for a country (e.g., Stage 2 prediction failed)
- **THEN** `gap_usd` and `giving_rate` are null and the country appears in `report.py` charts as `na_position="last"`
