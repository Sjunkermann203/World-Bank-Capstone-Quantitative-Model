## ADDED Requirements

### Requirement: Panel CSV validation
The system SHALL load `data/raw/heckman_panel.csv` and validate that the following columns are present: `country_iso3`, `replenishment_round`, `donate_dummy`, `donation_usd`, `log_gdp_per_capita`, `dac_member`, `un_voting_align`, `trade_openness`, `gov_effectiveness`, `peer_donor`, `log_gdp_level`, `fiscal_balance_pct_gdp`, `ida_vote_share_lag`, `trade_exposure_ida`, `log_donation_lag`, `us_eu_ally`, `sovereign_credit_rating`. It SHALL raise `SchemaValidationError` listing all missing columns if any are absent.

#### Scenario: All required columns present
- **WHEN** `heckman_panel.csv` contains all required columns
- **THEN** the panel loads successfully with no error raised

#### Scenario: One or more required columns missing
- **WHEN** `heckman_panel.csv` is missing one or more required columns
- **THEN** `SchemaValidationError` is raised naming every missing column

### Requirement: No-duplicate assertion
The system SHALL assert that no two rows share the same `(country_iso3, replenishment_round)` combination and SHALL raise `ValueError` identifying the duplicate keys if any exist.

#### Scenario: Unique panel keys
- **WHEN** every `(country_iso3, replenishment_round)` combination is unique
- **THEN** ingestion completes without error

#### Scenario: Duplicate panel keys detected
- **WHEN** two or more rows share the same `(country_iso3, replenishment_round)`
- **THEN** `ValueError` is raised identifying the duplicate pairs

### Requirement: WDI extension for trade openness and governance
The system SHALL extend `ingest.py` WDI_INDICATORS to include `NE.TRD.GNFS.ZS` (trade openness, % GDP) mapped to `trade_openness` and `GE.EST` (WGI government effectiveness estimate) mapped to `gov_effectiveness`.

#### Scenario: WDI fetch includes new indicators
- **WHEN** `build_master()` runs
- **THEN** `master.csv` contains `trade_openness` and `gov_effectiveness` columns

#### Scenario: WDI returns null for new indicators
- **WHEN** the World Bank API returns null for `trade_openness` or `gov_effectiveness` for a country
- **THEN** those values are null in `master.csv` and a warning is logged
