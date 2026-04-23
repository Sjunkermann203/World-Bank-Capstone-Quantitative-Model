## ADDED Requirements

### Requirement: PPP-adjusted GDP fetch from WDI
During the ingest stage, the pipeline SHALL fetch PPP-adjusted GDP (indicator `NY.GDP.MKTP.PP.KD`, constant 2017 international dollars) and PPP-adjusted GNI per capita (indicator `NY.GNP.PCAP.PP.CD`) from WDI for all countries. These SHALL be stored as `gdp_ppp` and `gni_per_capita_ppp` columns in the master dataframe.

#### Scenario: PPP data available for a country
- **WHEN** WDI returns valid PPP GDP data for a country
- **THEN** `gdp_ppp` is a positive non-null numeric value for that country

#### Scenario: PPP data missing for a country
- **WHEN** WDI returns no PPP GDP data for a country
- **THEN** `gdp_ppp` is NaN, `ppp_data_available` is False, and a warning is logged for that country

### Requirement: ppp_data_available flag column
The pipeline SHALL add a boolean column `ppp_data_available` to the master dataframe that is True when `gdp_ppp` is non-null and False otherwise. This column SHALL appear in `dri_output.csv`.

#### Scenario: Flag set correctly
- **WHEN** `gdp_ppp` is non-null for a country
- **THEN** `ppp_data_available` is True for that country

### Requirement: Benchmark ratio uses PPP-adjusted GDP
The peer-group benchmark contribution rate SHALL be computed as `contribution_usd / gdp_ppp` when `ppp_data_available` is True. For countries where `ppp_data_available` is False, the pipeline SHALL fall back to nominal GDP and log a warning.

#### Scenario: PPP-based benchmark for High Income peer group
- **WHEN** a High Income donor country has valid PPP GDP data
- **THEN** its contribution rate used in the weighted median computation is `contribution_usd / gdp_ppp`

#### Scenario: Nominal GDP fallback
- **WHEN** a donor country has `ppp_data_available = False`
- **THEN** its contribution rate is computed using nominal GDP and a warning is logged

### Requirement: gap_pct_ppp_gdp column
The pipeline SHALL compute `gap_pct_ppp_gdp = gap_usd_signed / gdp_ppp × 100` as a percentage of PPP-adjusted GDP for countries where `ppp_data_available` is True. For countries where PPP data is unavailable, this column SHALL be NaN.

#### Scenario: gap_pct_ppp_gdp in output
- **WHEN** a country has valid `gdp_ppp` and a computed `gap_usd_signed`
- **THEN** `gap_pct_ppp_gdp` is non-null and equals `gap_usd_signed / gdp_ppp * 100`

#### Scenario: Missing PPP data
- **WHEN** `ppp_data_available` is False
- **THEN** `gap_pct_ppp_gdp` is NaN for that country
