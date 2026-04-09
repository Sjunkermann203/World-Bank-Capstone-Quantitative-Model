## ADDED Requirements

### Requirement: Fetch World Bank WDI indicators
The system SHALL retrieve GDP (current USD), GDP per capita (current USD), and fiscal balance as % of GDP for all countries in the country universe from the World Bank WDI API for the most recent available year. Data SHALL be cached locally in `data/cache/wdi.csv` to avoid repeated API calls during a single run.

#### Scenario: Successful WDI fetch
- **WHEN** the ingestion pipeline runs and the WDI API is reachable
- **THEN** the system downloads GDP, GDP per capita, and fiscal balance for all target countries and writes them to `data/cache/wdi.csv`

#### Scenario: Cached WDI data is reused
- **WHEN** `data/cache/wdi.csv` already exists and the `--refresh` flag is not passed
- **THEN** the system reads from the cache file and skips the API call

#### Scenario: Missing indicator for a country
- **WHEN** a WDI indicator is unavailable for a specific country
- **THEN** the system SHALL log a warning with the country ISO code and indicator name, and record a null value for that field; it SHALL NOT abort the pipeline

### Requirement: Load IMF WEO government debt data
The system SHALL load government gross debt as % of GDP from an IMF WEO dataset file (Excel or CSV) stored at `data/raw/imf_weo.csv`. The ingestion stage SHALL validate that the expected columns are present and raise a descriptive error if the schema does not match.

#### Scenario: Valid IMF WEO file loaded
- **WHEN** `data/raw/imf_weo.csv` exists and contains the expected columns
- **THEN** the system reads government debt (% GDP) per country and merges it into the master dataset

#### Scenario: IMF WEO file has unexpected schema
- **WHEN** `data/raw/imf_weo.csv` is missing expected columns
- **THEN** the system SHALL raise a `SchemaValidationError` with the names of the missing columns and halt the pipeline

### Requirement: Load IDA contribution records
The system SHALL load IDA20 and IDA21 actual contribution amounts (USD) from `data/raw/ida_contributions.csv`. Each row SHALL represent one country-cycle pair (country ISO, cycle, contribution_usd). The file is hand-curated and versioned in the repository.

#### Scenario: IDA contributions file loaded
- **WHEN** `data/raw/ida_contributions.csv` exists with valid structure
- **THEN** the system reads actual contribution amounts for IDA20 and IDA21 and merges them into the master dataset per country

#### Scenario: Country present in IDA file but not in country universe
- **WHEN** a country appears in `ida_contributions.csv` but is not in the canonical country list
- **THEN** the system SHALL log a warning and skip that row

### Requirement: Resolve country identities via canonical mapping
The system SHALL use a canonical country mapping table (`data/country_map.csv`) keyed on ISO 3166-1 alpha-3 codes to reconcile country name differences across WDI, IMF, and IDA data sources. All output datasets SHALL use ISO alpha-3 codes as the primary country identifier.

#### Scenario: Country name mismatch resolved
- **WHEN** a country appears under different names in WDI and IDA sources
- **THEN** the canonical map resolves both to the same ISO alpha-3 code and the merge proceeds correctly

#### Scenario: Unknown country encountered
- **WHEN** a country name cannot be resolved to an ISO alpha-3 code via the mapping table
- **THEN** the system SHALL log a warning with the unresolved name and exclude that country from the output

### Requirement: Produce a normalized master dataset
The system SHALL output a single normalized per-country CSV (`data/processed/master.csv`) containing all ingested fields joined on ISO alpha-3 code. The file SHALL include: `iso3`, `country_name`, `income_group`, `gdp_usd`, `gdp_per_capita_usd`, `fiscal_balance_pct_gdp`, `govt_debt_pct_gdp`, `ida20_contribution_usd`, `ida21_contribution_usd`.

#### Scenario: Successful master dataset produced
- **WHEN** all ingestion stages complete without fatal errors
- **THEN** `data/processed/master.csv` is written with one row per country and all required columns present

#### Scenario: Country missing all economic data
- **WHEN** a country in the universe has no WDI, IMF, or IDA data
- **THEN** the country SHALL be excluded from `master.csv` and logged as excluded
