## ADDED Requirements

### Requirement: Income-tier peer group assignment
Each country in the dataset SHALL be assigned to a World Bank income-tier peer group (High Income, Upper-Middle Income, Lower-Middle Income, or Low Income) using the `income_level` field from `wbgapi.economy` metadata fetched during ingest. The assignment SHALL be stored in a `peer_group` column on the master dataframe.

#### Scenario: Country with known income classification
- **WHEN** a country has a valid `income_level` code in the WDI economy metadata
- **THEN** its `peer_group` column contains the corresponding income-tier label (e.g., "High Income")

#### Scenario: Country with missing income classification
- **WHEN** a country has no `income_level` code available
- **THEN** its `peer_group` is set to "Unclassified" and a warning is logged

### Requirement: GDP-weighted peer-group median benchmark
Within each peer group containing current IDA donors, the pipeline SHALL compute a GDP-weighted median contribution rate (contribution_usd / gdp_usd). The weighted median SHALL use each donor's nominal GDP as the weight and the weighted 50th percentile as the central value. This peer-group benchmark rate SHALL replace the existing global unweighted median when computing `benchmark_ratio` for each country.

#### Scenario: Peer group with three or more donors
- **WHEN** a peer group contains at least 3 current IDA donors
- **THEN** the GDP-weighted median contribution rate is computed for that group and used as the benchmark for all countries in that group

#### Scenario: Peer group with fewer than three donors
- **WHEN** a peer group contains fewer than 3 current IDA donors
- **THEN** a warning is logged specifying the group name and count, and the global GDP-weighted median is used as the fallback benchmark for that group

#### Scenario: Benchmark applied to non-donor countries
- **WHEN** a country is not a current IDA donor
- **THEN** it is assigned the benchmark of its peer group (or global fallback) to compute its gap_usd

### Requirement: peer_group column in output
The `peer_group` column SHALL be present in `outputs/dri_output.csv` and in all intermediate dataframes passed between pipeline stages.

#### Scenario: Output file contains peer_group
- **WHEN** the pipeline runs successfully
- **THEN** `dri_output.csv` contains a non-null `peer_group` column for every row
