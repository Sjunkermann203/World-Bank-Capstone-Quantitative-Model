## ADDED Requirements

### Requirement: Score UNGA voting alignment
The system SHALL score each country's UNGA voting alignment with a reference group (e.g., Western donors or World Bank shareholder consensus) using a published UNGA voting dataset. The score SHALL be normalized to [0, 100], where 100 represents perfect alignment. Countries with insufficient voting history (fewer than 10 resolutions) SHALL receive a null UNGA score.

#### Scenario: UNGA alignment score computed
- **WHEN** a country has sufficient UNGA voting history in the source dataset
- **THEN** the system computes an alignment score normalized to [0, 100]

#### Scenario: Insufficient voting history
- **WHEN** a country has fewer than 10 resolutions in the UNGA dataset
- **THEN** the UNGA alignment score is null and a warning is logged

### Requirement: Score World Bank vote share
The system SHALL score each country's World Bank vote share as a proportion of total IBRD/IDA votes, normalized to [0, 100]. The source SHALL be the most recent WB voting power disclosure. Countries with zero or no vote share record SHALL receive a score of 0.

#### Scenario: WB vote share score computed
- **WHEN** a country appears in the WB voting power data
- **THEN** the system normalizes its vote share against the maximum in the dataset and assigns a score in [0, 100]

#### Scenario: Country has no WB vote share record
- **WHEN** a country does not appear in the WB voting power data
- **THEN** its WB vote share score is 0

### Requirement: Score IFC presence
The system SHALL score IFC presence as a binary indicator: 1 (IFC has an active portfolio in the country) or 0 (no active portfolio). The score SHALL be scaled to [0, 100] (i.e., 0 or 100). Source: IFC country portal or a curated data file at `data/raw/ifc_presence.csv`.

#### Scenario: Country has active IFC portfolio
- **WHEN** a country appears in the IFC presence dataset with an active portfolio flag
- **THEN** its IFC presence score is 100

#### Scenario: Country has no IFC portfolio
- **WHEN** a country does not appear in the IFC presence dataset or is flagged as inactive
- **THEN** its IFC presence score is 0

### Requirement: Compute composite strategic alignment score
The system SHALL compute a composite strategic alignment score as the equally-weighted average of the three component scores (UNGA alignment, WB vote share, IFC presence), each on [0, 100]. If any component score is null, it SHALL be excluded from the average (i.e., the average is computed over available components only). The composite SHALL be recorded as `alignment_score` in [0, 100].

#### Scenario: All three components available
- **WHEN** all three component scores are non-null
- **THEN** `alignment_score = (unga_score + wb_vote_score + ifc_score) / 3`

#### Scenario: One component is null
- **WHEN** one component score is null (e.g., insufficient UNGA history)
- **THEN** `alignment_score` is the average of the two available components, and the excluded component is noted in the output

### Requirement: Output alignment scores dataset
The system SHALL write `data/processed/alignment_scores.csv` containing: `iso3`, `country_name`, `unga_alignment_score`, `wb_vote_share_score`, `ifc_presence_score`, `alignment_score`.

#### Scenario: Alignment scores file produced
- **WHEN** the alignment scoring stage completes
- **THEN** `data/processed/alignment_scores.csv` is written with one row per country in the universe
