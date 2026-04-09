## ADDED Requirements

### Requirement: Merge scores into final DRI output dataset
The system SHALL join `capacity_scores.csv` and `alignment_scores.csv` on `iso3` to produce a unified per-country output dataset containing all scoring fields. The merged dataset SHALL be ranked by `gap_usd` descending (largest gap first). The result SHALL be written to `outputs/dri_output.csv`.

#### Scenario: Successful merge and ranking
- **WHEN** both `capacity_scores.csv` and `alignment_scores.csv` exist and share common `iso3` values
- **THEN** the system produces `outputs/dri_output.csv` with one row per country, sorted by `gap_usd` descending, containing all capacity and alignment fields

#### Scenario: Country present in capacity scores but missing from alignment scores
- **WHEN** a country has a capacity score but no alignment score
- **THEN** alignment fields for that country are null in the output; the country is still included

### Requirement: Generate Chart 1 — Gap ranking bar chart
The system SHALL generate a horizontal bar chart showing the top N countries (default N=30) by `gap_usd` descending. Bars SHALL be color-coded by income group. The chart SHALL be saved as `outputs/charts/chart1_gap_ranking.png` at a minimum resolution of 150 DPI.

#### Scenario: Gap chart produced
- **WHEN** the reporting stage runs with a valid DRI output dataset
- **THEN** `outputs/charts/chart1_gap_ranking.png` is created showing the top 30 countries by gap

#### Scenario: Fewer than N countries with valid gap data
- **WHEN** fewer than N countries have a non-null `gap_usd`
- **THEN** the chart shows all countries with valid gap data and logs the actual count

### Requirement: Generate Chart 2 — Giving rate bar chart
The system SHALL generate a horizontal bar chart showing `giving_rate` for all scored countries, sorted ascending (lowest giving rate first). A vertical reference line SHALL be drawn at `giving_rate = 1.0`. Countries above 1.0 SHALL be visually distinguished. The chart SHALL be saved as `outputs/charts/chart2_giving_rate.png` at minimum 150 DPI.

#### Scenario: Giving rate chart produced
- **WHEN** the reporting stage runs with valid giving rate data
- **THEN** `outputs/charts/chart2_giving_rate.png` is created with a reference line at 1.0 and countries color-coded by above/below benchmark

### Requirement: Generate Chart 3 — Capacity vs. giving rate scatter plot
The system SHALL generate a scatter plot with `adjusted_target_usd` (or a normalized capacity score) on the x-axis and `giving_rate` on the y-axis. Each point represents one country, labeled with its ISO3 code. A horizontal reference line at `giving_rate = 1.0` and a vertical reference line at the median capacity score SHALL be drawn. The chart SHALL be saved as `outputs/charts/chart3_capacity_vs_giving_rate.png` at minimum 150 DPI.

#### Scenario: Capacity vs. giving rate scatter produced
- **WHEN** the reporting stage runs with valid capacity and giving rate data
- **THEN** `outputs/charts/chart3_capacity_vs_giving_rate.png` is created with per-country ISO3 labels and reference lines

### Requirement: Generate Chart 4 — Alignment vs. gap scatter plot
The system SHALL generate a scatter plot with `alignment_score` on the x-axis and `gap_usd` on the y-axis. Each point represents one country, labeled with its ISO3 code. Points SHALL be sized proportionally to `gdp_usd`. The chart SHALL be saved as `outputs/charts/chart4_alignment_vs_gap.png` at minimum 150 DPI.

#### Scenario: Alignment vs. gap scatter produced
- **WHEN** the reporting stage runs with valid alignment and gap data
- **THEN** `outputs/charts/chart4_alignment_vs_gap.png` is created with ISO3 labels and GDP-proportional point sizes

#### Scenario: Country missing alignment score
- **WHEN** a country has a gap but no alignment score
- **THEN** the country is excluded from Chart 4 only; it remains in Charts 1, 2, and 3

### Requirement: Ensure output directories exist before writing
The system SHALL create `outputs/` and `outputs/charts/` directories if they do not already exist before writing any output files.

#### Scenario: Output directories created on first run
- **WHEN** the reporting stage runs and `outputs/charts/` does not exist
- **THEN** the directories are created automatically and files are written successfully
