## ADDED Requirements

### Requirement: Choropleth map generation
The pipeline SHALL produce a self-contained interactive HTML choropleth world map at `outputs/charts/chart5_world_map.html` during Stage 4 (report generation). The map SHALL require no internet connection to render and SHALL be generated using Plotly with `locationmode="ISO-3"` and the built-in Natural Earth geometry dataset.

#### Scenario: Map generated on full pipeline run
- **WHEN** `uv run python main.py` completes successfully
- **THEN** `outputs/charts/chart5_world_map.html` exists and is a valid standalone HTML file

#### Scenario: Map generated on report-only call
- **WHEN** `generate_report()` is called with a valid `dri` DataFrame
- **THEN** `generate_world_map(dri)` is called and `chart5_world_map.html` is written

### Requirement: Gap-based choropleth color encoding
Countries SHALL be shaded by `gap_usd` using a diverging RdYlGn color scale centered at 0. Over-contributors (negative gap) SHALL render in red; countries with the largest positive gaps SHALL render in green. `zmin` SHALL be the minimum `gap_usd` value in the dataset. `zmax` SHALL be the 95th percentile of positive `gap_usd` values. `zmid` SHALL be 0 so the color midpoint aligns with the break-even point.

#### Scenario: Over-contributor renders red
- **WHEN** a country has a negative `gap_usd` (actual > target)
- **THEN** it renders in red on the map

#### Scenario: Large-gap country renders green
- **WHEN** a country has a large positive `gap_usd`
- **THEN** it renders in green, with deeper green for larger gaps

#### Scenario: Color scale anchored to 95th percentile on positive side
- **WHEN** the map is generated with a dataset containing countries with gaps ranging from −$935M to $5.8B
- **THEN** the colorbar maximum is set to the 95th percentile of positive gaps, not $5.8B

### Requirement: Hover tooltip
On cursor hover, countries with valid data SHALL display a tooltip containing: country name, gap formatted as `$X.XXB` (billions) or `$X.XXM` (millions), giving rate as a percentage, alignment score (0–100 or "N/A"), and capacity target (adjusted_target_usd formatted as `$X.XXB`/`$X.XXM` or "N/A").

#### Scenario: Hover on a country with full data
- **WHEN** the user hovers over a country that has all five fields populated
- **THEN** the tooltip displays all five fields with correct formatting

#### Scenario: Hover on a country with missing alignment score
- **WHEN** the user hovers over a country where `alignment_score` is null
- **THEN** the tooltip shows "N/A" for alignment score rather than crashing or showing NaN

### Requirement: No-data countries rendered in gray
Countries not present in `dri_output.csv` SHALL render in neutral gray via Plotly's built-in `showland=True` geography layer rather than appearing as blank/white voids.

#### Scenario: Territory not in DRI output
- **WHEN** the map is rendered and a territory has no row in `dri_output.csv`
- **THEN** that territory appears in gray on the map

### Requirement: Unmatched ISO-3 codes logged as warnings
For every `iso3` value in `dri_output.csv` that has no corresponding geometry in Plotly's Natural Earth dataset, the pipeline SHALL log a warning at the WARNING level.

#### Scenario: Unrecognized ISO-3 code in DRI output
- **WHEN** `dri_output.csv` contains an `iso3` code not recognized by Plotly's Natural Earth geometry
- **THEN** a WARNING-level log message identifying that `iso3` code is emitted

### Requirement: Main pipeline summary includes map output path
The `main.py` summary block printed after pipeline completion SHALL list `outputs/charts/chart5_world_map.html` alongside the existing output files.

#### Scenario: Summary block after successful run
- **WHEN** `uv run python main.py` completes
- **THEN** the printed summary includes the path `outputs/charts/chart5_world_map.html`
