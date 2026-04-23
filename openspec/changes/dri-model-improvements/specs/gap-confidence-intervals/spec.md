## ADDED Requirements

### Requirement: 90% confidence interval columns on gap estimates
The pipeline SHALL extract the HC3 robust standard error on the predicted log-contribution from the Heckman second-stage OLS fit (`se_predicted`). It SHALL compute 90% confidence bounds on `gap_usd_signed` as:
- `gap_usd_lower = gap_usd_signed - 1.645 × se_predicted × gdp`
- `gap_usd_upper = gap_usd_signed + 1.645 × se_predicted × gdp`

Both columns SHALL appear in `dri_output.csv`. A metadata note SHALL document that these bounds are an approximation that does not account for uncertainty in the selection equation.

#### Scenario: CI columns present in output
- **WHEN** the pipeline runs successfully and the Heckman model fits without error
- **THEN** `gap_usd_lower` and `gap_usd_upper` are non-null for all countries where `p_donate` is available

#### Scenario: Heckman model fit fails
- **WHEN** the Heckman second-stage OLS cannot be fit (e.g., insufficient data)
- **THEN** `gap_usd_lower` and `gap_usd_upper` are NaN for all countries and a warning is logged

#### Scenario: CI bounds direction
- **WHEN** a country has a positive gap_usd_signed (shortfall)
- **THEN** `gap_usd_lower < gap_usd_signed < gap_usd_upper`

### Requirement: Error bars on gap bar chart
The gap bar chart (Chart 1) SHALL display 90% confidence interval error bars derived from `gap_usd_lower` and `gap_usd_upper`. Error bars SHALL be clipped at ±200% of the point estimate to prevent unreadable charts. A subtitle note SHALL indicate clipping when it occurs.

#### Scenario: Error bars rendered on chart
- **WHEN** Chart 1 is generated and CI columns are available
- **THEN** each bar displays an error bar spanning [gap_usd_lower, gap_usd_upper]

#### Scenario: Error bar clipping
- **WHEN** an error bar half-width exceeds 200% of the absolute point estimate
- **THEN** the error bar is clipped to ±200% of the point estimate and a note appears in the chart subtitle

### Requirement: Fix Chart 3 double legend bug
The chart generation code for Chart 3 (capacity vs giving rate) SHALL call `ax.legend()` exactly once, after all series have been plotted. The intermediate `ax.legend()` call SHALL be removed. The consolidated call SHALL use `handles, labels = ax.get_legend_handles_labels()` to collect all series before rendering.

#### Scenario: Chart 3 legend contains all series
- **WHEN** Chart 3 is generated with multiple donor-segment series
- **THEN** the legend contains an entry for every series that was plotted, with no entries missing or duplicated

#### Scenario: No intermediate legend call
- **WHEN** the chart generation code is inspected
- **THEN** there is exactly one `ax.legend()` call in the Chart 3 generation function

### Requirement: Choropleth tooltip confidence band
The world-map choropleth (Chart 5) tooltip SHALL display `gap_usd_lower` and `gap_usd_upper` as a confidence range alongside the point estimate `gap_usd_signed`, formatted as: "Gap: $X.XB [$Y.YB – $Z.ZB 90% CI]".

#### Scenario: Tooltip shows CI range
- **WHEN** a user hovers over a country on the choropleth
- **THEN** the tooltip displays the point estimate and 90% CI bounds in the specified format

#### Scenario: Missing CI data in tooltip
- **WHEN** CI columns are NaN for a country
- **THEN** the tooltip displays only the point estimate without a CI range
