## 1. Dependency Setup

- [ ] 1.1 Run `uv add plotly` to add plotly as a project dependency
- [ ] 1.2 Verify `pyproject.toml` and `uv.lock` updated correctly

## 2. Core Map Function

- [ ] 2.1 Add `generate_world_map(dri)` function to `src/report.py` with `import plotly.graph_objects as go`
- [ ] 2.2 Pre-format tooltip columns: gap as `$X.XXB`/`$X.XXM`, giving rate as `X.X%`, alignment score as `X.X` or `N/A`, adjusted_target as `$X.XXB`/`$X.XXM` or `N/A`
- [ ] 2.3 Clamp negative `gap_usd` values to 0 for `zmin` compliance; compute `zmax` as 95th percentile of positive gap values
- [ ] 2.4 Build `go.Choropleth` with `locationmode="ISO-3"`, `colorscale="YlOrRd"`, `zmin=0`, computed `zmax`, and `customdata` array for tooltip fields
- [ ] 2.5 Set `hovertemplate` to display all five fields (country name, gap, giving rate, alignment score, capacity target)
- [ ] 2.6 Configure `go.Figure` layout: `geo=dict(showland=True, landcolor="lightgray", showframe=False, showcoastlines=True)`, title, and colorbar with labeled breakpoints
- [ ] 2.7 Write output with `fig.write_html(path, include_plotlyjs=True)` to produce a self-contained offline HTML file

## 3. Unmatched ISO-3 Warning

- [ ] 3.1 After building the figure, identify any `iso3` codes in the input `dri` DataFrame that Plotly's Natural Earth geometry does not recognize and log a WARNING for each

## 4. Pipeline Integration

- [ ] 4.1 Call `generate_world_map(dri)` from `generate_report()` in `src/report.py`, after the existing `chart4_alignment_vs_gap()` call
- [ ] 4.2 Update the `main.py` summary block to include `outputs/charts/chart5_world_map.html` in the listed output files

## 5. Verification

- [ ] 5.1 Run `uv run python main.py` and confirm no errors; verify `outputs/charts/chart5_world_map.html` is created
- [ ] 5.2 Open `chart5_world_map.html` in a browser and confirm choropleth shading is visible, colorbar is present, and hovering a country shows all five tooltip fields
- [ ] 5.3 Confirm over-contributor countries (UK, Japan, Germany, etc.) render in the lightest color rather than being excluded
- [ ] 5.4 Confirm countries not in `dri_output.csv` appear in gray rather than blank
- [ ] 5.5 Confirm existing charts (chart1–chart4 PNGs) and `dri_output.csv` are unchanged
