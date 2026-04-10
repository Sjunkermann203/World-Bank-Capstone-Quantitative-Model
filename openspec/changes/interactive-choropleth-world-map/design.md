## Context

The DRI pipeline (4 stages: ingest → capacity → alignment → report) currently outputs static PNG charts via matplotlib/seaborn. Stage 4 (`src/report.py`) already reads `outputs/dri_output.csv` and generates four charts. The new world map is a fifth output from the same stage using the same merged `dri` DataFrame. Plotly is not yet a dependency; the project uses `uv` for package management.

## Goals / Non-Goals

**Goals:**
- Produce a single self-contained HTML file renderable offline in any modern browser
- Color countries by `gap_usd` using a YlOrRd choropleth with a 95th-percentile cap on `zmax`
- Show a hover tooltip with: country name, gap (formatted $X.XXB/$X.XXM), giving rate (%), alignment score, and capacity score (target_usd-based)
- Log a warning for any `iso3` code in `dri_output.csv` that has no matching Plotly geometry
- Integrate cleanly into `generate_report()` with no side effects on existing outputs

**Non-Goals:**
- Click-to-lock tooltip (listed as optional in spec; deferred — Plotly's native hover covers the core need without custom JS)
- Server-side rendering or live data updates
- Modifying existing chart functions or CSV outputs

## Decisions

**Plotly `go.Choropleth` over `px.choropleth`**
`go.Choropleth` gives direct control over `colorscale`, `zmin`/`zmax`, `hovertemplate`, and `colorbar` ticks. `px.choropleth` is a thin wrapper that obscures these options and makes custom tooltip formatting harder. Use `go.Figure(go.Choropleth(...))`.

**`locationmode="ISO-3"` with Plotly's built-in Natural Earth geometries**
No external GeoJSON file needed — Plotly bundles Natural Earth at 110m resolution. Countries missing from `dri_output.csv` automatically render in gray when `showland=True` on the geo layout.

**`zmax` = 95th percentile of positive `gap_usd`**
Without a cap, China's $5.8B gap would compress all other countries into the low end of the color scale. The 95th percentile preserves visible variation for the majority while still rendering outliers at maximum color saturation (not clipped from the map — just saturated).

**`include_plotlyjs="cdn"` vs `"inline"`**
The spec requires offline rendering → use `include_plotlyjs=True` (default, inlines the ~3MB JS bundle). `"cdn"` would fail without internet. The resulting HTML is ~3–4 MB, acceptable for a pipeline output.

**Tooltip format via `hovertemplate`**
Plotly's `customdata` array lets us attach arbitrary columns per country. `hovertemplate` references them with `%{customdata[N]}`. Formatting (B/M suffix for gap) is done in Python before passing to Plotly, stored as a pre-formatted string column.

## Risks / Trade-offs

- **Plotly bundle size (~3 MB HTML)**: Larger than PNG charts but acceptable for an interactive output. → No mitigation needed; clearly documented.
- **ISO-3 code mismatches**: Some DRI `iso3` codes (e.g., territories, breakaway states) may not exist in Natural Earth. → Log warnings per the spec; these countries simply won't appear on the map.
- **`uv add plotly` changes lock file**: Minor; standard dependency workflow. → Committed alongside code changes.

## Migration Plan

1. `uv add plotly` — adds dependency, updates `pyproject.toml` and `uv.lock`
2. Add `generate_world_map(dri)` to `src/report.py`
3. Call it from `generate_report()` after `chart4_alignment_vs_gap()`
4. Update `main.py` summary block to list the new output path
5. Run `uv run python main.py` to verify end-to-end; open the HTML in a browser to confirm interactivity
