## Why

The DRI pipeline produces tabular and static chart outputs that are hard to scan globally — users must read ranked lists or individual bar charts to understand geographic distribution. An interactive choropleth map gives stakeholders an immediate spatial view of where contribution gaps are largest, enabling faster prioritization and communication.

## What Changes

- Add a new `generate_world_map(dri)` function in `src/report.py` that produces a self-contained interactive HTML choropleth map
- Wire the new function into `generate_report()` so it runs as part of the existing Stage 4 pipeline
- Update the `main.py` summary block to include the new output path
- Add `plotly` as a project dependency

## Capabilities

### New Capabilities

- `world-map`: Interactive choropleth map of all countries shaded by `gap_usd`, with hover tooltips showing country name, contribution gap, capacity score, alignment score, and giving rate. Output is a standalone HTML file at `outputs/charts/chart5_world_map.html`.

### Modified Capabilities

## Impact

- **`src/report.py`**: New `generate_world_map()` function added; `generate_report()` updated to call it
- **`main.py`**: Summary block updated to list `outputs/charts/chart5_world_map.html`
- **`pyproject.toml` / `uv.lock`**: `plotly` added as a dependency
- No changes to existing chart functions, capacity/alignment scoring, or CSV outputs
