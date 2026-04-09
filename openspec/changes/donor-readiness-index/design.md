## Context

The Donor Readiness Index (DRI) is a standalone Python analytical model. There is no existing codebase to extend — this design establishes the architecture from scratch. The primary consumers are policy analysts who will run the model periodically (around IDA replenishment cycles) and interpret the output charts and tables. The model must be reproducible, auditable, and easy to re-run when source data is updated.

Key constraints:
- Data sources are semi-structured (World Bank WDI API, IMF WEO Excel/CSV, IDA contribution PDFs/CSVs)
- Country universe is ~80 nations; performance is not a concern
- Output must include both machine-readable data (CSV) and human-readable charts (PNG/SVG)
- No deployment infrastructure needed — runs locally via CLI or Jupyter notebook

## Goals / Non-Goals

**Goals:**
- Clean pipeline from raw source data → normalized per-country dataset → scores → ranked output
- Deterministic, reproducible scoring: same inputs always yield same outputs
- Four charts matching the specified output requirements
- Per-country CSV with all intermediate scores (capacity score, fiscal modifier, gap, giving rate, alignment score)
- Modular code so individual scoring components can be updated independently

**Non-Goals:**
- Real-time or automated data refresh (manual data pull per replenishment cycle is fine)
- Web application or interactive dashboard
- Forecasting or predictive modeling beyond the current-period capacity benchmark
- Coverage of non-sovereign entities or private donors

## Decisions

### D1: Python scripts with a pipeline runner (not a Jupyter-first approach)

**Decision:** Implement as a Python package with discrete pipeline stages callable from a `main.py` runner, with an optional Jupyter notebook for exploration.

**Rationale:** Scripts are easier to version-control, diff, and re-run reproducibly than notebooks. Notebooks accumulate hidden state and are harder to test. The runner calls stages in order: `ingest → score → align → report`.

**Alternative considered:** Jupyter notebook as primary artifact. Rejected because cell execution order is fragile and output cells bloat git history.

### D2: Income-tier stratification for the benchmark IDA/GDP ratio

**Decision:** Compute the median IDA/GDP ratio separately for each World Bank income group (High Income, Upper-Middle Income) among current donors, then apply the appropriate tier median as the target rate for each candidate country.

**Rationale:** A single global median would unfairly penalize lower-income emerging donors. Tiering by income group produces a fairer, peer-benchmarked target.

**Alternative considered:** Single global median across all current donors. Rejected — conflates structural differences in fiscal capacity across income levels.

### D3: Fiscal modifier as a linear scale capped at ±20%

**Decision:** Map fiscal balance as % of GDP to a modifier in the range [−20%, +20%] using a linear scale anchored at 0% fiscal balance = 0 modifier. Bounds are hard-capped.

**Rationale:** Simple, interpretable, and auditable. A ±20% cap prevents extreme fiscal outliers from dominating the score.

**Alternative considered:** Non-linear (sigmoid) modifier. Rejected — adds complexity without meaningful improvement for the ~80-country universe.

### D4: Strategic alignment as an additive composite (not a multiplier)

**Decision:** Score strategic alignment as a separate additive composite (UNGA voting + WB vote share + IFC presence), normalized 0–100. Report it alongside the capacity-based gap, not embedded in the target calculation.

**Rationale:** Keeps the economic capacity signal (gap, giving rate) clean and interpretable. Strategic alignment is an overlay for prioritization, not a modifier on the financial benchmark.

**Alternative considered:** Fold alignment into the target contribution as a multiplier. Rejected — conflates economic capacity with geopolitical factors, making the target harder to explain to stakeholders.

### D5: Static data files for IDA contribution records

**Decision:** Store IDA20 and IDA21 contribution records as CSV files in `data/raw/`. WDI and IMF WEO data are fetched via API/download at pipeline run time and cached in `data/cache/`.

**Rationale:** IDA contribution records are released as PDFs/tables; they don't change retroactively and are small enough to commit. Live API data should be re-fetched to stay current but cached locally to avoid repeated calls during development.

## Risks / Trade-offs

- **IMF WEO data format changes** → IMF occasionally restructures WEO Excel files. Mitigation: pin the WEO vintage used, document the expected column names, add a schema validation step in the ingestion stage.
- **IDA contribution records require manual extraction** → These are often released as formatted PDFs. Mitigation: maintain a clean hand-curated CSV; document the source URL and extraction date in the file header.
- **Country name mismatches across sources** → WDI, IMF, and IDA use different country name conventions. Mitigation: maintain a canonical country mapping table (`data/country_map.csv`) keyed on ISO 3166-1 alpha-3 codes.
- **Median IDA/GDP ratio sensitive to donor composition** → Adding or removing one large donor can shift the benchmark. Mitigation: log and version the donor set used for each run; include it in the output report.

## Migration Plan

Not applicable — this is a new, standalone model with no existing system to migrate from or roll back.

## Open Questions

- **UNGA voting data source**: Which dataset will be used for UNGA voting alignment scores (UN Voting Correlation dataset, UCDP, or a custom pull)? Need to confirm source and reference year.
- **IFC presence definition**: Binary (has IFC portfolio ≥ $0?) or continuous (IFC exposure as % of GDP)? To be decided based on data availability.
- **Replenishment cycle scope**: Should the model score against IDA21 actuals, IDA20 actuals, or an average of both? Affects gap calculation for countries that joined mid-cycle.
