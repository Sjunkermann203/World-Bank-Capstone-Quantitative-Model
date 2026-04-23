## Context

The DRI pipeline is a Python data pipeline with four stages: ingest → capacity → alignment → report. The `capacity.py` stage is the primary site of scoring logic (benchmark_ratio, gap_usd, giving_rate, Heckman). The `report.py` stage assembles the final ranked output and produces charts. Currently, the final ranking is derived solely from `gap_usd`, and several intermediate computations contain correctness bugs. The model produces a `dri_output.csv` and five HTML/PNG charts.

Key constraints:
- World Bank WDI data is fetched via `wbgapi` (already a dependency); new indicators can be added to the existing fetch calls.
- The Heckman two-stage estimator is implemented in `capacity.py` using `statsmodels`; standard errors are already available on the fitted model object.
- All pipeline configuration currently lives in-file; we will introduce a `config.yaml` for DRI weights to avoid requiring code changes for sensitivity analysis.

## Goals / Non-Goals

**Goals:**
- Fix benchmark_ratio to use GDP-weighted peer-group median (per income tier)
- Fix giving_rate segmentation for over-contributors; add signed gap column
- Introduce composite DRI_score (gap + alignment + p_donate) with configurable weights
- Add PPP-adjusted GDP fetch and `gap_pct_ppp_gdp` column
- Propagate p_donate into scoring and add `gap_usd_expected`
- Fix Chart 3 double-legend bug
- Add 90% CI columns and error bars on gap chart

**Non-Goals:**
- Changing the underlying data sources (WDI is the authoritative source)
- Re-implementing the Heckman estimator (use existing statsmodels fit)
- Modifying the alignment scoring methodology
- Adding new chart types beyond bug fixes and error bars

## Decisions

### D1 — config.yaml for DRI weights
**Decision**: Introduce `config.yaml` at the project root with a `dri_weights` section. Load it in `report.py` (or a new `score.py`) at pipeline start; fall back to defaults (α=0.5, β=0.3, γ=0.2) if the file is absent.

**Rationale**: Researchers need to run robustness checks by varying weights without touching code. YAML is readable and already common in Python data projects. Alternatives considered: CLI flags (poor for reproducibility), in-code constants (require code edits), environment variables (awkward for tuples of values).

### D2 — Where composite scoring lives
**Decision**: Add a new `score.py` stage between `alignment.py` and `report.py`, or extend `report.py`. Given the pipeline is already four stages and the scoring is tightly coupled to the final merge, we will add a `compute_dri_score()` function in `report.py` (or a thin `score.py` imported by `report.py`).

**Rationale**: Minimizes structural changes while isolating composite scoring logic.

### D3 — Signed gap column naming
**Decision**: Add `gap_usd_signed` as a new column (negative = over-contribution). Retain `gap_usd` as a non-negative absolute value for backwards compatibility with existing chart code; update charts to use `gap_usd_signed` where directionality matters.

**Rationale**: Breaking rename of `gap_usd` would require updating all chart and downstream references. Adding `gap_usd_signed` is additive. Document the distinction clearly in column metadata.

### D4 — Peer-group benchmark computation
**Decision**: Use World Bank `income_level` classifications pulled from `wbgapi.economy` metadata (already fetched during ingest). Sort donors within each peer group by contribution-per-GDP, then compute weighted 50th percentile using GDP as weights (numpy-based weighted quantile).

**Rationale**: `wbgapi` exposes income level natively; no additional API call needed. Weighted median is a standard robust central tendency measure.

### D5 — Confidence interval source
**Decision**: Use the HC3 robust standard errors from the Heckman second-stage OLS on the predicted log-contribution coefficient. Propagate SE to gap_usd via: `gap_usd ± 1.645 × se_predicted × gdp`. This is an approximation (it ignores SE on the selection equation), acknowledged in output metadata.

**Rationale**: Full two-stage SE propagation requires bootstrapping; acceptable approximation for policy-analysis use.

## Risks / Trade-offs

- **PPP GDP coverage gaps** → Some countries (especially small/fragile states) may have missing PPP GDP data. Mitigation: fall back to nominal GDP for those countries; flag missing values in output with a `ppp_data_available` boolean column.
- **Weighted median edge cases** → A peer group with only one or two donors produces a degenerate weighted median equal to that single value. Mitigation: log a warning when peer group has fewer than 3 donors; fall back to global weighted median.
- **Composite score sensitivity** → Changing weights (α, β, γ) can significantly reorder rankings. Mitigation: document default weights prominently; include a sensitivity table in output metadata (future work, out of scope here).
- **Chart error bars** → Very wide CIs may make the gap chart unreadable. Mitigation: clip error bar display at ±200% of point estimate; note clipping in chart subtitle.

## Migration Plan

1. Run `uv sync` after adding any new dependencies (none expected; all libraries already present).
2. Add `config.yaml` to project root; existing runs without it use defaults.
3. New output columns are additive; existing downstream consumers reading `gap_usd` are unaffected.
4. `gap_usd_signed` replaces the semantic role of `gap_usd` for ranking; users relying on sort order should switch to `DRI_score`.
5. No database migrations; all outputs are flat files.

## Open Questions

- Should `gap_usd` be preserved exactly as today (absolute, non-negative) or should we rename the old column to `gap_usd_abs`? Current decision: preserve `gap_usd` as-is, add `gap_usd_signed`.
- Should the composite DRI_score be inverted so that higher = more ready (larger gap + high alignment + high p_donate)? Needs confirmation from research team. Assumed: higher DRI_score = higher priority target.
