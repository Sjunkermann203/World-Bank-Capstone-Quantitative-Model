## Why

The Donor Readiness Index (DRI) model contains seven accuracy and precision issues that undermine its credibility as an analytical tool: an unweighted global median distorts fair-share benchmarks, giving rates can exceed 1.0 without proper handling, alignment scores are computed but never incorporated into rankings, gap estimates use nominal rather than PPP-adjusted GDP, the Heckman selection probability (p_donate) is silently dropped from scoring, a chart legend bug discards series labels, and no confidence intervals are reported on gap estimates. These issues collectively mean the DRI currently ranks countries on a distorted gap index rather than a true readiness composite.

## What Changes

- Replace unweighted global median benchmark with GDP-weighted peer-group medians (High / Upper-Middle / Lower-Middle income)
- Add `peer_group` column to output; rename benchmark to reflect peer-specificity
- Cap `giving_rate` at 1.0 for segment assignment; preserve raw value as `giving_rate_raw`; add "Exceeded Target" segment; sign `gap_usd` (negative = over-contribution) and rename to `gap_usd_signed`
- **BREAKING**: Replace raw `gap_usd` ranking with composite `DRI_score = α·normalized_gap + β·alignment_score + γ·p_donate` (default α=0.5, β=0.3, γ=0.2); weights configurable via `config.yaml`
- Pull PPP-adjusted GDP (NY.GDP.MKTP.PP.KD) and GNI per capita PPP (NY.GNP.PCAP.PP.CD) from WDI; add `gap_pct_ppp_gdp` output column
- Add `gap_usd_expected = gap_usd × p_donate` as probability-adjusted gap column
- Fix Chart 3 double `ax.legend()` call — consolidate into single call after all series plotted
- Add 90% confidence intervals on gap estimates (`gap_usd_lower`, `gap_usd_upper`); display as error bars on gap chart and tooltip bands on choropleth

## Capabilities

### New Capabilities
- `peer-group-benchmark`: GDP-weighted median contribution rate computed within World Bank income-tier peer groups, replacing the global unweighted median
- `composite-dri-score`: Blended ranking score combining normalized gap, alignment score, and p_donate with configurable weights
- `ppp-gap-metrics`: PPP-adjusted GDP fetch, `gap_pct_ppp_gdp` column, and benchmark ratio recomputed against PPP GDP
- `gap-confidence-intervals`: 90% CI bounds on gap estimates derived from Heckman second-stage standard errors; error bars on charts
- `giving-rate-normalization`: Capped giving_rate for segmentation, `giving_rate_raw` preservation, "Exceeded Target" segment, signed gap column

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- `src/ingest.py`: Add WDI fetch for NY.GDP.MKTP.PP.KD and NY.GNP.PCAP.PP.CD
- `src/capacity.py`: Peer-group benchmark logic, giving_rate normalization, signed gap, PPP metrics, confidence intervals, p_donate weighting
- `src/report.py` (or equivalent scoring stage): Composite DRI score construction; Chart 3 legend fix; error bars on gap chart; choropleth tooltip updates
- `config.yaml` (new): DRI weight parameters (α, β, γ)
- `outputs/dri_output.csv`: New columns — `peer_group`, `giving_rate_raw`, `gap_usd_signed`, `gap_pct_ppp_gdp`, `gap_usd_expected`, `gap_usd_lower`, `gap_usd_upper`, `DRI_score`
