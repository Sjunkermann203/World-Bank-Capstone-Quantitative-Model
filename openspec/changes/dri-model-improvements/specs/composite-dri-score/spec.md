## ADDED Requirements

### Requirement: Configurable DRI weights via config.yaml
The pipeline SHALL read DRI composite weights (α, β, γ) from a `config.yaml` file at the project root under a `dri_weights` key. If `config.yaml` is absent or the key is missing, the pipeline SHALL fall back to default weights: α=0.5, β=0.3, γ=0.2. Weights SHALL sum to 1.0; the pipeline SHALL raise a ValueError if loaded weights do not sum to 1.0 (within floating-point tolerance of 1e-6).

#### Scenario: config.yaml present with valid weights
- **WHEN** `config.yaml` exists and `dri_weights: {alpha: 0.4, beta: 0.4, gamma: 0.2}` is set
- **THEN** the pipeline uses those weights and they sum to 1.0 without error

#### Scenario: config.yaml absent
- **WHEN** no `config.yaml` file exists at the project root
- **THEN** the pipeline uses default weights α=0.5, β=0.3, γ=0.2 and logs a message indicating defaults are in use

#### Scenario: Weights do not sum to 1.0
- **WHEN** `config.yaml` specifies weights that sum to a value outside [1.0 ± 1e-6]
- **THEN** the pipeline raises a ValueError with a descriptive message before any scoring occurs

### Requirement: Component normalization to [0, 1]
Before combining into DRI_score, each component SHALL be normalized to the [0, 1] range using min-max normalization across all countries in the dataset. Normalization SHALL be applied independently to: (1) `gap_usd_signed` transformed to a gap urgency score (larger positive gap → higher score), (2) `alignment_score`, and (3) `p_donate`.

#### Scenario: Normalization of gap urgency
- **WHEN** gap_usd_signed values span a positive range
- **THEN** the country with the largest shortfall receives a normalized gap score of 1.0 and the smallest receives 0.0

#### Scenario: Single-country edge case
- **WHEN** all countries have identical values for a component
- **THEN** the normalized component is set to 0.5 for all countries (avoid division by zero)

### Requirement: Composite DRI_score computation
The pipeline SHALL compute `DRI_score = α × norm_gap + β × alignment_score_norm + γ × p_donate_norm` for every country. Higher DRI_score indicates a higher-priority engagement target (large unrealized gap, strong alignment, high likelihood of donating).

#### Scenario: DRI_score range
- **WHEN** the pipeline completes scoring
- **THEN** all `DRI_score` values are in the range [0, 1]

#### Scenario: DRI_score in output
- **WHEN** the pipeline writes `dri_output.csv`
- **THEN** every row contains a non-null `DRI_score` value

### Requirement: DRI_score used for ranking
The final output SHALL be sorted by `DRI_score` descending. The `rank` column SHALL reflect DRI_score rank order, not raw `gap_usd` order.

#### Scenario: Ranking by DRI_score
- **WHEN** two countries have different DRI_score values
- **THEN** the country with the higher DRI_score receives a lower (better) rank number

### Requirement: gap_usd_expected probability-adjusted gap
The pipeline SHALL compute `gap_usd_expected = gap_usd_signed × p_donate` as a probability-weighted expected unrealized contribution. This column SHALL appear in `dri_output.csv`.

#### Scenario: gap_usd_expected for low p_donate country
- **WHEN** a country has a large gap_usd_signed but low p_donate (e.g., 0.1)
- **THEN** gap_usd_expected is substantially lower than gap_usd_signed, reflecting low donation likelihood
