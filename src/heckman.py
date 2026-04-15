"""
Heckman selection model for the Donor Readiness Index.

Replaces the rule-based capacity.py with a two-stage Heckman estimator that
corrects for selection bias in IDA contribution data and produces predictions
for non-donors as well as donors.

Stage 1 — Probit (selection equation):
    Models whether a country donates in a given replenishment round.
    Variables: log_gdp_per_capita, dac_member, un_voting_align, trade_openness,
               gov_effectiveness, peer_donor

Stage 2 — OLS with IMR correction (outcome equation):
    Models log-donation amount conditional on donating.
    Variables: log_gdp_level, fiscal_balance_pct_gdp, ida_vote_share_lag,
               trade_exposure_ida, log_donation_lag, us_eu_ally,
               sovereign_credit_rating, imr, round dummies

Exclusion restrictions (Stage 1 only):
    un_voting_align, peer_donor, dac_member

Reads:  data/raw/heckman_panel.csv
        data/processed/master.csv (passed in as `master` DataFrame)
Writes: data/processed/capacity_scores.csv
        outputs/heckman_diagnostics.txt
        outputs/charts/heckman_residuals.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ingest import SchemaValidationError

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
CHARTS = OUTPUTS / "charts"

PANEL_PATH = DATA_RAW / "heckman_panel.csv"
CAPACITY_SCORES_PATH = DATA_PROCESSED / "capacity_scores.csv"
DIAGNOSTICS_PATH = OUTPUTS / "heckman_diagnostics.txt"
RESIDUALS_PLOT_PATH = CHARTS / "heckman_residuals.png"

PANEL_REQUIRED_COLUMNS = {
    "country_iso3", "replenishment_round", "donate_dummy", "donation_usd",
    "log_gdp_per_capita", "dac_member", "un_voting_align", "trade_openness",
    "gov_effectiveness", "peer_donor", "log_gdp_level", "fiscal_balance_pct_gdp",
    "ida_vote_share_lag", "trade_exposure_ida", "log_donation_lag",
    "us_eu_ally", "sovereign_credit_rating",
}

STAGE1_VARS = [
    "log_gdp_per_capita", "dac_member", "un_voting_align",
    "trade_openness", "gov_effectiveness", "peer_donor",
]

STAGE2_VARS = [
    "log_gdp_level", "fiscal_balance_pct_gdp", "ida_vote_share_lag",
    "trade_exposure_ida", "log_donation_lag", "us_eu_ally",
    "sovereign_credit_rating",
]

STAGE2_CONTINUOUS = [
    "log_gdp_level", "fiscal_balance_pct_gdp", "ida_vote_share_lag",
    "trade_exposure_ida", "log_donation_lag", "sovereign_credit_rating",
]

CREDIT_RATING_MAP = {
    "AAA": 20, "AA+": 19, "AA": 18, "AA-": 17,
    "A+": 16, "A": 15, "A-": 14,
    "BBB+": 13, "BBB": 12, "BBB-": 11,
    "BB+": 10, "BB": 9, "BB-": 8,
    "B+": 7, "B": 6, "B-": 5,
    "CCC+": 4, "CCC": 3, "CCC-": 2, "D": 1,
}

TRAIN_ROUNDS = {f"IDA{i}" for i in range(1, 18)}
TEST_ROUNDS = {"IDA18", "IDA19", "IDA20"}


# ---------------------------------------------------------------------------
# Panel ingestion
# ---------------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    """Load and validate heckman_panel.csv."""
    panel = pd.read_csv(PANEL_PATH, comment="#")
    panel.columns = panel.columns.str.strip().str.lower()

    missing = PANEL_REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise SchemaValidationError(
            f"heckman_panel.csv is missing required columns: {sorted(missing)}"
        )

    panel["country_iso3"] = panel["country_iso3"].str.strip().str.upper()
    panel["replenishment_round"] = panel["replenishment_round"].str.strip().str.upper()

    dupes = panel[panel.duplicated(subset=["country_iso3", "replenishment_round"], keep=False)]
    if not dupes.empty:
        pairs = dupes[["country_iso3", "replenishment_round"]].drop_duplicates().values.tolist()
        raise ValueError(f"Duplicate (country_iso3, replenishment_round) pairs found: {pairs}")

    logger.info("Panel loaded: %d observations", len(panel))
    return panel


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _encode_credit_ratings(panel: pd.DataFrame) -> pd.DataFrame:
    """Convert sovereign_credit_rating strings to ordinal integers."""
    if panel["sovereign_credit_rating"].dtype == object:
        panel["sovereign_credit_rating"] = panel["sovereign_credit_rating"].map(
            CREDIT_RATING_MAP
        )
    return panel


def preprocess_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing transformations to the panel."""
    panel = panel.copy()

    # Log-transform raw donation_usd for donors (if log_donation_usd not already present)
    if "log_donation_usd" not in panel.columns:
        panel["log_donation_usd"] = np.where(
            panel["donate_dummy"] == 1,
            np.log(panel["donation_usd"].clip(lower=1)),
            np.nan,
        )

    # Credit rating encoding
    panel = _encode_credit_ratings(panel)

    # log_donation_lag: set to 0 for first-time donors / non-donors with no prior record
    panel["log_donation_lag"] = panel["log_donation_lag"].fillna(0.0)

    return panel


def split_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split panel into train (IDA1–IDA17) and test (IDA18–IDA20) by round label."""
    train = panel[panel["replenishment_round"].isin(TRAIN_ROUNDS)].copy()
    test = panel[panel["replenishment_round"].isin(TEST_ROUNDS)].copy()
    logger.info("Split: %d train obs, %d test obs", len(train), len(test))
    return train, test


def standardize(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fit standardization on train, apply to both train and test.

    Returns (train_scaled, test_scaled, scaler_params) where scaler_params is
    {col: (mean, std)} for inverse transformation.
    """
    scaler_params: dict[str, tuple[float, float]] = {}
    train = train.copy()
    test = test.copy()
    for col in cols:
        if col not in train.columns:
            continue
        mu = float(train[col].mean())
        sigma = float(train[col].std())
        if sigma == 0:
            sigma = 1.0
        train[col] = (train[col] - mu) / sigma
        test[col] = (test[col] - mu) / sigma
        scaler_params[col] = (mu, sigma)
    return train, test, scaler_params


# ---------------------------------------------------------------------------
# Stage 1: Probit selection model
# ---------------------------------------------------------------------------

def fit_stage1(train: pd.DataFrame) -> tuple:
    """
    Fit probit on full training sample (all rounds, all countries).

    Returns (result, imr_train) where imr_train is a Series of inverse Mills
    ratios indexed by the training DataFrame's index.
    """
    stage1_data = train[STAGE1_VARS].fillna(0).astype(float)

    # Drop near-zero-variance columns — they make the Hessian singular
    low_var = [c for c in stage1_data.columns if stage1_data[c].std() < 0.01]
    if low_var:
        logger.warning("Dropping near-zero-variance Stage 1 variables: %s", low_var)
        stage1_data = stage1_data.drop(columns=low_var)

    X = sm.add_constant(stage1_data)
    y = train["donate_dummy"].astype(float)

    probit_model = sm.Probit(y, X)
    result = probit_model.fit(disp=False, maxiter=200)

    if not result.mle_retvals.get("converged", True):
        logger.warning("Stage 1 probit did not converge — results may be unreliable")

    # IMR = φ(Xβ) / Φ(Xβ)
    xb = result.predict(X, which="linear")
    phi = scipy_stats.norm.pdf(xb)
    Phi = scipy_stats.norm.cdf(xb)
    imr = phi / np.where(Phi > 1e-10, Phi, 1e-10)
    imr_series = pd.Series(imr, index=train.index, name="imr")

    logger.info("Stage 1 probit fitted. Pseudo-R²: %.4f", result.prsquared)
    return result, imr_series


# ---------------------------------------------------------------------------
# Stage 2: OLS outcome model
# ---------------------------------------------------------------------------

class _RobustResultWrapper:
    """Wraps a statsmodels HC3 result to ensure params/pvalues are pandas Series."""

    def __init__(self, result, param_names):
        self._result = result
        self.params = pd.Series(np.asarray(result.params), index=param_names)
        self.pvalues = pd.Series(np.asarray(result.pvalues), index=param_names)

    def __getattr__(self, name):
        return getattr(self._result, name)


def _add_round_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add replenishment_round dummies, dropping the most common as reference."""
    dummies = pd.get_dummies(df["replenishment_round"], prefix="round", drop_first=True)
    return pd.concat([df, dummies], axis=1)


def fit_stage2(train: pd.DataFrame, imr_train: pd.Series) -> tuple:
    """
    Fit OLS on donor subsample with IMR correction and round dummies.

    Returns (result, use_robust, round_dummy_cols).
    """
    donors = train[train["donate_dummy"] == 1].copy()
    donors["imr"] = imr_train.reindex(donors.index)
    donors = _add_round_dummies(donors)

    round_cols = [c for c in donors.columns if c.startswith("round_")]
    all_vars = STAGE2_VARS + ["imr"] + round_cols
    X = sm.add_constant(donors[all_vars].fillna(0).astype(float))
    y = donors["log_donation_usd"].astype(float)

    # Initial OLS to get residuals for BP test
    ols = sm.OLS(y, X).fit()
    bp_lm, bp_pval, _, _ = het_breuschpagan(ols.resid, X)
    use_robust = bp_pval < 0.05

    if use_robust:
        hc3 = ols.get_robustcov_results(cov_type="HC3")
        result = _RobustResultWrapper(hc3, X.columns)
        logger.info("Breusch-Pagan p=%.4f — using robust standard errors (HC3)", bp_pval)
    else:
        result = ols
        logger.info("Breusch-Pagan p=%.4f — using standard OLS standard errors", bp_pval)

    logger.info("Stage 2 OLS fitted on %d donor observations. R²=%.4f", len(donors), result.rsquared)
    return result, use_robust, round_cols, bp_pval


# ---------------------------------------------------------------------------
# MLE Heckman for comparison
# ---------------------------------------------------------------------------

def fit_mle_heckman(train: pd.DataFrame) -> object:
    """
    Fit statsmodels Heckman MLE variant.

    Returns the fitted result (or None if statsmodels Heckman is unavailable).
    """
    try:
        from statsmodels.duration.hazard_regression import PHReg  # noqa: F401
    except ImportError:
        pass

    # statsmodels 0.14+ has experimental Heckman in sandbox
    try:
        from statsmodels.regression.linear_model import OLS as _OLS  # noqa: F401
        # Use manual two-step as the "MLE" proxy via statsmodels Heckman if available
        from statsmodels.treatment.treatment_effects import Heckman as StatsHeckman
        donors = train[train["donate_dummy"] == 1].copy()
        X_out = sm.add_constant(donors[STAGE2_VARS].fillna(0))
        X_sel = sm.add_constant(train[STAGE1_VARS].fillna(0))
        model = StatsHeckman(
            endog=donors["log_donation_usd"],
            exog=X_out,
            endog_select=train["donate_dummy"],
            exog_select=X_sel,
        )
        result = model.fit(method="mle", disp=False)
        logger.info("MLE Heckman fitted successfully")
        return result
    except Exception as exc:
        logger.warning("MLE Heckman fitting failed or unavailable: %s — skipping MLE comparison", exc)
        return None


def check_coefficient_divergence(
    stage2_result,
    mle_result,
    vars_to_check: list[str],
) -> None:
    """Log a warning for any coefficient differing >20% between two-step and MLE."""
    if mle_result is None:
        return
    for var in vars_to_check:
        try:
            ts_coef = stage2_result.params.get(var, None)
            mle_coef = mle_result.params.get(var, None)
            if ts_coef is None or mle_coef is None:
                continue
            if abs(ts_coef) < 1e-10:
                continue
            pct_change = abs((mle_coef - ts_coef) / ts_coef)
            if pct_change > 0.20:
                logger.warning(
                    "Coefficient divergence >20%% for %s: two-step=%.4f, MLE=%.4f",
                    var, ts_coef, mle_coef,
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _compute_imr_for_all(stage1_result, panel: pd.DataFrame) -> pd.Series:
    """Compute IMR for any set of observations using Stage 1 coefficients."""
    # has_constant='add' prevents sm from skipping the intercept when a column
    # in the data happens to be constant (e.g. peer_donor=1 at prediction time)
    X = sm.add_constant(panel[STAGE1_VARS].fillna(0).astype(float), has_constant="add")
    xb = stage1_result.predict(X, which="linear")
    phi = scipy_stats.norm.pdf(xb)
    Phi = scipy_stats.norm.cdf(xb)
    imr = phi / np.where(Phi > 1e-10, Phi, 1e-10)
    return pd.Series(imr, index=panel.index, name="imr")


def predict_all(
    stage1_result,
    stage2_result,
    panel_latest: pd.DataFrame,
    stage2_residuals: pd.Series,
    round_cols: list[str],
) -> pd.DataFrame:
    """
    Generate predictions for all countries in panel_latest (most recent round).

    Returns a DataFrame with p_donate, imr, pred_log_donation,
    pred_donation_usd, expected_contribution.
    """
    out = panel_latest.copy()

    # Stage 1: p_donate
    X1 = sm.add_constant(out[STAGE1_VARS].fillna(0).astype(float), has_constant="add")
    out["p_donate"] = stage1_result.predict(X1)

    # IMR for Stage 2 prediction
    out["imr"] = _compute_imr_for_all(stage1_result, out)

    # Stage 2: pred_log_donation
    # Add round dummies matching training columns (fill zeros for missing rounds)
    for col in round_cols:
        if col not in out.columns:
            out[col] = 0
    all_vars = STAGE2_VARS + ["imr"] + round_cols
    X2 = sm.add_constant(out[all_vars].fillna(0).astype(float), has_constant="add")
    out["pred_log_donation"] = stage2_result.predict(X2)

    # Duan smearing correction
    smearing_factor = float(np.exp(stage2_residuals).mean())
    out["pred_donation_usd"] = np.exp(out["pred_log_donation"]) * smearing_factor

    # Expected contribution: E[donation] = P(donate) × pred_donation_usd
    out["expected_contribution"] = out["p_donate"] * out["pred_donation_usd"]

    return out


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Assign donor_segment labels based on p_donate and gap."""
    df = df.copy()
    segments = []
    for _, row in df.iterrows():
        is_donor = row.get("is_current_donor", 0) == 1
        expected = row.get("expected_contribution", None)
        actual = row.get("actual_contribution_usd", 0.0) or 0.0
        p = row.get("p_donate", 0.0) or 0.0

        if is_donor and expected and expected > 0:
            gap_pct = (expected - actual) / expected
            if gap_pct > 0.20:
                segments.append("Under-Contributing Donor")
            else:
                segments.append("Reliable Donor")
        elif not is_donor and p >= 0.50:
            segments.append("High-Potential Prospect")
        elif not is_donor and p >= 0.20:
            segments.append("Emerging Prospect")
        else:
            segments.append("Low Probability")
    df["donor_segment"] = segments
    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _naive_ols(train: pd.DataFrame, round_cols: list[str]) -> object:
    """OLS Stage 2 without IMR — used for naive comparison."""
    donors = train[train["donate_dummy"] == 1].copy()
    donors = _add_round_dummies(donors)
    X = sm.add_constant(donors[STAGE2_VARS + round_cols].fillna(0).astype(float))
    y = donors["log_donation_usd"].astype(float)
    return sm.OLS(y, X).fit()


def run_diagnostics(
    stage1_result,
    stage2_result,
    train: pd.DataFrame,
    test: pd.DataFrame,
    stage1_result_for_test,
    stage2_result_for_test,
    round_cols: list[str],
    bp_pval: float,
    mle_result=None,
) -> str:
    """
    Run all post-estimation diagnostic checks.

    Returns a formatted string written to heckman_diagnostics.txt.
    """
    lines = ["=" * 70, "HECKMAN SELECTION MODEL — DIAGNOSTICS REPORT", "=" * 70, ""]

    # ── IMR significance ────────────────────────────────────────────────────
    lines.append("1. IMR Significance Test")
    lines.append("-" * 40)
    if "imr" in stage2_result.params.index:
        imr_pval = stage2_result.pvalues["imr"]
        imr_coef = stage2_result.params["imr"]
        lines.append(f"   IMR coefficient: {imr_coef:.4f}  p-value: {imr_pval:.4f}")
        if imr_pval > 0.10:
            msg = f"   WARNING: IMR p-value={imr_pval:.4f} > 0.10 — selection bias may be negligible"
            lines.append(msg)
            logger.warning("IMR p-value=%.4f: selection bias may be negligible", imr_pval)
        else:
            lines.append("   OK: IMR is statistically significant at 10% level")
    else:
        lines.append("   IMR not found in Stage 2 parameters")
    lines.append("")

    # ── Exclusion restriction LR test ──────────────────────────────────────
    lines.append("2. Exclusion Restriction Strength (LR Test)")
    lines.append("-" * 40)
    try:
        excl_vars = ["un_voting_align", "peer_donor", "dac_member"]
        X_full = sm.add_constant(train[STAGE1_VARS].fillna(0).astype(float))
        y_sel = train["donate_dummy"].astype(float)
        restricted_vars = [v for v in STAGE1_VARS if v not in excl_vars]
        X_restricted = sm.add_constant(train[restricted_vars].fillna(0).astype(float))

        full_probit = sm.Probit(y_sel, X_full).fit(disp=False)
        restr_probit = sm.Probit(y_sel, X_restricted).fit(disp=False)

        lr_stat = 2 * (full_probit.llf - restr_probit.llf)
        lr_pval = scipy_stats.chi2.sf(lr_stat, df=len(excl_vars))
        lines.append(f"   LR statistic: {lr_stat:.4f}  p-value: {lr_pval:.4f}")
        if lr_pval > 0.10:
            msg = f"   WARNING: Exclusion restrictions may be weak — LR p-value={lr_pval:.4f}"
            lines.append(msg)
            logger.warning("Exclusion restrictions may be weak: LR p-value=%.4f", lr_pval)
        else:
            lines.append("   OK: Excluded variables are jointly significant")
    except Exception as exc:
        lines.append(f"   LR test failed: {exc}")
    lines.append("")

    # ── Naive OLS comparison ───────────────────────────────────────────────
    lines.append("3. Naive OLS vs. Heckman Coefficient Comparison")
    lines.append("-" * 40)
    try:
        naive = _naive_ols(train, round_cols)
        comparison_rows = []
        for var in STAGE2_VARS:
            h_coef = stage2_result.params.get(var, np.nan)
            n_coef = naive.params.get(var, np.nan)
            if not np.isnan(h_coef) and not np.isnan(n_coef) and abs(n_coef) > 1e-10:
                pct = (h_coef - n_coef) / abs(n_coef) * 100
            else:
                pct = np.nan
            comparison_rows.append((var, h_coef, n_coef, pct))
        lines.append(f"   {'Variable':<35} {'Heckman':>10} {'Naive OLS':>10} {'% Change':>10}")
        lines.append("   " + "-" * 70)
        for var, h, n, p in comparison_rows:
            pct_str = f"{p:+.1f}%" if not np.isnan(p) else "N/A"
            lines.append(f"   {var:<35} {h:>10.4f} {n:>10.4f} {pct_str:>10}")
    except Exception as exc:
        lines.append(f"   Comparison failed: {exc}")
    lines.append("")

    # ── Heteroskedasticity ─────────────────────────────────────────────────
    lines.append("4. Heteroskedasticity (Breusch-Pagan)")
    lines.append("-" * 40)
    lines.append(f"   BP p-value: {bp_pval:.4f}")
    if bp_pval < 0.05:
        lines.append("   Robust SEs applied (HC3)")
    else:
        lines.append("   Standard OLS SEs used")
    lines.append("")

    # ── VIF ────────────────────────────────────────────────────────────────
    lines.append("5. Variance Inflation Factors (Stage 2)")
    lines.append("-" * 40)
    try:
        donors = train[train["donate_dummy"] == 1].copy()
        donors["imr"] = _compute_imr_for_all(stage1_result, donors)
        donors = _add_round_dummies(donors)
        vif_vars = STAGE2_VARS + ["imr"] + round_cols
        X_vif = sm.add_constant(donors[vif_vars].fillna(0).astype(float))
        vif_data = pd.DataFrame({
            "variable": X_vif.columns,
            "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
        })
        for _, row_vif in vif_data[vif_data["variable"] != "const"].iterrows():
            flag = " *** HIGH VIF ***" if row_vif["VIF"] > 10 else ""
            lines.append(f"   {row_vif['variable']:<35} VIF={row_vif['VIF']:.2f}{flag}")
            if row_vif["VIF"] > 10:
                logger.warning("High VIF for %s: %.2f", row_vif["variable"], row_vif["VIF"])
    except Exception as exc:
        lines.append(f"   VIF computation failed: {exc}")
    lines.append("")

    # ── OOS accuracy ───────────────────────────────────────────────────────
    lines.append("6. Out-of-Sample Accuracy (IDA18–IDA20 holdout)")
    lines.append("-" * 40)
    try:
        test_donors = test[test["donate_dummy"] == 1].copy()
        test_donors = _add_round_dummies(test_donors)
        test_donors["imr"] = _compute_imr_for_all(stage1_result, test_donors)
        for col in round_cols:
            if col not in test_donors.columns:
                test_donors[col] = 0
        X_test = sm.add_constant(test_donors[STAGE2_VARS + ["imr"] + round_cols].fillna(0).astype(float))
        pred_heckman = stage2_result.predict(X_test)
        y_test = test_donors["log_donation_usd"]
        valid = y_test.notna() & pred_heckman.notna()

        mae_h = float(np.abs(pred_heckman[valid] - y_test[valid]).mean())
        rmse_h = float(np.sqrt(((pred_heckman[valid] - y_test[valid]) ** 2).mean()))

        # Naive OLS predictions
        naive_test = _naive_ols(train, round_cols)
        X_naive = sm.add_constant(test_donors[STAGE2_VARS + round_cols].fillna(0).astype(float))
        pred_naive = naive_test.predict(X_naive)
        mae_n = float(np.abs(pred_naive[valid] - y_test[valid]).mean())
        rmse_n = float(np.sqrt(((pred_naive[valid] - y_test[valid]) ** 2).mean()))

        lines.append(f"   {'Model':<20} {'MAE':>10} {'RMSE':>10}")
        lines.append("   " + "-" * 45)
        lines.append(f"   {'Heckman':<20} {mae_h:>10.4f} {rmse_h:>10.4f}")
        lines.append(f"   {'Naive OLS':<20} {mae_n:>10.4f} {rmse_n:>10.4f}")
        lines.append(f"   (OOS sample: {int(valid.sum())} donor observations)")
    except Exception as exc:
        lines.append(f"   OOS evaluation failed: {exc}")
    lines.append("")

    lines.append("=" * 70)
    report = "\n".join(lines)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_PATH.write_text(report)
    logger.info("Diagnostics written to %s", DIAGNOSTICS_PATH)

    return report


def generate_residuals_plot(stage2_result, train: pd.DataFrame, imr_train: pd.Series) -> None:
    """Two-subplot residuals figure: actual vs. predicted and IMR distribution."""
    CHARTS.mkdir(parents=True, exist_ok=True)

    donors = train[train["donate_dummy"] == 1].copy()
    donors["imr"] = imr_train.reindex(donors.index)
    donors = _add_round_dummies(donors)
    round_cols_plot = [c for c in donors.columns if c.startswith("round_")]
    all_vars = STAGE2_VARS + ["imr"] + round_cols_plot
    X = sm.add_constant(donors[all_vars].fillna(0).astype(float))
    predicted = stage2_result.predict(X)
    actual = donors["log_donation_usd"]
    valid = actual.notna() & predicted.notna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(predicted[valid], actual[valid], alpha=0.6, s=20)
    lims = [
        min(predicted[valid].min(), actual[valid].min()),
        max(predicted[valid].max(), actual[valid].max()),
    ]
    axes[0].plot(lims, lims, "r--", linewidth=1, label="45° line")
    axes[0].set_xlabel("Predicted log(donation)")
    axes[0].set_ylabel("Actual log(donation)")
    axes[0].set_title("Stage 2: Actual vs. Predicted (training donors)")
    axes[0].legend(fontsize=9)

    axes[1].hist(imr_train.dropna(), bins=40, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Inverse Mills Ratio (λ)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("IMR Distribution (training set)")

    plt.tight_layout()
    fig.savefig(RESIDUALS_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Residuals plot saved to %s", RESIDUALS_PLOT_PATH)


# ---------------------------------------------------------------------------
# Donor-type giving rates
# ---------------------------------------------------------------------------

def compute_donor_type_rates(
    train: pd.DataFrame,
    recent_panel: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Compute per-country giving rate (donation / GDP) from training data.

    If recent_panel is provided (e.g. IDA18–IDA20 rows), per-country rates
    are overridden with each country's most-recent actual rate — eliminating
    the need for a uniform scale factor for known donors.

    Returns a dict {iso3: giving_rate_as_fraction_of_gdp}.
    Special keys for non-donor archetypes:
      '_new_oecd'  — DAC member not yet in donor history
      '_emerging'  — High-potential non-DAC prospect
      '_low_prob'  — Low-probability non-donor
    """
    donors = train[train["donate_dummy"] == 1].copy()
    donors["gdp_usd"] = np.exp(donors["log_gdp_level"])
    donors["giving_rate"] = donors["donation_usd"] / donors["gdp_usd"]

    # Historical median per country from training rounds
    country_rates: dict[str, float] = (
        donors.groupby("country_iso3")["giving_rate"]
        .median()
        .to_dict()
    )

    # Override with most-recent actual rate where available
    if recent_panel is not None:
        recent_donors = recent_panel[recent_panel["donate_dummy"] == 1].copy()
        if not recent_donors.empty:
            recent_donors["gdp_usd"] = np.exp(recent_donors["log_gdp_level"])
            recent_donors["giving_rate"] = recent_donors["donation_usd"] / recent_donors["gdp_usd"]
            # Keep the most recent round for each country
            round_order = {r: i for i, r in enumerate(
                ["IDA18", "IDA19", "IDA20", "IDA21"]
            )}
            recent_donors["round_ord"] = recent_donors["replenishment_round"].map(round_order).fillna(-1)
            latest = (
                recent_donors.sort_values("round_ord")
                .groupby("country_iso3")
                .last()
                .reset_index()
            )
            for _, row in latest.iterrows():
                country_rates[row["country_iso3"]] = float(row["giving_rate"])
            logger.info(
                "Donor rates: overrode %d countries with most-recent round actual rates",
                len(latest),
            )

    # Group archetype rates for non-donors (from training, not recent)
    rounds_donated = donors.groupby("country_iso3")["donate_dummy"].count()
    new_donor_idx = rounds_donated[rounds_donated <= rounds_donated.quantile(0.35)].index
    if len(new_donor_idx) > 0:
        new_oecd_rate = donors[donors["country_iso3"].isin(new_donor_idx)]["giving_rate"].median()
    else:
        new_oecd_rate = donors["giving_rate"].quantile(0.25)

    country_rates["_new_oecd"] = float(new_oecd_rate)
    country_rates["_emerging"] = float(donors["giving_rate"].quantile(0.15))
    country_rates["_low_prob"] = float(donors["giving_rate"].quantile(0.05))

    return country_rates


def resolve_giving_rate(
    iso3: str,
    p_donate: float,
    dac_member: int,
    country_rates: dict[str, float],
    ida21_scale_factor: float,
) -> float:
    """
    Return the giving rate for a country, scaled to the current replenishment.

    Known donors with recent-round rates (set via recent_panel) are used
    directly (scale_factor = 1.0 effectively, since the rate is already current).
    Historical-only donors are scaled by ida21_scale_factor.
    Non-donors use archetype rates × scale_factor.
    """
    if iso3 in country_rates:
        return country_rates[iso3]   # already calibrated to most-recent round

    if dac_member or p_donate >= 0.50:
        archetype = "_new_oecd"
    elif p_donate >= 0.20:
        archetype = "_emerging"
    else:
        archetype = "_low_prob"

    return country_rates[archetype] * ida21_scale_factor


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def score_capacity(
    master: pd.DataFrame | None = None,
    fiscal_modifier: bool = True,
) -> pd.DataFrame:
    """
    Compute Heckman-based capacity scores for all countries.

    Signature-compatible replacement for capacity.score_capacity().

    Parameters
    ----------
    master : DataFrame, optional
        Per-country snapshot from ingest.build_master(). Used to join
        country metadata (iso3, country_name, income_group, gdp_usd, actuals).
        If None, loads from data/processed/master.csv.
    fiscal_modifier : bool
        Accepted for interface compatibility; not used in Heckman estimation.

    Returns
    -------
    DataFrame written to data/processed/capacity_scores.csv.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    if master is None:
        master = pd.read_csv(DATA_PROCESSED / "master.csv")

    # ── 1. Load and validate panel ─────────────────────────────────────────
    logger.info("Heckman: loading panel data from %s", PANEL_PATH)
    panel = load_panel()

    # ── 2. Preprocess ──────────────────────────────────────────────────────
    panel = preprocess_panel(panel)

    # ── 3. Split ───────────────────────────────────────────────────────────
    train, test = split_panel(panel)

    # ── 3b. Compute donor-type giving rates BEFORE standardization ─────────
    # log_gdp_level and donation_usd are still in natural units here.
    # Pass IDA19–20 as recent_panel so known donors use their actual recent
    # rate rather than historical median × scale_factor.
    recent_rounds = panel[panel["replenishment_round"].isin({"IDA19", "IDA20"})]
    country_rates = compute_donor_type_rates(train, recent_panel=recent_rounds)

    # ── 4. Standardize Stage 2 continuous variables ────────────────────────
    train, test, scaler_params = standardize(train, test, STAGE2_CONTINUOUS)

    # ── 5. Fit Stage 1 ─────────────────────────────────────────────────────
    logger.info("Heckman: fitting Stage 1 probit...")
    stage1_result, imr_train = fit_stage1(train)

    # ── 6. Fit Stage 2 ─────────────────────────────────────────────────────
    logger.info("Heckman: fitting Stage 2 OLS...")
    stage2_result, use_robust, round_cols, bp_pval = fit_stage2(train, imr_train)

    # ── 7. MLE comparison ─────────────────────────────────────────────────
    mle_result = fit_mle_heckman(train)
    check_coefficient_divergence(stage2_result, mle_result, STAGE2_VARS + ["imr"])

    # ── 8. Build prediction input from master (real countries) ────────────
    pred_input = master.copy()
    pred_input["country_iso3"] = pred_input["iso3"]

    # Derive Heckman features available from master
    pred_input["log_gdp_per_capita"] = np.log(
        pred_input["gdp_per_capita_usd"].clip(lower=1).fillna(1)
    )
    pred_input["log_gdp_level"] = np.log(
        pred_input["gdp_usd"].clip(lower=1).fillna(1)
    )

    # Pull in per-country metadata from the same lookup tables used in build_panel.py
    from build_panel import (
        DAC_JOIN_YEAR, US_EU_ALLY, UN_VOTING_ALIGN,
        IDA_VOTE_SHARE, TRADE_EXPOSURE_IDA, SOVEREIGN_RATING,
    )

    iso = pred_input["iso3"]
    pred_input["dac_member"] = iso.map(lambda x: 1 if DAC_JOIN_YEAR.get(x, 9999) <= 2024 else 0)
    pred_input["us_eu_ally"] = iso.map(lambda x: US_EU_ALLY.get(x, 0))
    pred_input["un_voting_align"] = iso.map(lambda x: UN_VOTING_ALIGN.get(x, 0.45))
    pred_input["ida_vote_share_lag"] = iso.map(lambda x: IDA_VOTE_SHARE.get(x, 0.05))
    pred_input["trade_exposure_ida"] = iso.map(lambda x: TRADE_EXPOSURE_IDA.get(x, 0.15))
    # At IDA21, virtually all established peers are contributing — strong peer signal
    pred_input["peer_donor"] = 1

    # trade_openness and gov_effectiveness come from master (WDI cache)
    if "trade_openness" not in pred_input.columns:
        pred_input["trade_openness"] = np.nan
    if "gov_effectiveness" not in pred_input.columns:
        pred_input["gov_effectiveness"] = np.nan

    # log_donation_lag: use most recent actual IDA contribution
    actual = (
        pred_input["ida21_contribution_usd"]
        .combine_first(pred_input["ida20_contribution_usd"])
    )
    pred_input["log_donation_lag"] = np.where(
        actual.notna() & (actual > 0),
        np.log(actual.clip(lower=1)),
        0.0,
    )

    # sovereign_credit_rating: encode from string ratings via CREDIT_RATING_MAP
    pred_input["sovereign_credit_rating"] = (
        iso.map(lambda x: SOVEREIGN_RATING.get(x, "BBB"))
           .map(CREDIT_RATING_MAP)
           .fillna(12)  # BBB default
    )

    # Round column: IDA21 (no training round dummies will fire)
    pred_input["replenishment_round"] = "IDA21"

    # Apply same standardization fitted on training data
    for col in STAGE2_CONTINUOUS:
        if col in pred_input.columns and col in scaler_params:
            mu, sigma = scaler_params[col]
            pred_input[col] = (pred_input[col].fillna(mu) - mu) / sigma

    predictions = predict_all(
        stage1_result,
        stage2_result,
        pred_input,
        stage2_result.resid,
        round_cols,
    )

    # ── 9. Merge predictions back onto master metadata + actuals ──────────
    master_slim = master[[
        "iso3", "country_name", "income_group", "gdp_usd",
        "ida21_contribution_usd", "ida20_contribution_usd", "is_current_donor",
    ]].copy()

    # Actual contribution: prefer IDA21, fall back to IDA20
    master_slim["actual_contribution_usd"] = (
        master_slim["ida21_contribution_usd"]
        .combine_first(master_slim["ida20_contribution_usd"])
        .fillna(0.0)
    )

    merged = master_slim.merge(
        predictions[["country_iso3", "p_donate", "imr", "pred_log_donation",
                      "pred_donation_usd", "expected_contribution"]],
        left_on="iso3",
        right_on="country_iso3",
        how="left",
    )

    # ── 10. Calibrate expected_contribution using donor-type giving rates ────
    # Stage 2 outcome predictions are anchored to historical contribution levels.
    # Instead, we use p_donate from Stage 1 (well-calibrated) and multiply by
    # each country's donor-type giving rate scaled to the IDA21 replenishment.
    #
    # Known donors → personal historical rate (from training rounds)
    # Non-donors   → archetype rate by p_donate / DAC membership
    # All rates scaled by (IDA21 median rate / training median rate).

    # Per-country historical rates (computed pre-standardization in step 3b)
    train_median_rate = float(
        pd.Series([v for k, v in country_rates.items() if not k.startswith("_")]).median()
    )

    # IDA21 actual giving rates for current donors
    donors_mask = (
        (merged["is_current_donor"] == 1)
        & (merged["gdp_usd"] > 0)
        & (merged["actual_contribution_usd"] > 0)
    )
    ida21_rates = (
        merged.loc[donors_mask, "actual_contribution_usd"]
        / merged.loc[donors_mask, "gdp_usd"]
    )
    ida21_median_rate = float(ida21_rates.median())
    ida21_scale_factor = ida21_median_rate / train_median_rate if train_median_rate > 0 else 1.0

    logger.info(
        "Donor-type calibration: training median rate=%.4f%% GDP, "
        "IDA21 median rate=%.4f%% GDP, scale factor=%.3f",
        train_median_rate * 100, ida21_median_rate * 100, ida21_scale_factor,
    )

    # Build lookup of dac_member per iso3 from prediction input
    dac_lookup = pred_input.set_index("iso3")["dac_member"].to_dict()

    def _expected(row: pd.Series) -> float:
        # Known donors have already answered the selection question — use p=1.0
        # so the rate × GDP drives the prediction rather than a miscalibrated probit.
        is_donor = row.get("is_current_donor", 0) == 1
        p = 1.0 if is_donor else float(row.get("p_donate", 0) or 0)
        rate = resolve_giving_rate(
            iso3=row["iso3"],
            p_donate=p,
            dac_member=int(dac_lookup.get(row["iso3"], 0)),
            country_rates=country_rates,
            ida21_scale_factor=ida21_scale_factor,
        )
        return p * rate * float(row.get("gdp_usd") or 0)

    merged["expected_contribution"] = merged.apply(_expected, axis=1)

    # ── 11. Assign segments and map columns ────────────────────────────────
    merged = assign_segments(merged)
    merged["adjusted_target_usd"] = merged["expected_contribution"]
    merged["gap_usd"] = merged["expected_contribution"] - merged["actual_contribution_usd"]
    merged["giving_rate"] = np.where(
        merged["expected_contribution"].notna() & (merged["expected_contribution"] > 0),
        merged["actual_contribution_usd"] / merged["expected_contribution"],
        np.nan,
    )

    # ── 11. Log segmentation summary ───────────────────────────────────────
    for seg, grp in merged.groupby("donor_segment"):
        logger.info(
            "Segment %-30s: %d countries, mean expected contribution $%.1fM",
            seg,
            len(grp),
            grp["expected_contribution"].mean() / 1e6 if grp["expected_contribution"].notna().any() else 0,
        )

    # ── 12. Diagnostics ───────────────────────────────────────────────────
    logger.info("Heckman: running diagnostics...")
    run_diagnostics(
        stage1_result=stage1_result,
        stage2_result=stage2_result,
        train=train,
        test=test,
        stage1_result_for_test=stage1_result,
        stage2_result_for_test=stage2_result,
        round_cols=round_cols,
        bp_pval=bp_pval,
        mle_result=mle_result,
    )
    generate_residuals_plot(stage2_result, train, imr_train)

    # ── 13. Write output ───────────────────────────────────────────────────
    output_cols = [
        "iso3", "country_name", "income_group", "gdp_usd",
        "actual_contribution_usd", "adjusted_target_usd", "gap_usd", "giving_rate",
        "p_donate", "pred_donation_usd", "expected_contribution",
        "donor_segment", "imr",
    ]
    output_cols = [c for c in output_cols if c in merged.columns]
    scores = merged[output_cols].sort_values("gap_usd", ascending=False, na_position="last")
    scores = scores.reset_index(drop=True)

    scores.to_csv(CAPACITY_SCORES_PATH, index=False)
    logger.info(
        "Heckman capacity scores written to %s (%d countries, %d with valid gap)",
        CAPACITY_SCORES_PATH,
        len(scores),
        scores["gap_usd"].notna().sum(),
    )
    return scores
