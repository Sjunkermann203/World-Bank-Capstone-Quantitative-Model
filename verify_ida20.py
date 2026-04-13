#!/usr/bin/env python3
"""
IDA20 Out-of-Sample Accuracy Verification

Trains the Heckman model on IDA1–IDA17, predicts for IDA20 donors,
and reports accuracy / precision at USD scale.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent / "src"))
from heckman import (
    load_panel, preprocess_panel, split_panel, standardize,
    fit_stage1, fit_stage2, STAGE1_VARS, STAGE2_VARS, STAGE2_CONTINUOUS,
    TRAIN_ROUNDS, _compute_imr_for_all, _add_round_dummies,
)

logging.basicConfig(level=logging.WARNING)

# ── 1. Load + preprocess ────────────────────────────────────────────────────
panel = load_panel()
panel = preprocess_panel(panel)

train = panel[panel["replenishment_round"].isin(TRAIN_ROUNDS)].copy()
ida20 = panel[panel["replenishment_round"] == "IDA20"].copy()
ida20_donors = ida20[ida20["donate_dummy"] == 1].copy()

# ── 2. Standardize on train, apply to IDA20 ─────────────────────────────────
# Need a dummy "test" for the standardize() function signature
train_s, ida20_s, scaler = standardize(train, ida20_donors, STAGE2_CONTINUOUS)

# ── 3. Fit model on train ─────────────────────────────────────────────────────
stage1_result, imr_train = fit_stage1(train_s)
stage2_result, use_robust, round_cols, bp_pval = fit_stage2(train_s, imr_train)

# ── 4. Predict on IDA20 donors ───────────────────────────────────────────────
ida20_s["imr"] = _compute_imr_for_all(stage1_result, ida20_s)
ida20_s = _add_round_dummies(ida20_s)
for col in round_cols:
    if col not in ida20_s.columns:
        ida20_s[col] = 0

all_vars = STAGE2_VARS + ["imr"] + round_cols
X_pred = sm.add_constant(ida20_s[all_vars].fillna(0).astype(float), has_constant="add")
pred_log = stage2_result.predict(X_pred)

# Duan smearing correction (from training residuals)
smearing_factor = float(np.exp(stage2_result.resid).mean())
pred_usd = np.exp(pred_log) * smearing_factor

# Stage 1: p_donate
X1 = sm.add_constant(ida20_s[STAGE1_VARS].fillna(0).astype(float), has_constant="add")
p_donate = stage1_result.predict(X1)

# Expected = p_donate × pred_donation_usd
expected_usd = p_donate.values * pred_usd.values

actual_usd = ida20_donors["donation_usd"].values
countries   = ida20_donors["country_iso3"].values

# ── 5. Metrics ────────────────────────────────────────────────────────────────
log_actual = np.log(actual_usd.clip(1))
log_pred   = pred_log.values

# Log-scale (model native)
mae_log  = float(np.abs(log_pred - log_actual).mean())
rmse_log = float(np.sqrt(((log_pred - log_actual) ** 2).mean()))

# USD-scale (predicted raw, before p_donate weighting)
ratio_raw     = pred_usd.values / actual_usd          # pred / actual
pct_err_raw   = (pred_usd.values - actual_usd) / actual_usd * 100
mae_usd_raw   = float(np.abs(pred_usd.values - actual_usd).mean())
rmse_usd_raw  = float(np.sqrt(((pred_usd.values - actual_usd) ** 2).mean()))

# USD-scale (expected = p_donate × pred)
pct_err_exp   = (expected_usd - actual_usd) / actual_usd * 100
mae_usd_exp   = float(np.abs(expected_usd - actual_usd).mean())
rmse_usd_exp  = float(np.sqrt(((expected_usd - actual_usd) ** 2).mean()))

# Within-factor-2 accuracy
within_2x_raw = float(np.mean((ratio_raw >= 0.5) & (ratio_raw <= 2.0)) * 100)
within_50pct  = float(np.mean(np.abs(pct_err_raw) <= 50) * 100)

print("=" * 70)
print("  IDA20 OUT-OF-SAMPLE VERIFICATION  (model trained on IDA1–IDA17)")
print("=" * 70)
print(f"  Donors evaluated: {len(ida20_donors)}")
print()

print("  LOG-SCALE ACCURACY (Stage 2 native output)")
print("  " + "-" * 40)
print(f"  MAE  (log):   {mae_log:.4f}")
print(f"  RMSE (log):   {rmse_log:.4f}")
print()

print("  USD-SCALE — pred_donation_usd (before p_donate weighting)")
print("  " + "-" * 40)
print(f"  MAE:           ${mae_usd_raw/1e6:,.0f}M")
print(f"  RMSE:          ${rmse_usd_raw/1e6:,.0f}M")
print(f"  Median % err:  {float(np.median(pct_err_raw)):+.1f}%")
print(f"  Within 2×:     {within_2x_raw:.1f}% of donors")
print(f"  Within ±50%:   {within_50pct:.1f}% of donors")
print()

print("  USD-SCALE — expected_contribution (p_donate × pred_donation_usd)")
print("  " + "-" * 40)
print(f"  MAE:           ${mae_usd_exp/1e6:,.0f}M")
print(f"  RMSE:          ${rmse_usd_exp/1e6:,.0f}M")
print(f"  Median % err:  {float(np.median(pct_err_exp)):+.1f}%")
print()

# Correlation
corr_raw = float(np.corrcoef(pred_usd.values, actual_usd)[0, 1])
corr_exp = float(np.corrcoef(expected_usd, actual_usd)[0, 1])
print(f"  Pearson r (pred vs actual):     {corr_raw:.4f}")
print(f"  Pearson r (expected vs actual): {corr_exp:.4f}")
print()

# ── 6. Per-country table ──────────────────────────────────────────────────────
print("  PER-COUNTRY BREAKDOWN")
print("  " + "-" * 70)
print(f"  {'ISO3':<6} {'Actual':>12} {'Pred (raw)':>12} {'Expected':>12} {'%Err raw':>9} {'p_donate':>9}")
print("  " + "-" * 70)

order = np.argsort(-actual_usd)
for i in order:
    c   = countries[i]
    act = actual_usd[i]
    pr  = pred_usd.values[i]
    ex  = expected_usd[i]
    pe  = pct_err_raw[i]
    pd_ = p_donate.values[i]
    print(f"  {c:<6} ${act/1e6:>10,.0f}M  ${pr/1e6:>10,.0f}M  ${ex/1e6:>10,.0f}M  {pe:>+8.1f}%  {pd_:>8.3f}")

print("=" * 70)

# ── 7. Bias check: are we systematically over/under? ─────────────────────────
print()
print("  BIAS ANALYSIS")
print("  " + "-" * 40)
mean_err_raw = float(np.mean(pct_err_raw))
std_err_raw  = float(np.std(pct_err_raw))
t_stat, t_pval = scipy_stats.ttest_1samp(pct_err_raw, 0)
print(f"  Mean % error (pred raw): {mean_err_raw:+.1f}%  (std={std_err_raw:.1f}%)")
print(f"  t-test H0=0:  t={t_stat:.3f}, p={t_pval:.4f}", end="")
print("  [SIGNIFICANT BIAS]" if t_pval < 0.05 else "  [no significant bias]")

t_stat2, t_pval2 = scipy_stats.ttest_1samp(pct_err_exp, 0)
mean_err_exp = float(np.mean(pct_err_exp))
print(f"  Mean % error (expected): {mean_err_exp:+.1f}%  t={t_stat2:.3f}, p={t_pval2:.4f}", end="")
print("  [SIGNIFICANT BIAS]" if t_pval2 < 0.05 else "  [no significant bias]")
print()
