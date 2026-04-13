#!/usr/bin/env python3
"""
IDA21 Accuracy & Precision Verification

Compares the Heckman model's IDA21 predictions (capacity_scores.csv)
against actual IDA21 contributions from the panel and master datasets.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── 1. Ground-truth IDA21 actuals ────────────────────────────────────────────
# Panel has 32 IDA21 donors; master has 25 with ida21_contribution_usd.
# Prefer panel (wider coverage for known donors); fill gaps from master.
panel = pd.read_csv("data/raw/heckman_panel.csv")
ida21_panel = (
    panel[panel["replenishment_round"] == "IDA21"]
    .rename(columns={"country_iso3": "iso3", "donation_usd": "actual_panel_usd"})
    [["iso3", "donate_dummy", "actual_panel_usd"]]
    .copy()
)

master = pd.read_csv("data/processed/master.csv")
ida21_master = master[["iso3", "country_name", "ida21_contribution_usd"]].copy()

# ── 2. Model predictions ──────────────────────────────────────────────────────
cap = pd.read_csv("data/processed/capacity_scores.csv")
cap_slim = cap[[
    "iso3", "country_name", "income_group", "gdp_usd",
    "actual_contribution_usd",   # IDA21 actual used by model (may be IDA20 fallback)
    "expected_contribution",     # model prediction (p_donate × giving_rate × GDP)
    "adjusted_target_usd",       # same as expected_contribution for Heckman path
    "p_donate", "pred_donation_usd", "donor_segment",
]].copy()

# ── 3. Merge all sources ──────────────────────────────────────────────────────
df = cap_slim.merge(ida21_panel, on="iso3", how="left")
df = df.merge(ida21_master[["iso3", "ida21_contribution_usd"]], on="iso3", how="left")

# Best ground-truth: panel IDA21 donation > master ida21 > model's actual_contribution_usd
df["ground_truth_usd"] = (
    df["actual_panel_usd"]
    .combine_first(df["ida21_contribution_usd"])
    .combine_first(df["actual_contribution_usd"])
)
df["gt_source"] = np.where(
    df["actual_panel_usd"].notna(), "panel",
    np.where(df["ida21_contribution_usd"].notna(), "master", "model_fallback")
)

# Only evaluate countries where we have a real IDA21 ground truth
eval_df = df[df["ground_truth_usd"].notna() & (df["ground_truth_usd"] > 0)].copy()

# Separate: donors with IDA21 ground truth vs rest
panel_donors = eval_df[eval_df["gt_source"] == "panel"].copy()
all_known    = eval_df.copy()

# ── 4. Metrics helper ─────────────────────────────────────────────────────────
def metrics(actual, pred, label):
    actual, pred = np.array(actual), np.array(pred)
    pct_err      = (pred - actual) / actual * 100
    ratio        = pred / actual
    mae          = float(np.abs(pred - actual).mean())
    rmse         = float(np.sqrt(((pred - actual) ** 2).mean()))
    median_pct   = float(np.median(pct_err))
    mean_pct     = float(np.mean(pct_err))
    within_2x    = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100)
    within_50pct = float(np.mean(np.abs(pct_err) <= 50) * 100)
    corr         = float(np.corrcoef(pred, actual)[0, 1])
    t_stat, t_p  = scipy_stats.ttest_1samp(pct_err, 0)

    print(f"  {label}  (n={len(actual)})")
    print(f"    MAE:             ${mae/1e6:>9,.0f}M")
    print(f"    RMSE:            ${rmse/1e6:>9,.0f}M")
    print(f"    Median % err:    {median_pct:>+8.1f}%")
    print(f"    Mean % err:      {mean_pct:>+8.1f}%  (bias t={t_stat:.2f}, p={t_p:.4f}{'  [BIAS]' if t_p < 0.05 else ''})")
    print(f"    Pearson r:       {corr:>9.4f}")
    print(f"    Within ±50%:     {within_50pct:>7.1f}% of countries")
    print(f"    Within 2×:       {within_2x:>7.1f}% of countries")
    return pct_err

# ── 5. Report ─────────────────────────────────────────────────────────────────
print("=" * 72)
print("  IDA21 ACCURACY & PRECISION VERIFICATION")
print("=" * 72)
print(f"  Panel IDA21 donors:            {len(panel_donors)}")
print(f"  Countries with any IDA21 data: {len(all_known)}")
print()

print("  AGGREGATE METRICS — expected_contribution vs IDA21 actual")
print("  " + "-" * 60)
pct_all = metrics(
    all_known["ground_truth_usd"],
    all_known["expected_contribution"],
    "All countries with IDA21 ground truth",
)
print()
pct_panel = metrics(
    panel_donors["ground_truth_usd"],
    panel_donors["expected_contribution"],
    "Panel IDA21 donors only (32 countries)",
)

# ── 6. Per-country table ──────────────────────────────────────────────────────
print()
print("  PER-COUNTRY BREAKDOWN (sorted by actual, descending)")
print("  " + "-" * 78)
print(f"  {'ISO3':<6} {'Actual':>10} {'Predicted':>12} {'%Err':>8} {'p_donate':>9} {'Segment':<28} {'GT'}")
print("  " + "-" * 78)

ordered = all_known.sort_values("ground_truth_usd", ascending=False)
for _, row in ordered.iterrows():
    act  = row["ground_truth_usd"]
    pred = row["expected_contribution"]
    pe   = (pred - act) / act * 100
    pd_  = row["p_donate"]
    seg  = str(row["donor_segment"])[:27] if pd.notna(row["donor_segment"]) else ""
    src  = row["gt_source"]
    print(f"  {row['iso3']:<6} ${act/1e6:>8,.0f}M  ${pred/1e6:>9,.0f}M  {pe:>+7.1f}%  {pd_:>8.3f}  {seg:<28} {src}")

# ── 7. Segment-level summary ──────────────────────────────────────────────────
print()
print("  ACCURACY BY DONOR SEGMENT")
print("  " + "-" * 60)
for seg, grp in all_known.groupby("donor_segment"):
    if len(grp) == 0:
        continue
    act  = grp["ground_truth_usd"].values
    pred = grp["expected_contribution"].values
    pe   = (pred - act) / act * 100
    print(f"  {seg:<32}  n={len(grp):>2}  median_err={float(np.median(pe)):>+7.1f}%  "
          f"within±50%={float(np.mean(np.abs(pe)<=50)*100):.0f}%")

# ── 8. Panel vs master discrepancy note ──────────────────────────────────────
print()
print("  PANEL vs MASTER IDA21 DATA DISCREPANCY")
print("  " + "-" * 60)
both = df[df["actual_panel_usd"].notna() & df["ida21_contribution_usd"].notna()].copy()
if not both.empty:
    both["discrepancy_pct"] = (both["actual_panel_usd"] - both["ida21_contribution_usd"]) / both["ida21_contribution_usd"] * 100
    notable = both[both["discrepancy_pct"].abs() > 5].sort_values("discrepancy_pct")
    if notable.empty:
        print("  All overlapping countries agree within 5%")
    else:
        print(f"  {'ISO3':<6} {'Panel':>12} {'Master':>12} {'Diff%':>8}")
        for _, row in notable.iterrows():
            print(f"  {row['iso3']:<6} ${row['actual_panel_usd']/1e6:>10,.0f}M  ${row['ida21_contribution_usd']/1e6:>10,.0f}M  {row['discrepancy_pct']:>+7.1f}%")

print("=" * 72)
