#!/usr/bin/env python3
"""
compare_predictions_to_market.py

Purpose
-------
Compare your pipeline label y (realized return) and predictions (y_hat) against:
1) realized return y (always)
2) realized price moves (optional, if you provide price columns or fetch via yfinance)
3) simple economic backtests (long/cash and long/short)

Inputs
------
- Step3_MatchFeatures.xlsx: must contain match_id, match_date, y (realized return), and optionally price columns.
- Step3_Predictions.xlsx: must contain match_id and prediction columns (e.g., y_hat_baseline, y_hat_augmented).

Outputs
-------
- Excel workbook with:
  - merged_data
  - metrics
  - calibration (binned)
  - backtest_long_cash
  - backtest_long_short
  - price_checks (if price available)

Notes
-----
- Comments are in English by request.
- This script does not assume any specific ticker or market calendar.
  If you provide price series, we align dates by searching the nearest trading day in the price index.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------
# Helpers
# -------------------------

def _detect_price_cols(df: pd.DataFrame):
    """
    Try to detect common price columns.
    Returns (price_t0_col, price_t1_col) or (None, None) if not found.
    Candidate list includes both standardised names (close_t0, close_tplus1)
    and the raw source names from ManU_match_stock_merged.xlsx
    (pre_price, post_price, event_price_t0).
    """
    candidates_t0 = ["close_t0", "price_t0", "close0", "close_prev",
                     "adj_close_t0", "adjclose_t0",
                     "pre_price", "event_price_t0"]
    candidates_t1 = ["close_tplus1", "price_tplus1", "close1", "close_next",
                     "adj_close_tplus1", "adjclose_tplus1",
                     "post_price"]

    t0 = next((c for c in candidates_t0 if c in df.columns), None)
    t1 = next((c for c in candidates_t1 if c in df.columns), None)
    return t0, t1


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan

def _mae(x):
    return float(np.mean(np.abs(x))) if len(x) else np.nan

def _r2_oos(y, yhat):
    """
    Out-of-sample R^2 using mean(y) baseline.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    denom = np.sum((y - np.mean(y))**2)
    if denom == 0:
        return np.nan
    return float(1 - np.sum((y - yhat)**2) / denom)

def _hit_rate(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.mean((y > 0) == (yhat > 0))) if len(y) else np.nan


def _calibration_table(df, y_col, yhat_col, n_bins=5):
    """
    Bin by predicted yhat (quantiles) and compute mean actual y within each bin.
    """
    d = df[[y_col, yhat_col]].dropna().copy()
    if len(d) < n_bins:
        return pd.DataFrame()
    d["bin"] = pd.qcut(d[yhat_col], q=n_bins, duplicates="drop")
    out = d.groupby("bin").agg(
        n=("bin", "size"),
        yhat_mean=(yhat_col, "mean"),
        y_mean=(y_col, "mean"),
        y_median=(y_col, "median"),
    ).reset_index()
    return out


def _backtest(df, y_col, yhat_col, mode="long_cash", fee_bps=0.0):
    """
    Simple 1-step strategy based on the sign of yhat.

    mode:
      - long_cash: if yhat>0 -> take y ; else -> 0
      - long_short: if yhat>0 -> +y ; else -> -y
    fee_bps: transaction cost in basis points (e.g., 10 = 0.10%)
    """
    d = df[["match_id", "match_date", y_col, yhat_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame()

    y = d[y_col].astype(float).to_numpy()
    s = np.sign(d[yhat_col].astype(float).to_numpy())
    s[s == 0] = 0  # keep zeros as no position

    if mode == "long_cash":
        pos = (s > 0).astype(float)  # 1 if buy, else 0
        strat = pos * y
        # fee charged when entering a position (approx)
        fees = (pos > 0).astype(float) * (fee_bps / 10000.0)
        strat_net = strat - fees
    elif mode == "long_short":
        pos = np.where(s > 0, 1.0, np.where(s < 0, -1.0, 0.0))
        strat = pos * y
        # fee charged when absolute position is 1
        fees = (np.abs(pos) > 0).astype(float) * (fee_bps / 10000.0)
        strat_net = strat - fees
    else:
        raise ValueError("Unknown mode")

    d["position"] = pos
    d["strategy_return"] = strat
    d["fee"] = fees
    d["strategy_return_net"] = strat_net
    d["cum_gross"] = (1.0 + d["strategy_return"]).cumprod()
    d["cum_net"] = (1.0 + d["strategy_return_net"]).cumprod()

    # Summary row (append)
    summary = pd.DataFrame([{
        "match_id": "SUMMARY",
        "match_date": pd.NaT,
        y_col: float(np.mean(y)),
        yhat_col: np.nan,
        "position": np.nan,
        "strategy_return": float(np.mean(strat)),
        "fee": float(np.mean(fees)),
        "strategy_return_net": float(np.mean(strat_net)),
        "cum_gross": float(d["cum_gross"].iloc[-1]),
        "cum_net": float(d["cum_net"].iloc[-1]),
    }])
    return pd.concat([d, summary], ignore_index=True)


def _compute_price_based_checks(df, price_t0_col, price_t1_col, y_col, yhat_col=None):
    """
    Compute realized return implied by prices and compare to y.
    Also compute predicted price if yhat_col is provided.
    """
    d = df[["match_id", "match_date", price_t0_col, price_t1_col, y_col] + ([yhat_col] if yhat_col else [])].copy()
    d = d.dropna(subset=[price_t0_col, price_t1_col]).copy()

    P0 = d[price_t0_col].astype(float)
    P1 = d[price_t1_col].astype(float)
    d["return_from_price"] = (P1 / P0) - 1.0
    d["return_diff_y_minus_price"] = d[y_col].astype(float) - d["return_from_price"]

    if yhat_col:
        d["pred_price_t1"] = P0 * (1.0 + d[yhat_col].astype(float))
        d["price_error"] = d["pred_price_t1"] - P1
        d["price_abs_error"] = np.abs(d["price_error"])
    return d


def _safe_read_excel(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_excel(path)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="Step3_MatchFeatures.xlsx")
    ap.add_argument("--preds", type=str, default="Step3_Predictions.xlsx")
    ap.add_argument("--y_col", type=str, default="y")
    ap.add_argument("--y_true_col_in_preds", type=str, default="y_true")  # in predictions file
    ap.add_argument("--pred_cols", type=str, default="y_hat_baseline,y_hat_augmented")
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--fee_bps", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="Compare_y_to_market.xlsx")
    args = ap.parse_args()

    features_path = Path(args.features)
    preds_path = Path(args.preds)
    out_path = Path(args.out)

    feat = _safe_read_excel(features_path)
    feat["match_date"] = pd.to_datetime(feat["match_date"])

    preds = _safe_read_excel(preds_path)
    if "match_date" in preds.columns:
        preds["match_date"] = pd.to_datetime(preds["match_date"])

    # Merge: prioritize match_id; if match_id missing, fall back to match_date
    if "match_id" in preds.columns and "match_id" in feat.columns:
        merged = feat.merge(preds, on="match_id", how="left", suffixes=("", "_pred"))
    else:
        merged = feat.merge(preds, on="match_date", how="left", suffixes=("", "_pred"))

    # Decide the realized y column to use:
    # - if predictions file has y_true, use it (it should match y)
    # - otherwise use feat[y]
    y_col = args.y_col
    if args.y_true_col_in_preds in merged.columns and merged[args.y_true_col_in_preds].notna().any():
        merged["y_realized"] = merged[args.y_true_col_in_preds]
    else:
        merged["y_realized"] = merged[y_col]

    merged = merged.sort_values("match_date").reset_index(drop=True)

    pred_cols = [c.strip() for c in args.pred_cols.split(",") if c.strip()]
    available_pred_cols = [c for c in pred_cols if c in merged.columns]

    # -------------------------
    # Metrics table
    # -------------------------
    metrics_rows = []
    for pc in available_pred_cols:
        d = merged.dropna(subset=["y_realized", pc]).copy()
        if d.empty:
            continue
        err = (d[pc].astype(float) - d["y_realized"].astype(float)).to_numpy()
        metrics_rows.append({
            "model": pc,
            "n": int(len(d)),
            "rmse": _rmse(err),
            "mae": _mae(err),
            "hit_rate": _hit_rate(d["y_realized"], d[pc]),
            "r2_oos": _r2_oos(d["y_realized"], d[pc]),
            "corr(y, yhat)": float(np.corrcoef(d["y_realized"].astype(float), d[pc].astype(float))[0,1]) if len(d) > 1 else np.nan
        })
    metrics = pd.DataFrame(metrics_rows)

    # -------------------------
    # Calibration tables
    # -------------------------
    calib_tabs = {}
    for pc in available_pred_cols:
        calib_tabs[pc] = _calibration_table(merged, "y_realized", pc, n_bins=args.n_bins)

    # -------------------------
    # Backtests
    # -------------------------
    backtests_long_cash = {}
    backtests_long_short = {}
    for pc in available_pred_cols:
        backtests_long_cash[pc]  = _backtest(merged, "y_realized", pc, mode="long_cash", fee_bps=args.fee_bps)
        backtests_long_short[pc] = _backtest(merged, "y_realized", pc, mode="long_short", fee_bps=args.fee_bps)

    # -------------------------
    # Optional price-based checks
    # -------------------------
    price_t0_col, price_t1_col = _detect_price_cols(merged)
    price_checks = {}
    if price_t0_col and price_t1_col:
        for pc in available_pred_cols:
            price_checks[pc] = _compute_price_based_checks(
                merged, price_t0_col, price_t1_col, "y_realized", yhat_col=pc
            )
    else:
        # still check return_from_price vs y if user later adds price columns
        price_checks = {}

    # -------------------------
    # Write outputs
    # -------------------------
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        merged.to_excel(w, sheet_name="merged_data", index=False)
        metrics.to_excel(w, sheet_name="metrics", index=False)

        for pc, tab in calib_tabs.items():
            if not tab.empty:
                tab.to_excel(w, sheet_name=f"calib_{pc[:22]}", index=False)

        for pc, bt in backtests_long_cash.items():
            if not bt.empty:
                bt.to_excel(w, sheet_name=f"bt_cash_{pc[:18]}", index=False)

        for pc, bt in backtests_long_short.items():
            if not bt.empty:
                bt.to_excel(w, sheet_name=f"bt_ls_{pc[:20]}", index=False)

        if price_t0_col and price_t1_col and price_checks:
            for pc, chk in price_checks.items():
                if not chk.empty:
                    chk.to_excel(w, sheet_name=f"price_{pc[:22]}", index=False)

    print(f"[DONE] Wrote comparison workbook: {out_path.resolve()}")
    if not available_pred_cols:
        print("[WARN] No prediction columns were found in the merged data. "
              "Check --pred_cols or your predictions file columns.")
    if not (price_t0_col and price_t1_col):
        print("[INFO] No price columns detected. Return-based comparison/backtests were still produced.\n"
              "       If you want price-based comparisons, add columns like close_t0 and close_tplus1 "
              "to either features or predictions input.")


if __name__ == "__main__":
    main()
