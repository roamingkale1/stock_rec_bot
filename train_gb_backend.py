#!/usr/bin/env python3
# ============================================================
# Backend: Train Gradient Boosting (+ DistilBERT sentiment),
# score upcoming quarter, save picks + per-stock contribution
# percentages (SHAP), and export TWO MOST RECENT quarters of
# inputs per ticker for the UI ( *_cur, *_prev, *_qoq ).
#
# Usage:
#   pip install numpy pandas scikit-learn shap
#   python train_gb_backend.py \
#     --fs data/fs_sp500.csv \
#     --prices data/ohlcv_sp500.csv \
#     --sentiment sentiment/headlines_with_sentiment_DistilBERT.csv \
#     --top-k 20 --out web_dist
#
# Outputs in --out:
#   - picks_latest.json
#   - predictions_latest.csv
#   - meta.json
#   - feature_importances.csv
# ============================================================

import os, json, re, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# ---- SHAP for per-stock contributions
try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

# ----------------------------
# Robust helpers / transforms
# ----------------------------
def pct_change_safe(s: pd.Series, periods: int) -> pd.Series:
    prev = s.shift(periods)
    return (s - prev) / prev.replace(0, np.nan)

def winsorize_df_trainwise(X_all: pd.DataFrame, train_mask: pd.Series, low_q=0.01, high_q=0.99) -> pd.DataFrame:
    X = X_all.copy()
    train_X = X.loc[train_mask]
    q_low = train_X.quantile(low_q, numeric_only=True)
    q_high = train_X.quantile(high_q, numeric_only=True)
    for c in X.columns:
        if c in q_low.index:
            X[c] = X[c].clip(lower=q_low[c], upper=q_high[c])
    return X

def safe_div(num, den):
    num = num.astype(float)
    den = den.astype(float)
    return pd.Series(np.where((den.notna()) & (den != 0), num/den, np.nan), index=num.index)

def safe_log1p(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    mask = x > -1
    out[mask] = np.log1p(x[mask])
    return out

def base_metric_name(feat: str) -> str:
    """Collapse engineered variants into a family name for SHAP grouping."""
    name = feat
    if name.startswith("log1p_"):                # log transforms
        name = name[len("log1p_"):]
    name = re.sub(r"_pctchg_(qoq|yoy)$", "", name)  # qoq/yoy deltas
    name = re.sub(r"_lag\d+$", "", name)            # lags
    return name

# ----------------------------
# Sentiment loader (generic)
# ----------------------------
def load_sentiment_quarterly(path: str) -> pd.DataFrame:
    """
    Expected columns (case-insensitive): ticker, date, sentiment
      - sentiment in {'positive','negative','neutral'} (any case)
    Returns (Ticker, Quarter, NewsCount, NewsPos, NewsNeg, NewsNeu, NewsPosMinusNeg)
    """
    s = pd.read_csv(path)
    colmap = {c.lower(): c for c in s.columns}
    for need in ["ticker", "date", "sentiment"]:
        if need not in colmap and need not in s.columns:
            raise ValueError(f"Missing column '{need}' in {path}")

    s["ticker"] = s[colmap.get("ticker","ticker")].astype(str).str.upper().str.strip()
    s["date"] = pd.to_datetime(s[colmap.get("date","date")], errors="coerce", utc=True).dt.tz_localize(None)
    s["sentiment"] = s[colmap.get("sentiment","sentiment")].astype(str).str.lower().str.strip()
    s = s.dropna(subset=["ticker","date","sentiment"])
    s["Quarter"] = s["date"].dt.to_period("Q").dt.to_timestamp("Q")

    s["is_pos"] = (s["sentiment"] == "positive").astype(int)
    s["is_neg"] = (s["sentiment"] == "negative").astype(int)
    s["is_neu"] = (s["sentiment"] == "neutral").astype(int)

    agg = (s.groupby(["ticker","Quarter"])
             .agg(NewsCount=("sentiment","size"),
                  NewsPos=("is_pos","sum"),
                  NewsNeg=("is_neg","sum"),
                  NewsNeu=("is_neu","sum"))
             .reset_index())
    agg["NewsPosMinusNeg"] = agg["NewsPos"] - agg["NewsNeg"]
    agg = agg.rename(columns={"ticker":"Ticker"})
    return agg

# ----------------------------
# Feature engineering
# ----------------------------
def build_feature_panel(fs_raw: pd.DataFrame, px_raw: pd.DataFrame, sent_q: pd.DataFrame):
    # Fundamentals labels
    LBL_ASSETS = "Assets"
    LBL_LIAB   = "Liabilities"
    LBL_NETINC = "Net Income (Loss) Attributable to Parent"
    LBL_SH_OUT_MAIN = "Entity Common Stock, Shares Outstanding"
    LBL_SH_OUT_ALT  = "Common Stock, Shares, Outstanding"
    LBL_CAPEX_1 = "Payments to Acquire Property, Plant, and Equipment"
    LBL_CAPEX_2 = "Capital Expenditures Incurred but Not yet Paid"
    needed = [LBL_ASSETS, LBL_LIAB, LBL_NETINC, LBL_SH_OUT_MAIN, LBL_SH_OUT_ALT, LBL_CAPEX_1, LBL_CAPEX_2]

    # Fundamentals
    fs = fs_raw[fs_raw["label"].isin(needed)].copy()
    fs["Quarter"] = fs["end"].dt.to_period("Q").dt.to_timestamp("Q")
    fs = (fs.sort_values(["ticker","label","end","filed"])
            .drop_duplicates(subset=["ticker","label","end"], keep="last"))
    fs = fs.dropna(subset=["ticker","label","end","val"])
    fs_wide = (fs.rename(columns={"ticker":"Ticker"})
                 .pivot_table(index=["Ticker","Quarter"], columns="label", values="val", aggfunc="last")
                 .reset_index())

    # Shares outstanding
    fs_wide[LBL_SH_OUT_MAIN] = fs_wide.get(LBL_SH_OUT_MAIN, np.nan)
    fs_wide[LBL_SH_OUT_ALT]  = fs_wide.get(LBL_SH_OUT_ALT,  np.nan)
    fs_wide["SharesOutstanding"] = fs_wide[LBL_SH_OUT_MAIN].fillna(fs_wide[LBL_SH_OUT_ALT])

    for col in [LBL_ASSETS, LBL_LIAB, LBL_NETINC, LBL_CAPEX_1, LBL_CAPEX_2]:
        if col not in fs_wide.columns:
            fs_wide[col] = np.nan

    # Prices/Volumes -> quarterly agg
    px = px_raw.sort_values(["Ticker","Date"]).copy()
    px["ret_d"] = px.groupby("Ticker")["Close"].pct_change()
    px["dollar_vol_d"] = px["Close"] * px["Volume"]
    px["Quarter"] = px["Date"].dt.to_period("Q").dt.to_timestamp("Q")

    q_px = (px.groupby(["Ticker","Quarter"])
              .agg(first_close=("Close","first"),
                   last_close=("Close","last"),
                   sum_volume=("Volume","sum"),
                   avg_volume=("Volume","mean"),
                   vol_vol_q=("Volume","std"),
                   n_days=("Date","count"),
                   ret_vol_q=("ret_d","std"),
                   dollar_vol_sum=("dollar_vol_d","sum"),
                   dollar_vol_mean=("dollar_vol_d","mean"))
              .reset_index())
    q_px["q_ret"] = (q_px["last_close"] / q_px["first_close"]) - 1.0

    # Equal-weight market
    mkt = (q_px.groupby("Quarter").agg(mkt_ret=("q_ret","mean")).reset_index())
    q_px = q_px.merge(mkt, on="Quarter", how="left")
    q_px["excess_ret"] = q_px["q_ret"] - q_px["mkt_ret"]

    # Merge wide fundamentals with quarter price/volume
    df = fs_wide.merge(q_px[[
        "Ticker","Quarter","last_close","excess_ret","ret_vol_q",
        "sum_volume","avg_volume","vol_vol_q","dollar_vol_mean","dollar_vol_sum"
    ]], on=["Ticker","Quarter"], how="left")

    # --- Merge sentiment aggregates (quarterly) ---
    if sent_q is not None and len(sent_q):
        df = df.merge(sent_q, on=["Ticker","Quarter"], how="left")
    else:
        for c in ["NewsCount","NewsPos","NewsNeg","NewsNeu","NewsPosMinusNeg"]:
            df[c] = np.nan

    # Core metrics
    df["BookValue"] = df[LBL_ASSETS] - df[LBL_LIAB]
    df["CapEx"] = df[LBL_CAPEX_1].fillna(0) + df[LBL_CAPEX_2].fillna(0)
    df["TotalLiabilities"] = df[LBL_LIAB]
    df["MarketCap"] = df["last_close"] * df["SharesOutstanding"]
    df["PB"] = safe_div(df["MarketCap"], df["BookValue"])

    df["PE"] = np.nan
    ni = df[LBL_NETINC]; pos = ni > 0
    df.loc[pos, "PE"] = safe_div(df.loc[pos, "MarketCap"], df.loc[pos, LBL_NETINC])

    df["RelativeReturn"] = df["excess_ret"]
    df["Turnover"] = safe_div(df["sum_volume"], df["SharesOutstanding"])

    core_cols = [
        "PB","PE","BookValue","CapEx","TotalLiabilities",
        "RelativeReturn","ret_vol_q",
        "sum_volume","avg_volume","vol_vol_q",
        "dollar_vol_mean","dollar_vol_sum","Turnover",
        # sentiment core
        "NewsCount","NewsPosMinusNeg","NewsPos","NewsNeg","NewsNeu",
    ]

    df = df[["Ticker","Quarter"] + core_cols + ["excess_ret"]].copy()
    df = df.sort_values(["Ticker","Quarter"]).reset_index(drop=True)

    # Momentum lags of excess return
    for L in [1,2,3,4]:
        df[f"RelativeReturn_lag{L}"] = df.groupby("Ticker")["RelativeReturn"].shift(L)

    # QoQ/YoY changes for features (model) – include key sentiment signals
    qoq_cols = ["PB","PE","BookValue","CapEx","TotalLiabilities",
                "RelativeReturn","ret_vol_q",
                "sum_volume","avg_volume","vol_vol_q",
                "dollar_vol_mean","dollar_vol_sum","Turnover",
                "NewsCount","NewsPosMinusNeg"]
    yoy_cols = qoq_cols[:]

    parts = []
    for _, g in df.groupby("Ticker", sort=False):
        g = g.copy()
        for c in qoq_cols:
            g[f"{c}_pctchg_qoq"] = pct_change_safe(g[c], 1)
        for c in yoy_cols:
            g[f"{c}_pctchg_yoy"] = pct_change_safe(g[c], 4)
        parts.append(g)
    df = pd.concat(parts, axis=0).sort_values(["Ticker","Quarter"]).reset_index(drop=True)

    # Logs for positive metrics (+ NewsCount)
    for c in ["PB","PE","BookValue","CapEx","TotalLiabilities",
              "sum_volume","avg_volume","dollar_vol_mean","dollar_vol_sum","Turnover",
              "NewsCount"]:
        df[f"log1p_{c}"] = safe_log1p(df[c])

    # Label (for training rows)
    df["next_excess_ret"] = df.groupby("Ticker")["excess_ret"].shift(-1)
    df["y_outperf"] = (df["next_excess_ret"] > 0).astype(float)

    # Final feature list (model)
    feature_cols = []
    feature_cols += core_cols
    feature_cols += [f"RelativeReturn_lag{L}" for L in [1,2,3,4]]
    feature_cols += [c for c in df.columns if c.endswith("_pctchg_qoq") or c.endswith("_pctchg_yoy")]
    feature_cols += [c for c in df.columns if c.startswith("log1p_")]

    return df, feature_cols, core_cols

# ----------------------------
# SHAP -> per-stock % contributions by metric family
# ----------------------------
def per_stock_contrib_percent(model, X_score: pd.DataFrame, feature_names, topn=6):
    out = []
    if not HAVE_SHAP:
        for _ in range(len(X_score)):
            out.append([{"metric": "Install SHAP", "percent": 100.0}])
        return out
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_score)
    shap_matrix = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) > 1 else shap_vals
    for i in range(shap_matrix.shape[0]):
        row = shap_matrix[i, :]
        abs_vals = np.abs(row)
        total = abs_vals.sum()
        if not np.isfinite(total) or total <= 0:
            out.append([])
            continue
        family = {}
        for v, f in zip(abs_vals, feature_names):
            fam = base_metric_name(f)
            family[fam] = family.get(fam, 0.0) + float(v)
        items = sorted([(k, v / total * 100.0) for k, v in family.items()],
                       key=lambda x: x[1], reverse=True)[:topn]
        out.append([{"metric": k, "percent": float(p)} for k, p in items])
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", default="data/fs_sp500.csv", help="Path to fundamentals CSV")
    ap.add_argument("--prices", default="data/ohlcv_sp500.csv", help="Path to ohlcv CSV")
    ap.add_argument("--sentiment", default="../sentiment/headlines_with_sentiment_DistilBERT.csv",
                    help="Path to DistilBERT sentiment CSV")
    ap.add_argument("--top-k", type=int, default=20, help="Top/Bottom K picks to export")
    ap.add_argument("--out", default="web_dist", help="Output folder (also place index.html here)")
    ap.add_argument("--min-prob", type=float, default=0.0, help="Min prob threshold for longs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    fs = pd.read_csv(args.fs)
    for c in ["ticker","label"]:
        fs[c] = fs[c].astype(str).str.strip()
    fs["ticker"] = fs["ticker"].str.upper()
    fs["val"] = pd.to_numeric(fs["val"], errors="coerce")
    for c in ["end","filed","start"]:
        if c in fs.columns:
            fs[c] = pd.to_datetime(fs[c], errors="coerce", utc=True).dt.tz_localize(None)

    px = pd.read_csv(args.prices)
    px["Ticker"] = px["Ticker"].astype(str).str.upper().str.strip()
    px["Date"] = pd.to_datetime(px["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    px["Close"] = pd.to_numeric(px["Close"], errors="coerce")
    px["Volume"] = pd.to_numeric(px["Volume"], errors="coerce")
    px = px.dropna(subset=["Ticker","Date","Close"])

    # Load & aggregate sentiment (DistilBERT)
    sent_q = load_sentiment_quarterly(args.sentiment)

    # Build features (+ sentiment)
    df, feature_cols, core_cols = build_feature_panel(fs, px, sent_q)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Train on all quarters < latest; score latest (common quarter)
    latest_q = df["Quarter"].max()
    train_mask = (df["Quarter"] < latest_q) & (df["y_outperf"].notna())
    score_mask = (df["Quarter"] == latest_q)

    # Winsorize & impute using train stats (for model only)
    X_all = df[feature_cols].copy()
    X_all = winsorize_df_trainwise(X_all, train_mask, 0.01, 0.99)
    train_medians = X_all[train_mask].median(numeric_only=True)
    X_all = X_all.fillna(train_medians)

    X_train = X_all[train_mask]
    y_train = df.loc[train_mask, "y_outperf"].astype(int)

    X_score = X_all[score_mask]
    meta_score = df.loc[score_mask, ["Ticker","Quarter"]].reset_index(drop=True)

    # Model (same params that performed best in your runs for GB)
    GB_PARAMS = dict(learning_rate=0.05, max_depth=3, n_estimators=200, random_state=42)
    model = GradientBoostingClassifier(**GB_PARAMS)
    model.fit(X_train, y_train)

    # Train AUC (sanity)
    try:
        p_train = model.predict_proba(X_train)[:, 1]
        auc_train = float(roc_auc_score(y_train, p_train))
    except Exception:
        auc_train = float("nan")

    # Score upcoming quarter
    p_score = model.predict_proba(X_score)[:, 1]
    scored = meta_score.copy()
    scored["p_outperf"] = p_score

    # SHAP per-stock contributions (percent by family)
    contrib_rows = per_stock_contrib_percent(model, X_score, feature_cols, topn=6)
    scored["contrib"] = contrib_rows

    # ===== Export TWO MOST RECENT quarters per ticker (actual if present, else imputed) =====
    metrics_for_values = [
        "PB","PE","BookValue","CapEx","TotalLiabilities",
        "RelativeReturn","ret_vol_q",
        "sum_volume","avg_volume","vol_vol_q",
        "dollar_vol_mean","dollar_vol_sum","Turnover",
        # include a couple of sentiment metrics for the UI table if desired
        "NewsCount","NewsPosMinusNeg"
    ]

    # Align IDs to pull imputed values when raw is NaN
    df_ids = df[["Ticker","Quarter"]].copy()
    df_ids["__row__"] = np.arange(len(df))
    X_all_ids = X_all.copy()
    X_all_ids["__row__"] = np.arange(len(X_all))
    X_all_with_ids = df_ids.merge(X_all_ids, on="__row__", how="left").set_index("__row__")

    rows = []
    for tkr in scored["Ticker"].unique():
        g = df[df["Ticker"] == tkr].sort_values("Quarter")
        if len(g) == 0:
            continue
        g_tail = g.tail(2)
        cur_row = g_tail.iloc[-1]
        prev_row = g_tail.iloc[-2] if len(g_tail) >= 2 else None

        cur_idx = int(cur_row.name)
        prev_idx = int(prev_row.name) if prev_row is not None else None

        cur_q = pd.to_datetime(cur_row["Quarter"]).date()
        prev_q = pd.to_datetime(prev_row["Quarter"]).date() if prev_row is not None else None

        out = {"Ticker": tkr, "CurQuarter": str(cur_q), "PrevQuarter": str(prev_q) if prev_q else ""}

        for m in metrics_for_values:
            cur_val = cur_row.get(m, np.nan)
            if pd.isna(cur_val):
                cur_val = X_all_with_ids.loc[cur_idx, m] if m in X_all_with_ids.columns else np.nan

            prev_val = np.nan
            if prev_row is not None:
                prev_val = prev_row.get(m, np.nan)
                if pd.isna(prev_val):
                    prev_val = X_all_with_ids.loc[prev_idx, m] if m in X_all_with_ids.columns else np.nan

            qoq = np.nan
            if pd.notna(cur_val) and pd.notna(prev_val) and prev_val != 0:
                qoq = (cur_val - prev_val) / prev_val

            out[f"{m}_cur"]  = float(cur_val) if pd.notna(cur_val) else np.nan
            out[f"{m}_prev"] = float(prev_val) if pd.notna(prev_val) else np.nan
            out[f"{m}_qoq"]  = float(qoq) if pd.notna(qoq) else np.nan

            # Back-compat columns (old frontend):
            out[m] = out[f"{m}_cur"]
            out[f"{m}_pctchg_qoq"] = out[f"{m}_qoq"]

        rows.append(out)

    vals_df = pd.DataFrame(rows)

    # Merge with scores (include common "Quarter" = global max for scoring)
    preds_flat = scored[["Ticker","Quarter","p_outperf"]].merge(vals_df, on="Ticker", how="left")
    preds_flat["Quarter"] = pd.to_datetime(preds_flat["Quarter"]).dt.date

    # Write CSV
    preds_flat.to_csv(out_dir / "predictions_latest.csv", index=False)

    # Global feature importances
    try:
        fi = pd.DataFrame({"feature": feature_cols,
                           "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        fi.to_csv(out_dir / "feature_importances.csv", index=False)
    except Exception:
        pass

    # Picks JSON (longs & shorts)
    K = int(args.top_k)
    min_prob = float(args.min_prob)
    scored_sorted = scored.sort_values("p_outperf", ascending=False).reset_index(drop=True)

    longs_df = scored_sorted[scored_sorted["p_outperf"] >= min_prob].head(K)
    shorts_df = scored_sorted.tail(K).sort_values("p_outperf", ascending=True)

    def pack_rows(dfpart):
        rows = []
        for _, r in dfpart.iterrows():
            rows.append({
                "ticker": str(r["Ticker"]),
                "prob": float(r["p_outperf"]),
                "contrib": r["contrib"]
            })
        return rows

    picks = {
        "quarter": str(pd.to_datetime(latest_q).date()),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": {"name": "GradientBoostingClassifier+DistilBERT",
                  "params": dict(learning_rate=0.05, max_depth=3, n_estimators=200, random_state=42)},
        "top_k": K,
        "min_prob": min_prob,
        "longs": pack_rows(longs_df),
        "shorts": pack_rows(shorts_df),
        "notes": "Contributions are SHAP-based percent impact aggregated by metric family; sentiment via DistilBERT."
    }
    with open(out_dir / "picks_latest.json", "w") as f:
        json.dump(picks, f, indent=2)

    # Meta
    meta = {
        "quarter": picks["quarter"],
        "generated_at": picks["generated_at"],
        "train_auc": auc_train,
        "n_train": int(len(X_train)),
        "n_score": int(len(X_score)),
        "feature_count": int(len(feature_cols)),
        "have_shap": HAVE_SHAP,
        "sentiment_source": str(Path(args.sentiment).name)
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Wrote to: {out_dir.resolve()}")
    print("  - picks_latest.json")
    print("  - predictions_latest.csv  (now includes sentiment + *_cur/*_prev/*_qoq)")
    print("  - meta.json")
    print("  - feature_importances.csv (if available)")
    print("\nServe from project root so the frontend can access:")
    print("  /web_dist/*.json|.csv, /data/SP500_2025.csv, /final_website_data/ohlcv_sp500.csv")

if __name__ == "__main__":
    main()
