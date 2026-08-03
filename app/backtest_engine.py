"""
backtest_engine.py

Core backtesting logic, ported from H1_backtest_hedge.py.
Supports long-only (no hedge) and long-only + long put overlay modes.
Notional matching for the put overlay supports three modes:
  - "real-beta": single beta computed over the full backtest window
  - "dynamic-beta": beta recalculated each period using only prior periods
  - "custom": fixed user-supplied beta value
"""

import os
import numpy as np
import pandas as pd


INPUT_DIR = "Input"
PORTFOLIO_FILE = "portfolio_BAM_default.csv"
RETURNS_FILE = "returns.csv"
INDEX_FILE = "index_returns.csv"
BENCHMARK_TICKER = "SPX"

FREQUENCY = "annual"
RISK_FREQUENCY = "monthly"
ROLLING_WINDOW = 3

TRANSACTION_COST_BPS = 10.0  # charged once at purchase; charged again ONLY if exercised (ITM at expiry)
COUNTERPARTY_HAIRCUT = 0.0   # % of payoff lost to settlement friction (0 = none)


def parse_period_col(col: str):
    col = col.strip()
    if "M" not in col:
        return None
    try:
        year, month = col.split("M")
        return pd.Period(f"{year.strip()}-{int(month.strip()):02d}", freq="M")
    except Exception:
        return None


def build_period_range(start_year, start_month, end_year, end_month):
    s = pd.Period(f"{start_year}-{start_month:02d}", freq="M")
    e = pd.Period(f"{end_year}-{end_month:02d}", freq="M")
    return pd.period_range(s, e, freq="M")


def periods_per_year(frequency):
    return {"monthly": 12, "quarterly": 4, "semi-annual": 2, "annual": 1}[frequency]


def periods_for_frequency(all_periods, frequency):
    if frequency == "monthly":
        return list(all_periods)
    if frequency == "quarterly":
        return [p for p in all_periods if p.month in (3, 6, 9, 12)]
    if frequency == "semi-annual":
        return [p for p in all_periods if p.month in (6, 12)]
    if frequency == "annual":
        return [p for p in all_periods if p.month == 12]
    raise ValueError(f"Unknown frequency {frequency!r}")


def period_start(rebal_period, frequency):
    if frequency == "monthly":
        return rebal_period
    if frequency == "quarterly":
        m = rebal_period.month - 2
        y = rebal_period.year
        if m <= 0:
            m += 12
            y -= 1
        return pd.Period(f"{y}-{m:02d}", freq="M")
    if frequency == "semi-annual":
        m = rebal_period.month - 5
        y = rebal_period.year
        if m <= 0:
            m += 12
            y -= 1
        return pd.Period(f"{y}-{m:02d}", freq="M")
    if frequency == "annual":
        return pd.Period(f"{rebal_period.year}-01", freq="M")
    raise ValueError(f"Unknown frequency {frequency!r}")


def months_in_period(rebal_period, frequency, available):
    start = period_start(rebal_period, frequency)
    return [p for p in pd.period_range(start, rebal_period, freq="M") if p in available]


def cagr(cum_return, n_periods, ppy):
    if n_periods <= 0:
        return np.nan
    return (1 + cum_return) ** (ppy / n_periods) - 1


def parse_val(val):
    if isinstance(val, str):
        try:
            return float(val.replace(",", ".").strip())
        except ValueError:
            return np.nan
    return np.nan


def normalise_ticker(t: str) -> str:
    return t.strip().split(" ")[0].upper()


def compound_series_to_frequency(monthly_series, target_frequency):
    if monthly_series is None or len(monthly_series) == 0:
        return pd.Series(dtype=float)
    monthly_series = monthly_series.sort_index()
    if target_frequency == "monthly":
        return monthly_series.dropna()
    if target_frequency == "quarterly":
        bucket_keys = monthly_series.index.asfreq("Q-DEC")
    elif target_frequency == "semi-annual":
        bucket_keys = pd.Index([f"{p.year}-H1" if p.month <= 6 else f"{p.year}-H2" for p in monthly_series.index])
    elif target_frequency == "annual":
        bucket_keys = monthly_series.index.year
    else:
        raise ValueError(f"Unknown target frequency {target_frequency!r}")

    temp = pd.DataFrame({"ret": monthly_series.values, "bucket": bucket_keys}, index=monthly_series.index)
    grouped, labels = [], []
    for bucket, g in temp.groupby("bucket", sort=True):
        vals = g["ret"].dropna().values
        if len(vals) == 0:
            continue
        grouped.append(np.prod(1 + vals) - 1)
        labels.append(bucket)
    return pd.Series(grouped, index=labels, dtype=float)


def resample_returns_for_risk(period_returns_series, frequency, risk_frequency, port_monthly_series=None):
    if risk_frequency == frequency:
        return period_returns_series, periods_per_year(frequency), frequency

    risk_ppy = periods_per_year(risk_frequency)
    freq_ppy = periods_per_year(frequency)

    if risk_ppy >= freq_ppy:
        if port_monthly_series is not None and len(port_monthly_series) > 0:
            if risk_frequency == "monthly":
                return port_monthly_series.dropna(), 12, "monthly"
            resampled = compound_series_to_frequency(port_monthly_series, risk_frequency)
            if len(resampled) > 0:
                return resampled, risk_ppy, risk_frequency
        return period_returns_series, freq_ppy, frequency

    group_size = int(freq_ppy // risk_ppy)
    vals = period_returns_series.values
    grouped = []
    for i in range(0, len(vals) - group_size + 1, group_size):
        chunk = vals[i:i + group_size]
        grouped.append(np.prod(1 + chunk) - 1)
    if not grouped:
        return period_returns_series, freq_ppy, frequency
    return pd.Series(grouped), risk_ppy, risk_frequency


def compute_metrics(results_df, return_col, frequency, risk_frequency, rolling_window, ppy,
                     monthly_returns_map=None):
    period_returns = results_df[return_col].dropna()
    n_periods = len(period_returns)
    cum_series = (1 + results_df[return_col]).cumprod()
    total_cumulative = cum_series.iloc[-1] - 1
    annualised_return = cagr(total_cumulative, n_periods, ppy)

    monthly_series = pd.Series(monthly_returns_map).sort_index() if monthly_returns_map else None
    risk_rets, risk_ppy, risk_freq_used = resample_returns_for_risk(
        period_returns, frequency, risk_frequency, port_monthly_series=monthly_series
    )

    std_risk = risk_rets.std()
    ann_std = std_risk * np.sqrt(risk_ppy)
    sharpe = annualised_return / ann_std if ann_std > 0 else np.nan

    neg_risk = risk_rets[risk_rets < 0]
    downside_dev = np.sqrt((neg_risk ** 2).mean()) * np.sqrt(risk_ppy) if len(neg_risk) > 0 else np.nan
    sortino = annualised_return / downside_dev if pd.notna(downside_dev) and downside_dev > 0 else np.nan

    rolling_peak = cum_series.cummax()
    drawdown_ser = cum_series / rolling_peak - 1
    max_drawdown = drawdown_ser.min()
    calmar = annualised_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    var95 = float(np.percentile(period_returns, 5)) if n_periods > 0 else np.nan
    hit_rate = (period_returns > 0).sum() / n_periods if n_periods else np.nan
    std_period_return = period_returns.std()

    rolling_cagrs = []
    for i in range(n_periods - rolling_window + 1):
        w = period_returns.iloc[i:i + rolling_window]
        rolling_cagrs.append(cagr(np.prod(1 + w) - 1, rolling_window, ppy))

    best_idx = period_returns.idxmax() if n_periods else None
    worst_idx = period_returns.idxmin() if n_periods else None
    best_period = str(results_df.loc[best_idx, "period"]) if best_idx is not None else None
    worst_period = str(results_df.loc[worst_idx, "period"]) if worst_idx is not None else None
    best_period_return = float(period_returns.loc[best_idx]) if best_idx is not None else None
    worst_period_return = float(period_returns.loc[worst_idx]) if worst_idx is not None else None

    return {
        "n_periods": n_periods,
        "total_cumulative": total_cumulative,
        "annualised_return": annualised_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "var_95": var95,
        "hit_rate": hit_rate,
        "std_period_return": std_period_return,
        "risk_freq_used": risk_freq_used,
        "best_rolling_cagr": max(rolling_cagrs) if rolling_cagrs else np.nan,
        "worst_rolling_cagr": min(rolling_cagrs) if rolling_cagrs else np.nan,
        "best_period": best_period,
        "best_period_return": best_period_return,
        "worst_period": worst_period,
        "worst_period_return": worst_period_return,
    }


def compute_benchmark_stats(results_df, return_col, index_df, benchmark_ticker, frequency, available, ppy, annualised_return):
    if index_df is None or benchmark_ticker not in index_df.index:
        return None

    bench_period_rets = {}
    for rp in results_df["period"]:
        sub = months_in_period(rp, frequency, available)
        if not sub:
            continue
        rets, ok = [], True
        for m in sub:
            if m not in index_df.columns:
                ok = False
                break
            r = index_df.loc[benchmark_ticker, m]
            if isinstance(r, pd.Series):
                r = r.iloc[0]
            if pd.isna(r):
                ok = False
                break
            rets.append(r)
        if ok and rets:
            bench_period_rets[rp] = np.prod([1 + r for r in rets]) - 1

    bench_series = pd.Series(bench_period_rets)
    port_series = results_df.set_index("period")[return_col]
    common = port_series.index.intersection(bench_series.index)

    if len(common) <= 1:
        return None

    port_aligned = port_series.loc[common]
    bench_aligned = bench_series.loc[common]
    n_common = len(common)
    bench_cum = np.prod(1 + bench_aligned) - 1
    bench_cagr = cagr(bench_cum, n_common, ppy)
    alpha = annualised_return - bench_cagr

    risk_ppy = periods_per_year(frequency)
    tracking_error = float(np.std(port_aligned.values - bench_aligned.values, ddof=1)) * np.sqrt(risk_ppy)
    info_ratio = alpha / tracking_error if tracking_error > 0 else np.nan

    cov_matrix = np.cov(port_aligned.values, bench_aligned.values)
    beta_est = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else np.nan

    up_idx = bench_aligned[bench_aligned > 0].index
    down_idx = bench_aligned[bench_aligned < 0].index
    up_capture = down_capture = np.nan
    if len(up_idx):
        pu = cagr(np.prod(1 + port_aligned.loc[up_idx]) - 1, len(up_idx), ppy)
        bu = cagr(np.prod(1 + bench_aligned.loc[up_idx]) - 1, len(up_idx), ppy)
        up_capture = pu / bu if bu != 0 else np.nan
    if len(down_idx):
        pd_ = cagr(np.prod(1 + port_aligned.loc[down_idx]) - 1, len(down_idx), ppy)
        bd = cagr(np.prod(1 + bench_aligned.loc[down_idx]) - 1, len(down_idx), ppy)
        down_capture = pd_ / bd if bd != 0 else np.nan

    outperf_rate = float((port_aligned.values > bench_aligned.values).sum() / n_common)

    return {
        "n_common": n_common,
        "cagr": bench_cagr,
        "alpha": alpha,
        "tracking_error": tracking_error,
        "info_ratio": info_ratio,
        "beta": beta_est,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "outperf_rate": outperf_rate,
    }, bench_series


def get_normalised_ticker_map(returns_df_index):
    norm_to_orig = {}
    for orig in returns_df_index:
        base = normalise_ticker(orig)
        if base not in norm_to_orig:
            norm_to_orig[base] = orig
    return norm_to_orig


def load_returns_csv(path):
    raw = pd.read_csv(path, sep=";", dtype=str)
    raw.columns = raw.columns.str.strip()
    first_col = raw.columns[0]
    raw[first_col] = raw[first_col].str.strip()
    raw = raw.set_index(first_col)

    period_cols = {}
    for col in raw.columns:
        p = parse_period_col(col)
        if p is not None:
            period_cols[col] = p
    raw = raw[list(period_cols.keys())]
    df = raw.map(lambda v: parse_val(v) / 100.0)
    df.columns = [period_cols[c] for c in df.columns]
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_index_csv(path):
    index_raw = pd.read_csv(path, sep=";", dtype=str)
    index_raw.columns = index_raw.columns.str.strip()
    index_raw["ticker"] = index_raw["ticker"].str.strip()
    index_raw = index_raw.set_index("ticker")
    index_raw = index_raw.drop(columns=["index"], errors="ignore")
    idx_period_cols = {}
    for col in index_raw.columns:
        p = parse_period_col(col)
        if p is not None:
            idx_period_cols[col] = p
    index_raw = index_raw[list(idx_period_cols.keys())]
    index_df = index_raw.map(lambda v: parse_val(v) / 100.0)
    index_df.columns = [idx_period_cols[c] for c in index_df.columns]
    index_df = index_df[~index_df.index.duplicated(keep="first")]
    return index_df


def load_portfolio_csv(path):
    portfolio = pd.read_csv(path, parse_dates=["date"])
    portfolio["period"] = portfolio["date"].dt.to_period("M")
    portfolio["ticker_orig"] = portfolio["ticker"].astype(str).str.strip()
    portfolio["ticker"] = portfolio["ticker_orig"].apply(normalise_ticker)
    return portfolio


# ---------- LONG PUT OVERLAY ----------
def compute_full_period_beta(port_returns, spx_returns):
    common = port_returns.index.intersection(spx_returns.index)
    if len(common) < 2:
        return np.nan
    p = port_returns.loc[common].values
    s = spx_returns.loc[common].values
    cov = np.cov(p, s)
    return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan


def compute_dynamic_beta_series(port_returns, spx_returns):
    """
    For each period, beta uses only periods strictly before it (expanding window).
    Returns a dict {period: beta_or_nan}.
    """
    common = sorted(port_returns.index.intersection(spx_returns.index))
    betas = {}
    for i, period in enumerate(common):
        if i < 2:
            betas[period] = np.nan
            continue
        hist_periods = common[:i]
        p = port_returns.loc[hist_periods].values
        s = spx_returns.loc[hist_periods].values
        cov = np.cov(p, s)
        betas[period] = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
    return betas


def apply_long_put_overlay(results_df, otm_pct, premium_pct, notional_method, custom_beta=None,
                            txn_cost_bps=TRANSACTION_COST_BPS, counterparty_haircut=COUNTERPARTY_HAIRCUT):
    """
    Adds protective put P&L on top of the long-only results_df.
    otm_pct: e.g. 0.10 means strike = 90% of spot (10% OTM).
    premium_pct: annual premium as % of notional, e.g. 0.0419.
    notional_method: "real-beta" | "dynamic-beta" | "custom".
    txn_cost_bps: one-way transaction cost in bps, charged once at purchase and again if exercised.
    counterparty_haircut: fraction of payoff lost to settlement friction (0 = none).
    Returns results_df with hedge fields populated.
    """
    results_df = results_df.copy()

    port_series = results_df.set_index("period")["portfolio_return_adjusted"]
    spx_series = results_df.set_index("period")["spx_return"].dropna()

    if notional_method == "real-beta":
        beta_val = compute_full_period_beta(port_series, spx_series)
        beta_map = {p: beta_val for p in results_df["period"]}
    elif notional_method == "dynamic-beta":
        beta_map = compute_dynamic_beta_series(port_series, spx_series)
    elif notional_method == "custom":
        beta_val = custom_beta if custom_beta is not None else 1.0
        beta_map = {p: beta_val for p in results_df["period"]}
    else:
        raise ValueError(f"Unknown notional_method {notional_method!r}")

    strike_pct = 1 - otm_pct  # e.g. otm=0.10 -> strike at 90% of spot
    txn_unit = txn_cost_bps / 10000.0

    exercised_list, premium_list, payoff_list, txn_list, net_pnl_list, hedged_ret_list = [], [], [], [], [], []

    for _, row in results_df.iterrows():
        period = row["period"]
        spx_ret = row["spx_return"]
        port_ret = row["portfolio_return_adjusted"]
        beta = beta_map.get(period, np.nan)

        if pd.isna(beta) or pd.isna(spx_ret):
            exercised_list.append(False)
            premium_list.append(0.0)
            payoff_list.append(0.0)
            txn_list.append(0.0)
            net_pnl_list.append(0.0)
            hedged_ret_list.append(port_ret)
            continue

        notional = beta
        premium_paid = premium_pct * notional

        spx_level_end = 1 + spx_ret
        itm = spx_level_end < strike_pct
        payoff_raw = max(strike_pct - spx_level_end, 0.0) * (1 - counterparty_haircut)
        payoff = payoff_raw * notional

        txn_cost_pct = txn_unit * (2 if itm else 1)
        txn_cost = txn_cost_pct * notional

        net_pnl = payoff - premium_paid - txn_cost
        hedged_ret = port_ret + net_pnl

        exercised_list.append(bool(itm))
        premium_list.append(premium_paid)
        payoff_list.append(payoff)
        txn_list.append(txn_cost)
        net_pnl_list.append(net_pnl)
        hedged_ret_list.append(hedged_ret)

    results_df["exercised"] = exercised_list
    results_df["premium_paid_pct_initial"] = premium_list
    results_df["payoff_pct_initial"] = payoff_list
    results_df["txn_cost_pct_initial"] = txn_list
    results_df["net_hedge_pnl_pct_port"] = net_pnl_list
    results_df["hedged_return"] = hedged_ret_list
    results_df["hedged_cumulative_return"] = (1 + results_df["hedged_return"]).cumprod() - 1

    return results_df


def run_long_only_backtest(
    start_year, start_month, end_year, end_month,
    input_dir=INPUT_DIR,
    portfolio_file=PORTFOLIO_FILE,
    returns_file=RETURNS_FILE,
    index_file=INDEX_FILE,
    benchmark_ticker=BENCHMARK_TICKER,
    frequency=FREQUENCY,
    risk_frequency=RISK_FREQUENCY,
    rolling_window=ROLLING_WINDOW,
    weight_pct=1.0,
    long_put=None,
):
    """
    Runs the long-only portfolio backtest, optionally with a long put overlay.

    long_put: optional dict {
        "otm_pct": float,            # e.g. 0.10 for 10% OTM
        "premium_pct": float,        # e.g. 0.0419
        "notional_method": str,      # "real-beta" | "dynamic-beta" | "custom"
        "custom_beta": float|None,
        "txn_cost_bps": float,       # optional, defaults to 10.0
        "counterparty_haircut": float,  # optional, defaults to 0.0
    }
    If None, hedged fields simply mirror unhedged fields (no overlay).

    Returns (results_df, metrics_dict) where metrics_dict has keys
    "unhedged", "hedged", "put_stats".
    """
    ppy = periods_per_year(frequency)

    returns_df = load_returns_csv(os.path.join(input_dir, returns_file))
    norm_to_orig = get_normalised_ticker_map(returns_df.index)

    all_periods = build_period_range(start_year, start_month, end_year, end_month)
    available = [p for p in all_periods if p in returns_df.columns]
    if not available:
        raise ValueError("No overlapping periods between returns file and requested window.")
    returns_df = returns_df[available]

    index_df = load_index_csv(os.path.join(input_dir, index_file))

    portfolio = load_portfolio_csv(os.path.join(input_dir, portfolio_file))
    portfolio = portfolio[(portfolio["period"] >= available[0]) & (portfolio["period"] <= available[-1])]

    rebalance_periods = periods_for_frequency(available, frequency)
    portfolio_by_period = {p: g for p, g in portfolio.groupby("period")}

    results = []
    monthly_port_rets = {}
    tickers_by_period = {}

    for rebal_period in rebalance_periods:
        weight_period = period_start(rebal_period, frequency)
        if weight_period not in portfolio_by_period:
            continue
        port_group = portfolio_by_period[weight_period]
        sub_months = months_in_period(rebal_period, frequency, available)
        if not sub_months:
            continue

        weighted_returns, total_weight_used = [], 0.0
        monthly_weighted = {m: [] for m in sub_months}
        monthly_weights = {m: 0.0 for m in sub_months}
        tickers_by_period[rebal_period] = set(port_group["ticker"].tolist())

        for _, prow in port_group.iterrows():
            ticker_norm = prow["ticker"]
            weight = prow["weight"] * weight_pct
            orig_ticker = norm_to_orig.get(ticker_norm)
            if orig_ticker is None:
                continue

            monthly_rets, skip = [], False
            for m in sub_months:
                if m not in returns_df.columns:
                    skip = True
                    break
                r = returns_df.loc[orig_ticker, m]
                if isinstance(r, pd.Series):
                    r = r.iloc[0]
                if pd.isna(r):
                    skip = True
                    break
                monthly_rets.append(r)
            if skip or not monthly_rets:
                continue

            compounded = np.prod([1 + r for r in monthly_rets]) - 1
            weighted_returns.append(compounded * weight)
            total_weight_used += weight
            for m, r in zip(sub_months, monthly_rets):
                monthly_weighted[m].append(r * weight)
                monthly_weights[m] += weight

        if weighted_returns:
            raw_return = sum(weighted_returns)
            adjusted_return = raw_return / total_weight_used if total_weight_used > 0 else np.nan
            for m in sub_months:
                if monthly_weights[m] > 0:
                    monthly_port_rets[m] = sum(monthly_weighted[m]) / monthly_weights[m]

            spx_ret = None
            if benchmark_ticker in index_df.index:
                spx_rets = []
                ok = True
                for m in sub_months:
                    if m not in index_df.columns:
                        ok = False
                        break
                    r = index_df.loc[benchmark_ticker, m]
                    if isinstance(r, pd.Series):
                        r = r.iloc[0]
                    if pd.isna(r):
                        ok = False
                        break
                    spx_rets.append(r)
                if ok and spx_rets:
                    spx_ret = np.prod([1 + r for r in spx_rets]) - 1

            results.append({
                "period": rebal_period,
                "date": rebal_period.to_timestamp(how="end").normalize(),
                "portfolio_return_adjusted": adjusted_return,
                "spx_return": spx_ret,
                "weight_coverage": total_weight_used,
                "n_holdings": len(weighted_returns),
            })

    if not results:
        raise RuntimeError("Backtest produced zero results. Check portfolio dates align with rebalance periods.")

    results_df = pd.DataFrame(results).sort_values("period").reset_index(drop=True)
    results_df["cumulative_return"] = (1 + results_df["portfolio_return_adjusted"]).cumprod() - 1

    if long_put is not None:
        results_df = apply_long_put_overlay(
            results_df,
            otm_pct=long_put["otm_pct"],
            premium_pct=long_put["premium_pct"],
            notional_method=long_put["notional_method"],
            custom_beta=long_put.get("custom_beta"),
            txn_cost_bps=long_put.get("txn_cost_bps", TRANSACTION_COST_BPS),
            counterparty_haircut=long_put.get("counterparty_haircut", COUNTERPARTY_HAIRCUT),
        )
    else:
        results_df["exercised"] = False
        results_df["premium_paid_pct_initial"] = 0.0
        results_df["payoff_pct_initial"] = 0.0
        results_df["txn_cost_pct_initial"] = 0.0
        results_df["net_hedge_pnl_pct_port"] = 0.0
        results_df["hedged_return"] = results_df["portfolio_return_adjusted"]
        results_df["hedged_cumulative_return"] = results_df["cumulative_return"]

    unhedged_metrics = compute_metrics(
        results_df, "portfolio_return_adjusted", frequency, risk_frequency, rolling_window, ppy,
        monthly_returns_map=monthly_port_rets,
    )
    hedged_metrics = compute_metrics(
        results_df, "hedged_return", frequency, risk_frequency, rolling_window, ppy,
        monthly_returns_map=None,  # matches original: no true monthly hedged-return series exists
    )

    bench_unhedged = compute_benchmark_stats(
        results_df, "portfolio_return_adjusted", index_df, benchmark_ticker, frequency, available, ppy,
        unhedged_metrics["annualised_return"],
    )
    bench_hedged = compute_benchmark_stats(
        results_df, "hedged_return", index_df, benchmark_ticker, frequency, available, ppy,
        hedged_metrics["annualised_return"],
    )
    if bench_unhedged is not None:
        unhedged_metrics["benchmark"] = bench_unhedged[0]
    if bench_hedged is not None:
        hedged_metrics["benchmark"] = bench_hedged[0]

    avg_holdings = float(results_df["n_holdings"].mean())
    sorted_ps = sorted(tickers_by_period.keys())
    turnover_rates = []
    for i in range(1, len(sorted_ps)):
        prev = tickers_by_period[sorted_ps[i - 1]]
        curr = tickers_by_period[sorted_ps[i]]
        union = len(prev | curr)
        if union > 0:
            turnover_rates.append((len(curr - prev) + len(prev - curr)) / union)
    avg_turnover = float(np.mean(turnover_rates)) if turnover_rates else None

    n_years = len(results_df)
    n_exercised = int(results_df["exercised"].sum())
    total_premium = float(results_df["premium_paid_pct_initial"].sum())
    total_payoff = float(results_df["payoff_pct_initial"].sum())
    net_cost = total_premium - total_payoff
    corr_eff = None
    if results_df["net_hedge_pnl_pct_port"].std() > 0:
        corr_eff = float(np.corrcoef(
            results_df["portfolio_return_adjusted"], results_df["net_hedge_pnl_pct_port"]
        )[0, 1])
    dd_reduction = None
    if pd.notna(unhedged_metrics["max_drawdown"]) and pd.notna(hedged_metrics["max_drawdown"]):
        dd_reduction = float(abs(unhedged_metrics["max_drawdown"]) - abs(hedged_metrics["max_drawdown"]))

    put_stats = {
        "exercise_rate": (n_exercised / n_years) if n_years else 0.0,
        "n_exercised": n_exercised,
        "n_years": n_years,
        "total_premium_paid_pct_initial": total_premium,
        "total_payoff_received_pct_initial": total_payoff,
        "net_hedge_cost_pct_initial": net_cost,
        "avg_annual_premium_drag": total_premium / n_years if n_years else 0.0,
        "hedge_effectiveness_corr": corr_eff,
        "drawdown_reduction": dd_reduction,
        "avg_holdings": avg_holdings,
        "avg_turnover": avg_turnover,
    }

    return results_df, {
        "unhedged": unhedged_metrics,
        "hedged": hedged_metrics,
        "put_stats": put_stats,
    }
