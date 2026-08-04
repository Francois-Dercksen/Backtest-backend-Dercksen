"""
backtest_engine.py

Modular, leg-based backtesting engine.

Architecture
------------
Every selected strategy (Long Portfolio, Long Put, Long Call, ...) is a "leg".
Each leg produces, per rebalance period:
  - capital_weight: fraction of $1 total equity this leg consumes that period
  - return_contribution: $ P&L this leg adds that period, as a fraction of $1 equity

The Net Portfolio is the sum of all active legs, plus a financing adjustment:
  total_capital = sum(capital_weight across legs)
  excess = total_capital - 1.0
  period_rf = (1 + risk_free_rate) ** (months_in_period / 12) - 1
  financing_return = -excess * period_rf      # expense if geared, income if holding cash
  net_return = sum(return_contribution across legs) + financing_return
  net_gearing = total_capital                 # 1.0 = fully invested, >1.0 = geared

Short Call and Short Put are premium-income overlays with NO margin/collateral
accounting: capital_weight = 0 for both, so they never appear on the gearing
chart. Their notional exposure is tracked separately as "short_exposure" on the
Net Portfolio results_df (options-only), so a dedicated short-exposure chart can
be built from it.

Short Portfolio DOES consume capital (capital_weight = weight_pct) and already
appears in the gearing calc, since shorting the portfolio ties up real
capital/collateral the same way Long Portfolio does. Its notional is ALSO
tracked separately as "short_portfolio_exposure" on the Net Portfolio
results_df, so the exposure chart can show portfolio-driven short exposure
alongside option-driven short exposure as two distinct series -- they are
economically different (one is capital-backed, one is not) and should not be
summed into a single number.

Leg-level metrics (compute_leg_metrics) measure each leg's return_contribution
directly as a fraction of $1 TOTAL equity -- NOT return_contribution divided by
capital_weight. Dividing by capital_weight is wrong for options legs: they spend
~100% of their allocated capital on premium every period, so any non-exercised
period would produce leg_return = -1.0 (-100%), and cumprod() would permanently
zero the series the first time that happens. Using return_contribution directly
avoids this and matches how the leg actually affects Net Portfolio wealth.

The Net Portfolio results_df also carries a "spx_return_used" column (the
benchmark return for that same period, when available) so the report's Returns
and Annual Alpha vs SPX charts can be built directly from the net results
without a second lookup.

This lets any combination of legs be run together (or standalone) without the
engine needing to special-case combinations.
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
RISK_FREE_RATE = 0.045  # effective annual rate; static input for now

TRANSACTION_COST_BPS = 10.0
COUNTERPARTY_HAIRCUT = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Period / date helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Risk resampling / metrics (unchanged math, generic over any return column)
# ─────────────────────────────────────────────────────────────────────────────

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
    if n_periods == 0:
        return {"n_periods": 0}

    cum_series = (1 + results_df[return_col].fillna(0)).cumprod()
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


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Shared per-period market context (built once, used by every leg)
# ─────────────────────────────────────────────────────────────────────────────

def build_market_context(rebalance_periods, frequency, available, returns_df, norm_to_orig,
                          index_df, benchmark_ticker, portfolio_by_period):
    """
    Computes, for every rebalance period:
      - raw_portfolio_return: weighted-average return of the long portfolio's
        holdings, using each holding's OWN relative weight (normalised to sum
        to the weight actually matched to data). This is "return per dollar
        allocated to the long-portfolio leg", independent of how much capital
        is allocated to that leg overall.
      - spx_return: benchmark return over the same period (used by options legs
        and carried through to the Net Portfolio results_df as "spx_return_used").
      - monthly_port_rets: monthly-frequency version of the portfolio return,
        used for risk-frequency resampling in metrics.
      - tickers_by_period: holdings set per period, used for turnover.
    """
    raw_return_map, spx_return_map, monthly_port_rets, tickers_by_period = {}, {}, {}, {}
    holdings_count = {}

    for rebal_period in rebalance_periods:
        weight_period = period_start(rebal_period, frequency)
        sub_months = months_in_period(rebal_period, frequency, available)
        if not sub_months:
            continue

        if weight_period in portfolio_by_period:
            port_group = portfolio_by_period[weight_period]
            tickers_by_period[rebal_period] = set(port_group["ticker"].tolist())

            weighted_returns, total_weight_used = [], 0.0
            monthly_weighted = {m: [] for m in sub_months}
            monthly_weights = {m: 0.0 for m in sub_months}

            for _, prow in port_group.iterrows():
                ticker_norm = prow["ticker"]
                weight = prow["weight"]
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

            if weighted_returns and total_weight_used > 0:
                raw_return_map[rebal_period] = sum(weighted_returns) / total_weight_used
                holdings_count[rebal_period] = len(weighted_returns)
                for m in sub_months:
                    if monthly_weights[m] > 0:
                        monthly_port_rets[m] = sum(monthly_weighted[m]) / monthly_weights[m]

        if benchmark_ticker in index_df.index:
            spx_rets, ok = [], True
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
                spx_return_map[rebal_period] = np.prod([1 + r for r in spx_rets]) - 1

    return {
        "raw_portfolio_return": raw_return_map,
        "spx_return": spx_return_map,
        "monthly_port_rets": monthly_port_rets,
        "tickers_by_period": tickers_by_period,
        "holdings_count": holdings_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Beta helpers (used by Long Put notional matching)
# ─────────────────────────────────────────────────────────────────────────────

def compute_full_period_beta(port_returns, spx_returns):
    common = port_returns.index.intersection(spx_returns.index)
    if len(common) < 2:
        return np.nan
    p = port_returns.loc[common].values
    s = spx_returns.loc[common].values
    cov = np.cov(p, s)
    return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan


def compute_dynamic_beta_series(port_returns, spx_returns):
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


# ─────────────────────────────────────────────────────────────────────────────
# LEG BUILDERS
#
# Every leg builder returns a DataFrame indexed by "period" with at least:
#   capital_weight       -- fraction of $1 equity consumed this period
#   return_contribution  -- $ P&L added this period, as a fraction of $1 equity
# plus leg-specific diagnostic columns for reporting (e.g. "notional" for
# options legs, used to build exposure charts independent of the capital ledger).
# ─────────────────────────────────────────────────────────────────────────────

def build_long_portfolio_leg(rebalance_periods, market, weight_pct=1.0):
    rows = []
    for rp in rebalance_periods:
        raw_ret = market["raw_portfolio_return"].get(rp, np.nan)
        n_holdings = market["holdings_count"].get(rp, 0)
        rows.append({
            "period": rp,
            "portfolio_return_raw": raw_ret,
            "notional": weight_pct,
            "capital_weight": weight_pct,
            "return_contribution": raw_ret * weight_pct if pd.notna(raw_ret) else 0.0,
            "n_holdings": n_holdings,
        })
    return pd.DataFrame(rows)


def build_short_portfolio_leg(rebalance_periods, market, weight_pct=1.0):
    rows = []
    for rp in rebalance_periods:
        raw_ret = market["raw_portfolio_return"].get(rp, np.nan)
        n_holdings = market["holdings_count"].get(rp, 0)
        rows.append({
            "period": rp,
            "portfolio_return_raw": raw_ret,
            "notional": weight_pct,
            "capital_weight": weight_pct,
            "return_contribution": -raw_ret * weight_pct if pd.notna(raw_ret) else 0.0,
            "n_holdings": n_holdings,
        })
    return pd.DataFrame(rows)


def build_long_put_leg(rebalance_periods, market, otm_pct, premium_pct, notional_method,
                        custom_beta=None, txn_cost_bps=TRANSACTION_COST_BPS,
                        counterparty_haircut=COUNTERPARTY_HAIRCUT):
    port_series = pd.Series(market["raw_portfolio_return"])
    spx_series = pd.Series(market["spx_return"]).dropna()

    if notional_method == "real-beta":
        beta_val = compute_full_period_beta(port_series, spx_series)
        beta_map = {p: beta_val for p in rebalance_periods}
    elif notional_method == "dynamic-beta":
        beta_map = compute_dynamic_beta_series(port_series, spx_series)
    elif notional_method == "custom":
        beta_val = custom_beta if custom_beta is not None else 1.0
        beta_map = {p: beta_val for p in rebalance_periods}
    else:
        raise ValueError(f"Unknown notional_method {notional_method!r}")

    strike_pct = 1 - otm_pct
    txn_unit = txn_cost_bps / 10000.0

    rows = []
    for rp in rebalance_periods:
        spx_ret = market["spx_return"].get(rp, np.nan)
        beta = beta_map.get(rp, np.nan)

        if pd.isna(beta) or pd.isna(spx_ret):
            rows.append({
                "period": rp, "exercised": False, "notional": 0.0,
                "premium_paid_pct_port": 0.0, "payoff_pct_port": 0.0, "txn_cost_pct_port": 0.0,
                "capital_weight": 0.0, "return_contribution": 0.0,
            })
            continue

        notional = beta
        premium_paid = premium_pct * notional
        spx_level_end = 1 + spx_ret
        itm = spx_level_end < strike_pct
        payoff_raw = max(strike_pct - spx_level_end, 0.0) * (1 - counterparty_haircut)
        payoff = payoff_raw * notional
        txn_cost = txn_unit * (2 if itm else 1) * notional
        net_pnl = payoff - premium_paid - txn_cost

        rows.append({
            "period": rp, "exercised": bool(itm), "notional": notional,
            "premium_paid_pct_port": premium_paid, "payoff_pct_port": payoff,
            "txn_cost_pct_port": txn_cost,
            "capital_weight": premium_paid + txn_cost,
            "return_contribution": net_pnl,
        })
    return pd.DataFrame(rows)


def build_long_call_leg(rebalance_periods, market, strike_pct, premium_pct, sizing_mode,
                         fixed_pct=None, weight_pct=None):
    """
    sizing_mode == "fixed":
      notional = fixed_pct (relative to $1 portfolio value), recalculated fresh
      each period. Capital spent = notional * premium_pct.
    sizing_mode == "weight":
      capital spent = weight_pct (spend this fraction of that period's
      portfolio value fully on premium). notional = capital_spent / premium_pct.
    """
    rows = []
    for rp in rebalance_periods:
        spx_ret = market["spx_return"].get(rp, np.nan)
        if pd.isna(spx_ret) or premium_pct <= 0:
            rows.append({
                "period": rp, "exercised": False, "notional": 0.0,
                "premium_paid_pct_port": 0.0, "payoff_pct_port": 0.0,
                "capital_weight": 0.0, "return_contribution": 0.0,
            })
            continue

        if sizing_mode == "fixed":
            notional = fixed_pct if fixed_pct is not None else 0.0
            premium_paid = notional * premium_pct
        elif sizing_mode == "weight":
            premium_paid = weight_pct if weight_pct is not None else 0.0
            notional = premium_paid / premium_pct
        else:
            raise ValueError(f"Unknown sizing_mode {sizing_mode!r}")

        spx_level_end = 1 + spx_ret
        payoff_pct_of_notional = max(spx_level_end - strike_pct, 0.0)
        payoff = payoff_pct_of_notional * notional
        itm = payoff_pct_of_notional > 0
        net_pnl = payoff - premium_paid

        rows.append({
            "period": rp, "exercised": bool(itm), "notional": notional,
            "premium_paid_pct_port": premium_paid, "payoff_pct_port": payoff,
            "capital_weight": premium_paid,
            "return_contribution": net_pnl,
        })
    return pd.DataFrame(rows)


def build_short_call_leg(rebalance_periods, market, strike_pct, premium_pct, notional_pct):
    """
    Premium-income overlay. No margin/collateral accounting: capital_weight is
    always 0, so this leg never contributes to net_gearing. "notional" is
    still tracked on the results_df for a dedicated short-exposure chart.
    notional_pct: fixed fraction of portfolio value, recalculated fresh each period.
    """
    rows = []
    for rp in rebalance_periods:
        spx_ret = market["spx_return"].get(rp, np.nan)
        if pd.isna(spx_ret) or premium_pct < 0:
            rows.append({
                "period": rp, "exercised": False, "notional": 0.0,
                "premium_received_pct_port": 0.0, "payoff_owed_pct_port": 0.0,
                "capital_weight": 0.0, "return_contribution": 0.0,
            })
            continue

        notional = notional_pct
        premium_received = notional * premium_pct
        spx_level_end = 1 + spx_ret
        payoff_pct_of_notional = max(spx_level_end - strike_pct, 0.0)
        payoff_owed = payoff_pct_of_notional * notional
        itm = payoff_pct_of_notional > 0
        net_pnl = premium_received - payoff_owed

        rows.append({
            "period": rp, "exercised": bool(itm), "notional": notional,
            "premium_received_pct_port": premium_received, "payoff_owed_pct_port": payoff_owed,
            "capital_weight": 0.0,
            "return_contribution": net_pnl,
        })
    return pd.DataFrame(rows)


def build_short_put_leg(rebalance_periods, market, strike_pct, premium_pct, notional_pct):
    """
    Premium-income overlay. No margin/collateral accounting: capital_weight is
    always 0, so this leg never contributes to net_gearing. "notional" is
    still tracked on the results_df for a dedicated short-exposure chart.
    notional_pct: fixed fraction of portfolio value, recalculated fresh each period.
    """
    rows = []
    for rp in rebalance_periods:
        spx_ret = market["spx_return"].get(rp, np.nan)
        if pd.isna(spx_ret) or premium_pct < 0:
            rows.append({
                "period": rp, "exercised": False, "notional": 0.0,
                "premium_received_pct_port": 0.0, "payoff_owed_pct_port": 0.0,
                "capital_weight": 0.0, "return_contribution": 0.0,
            })
            continue

        notional = notional_pct
        premium_received = notional * premium_pct
        spx_level_end = 1 + spx_ret
        payoff_pct_of_notional = max(strike_pct - spx_level_end, 0.0)
        payoff_owed = payoff_pct_of_notional * notional
        itm = payoff_pct_of_notional > 0
        net_pnl = premium_received - payoff_owed

        rows.append({
            "period": rp, "exercised": bool(itm), "notional": notional,
            "premium_received_pct_port": premium_received, "payoff_owed_pct_port": payoff_owed,
            "capital_weight": 0.0,
            "return_contribution": net_pnl,
        })
    return pd.DataFrame(rows)


LEG_BUILDERS = {
    "long_portfolio": build_long_portfolio_leg,
    "short_portfolio": build_short_portfolio_leg,
    "long_put": build_long_put_leg,
    "long_call": build_long_call_leg,
    "short_call": build_short_call_leg,
    "short_put": build_short_put_leg,
}

# Legs whose capital_weight participates in the gearing calc. Short options
# are intentionally excluded (see docstrings above) — kept here as an explicit
# allowlist so the exclusion is a visible design decision, not an accident.
GEARING_ELIGIBLE_TYPES = {"long_portfolio", "short_portfolio", "long_put", "long_call"}
SHORT_EXPOSURE_TYPES = {"short_call", "short_put"}


# ─────────────────────────────────────────────────────────────────────────────
# Net Portfolio combination (capital ledger + financing/gearing)
# ─────────────────────────────────────────────────────────────────────────────

def combine_legs_into_net_portfolio(leg_frames, leg_types, rebalance_periods, frequency, available,
                                     risk_free_rate, spx_return_map=None):
    """
    Tracks two distinct short-exposure series, since they are economically
    different and should not be summed:
      - short_exposure: notional from Short Call / Short Put (capital-free,
        no margin modelled -- excluded from the gearing calc entirely).
      - short_portfolio_exposure: notional from Short Portfolio (capital-backed,
        already counted in net_gearing via capital_weight).
    """
    spx_return_map = spx_return_map or {}
    rows = []
    for rp in rebalance_periods:
        total_capital = 0.0
        total_return = 0.0
        total_short_option_exposure = 0.0
        total_short_portfolio_exposure = 0.0

        for leg_key, leg_df in leg_frames.items():
            match = leg_df[leg_df["period"] == rp]
            if not len(match):
                continue
            row = match.iloc[0]
            total_return += float(row["return_contribution"])

            leg_type = leg_types.get(leg_key)
            if leg_type in GEARING_ELIGIBLE_TYPES:
                total_capital += float(row["capital_weight"])
            if leg_type in SHORT_EXPOSURE_TYPES:
                total_short_option_exposure += float(row.get("notional", 0.0))
            if leg_type == "short_portfolio":
                total_short_portfolio_exposure += float(row.get("notional", 0.0))

        n_months = len(months_in_period(rp, frequency, available))
        period_rf = (1 + risk_free_rate) ** (n_months / 12.0) - 1 if n_months else 0.0
        excess = total_capital - 1.0
        financing_return = -excess * period_rf
        net_return = total_return + financing_return

        rows.append({
            "period": rp,
            "net_capital_weight": total_capital,
            "net_gearing": total_capital,
            "excess_capital": excess,
            "period_risk_free_rate": period_rf,
            "financing_return": financing_return,
            "leg_return_contribution": total_return,
            "net_return": net_return,
            "short_exposure": total_short_option_exposure,
            "short_portfolio_exposure": total_short_portfolio_exposure,
            "spx_return_used": spx_return_map.get(rp, None),
        })

    net_df = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    net_df["net_cumulative_return"] = (1 + net_df["net_return"]).cumprod() - 1
    return net_df


# ─────────────────────────────────────────────────────────────────────────────
# Leg-level metrics (return contribution as a fraction of $1 TOTAL equity)
# ─────────────────────────────────────────────────────────────────────────────

def compute_leg_metrics(leg_df, frequency, risk_frequency, rolling_window, ppy):
    """
    Leg return is measured as return_contribution directly (fraction of $1
    TOTAL equity), NOT return_contribution / capital_weight. Dividing by
    capital_weight is wrong for options legs: they spend ~100% of their
    allocated capital on premium every period, so any non-exercised period
    would produce leg_return = -1.0 (-100%), and cumprod() would permanently
    zero the series the first time that happens. Using return_contribution
    directly avoids this and matches how the leg actually affects Net
    Portfolio wealth.
    """
    df = leg_df.copy()
    df["leg_return"] = df["return_contribution"]
    return compute_metrics(df, "leg_return", frequency, risk_frequency, rolling_window, ppy)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    start_year, start_month, end_year, end_month,
    legs,
    input_dir=INPUT_DIR,
    portfolio_file=PORTFOLIO_FILE,
    returns_file=RETURNS_FILE,
    index_file=INDEX_FILE,
    benchmark_ticker=BENCHMARK_TICKER,
    frequency=FREQUENCY,
    risk_frequency=RISK_FREQUENCY,
    rolling_window=ROLLING_WINDOW,
    risk_free_rate=RISK_FREE_RATE,
):
    """
    legs: list of {"key": str, "type": one of LEG_BUILDERS, "label": str, "params": dict}
    Every leg type's params map directly onto its builder function's kwargs
    (excluding rebalance_periods/market, which are supplied internally).

    Returns a dict:
      {
        "net": {"results_df": ..., "metrics": {...}},
        "legs": {leg_key: {"results_df": ..., "metrics": {...}, "label": ..., "type": ...}},
      }
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

    market = build_market_context(
        rebalance_periods, frequency, available, returns_df, norm_to_orig,
        index_df, benchmark_ticker, portfolio_by_period,
    )

    if not market["raw_portfolio_return"] and any(
        leg["type"] in ("long_portfolio", "short_portfolio") for leg in legs
    ):
        raise RuntimeError("Backtest produced zero results. Check portfolio dates align with rebalance periods.")

    leg_frames, leg_meta, leg_types = {}, {}, {}
    for leg_cfg in legs:
        leg_key = leg_cfg["key"]
        leg_type = leg_cfg["type"]
        builder = LEG_BUILDERS.get(leg_type)
        if builder is None:
            raise ValueError(f"Unknown leg type {leg_type!r}")
        leg_df = builder(rebalance_periods, market, **leg_cfg.get("params", {}))
        leg_frames[leg_key] = leg_df
        leg_meta[leg_key] = {"label": leg_cfg.get("label", leg_key), "type": leg_type}
        leg_types[leg_key] = leg_type

    net_df = combine_legs_into_net_portfolio(
        leg_frames, leg_types, rebalance_periods, frequency, available, risk_free_rate,
        spx_return_map=market["spx_return"],
    )

    net_metrics = compute_metrics(
        net_df.rename(columns={"net_return": "_ret"}), "_ret", frequency, risk_frequency, rolling_window, ppy,
        monthly_returns_map=market["monthly_port_rets"] if len(legs) == 1 and legs[0]["type"] == "long_portfolio" else None,
    )
    bench = compute_benchmark_stats(
        net_df.rename(columns={"net_return": "_ret"}), "_ret", index_df, benchmark_ticker, frequency, available, ppy,
        net_metrics.get("annualised_return", np.nan),
    )
    if bench is not None:
        net_metrics["benchmark"] = bench[0]

    sorted_ps = sorted(market["tickers_by_period"].keys())
    turnover_rates = []
    for i in range(1, len(sorted_ps)):
        prev = market["tickers_by_period"][sorted_ps[i - 1]]
        curr = market["tickers_by_period"][sorted_ps[i]]
        union = len(prev | curr)
        if union > 0:
            turnover_rates.append((len(curr - prev) + len(prev - curr)) / union)
    net_metrics["avg_turnover"] = float(np.mean(turnover_rates)) if turnover_rates else None
    if market["holdings_count"]:
        net_metrics["avg_holdings"] = float(np.mean(list(market["holdings_count"].values())))

    legs_out = {}
    for leg_key, leg_df in leg_frames.items():
        leg_metrics = compute_leg_metrics(leg_df, frequency, risk_frequency, rolling_window, ppy)
        if "exercised" in leg_df.columns:
            n_periods_leg = len(leg_df)
            n_exercised = int(leg_df["exercised"].sum())
            premium_col = "premium_paid_pct_port" if "premium_paid_pct_port" in leg_df.columns else "premium_received_pct_port"
            payoff_col = "payoff_pct_port" if "payoff_pct_port" in leg_df.columns else "payoff_owed_pct_port"
            total_premium = float(leg_df[premium_col].sum())
            total_payoff = float(leg_df[payoff_col].sum())
            leg_metrics["option_stats"] = {
                "n_periods": n_periods_leg,
                "n_exercised": n_exercised,
                "exercise_rate": (n_exercised / n_periods_leg) if n_periods_leg else 0.0,
                "total_premium_pct_port": total_premium,
                "total_payoff_pct_port": total_payoff,
                "net_cost_pct_port": total_premium - total_payoff,
                "avg_notional": float(leg_df["notional"].mean()) if "notional" in leg_df.columns else None,
            }
        legs_out[leg_key] = {
            "results_df": leg_df,
            "metrics": leg_metrics,
            "label": leg_meta[leg_key]["label"],
            "type": leg_meta[leg_key]["type"],
        }

    return {
        "net": {"results_df": net_df, "metrics": net_metrics},
        "legs": legs_out,
    }
