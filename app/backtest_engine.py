"""
backtest_engine.py

Modular, leg-based backtesting engine.

Architecture
------------
Every selected strategy (Long Portfolio, Long Put, Long Call, ...) is a "leg".
Each leg produces, per rebalance period:
  - capital_weight: fraction of $1 total equity this leg consumes that period
  - return_contribution: $ P&L this leg adds that period, as a fraction of $1 equity
  - notional: the position size for that period (e.g. weight_pct for portfolio
    legs, beta-matched or fixed/weight-sized notional for options legs)

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

Leg-level metrics (compute_leg_metrics) return TWO independent metric sets:

  "contribution": measures return_contribution directly (fraction of $1 TOTAL
    equity) -- how much
