"""schumann_backtest.py — Backtesting discipline from Schumann (2018), SSRN 3374195.

Implements the paper's three core validation tools, adapted for crypto:
  1. Random-data look-ahead control (F6): a correct strategy on zero-mean
     random walks must NOT systematically beat buy-and-hold.
  2. Overfitting demo (F1): grid-searching MA-crossover on random data
     "beats" the market ~95% of the time — pure data-snooping.
  3. Walk-forward (F2): tune on in-sample, evaluate on out-of-sample only.
  4. Sensitivity sweep (F3): a robust strategy is profitable across a
     parameter neighbourhood, not at one cherry-picked optimum.

All returns are net of a configurable round-trip fee (Kraken taker ~0.52%).
See research_db/ssrn-3374195.json for the full analysis.
"""

import math
import random as _random
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------

def random_price_series(length: int, vol: float = 0.01, demean: bool = True,
                        seed: int | None = None) -> list[float]:
    """Random walk of chained Gaussian returns, starting at 100.

    Mirrors the paper's randomPriceSeries. demean=True scales so total
    return is ~0 (the honest control: no exploitable drift).
    """
    rng = _random.Random(seed)
    rets = [rng.gauss(0.0, vol) for _ in range(length - 1)]
    if demean:
        # remove mean drift so buy-and-hold has ~0 expected return
        mu = sum(rets) / len(rets) if rets else 0.0
        rets = [r - mu for r in rets]
    prices = [100.0]
    for r in rets:
        prices.append(prices[-1] * (1 + r))
    return prices


def moving_average(series: list[float], window: int) -> list[float | None]:
    """Simple MA; first (window-1) entries are None (no look-ahead padding)."""
    out: list[float | None] = [None] * len(series)
    if window <= 0 or window > len(series):
        return out
    run = sum(series[:window])
    out[window - 1] = run / window
    for i in range(window, len(series)):
        run += series[i] - series[i - window]
        out[i] = run / window
    return out


def sharpe_ratio(returns: list[float], periods_per_year: int = 365 * 96) -> float:
    """Annualised Sharpe from per-bar returns. Default assumes 15m bars."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return (mean / sd) * math.sqrt(periods_per_year)


def max_drawdown(equity: list[float]) -> float:
    """Max peak-to-trough drawdown as a positive percentage."""
    if not equity:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


# ----------------------------------------------------------------------------
# MA-crossover simulator (the paper's worked example)
# ----------------------------------------------------------------------------

def simulate_ma_crossover(series: list[float], fast: int, slow: int,
                          fee_roundtrip: float = 0.0052) -> dict:
    """Long when fast MA > slow MA, else cash. Deducts fee on each flip.

    Returns {final_wealth, returns, equity, n_trades}. No look-ahead:
    the decision at bar t uses MAs computed through bar t, and the return
    is realised over bar t -> t+1.
    """
    m_fast = moving_average(series, fast)
    m_slow = moving_average(series, slow)
    wealth = 100.0
    equity = [wealth]
    per_bar_returns: list[float] = []
    position = 0  # 0 = cash, 1 = long
    n_trades = 0

    for t in range(len(series) - 1):
        mf, ms = m_fast[t], m_slow[t]
        desired = 1 if (mf is not None and ms is not None and mf > ms) else 0
        if desired != position:
            wealth *= (1 - fee_roundtrip / 2)  # one-way fee on the flip
            n_trades += 1
            position = desired
        # realise return over t -> t+1 if long
        bar_ret = (series[t + 1] / series[t] - 1) if position == 1 else 0.0
        wealth *= (1 + bar_ret)
        per_bar_returns.append(bar_ret)
        equity.append(wealth)

    return {
        "final_wealth": wealth,
        "returns": per_bar_returns,
        "equity": equity,
        "n_trades": n_trades,
    }


def buy_and_hold_wealth(series: list[float]) -> float:
    if not series or series[0] == 0:
        return 100.0
    return 100.0 * series[-1] / series[0]


# ----------------------------------------------------------------------------
# F6 + F1: random-data controls
# ----------------------------------------------------------------------------

def random_data_lookahead_control(n_runs: int = 200, length: int = 500,
                                  fast: int = 10, slow: int = 30,
                                  fee_roundtrip: float = 0.0,
                                  seed: int = 2552551) -> dict:
    """F6: fixed-param MA-crossover on zero-mean random walks.

    A correct (look-ahead-free) strategy should beat buy-and-hold ~50% of
    the time with mean excess ~0. A large positive mean => look-ahead bug.
    """
    rng = _random.Random(seed)
    excess = []
    wins = 0
    for _ in range(n_runs):
        s = random_price_series(length, seed=rng.randint(0, 10**9))
        strat = simulate_ma_crossover(s, fast, slow, fee_roundtrip)["final_wealth"]
        bh = buy_and_hold_wealth(s)
        d = strat - bh
        excess.append(d)
        if d > 0:
            wins += 1
    mean_excess = sum(excess) / len(excess)
    return {
        "n_runs": n_runs,
        "win_rate_vs_bh": wins / n_runs,
        "mean_excess": mean_excess,
        "verdict": ("LOOK-AHEAD SUSPECTED" if mean_excess > 2.0
                    else "clean (no systematic edge on noise)"),
    }


def overfitting_demo(n_runs: int = 100, length: int = 1000,
                     fast_range: range = range(1, 21),
                     slow_range: range = range(21, 61),
                     seed: int = 2552551) -> dict:
    """F1: grid-search MA-crossover on random data. Reproduces the paper's
    result that optimization 'beats' the market ~95% of the time on noise."""
    rng = _random.Random(seed)
    excess = []
    wins = 0
    for _ in range(n_runs):
        s = random_price_series(length, seed=rng.randint(0, 10**9))
        bh = buy_and_hold_wealth(s)
        best = -1e18
        for f in fast_range:
            for sl in slow_range:
                w = simulate_ma_crossover(s, f, sl, fee_roundtrip=0.0)["final_wealth"]
                if w > best:
                    best = w
        d = best - bh
        excess.append(d)
        if d > 0:
            wins += 1
    excess.sort()
    median = excess[len(excess) // 2]
    return {
        "n_runs": n_runs,
        "win_rate_vs_bh": wins / n_runs,
        "median_excess": median,
        "verdict": "OVERFIT — grid search finds winners on pure noise",
    }


# ----------------------------------------------------------------------------
# F2: walk-forward validation (in-sample tune, out-of-sample evaluate)
# ----------------------------------------------------------------------------

def walk_forward_ma(series: list[float], n_windows: int = 5,
                    is_frac: float = 0.7,
                    fast_range: range = range(2, 21, 2),
                    slow_range: range = range(24, 61, 4),
                    fee_roundtrip: float = 0.0052) -> dict:
    """Split series into n_windows. In each: tune (fast, slow) on the
    in-sample front (is_frac), then apply those params to the out-of-sample
    tail. Concatenate OOS returns and report OOS-only performance.

    Compares honest OOS to the naive in-sample-best (which the paper warns
    is worthless)."""
    N = len(series)
    if N < n_windows * 30:
        return {"error": f"series too short ({N}) for {n_windows} windows"}

    win = N // n_windows
    oos_returns: list[float] = []
    is_wealths, oos_wealths, chosen = [], [], []

    for w in range(n_windows):
        lo = w * win
        hi = N if w == n_windows - 1 else (w + 1) * win
        block = series[lo:hi]
        split = int(len(block) * is_frac)
        is_block = block[:split]
        oos_block = block[split:]
        if len(is_block) < max(slow_range) + 2 or len(oos_block) < 3:
            continue

        # tune on in-sample
        best_w, best_par = -1e18, (fast_range.start, slow_range.start)
        for f in fast_range:
            for sl in slow_range:
                if f >= sl:
                    continue
                res = simulate_ma_crossover(is_block, f, sl, fee_roundtrip)
                if res["final_wealth"] > best_w:
                    best_w = res["final_wealth"]
                    best_par = (f, sl)
        is_wealths.append(best_w)
        chosen.append(best_par)

        # evaluate on out-of-sample with the tuned params
        oos_res = simulate_ma_crossover(oos_block, best_par[0], best_par[1], fee_roundtrip)
        oos_wealths.append(oos_res["final_wealth"])
        oos_returns.extend(oos_res["returns"])

    if not oos_returns:
        return {"error": "no usable windows"}

    # concatenate OOS equity
    eq = [100.0]
    for r in oos_returns:
        eq.append(eq[-1] * (1 + r))
    oos_total_ret = (eq[-1] / eq[0] - 1) * 100

    return {
        "n_windows": len([c for c in chosen]),
        "chosen_params": chosen,
        "in_sample_mean_wealth": sum(is_wealths) / len(is_wealths) if is_wealths else 0,
        "oos_total_return_pct": round(oos_total_ret, 2),
        "oos_sharpe": round(sharpe_ratio(oos_returns), 3),
        "oos_max_drawdown_pct": round(max_drawdown(eq), 2),
        "oos_n_bars": len(oos_returns),
    }


# ----------------------------------------------------------------------------
# F3: sensitivity sweep (robust across a neighbourhood, not one optimum)
# ----------------------------------------------------------------------------

def sensitivity_sweep_ma(series: list[float],
                         fast_range: range = range(2, 21, 2),
                         slow_range: range = range(24, 61, 4),
                         fee_roundtrip: float = 0.0052) -> dict:
    """Evaluate MA-crossover across a parameter grid vs buy-and-hold.
    Robust => profitable across MOST of the grid, not one cherry-picked cell."""
    bh = buy_and_hold_wealth(series)
    cells, beat = 0, 0
    best_w, best_par = -1e18, None
    worst_w = 1e18
    for f in fast_range:
        for sl in slow_range:
            if f >= sl:
                continue
            w = simulate_ma_crossover(series, f, sl, fee_roundtrip)["final_wealth"]
            cells += 1
            if w > bh:
                beat += 1
            if w > best_w:
                best_w, best_par = w, (f, sl)
            if w < worst_w:
                worst_w = w
    if cells == 0:
        return {"error": "empty grid"}
    frac = beat / cells
    return {
        "grid_cells": cells,
        "buy_hold_wealth": round(bh, 2),
        "frac_beating_bh": round(frac, 3),
        "best_wealth": round(best_w, 2),
        "best_params": best_par,
        "worst_wealth": round(worst_w, 2),
        "verdict": ("ROBUST" if frac >= 0.6 else
                    "FRAGILE — only works at cherry-picked params" if frac < 0.35
                    else "MIXED"),
    }
