"""
mcpt_validation.py — Monte Carlo Permutation Testing for strategy validation.

Uses bar_permute.py to create realistic shuffled data that preserves return
distributions but destroys exploitable patterns. If a strategy's profit factor
on real data isn't significantly better than on permuted data, the "edge" is
curve-fit noise, not a genuine signal.

Based on: https://github.com/neurotrader888/mcpt
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from .backtest.bar_permute import get_permutation


def run_mcpt(ohlc: pd.DataFrame, strategy_fn, n_permutations: int = 1000,
             profit_metric: str = "sharpe", seed: int = 42) -> dict:
    """
    Run Monte Carlo Permutation Test on a strategy.

    Args:
        ohlc: DataFrame with open/high/low/close columns, datetime index
        strategy_fn: function(ohlc_df) -> (signal_series, metric_value)
                     signal = -1/0/1 positions, metric = profit factor or sharpe
        n_permutations: number of shuffled datasets to test
        profit_metric: "sharpe" or "profit_factor"
        seed: random seed for reproducibility

    Returns:
        {
            "real_metric": float,
            "p_value": float,
            "permutation_metrics": list,
            "verdict": str,
            "significant": bool,
        }
    """
    np.random.seed(seed)

    # Run on real data
    _, real_metric = strategy_fn(ohlc)

    # Run on permuted data
    perm_metrics = []
    better_count = 0

    for i in tqdm(range(n_permutations), desc="MCPT permutations", leave=False):
        perm_data = get_permutation(ohlc, seed=i)
        _, perm_metric = strategy_fn(perm_data)

        perm_metrics.append(perm_metric)
        if perm_metric >= real_metric:
            better_count += 1

    p_value = (better_count + 1) / (n_permutations + 1)

    # Verdict
    if p_value < 0.01:
        verdict = "HIGHLY SIGNIFICANT — genuine edge (p<0.01)"
    elif p_value < 0.05:
        verdict = "SIGNIFICANT — likely real edge (p<0.05)"
    elif p_value < 0.10:
        verdict = "MARGINAL — weak edge, proceed with caution (p<0.10)"
    else:
        verdict = "NOT SIGNIFICANT — likely curve-fit noise (p>=0.10)"

    return {
        "real_metric": real_metric,
        "p_value": p_value,
        "permutation_metrics": perm_metrics,
        "permutation_mean": float(np.mean(perm_metrics)) if perm_metrics else 0,
        "permutation_std": float(np.std(perm_metrics)) if perm_metrics else 0,
        "verdict": verdict,
        "significant": p_value < 0.05,
        "n_permutations": n_permutations,
        "metric_name": profit_metric,
    }


def strategy_adapter_sharpe(ohlc: pd.DataFrame, signal_fn) -> tuple:
    """
    Adapt a signal function to MCPT's expected format.
    signal_fn(ohlc_df) -> signal_series (-1/0/1)
    Returns (signal, sharpe_ratio)
    """
    signal = signal_fn(ohlc)
    returns = np.log(ohlc["close"]).diff().shift(-1)
    strategy_returns = signal * returns

    valid = strategy_returns.dropna()
    if len(valid) < 10 or valid.std() == 0:
        return signal, 0.0

    sharpe = valid.mean() / valid.std() * np.sqrt(365 * 24)  # annualized for hourly
    return signal, float(sharpe)


def strategy_adapter_profit_factor(ohlc: pd.DataFrame, signal_fn) -> tuple:
    """
    Adapt a signal function to MCPT's expected format.
    Returns (signal, profit_factor)
    """
    signal = signal_fn(ohlc)
    returns = np.log(ohlc["close"]).diff().shift(-1)
    strategy_returns = signal * returns

    valid = strategy_returns.dropna()
    gains = valid[valid > 0].sum()
    losses = abs(valid[valid < 0].sum())

    pf = gains / losses if losses > 0 else 0.0
    return signal, float(pf)


# Pre-built signal functions for common strategies

def rsi_signal(ohlc: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """RSI mean-reversion signal for MCPT."""
    delta = ohlc["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    signal = pd.Series(0, index=ohlc.index)
    signal[rsi < oversold] = 1   # oversold → long
    signal[rsi > overbought] = -1 # overbought → short
    return signal


def ema_cross_signal(ohlc: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """EMA crossover signal for MCPT."""
    fast_ma = ohlc["close"].rolling(fast).mean()
    slow_ma = ohlc["close"].rolling(slow).mean()
    signal = pd.Series(np.where(fast_ma > slow_ma, 1, -1), index=ohlc.index)
    return signal


def donchian_signal(ohlc: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Donchian breakout signal for MCPT."""
    upper = ohlc["close"].rolling(lookback - 1).max().shift(1)
    lower = ohlc["close"].rolling(lookback - 1).min().shift(1)
    signal = pd.Series(0, index=ohlc.index)
    signal[ohlc["close"] > upper] = 1
    signal[ohlc["close"] < lower] = -1
    return signal.ffill()


def momentum_signal(ohlc: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Simple momentum (trailing return direction)."""
    ret = ohlc["close"].pct_change(lookback)
    signal = pd.Series(0, index=ohlc.index)
    signal[ret > 0] = 1
    signal[ret < 0] = -1
    return signal
