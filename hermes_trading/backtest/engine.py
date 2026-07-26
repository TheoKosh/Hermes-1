"""
Hermes Backtesting Engine
==========================
Implements the framework specified in the market-analysis and
strategy-search prompts:

  - Regime classification (trend / range, volatility state, volume
    confirmation) computed only from data available at each bar
    (point-in-time -> no look-ahead bias).
  - A composite signal built from orthogonal components: momentum,
    mean-reversion, volatility-normalized confirmation.
  - Fractional-Kelly position sizing derived from a rolling sample of
    the strategy's own realized trades (not from theoretical/backtest
    win rates).
  - Hard guardrails: no duplicate entries into an already-open symbol,
    mandatory stop-loss, portfolio-level max-drawdown circuit breaker.
  - Walk-forward split: parameters are only ever fit on the in-sample
    slice; the out-of-sample slice is evaluated once, untouched.

This is a single-portfolio, single-run engine. `run_walkforward()` at
the bottom wraps it to produce the IS/OOS comparison that the
strategy-search prompt requires before trusting any result.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(df: pd.DataFrame, trend_window: int = 50,
                     vol_window: int = 20, vol_lookback: int = 252) -> pd.DataFrame:
    """
    Adds regime columns to a per-symbol OHLCV dataframe. Every value at
    row t uses only data up to and including t (rolling windows), so
    nothing here can leak future information into a decision made at t.

    Columns added:
      trend_state   : 'trending' | 'ranging'
      vol_pctile    : realized vol percentile vs its own trailing history
      vol_state     : 'low' | 'normal' | 'high'
      volume_confirm: bool, current volume above its rolling median
    """
    out = df.copy()
    price = out['close']

    # Trend: slope of a rolling linear fit on log-price over trend_window,
    # normalized by rolling volatility so it's comparable across assets.
    log_price = np.log(price)
    def _slope(x):
        y = x.values
        t = np.arange(len(y))
        if np.std(y) == 0:
            return 0.0
        b = np.polyfit(t, y, 1)[0]
        return b
    rolling_slope = log_price.rolling(trend_window).apply(_slope, raw=False)
    ret = price.pct_change()
    rolling_vol = ret.rolling(vol_window).std()
    norm_slope = rolling_slope / rolling_vol.replace(0, np.nan)
    out['trend_score'] = norm_slope
    out['trend_state'] = np.where(norm_slope.abs() > 0.15, 'trending', 'ranging')

    # Volatility regime: percentile rank of current rolling vol against
    # its own trailing history (so it's asset-relative, not a fixed cutoff).
    vol_pctile = rolling_vol.rolling(vol_lookback, min_periods=vol_window).rank(pct=True)
    out['vol_pctile'] = vol_pctile
    out['vol_state'] = pd.cut(vol_pctile, bins=[0, 0.33, 0.67, 1.0],
                               labels=['low', 'normal', 'high'])

    # Volume confirmation: today's volume vs its own rolling median.
    if 'volume' in out.columns:
        vol_med = out['volume'].rolling(vol_window).median()
        out['volume_confirm'] = out['volume'] > vol_med
    else:
        out['volume_confirm'] = True  # no volume data available -> don't gate on it

    return out


# ---------------------------------------------------------------------------
# Signal components (orthogonal: momentum, mean-reversion, vol-normalized)
# ---------------------------------------------------------------------------

def momentum_signal(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Z-scored trailing return -> continuation bet. Positive = long bias."""
    ret = df['close'].pct_change(lookback)
    z = (ret - ret.rolling(252, min_periods=lookback).mean()) / \
        ret.rolling(252, min_periods=lookback).std().replace(0, np.nan)
    return z.clip(-3, 3)


def mean_reversion_signal(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """
    Distance from rolling mean in standard deviations, sign-flipped so
    the signal points toward reversion (price far above mean -> negative
    / short-biased signal, and vice versa).
    """
    mean = df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std().replace(0, np.nan)
    z = (df['close'] - mean) / std
    return (-z).clip(-3, 3)


def composite_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regime-routed composite: trending regimes weight momentum higher;
    ranging regimes weight mean-reversion higher. Volatility state and
    volume confirmation gate confidence rather than feed the score
    directly (per the market-analysis prompt: vol should raise the bar,
    not just be one more additive input).
    """
    out = df.copy()
    mom = momentum_signal(out)
    mr = mean_reversion_signal(out)

    is_trending = out['trend_state'] == 'trending'
    w_mom = np.where(is_trending, 0.75, 0.25)
    w_mr = 1 - w_mom
    composite = w_mom * mom.fillna(0) + w_mr * mr.fillna(0)

    # Confidence: both components agreeing in sign raises confidence;
    # disagreement lowers it. High-vol regime lowers confidence further.
    agree = np.sign(mom.fillna(0)) == np.sign(mr.fillna(0))
    conf = np.where(agree, 1.0, 0.5)
    conf = np.where(out['vol_state'] == 'high', conf * 0.6, conf)
    conf = np.where(~out['volume_confirm'].astype(bool), conf * 0.7, conf)

    out['mom_signal'] = mom
    out['mr_signal'] = mr
    out['composite'] = composite
    out['confidence'] = conf
    return out
