"""
Performance metrics + the "is this too good to be true" bias checklist
from the market-analysis prompt.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_metrics(equity_curve: pd.Series, trades: list[dict],
                     periods_per_year: int = 365) -> dict:
    equity_curve = equity_curve.dropna()
    rets = equity_curve.pct_change().dropna()

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if len(equity_curve) > 1 else 0.0
    ann_return = (1 + total_return) ** (periods_per_year / max(len(equity_curve), 1)) - 1

    sharpe = _annualized_ratio(rets, rets.std(), periods_per_year)
    downside = rets[rets < 0]
    sortino = _annualized_ratio(rets, downside.std(), periods_per_year) if len(downside) else np.nan

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_dd = drawdown.min()

    n_trades = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    win_rate = len(wins) / n_trades if n_trades else np.nan
    gross_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.nan

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
    }


def _annualized_ratio(rets: pd.Series, denom: float, periods_per_year: int) -> float:
    if denom is None or denom == 0 or np.isnan(denom):
        return np.nan
    return float(rets.mean() / denom * np.sqrt(periods_per_year))


def deflated_sharpe_ratio(sharpe: float, n_trials: int, n_obs: int,
                           skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    Approximate Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014):
    the probability that the observed Sharpe is genuinely > 0 after
    correcting for having tested n_trials candidate strategies. This is
    what "unlimited portfolio search" *requires* before trusting a
    winner -- more trials should raise the bar, not just increase the
    chance something looks good by luck.
    """
    if n_obs <= 1 or sharpe is None or np.isnan(sharpe):
        return np.nan
    from scipy.stats import norm
    if n_trials <= 1:
        # No multiple-testing correction to apply (single candidate tested) ->
        # this reduces to the plain Probabilistic Sharpe Ratio (PSR vs. 0).
        sr0 = 0.0
    else:
        # Expected max Sharpe under the null across n_trials independent trials
        euler_mascheroni = 0.5772156649
        e_max_z = ((1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials) +
                   euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e)))
        sr0 = e_max_z / np.sqrt(n_obs)  # expected Sharpe under null, scaled to sample
    denom = np.sqrt(max(1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2, 1e-9) / n_obs)
    dsr_stat = (sharpe - sr0) / denom
    return float(norm.cdf(dsr_stat))


def overfitting_flags(metrics: dict) -> list[str]:
    """Red flags per the market-analysis prompt's suspicious-result thresholds."""
    flags = []
    sharpe = metrics.get('sharpe', np.nan)
    max_dd = metrics.get('max_drawdown', np.nan)
    win_rate = metrics.get('win_rate', np.nan)
    n_trades = metrics.get('n_trades', 0)

    if not np.isnan(sharpe) and sharpe > 3:
        flags.append(f"Sharpe {sharpe:.2f} > 3 — suspicious, real systematic strategies "
                      f"typically run 1-2.5. Check for look-ahead/survivorship bias before trusting this.")
    if not np.isnan(max_dd) and abs(max_dd) < 0.05:
        flags.append(f"Max drawdown {max_dd:.1%} is unrealistically shallow — check for a bug "
                      f"suppressing losing trades, or a too-short/benign test window.")
    if not np.isnan(win_rate) and win_rate > 0.75:
        flags.append(f"Win rate {win_rate:.1%} > 75% is unusually high — verify reward:risk "
                      f"isn't being hidden (e.g. tiny wins, rare huge losses not yet realized).")
    if n_trades < 50:
        flags.append(f"Only {n_trades} trades — far too small a sample to trust the Sharpe/win-rate "
                      f"estimate; treat all metrics here as provisional.")
    if not flags:
        flags.append("No obvious red flags, but that alone doesn't make the result trustworthy — "
                      "still requires out-of-sample / holdout confirmation before allocating capital.")
    return flags
