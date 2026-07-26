"""
Position sizing and portfolio guardrails.

Implements:
  - Fractional Kelly sizing from a rolling sample of the strategy's own
    realized trades (win rate p, reward:risk b), not from theoretical
    numbers. Falls back to a small flat size until enough trades exist
    (50-trade minimum per the market-analysis prompt).
  - Correlation-aware exposure cap: positions that are highly correlated
    (e.g. alts vs BTC) are sized down as a group so "8 positions" can't
    secretly be one leveraged bet on the same underlying move.
  - Hard guardrails: no duplicate entry into an already-open symbol,
    mandatory non-zero stop distance, portfolio max-drawdown circuit
    breaker.
"""

from __future__ import annotations
import numpy as np


class KellySizer:
    def __init__(self, kelly_fraction: float = 0.25, min_trades: int = 50,
                 fallback_risk_pct: float = 0.01, max_risk_pct: float = 0.05):
        """
        kelly_fraction : fraction of full-Kelly to actually use
                          (0.25 = quarter-Kelly, the level recommended in
                          the market-analysis prompt for a bot holding
                          multiple correlated positions at once).
        min_trades     : minimum realized trades before trusting p/b
                          estimates; below this, use fallback_risk_pct.
        fallback_risk_pct : flat risk-per-trade (% of equity) used before
                          min_trades is reached.
        max_risk_pct   : absolute ceiling on risk-per-trade regardless of
                          what Kelly suggests (protects against a noisy
                          early estimate recommending something absurd).
        """
        self.kelly_fraction = kelly_fraction
        self.min_trades = min_trades
        self.fallback_risk_pct = fallback_risk_pct
        self.max_risk_pct = max_risk_pct
        self.trade_history: list[float] = []  # realized R-multiples (pnl / risked)

    def record_trade(self, r_multiple: float):
        self.trade_history.append(r_multiple)

    def _estimate_p_b(self):
        wins = [r for r in self.trade_history if r > 0]
        losses = [r for r in self.trade_history if r <= 0]
        if not wins or not losses:
            return None
        p = len(wins) / len(self.trade_history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        if avg_loss == 0:
            return None
        b = avg_win / avg_loss
        return p, b

    def risk_pct(self) -> float:
        """Return the fraction of equity to risk on the next trade."""
        if len(self.trade_history) < self.min_trades:
            return self.fallback_risk_pct
        est = self._estimate_p_b()
        if est is None:
            return self.fallback_risk_pct
        p, b = est
        q = 1 - p
        f_full = (b * p - q) / b
        f_full = max(f_full, 0.0)  # never size negative -> no edge, sit out
        f_used = f_full * self.kelly_fraction
        return float(min(f_used, self.max_risk_pct))


def position_size(equity: float, entry_price: float, stop_price: float,
                   risk_pct: float, cluster_scale: float = 1.0) -> float:
    """
    Convert a risk-% and stop distance into a position size (units of
    the asset). cluster_scale (<=1) down-weights positions that are
    correlated with other currently-open positions.
    """
    if stop_price is None or stop_price == entry_price:
        raise ValueError("Refusing to size a position with no valid stop distance.")
    risk_amount = equity * risk_pct * cluster_scale
    per_unit_risk = abs(entry_price - stop_price)
    return risk_amount / per_unit_risk


class PortfolioGuardrails:
    """
    Hard, signal-independent guardrails from the market-analysis prompt.
    """
    def __init__(self, max_drawdown_pct: float = 0.10, max_positions: int = 8,
                 cluster_map: dict | None = None, cluster_cap_pct: float = 0.30):
        self.max_drawdown_pct = max_drawdown_pct
        self.max_positions = max_positions
        self.cluster_map = cluster_map or {}   # symbol -> cluster id (e.g. "btc_beta")
        self.cluster_cap_pct = cluster_cap_pct  # max total equity risked in one cluster
        self.halted = False
        self.peak_equity = None

    def check_drawdown(self, equity: float) -> bool:
        """Returns True if trading should halt (breaker tripped)."""
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        dd = 1 - (equity / self.peak_equity)
        if dd >= self.max_drawdown_pct:
            self.halted = True
        return self.halted

    def can_enter(self, symbol: str, open_positions: dict, cluster_risk: dict) -> tuple[bool, str]:
        if self.halted:
            return False, "halted: max drawdown circuit breaker tripped"
        if symbol in open_positions:
            return False, "skip: already have an open position in this symbol"
        if len(open_positions) >= self.max_positions:
            return False, "skip: max positions reached"
        cluster = self.cluster_map.get(symbol, symbol)  # default: each symbol its own cluster
        if cluster_risk.get(cluster, 0.0) >= self.cluster_cap_pct:
            return False, f"skip: cluster '{cluster}' exposure cap reached"
        return True, "ok"
