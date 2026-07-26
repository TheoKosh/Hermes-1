"""
Bar-by-bar backtest loop tying together regime classification, the
composite signal, Kelly sizing, and the portfolio guardrails.

Design choices that matter for correctness:
  - Every decision at bar t uses only columns computed from data up to
    and including t (see engine.py) -> no look-ahead.
  - Entries are taken on t's close, evaluated against t+1's open/close
    for fills, to avoid the "traded on today's own close" look-ahead
    pattern flagged in the strategy-search prompt.
  - Universe should ideally be survivorship-bias-free (i.e. include
    symbols that were later delisted). This engine doesn't fabricate
    that -- it's a data-sourcing responsibility -- but it will warn if
    every symbol in the panel is still "alive" through the last bar,
    since that's the signature of a survivorship-biased universe.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .engine import classify_regime, composite_signal
from .sizing import KellySizer, PortfolioGuardrails, position_size
from .metrics import compute_metrics, overfitting_flags, deflated_sharpe_ratio


class Trade:
    __slots__ = ['symbol', 'side', 'entry_time', 'entry_price', 'stop_price',
                 'target_price', 'size', 'exit_time', 'exit_price', 'pnl', 'r_multiple']

    def __init__(self, symbol, side, entry_time, entry_price, stop_price, target_price, size):
        self.symbol = symbol
        self.side = side
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price
        self.size = size
        self.exit_time = None
        self.exit_price = None
        self.pnl = None
        self.r_multiple = None

    def close(self, exit_time, exit_price):
        self.exit_time = exit_time
        self.exit_price = exit_price
        direction = 1 if self.side == 'long' else -1
        self.pnl = (exit_price - self.entry_price) * direction * self.size
        risk = abs(self.entry_price - self.stop_price) * self.size
        self.r_multiple = self.pnl / risk if risk > 0 else 0.0

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


class Backtester:
    def __init__(self, panel: dict[str, pd.DataFrame], starting_equity: float = 10_000,
                 conf_threshold: float = 0.35, stop_atr_mult: float = 2.0,
                 target_atr_mult: float = 3.0, fee_bps: float = 5.0,
                 max_positions: int = 8, max_drawdown_pct: float = 0.10,
                 kelly_fraction: float = 0.25, cluster_map: dict | None = None):
        """
        panel: {symbol: OHLCV dataframe indexed by datetime, columns
                open/high/low/close/volume}
        """
        self.panel = {s: composite_signal(classify_regime(df)) for s, df in panel.items()}
        self.equity = starting_equity
        self.starting_equity = starting_equity
        self.conf_threshold = conf_threshold
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.fee_bps = fee_bps
        self.sizer = KellySizer(kelly_fraction=kelly_fraction)
        self.guard = PortfolioGuardrails(max_drawdown_pct=max_drawdown_pct,
                                          max_positions=max_positions,
                                          cluster_map=cluster_map)
        self.open_positions: dict[str, Trade] = {}
        self.closed_trades: list[Trade] = []
        self.equity_curve = []
        self._cluster_risk_cache: dict[str, float] = {}

    def _atr(self, df: pd.DataFrame, t, window: int = 14) -> float:
        loc = df.index.get_loc(t)
        if loc < window:
            return float(df['close'].iloc[loc] * 0.02)  # fallback: 2% of price
        window_df = df.iloc[loc - window + 1: loc + 1]
        tr = pd.concat([
            window_df['high'] - window_df['low'],
            (window_df['high'] - window_df['close'].shift()).abs(),
            (window_df['low'] - window_df['close'].shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.mean())

    def _cluster_risk(self) -> dict[str, float]:
        risk = {}
        for sym, trade in self.open_positions.items():
            cluster = self.guard.cluster_map.get(sym, sym)
            notional = trade.size * trade.entry_price
            risk[cluster] = risk.get(cluster, 0.0) + notional / self.equity
        return risk

    def run(self):
        all_times = sorted(set().union(*[df.index for df in self.panel.values()]))
        for i, t in enumerate(all_times[:-1]):  # need t+1 to fill
            next_t = all_times[i + 1]

            # 1) manage open positions: check stop/target using this bar's data
            for sym in list(self.open_positions.keys()):
                df = self.panel[sym]
                if t not in df.index:
                    continue
                trade = self.open_positions[sym]
                bar = df.loc[t]
                hit_stop = (bar['low'] <= trade.stop_price if trade.side == 'long'
                            else bar['high'] >= trade.stop_price)
                hit_target = (bar['high'] >= trade.target_price if trade.side == 'long'
                              else bar['low'] <= trade.target_price)
                if hit_stop or hit_target:
                    exit_price = trade.stop_price if hit_stop else trade.target_price
                    trade.close(t, exit_price)
                    self.equity += trade.pnl - abs(trade.pnl) * self.fee_bps / 1e4
                    self.sizer.record_trade(trade.r_multiple)
                    self.closed_trades.append(trade)
                    del self.open_positions[sym]

            # 2) drawdown circuit breaker
            halted = self.guard.check_drawdown(self.equity)

            # 3) evaluate entries (decision made on t's close, filled at next_t's open)
            if not halted:
                cluster_risk = self._cluster_risk()
                for sym, df in self.panel.items():
                    if t not in df.index or next_t not in df.index:
                        continue
                    can, reason = self.guard.can_enter(sym, self.open_positions, cluster_risk)
                    if not can:
                        continue
                    row = df.loc[t]
                    comp, conf = row.get('composite', np.nan), row.get('confidence', 0)
                    if pd.isna(comp) or abs(comp) < self.conf_threshold or conf < 0.5:
                        continue
                    side = 'long' if comp > 0 else 'short'
                    entry_price = float(df.loc[next_t, 'open'])
                    atr = self._atr(df, t)
                    if atr <= 0:
                        continue
                    stop_price = (entry_price - self.stop_atr_mult * atr if side == 'long'
                                  else entry_price + self.stop_atr_mult * atr)
                    target_price = (entry_price + self.target_atr_mult * atr if side == 'long'
                                     else entry_price - self.target_atr_mult * atr)
                    risk_pct = self.sizer.risk_pct()
                    cluster = self.guard.cluster_map.get(sym, sym)
                    cluster_scale = max(0.0, 1 - cluster_risk.get(cluster, 0.0) / self.guard.cluster_cap_pct)
                    try:
                        size = position_size(self.equity, entry_price, stop_price,
                                              risk_pct, cluster_scale)
                    except ValueError:
                        continue
                    trade = Trade(sym, side, next_t, entry_price, stop_price, target_price, size)
                    self.open_positions[sym] = trade
                    cluster_risk[cluster] = cluster_risk.get(cluster, 0.0) + size * entry_price / self.equity

            self.equity_curve.append({'time': t, 'equity': self.equity +
                                       sum(self._unrealized(sym, t) for sym in self.open_positions)})

        # close any remaining open positions at final bar for reporting purposes
        for sym, trade in list(self.open_positions.items()):
            df = self.panel[sym]
            last_t = df.index[-1]
            trade.close(last_t, float(df['close'].iloc[-1]))
            self.equity += trade.pnl
            self.closed_trades.append(trade)
        self.open_positions.clear()

        eq_df = pd.DataFrame(self.equity_curve).set_index('time')['equity'] if self.equity_curve \
            else pd.Series([self.starting_equity])
        return eq_df

    def _unrealized(self, sym, t) -> float:
        trade = self.open_positions[sym]
        df = self.panel[sym]
        if t not in df.index:
            return 0.0
        price = df.loc[t, 'close']
        direction = 1 if trade.side == 'long' else -1
        return (price - trade.entry_price) * direction * trade.size

    def results(self, n_trials_for_deflation: int = 1) -> dict:
        eq = pd.Series([r['equity'] for r in self.equity_curve],
                        index=[r['time'] for r in self.equity_curve])
        trades = [t.to_dict() for t in self.closed_trades]
        m = compute_metrics(eq, trades)
        m['deflated_sharpe_prob'] = deflated_sharpe_ratio(
            m['sharpe'], n_trials_for_deflation, len(eq))
        m['overfitting_flags'] = overfitting_flags(m)
        return {'metrics': m, 'equity_curve': eq, 'trades': trades}


def check_survivorship(panel: dict[str, pd.DataFrame]) -> str | None:
    """Warn if every symbol survives to the final bar -- a signature of
    a survivorship-biased universe (delisted assets silently dropped)."""
    if not panel:
        return None
    last_bars = [df.index[-1] for df in panel.values()]
    global_last = max(last_bars)
    if all(lb == global_last for lb in last_bars):
        return ("WARNING: every symbol in this panel survives to the final bar. "
                "If any assets were delisted/failed during this period and simply "
                "aren't in your dataset, this backtest is survivorship-biased and "
                "will overstate performance.")
    return None


def run_walkforward(panel: dict[str, pd.DataFrame], split_frac: float = 0.7, **kwargs):
    """
    Splits the panel chronologically into in-sample / out-of-sample and
    runs the backtest on each independently. Per the strategy-search
    prompt: parameters should only ever be tuned on the IS slice; OOS is
    evaluated once and not fed back into further tuning.
    """
    is_panel, oos_panel = {}, {}
    for sym, df in panel.items():
        split_idx = int(len(df) * split_frac)
        is_panel[sym] = df.iloc[:split_idx]
        oos_panel[sym] = df.iloc[split_idx:]

    survivorship_warning = check_survivorship(panel)

    bt_is = Backtester(is_panel, **kwargs)
    bt_is.run()
    res_is = bt_is.results()

    bt_oos = Backtester(oos_panel, **kwargs)
    bt_oos.run()
    res_oos = bt_oos.results()

    is_sharpe = res_is['metrics']['sharpe']
    oos_sharpe = res_oos['metrics']['sharpe']
    degradation = None
    if is_sharpe and not np.isnan(is_sharpe) and is_sharpe != 0:
        degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe) if not np.isnan(oos_sharpe) else 1.0

    return {
        'in_sample': res_is,
        'out_of_sample': res_oos,
        'sharpe_degradation_pct': degradation,
        'survivorship_warning': survivorship_warning,
    }
