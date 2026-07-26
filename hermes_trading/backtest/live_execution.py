"""
Live / paper-trading execution layer.

Complements — and deliberately does not duplicate the logic of —
engine.py / backtester.py / sizing.py. Inspired by the streaming
Tick/Window architecture and Decimal-precision Order/Position/Account/
Exchange model from the "Let's Build a Quant Trading Strategy" video
series (github.com/memlabs-research/build-a-quant-trading-strategy),
adapted to plug directly into this package's existing regime/composite
signal and risk framework instead of introducing a second, separate one.

WHY THIS EXISTS
----------------
engine.py's regime/composite logic is vectorized (pandas) — great for
research, but a live system that reimplements the same math tick-by-tick
from scratch can silently drift from what the backtest actually computed.
That backtest/live parity gap is a close cousin of the original bug this
project started from (Hermes re-entering the same symbol every cycle
because live state wasn't tracked correctly) — same root problem, "does
what's running live match what was actually validated", just showing up
at a different layer.

StreamingSignalEngine below solves this by NOT reimplementing the regime/
composite math. It keeps a bounded rolling buffer per symbol and calls
the exact same classify_regime()/composite_signal() functions from
engine.py on it. Whatever a backtest run through backtester.py computes
for a given bar is, by construction, what this will compute too.

WHAT WAS DELIBERATELY *NOT* CARRIED OVER
------------------------------------------
The source video demonstrates 8x leverage combined with compounding
position sizing to produce outsized returns, then separately shows the
liquidation-price math almost as an afterthought. That's a demonstration
of risk mechanics, not a sizing recommendation — stacked with
compounding, it's exactly the kind of setup that blows through this
package's max_drawdown_pct guardrail and defeats the whole point of
quarter-Kelly sizing (sizing.py). So here, leverage is: opt-in, capped
low by default, liquidation-aware, and computed as a multiplier on top
of the existing KellySizer/PortfolioGuardrails output — never a
replacement for them.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import Optional
import pandas as pd

from .engine import classify_regime, composite_signal
from .sizing import KellySizer, PortfolioGuardrails, position_size

getcontext().prec = 28  # plenty for crypto price/qty precision


# ---------------------------------------------------------------------------
# Streaming signal engine: same math as engine.py, tick-by-tick interface
# ---------------------------------------------------------------------------

class StreamingSignalEngine:
    """
    Tick-by-tick wrapper around classify_regime()/composite_signal().
    Maintains a bounded per-symbol OHLCV buffer and recomputes the
    identical vectorized functions on it each tick, so results are
    guaranteed consistent with a backtest over the same data — no
    separately-maintained incremental math to drift out of sync.
    """

    def __init__(self, lookback: int = 300):
        self.lookback = lookback  # bars kept; must exceed the longest
                                   # rolling window used in engine.py
                                   # (vol_lookback=252 by default)
        self._buffers: dict[str, deque] = {}

    def on_tick(self, symbol: str, bar_time, bar: dict) -> Optional[dict]:
        """
        bar: {'open','high','low','close','volume'(optional)}
        Returns the latest computed signal row as a dict, or None if
        there isn't yet enough history to compute it.
        """
        buf = self._buffers.setdefault(symbol, deque(maxlen=self.lookback))
        buf.append({'time': bar_time, **bar})
        if len(buf) < 30:  # smallest window used in engine.py's calcs
            return None

        df = pd.DataFrame(list(buf)).set_index('time')
        result = composite_signal(classify_regime(df))
        last = result.iloc[-1]
        if pd.isna(last.get('composite')):
            return None
        return last.to_dict()


# ---------------------------------------------------------------------------
# Money-safe primitives (Decimal, per the video's "don't use float for
# money" point — a real and easy-to-hit bug class)
# ---------------------------------------------------------------------------

def _sign(d: Decimal) -> int:
    return 1 if d > 0 else (-1 if d < 0 else 0)


@dataclass(frozen=True)
class Order:
    sym: str
    signed_qty: Decimal
    reason: str = ""

    def __str__(self):
        side = "LONG" if self.signed_qty > 0 else "SHORT"
        return f"Order({side} {abs(self.signed_qty)} {self.sym}: {self.reason})"


@dataclass(frozen=True)
class Trade:
    sym: str
    signed_qty: Decimal
    price: Decimal
    pnl: Decimal
    fee: Decimal = Decimal(0)


@dataclass
class Position:
    sym: str
    signed_qty: Decimal
    entry_price: Decimal
    leverage: int = 1
    liquidation_price: Optional[Decimal] = None

    def is_long(self) -> bool:
        return self.signed_qty > 0

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        return (current_price - self.entry_price) * self.signed_qty

    def close_order(self) -> Order:
        return Order(self.sym, -self.signed_qty, reason="close")


# ---------------------------------------------------------------------------
# Leverage & liquidation — opt-in, capped, wired to the existing guardrails
# ---------------------------------------------------------------------------

def liquidation_price(entry_price: Decimal, leverage: int, side: str,
                       maintenance_margin: Decimal = Decimal("0.005")) -> Decimal:
    """Same formulas as the source material, isolated so they can be
    tested and reused independently of any specific exchange model."""
    p, l, mmr = entry_price, Decimal(leverage), maintenance_margin
    if side == 'long':
        return (p * l) / (l + 1 - mmr * l)
    else:
        return (p * l) / (l - 1 + mmr * l)


class LeverageManager:
    """
    Deliberately conservative by default: leverage is capped, and the
    cap itself shrinks as the portfolio approaches its max-drawdown
    limit, rather than staying fixed regardless of how much drawdown
    budget is left. This is the guardrail that the source video's 8x
    "look how much money we made" segment doesn't have.
    """
    def __init__(self, max_leverage: int = 3, guard: PortfolioGuardrails = None):
        self.max_leverage = max_leverage
        self.guard = guard

    def allowed_leverage(self, equity: float) -> int:
        if self.guard is None or self.guard.peak_equity is None:
            return self.max_leverage
        dd_used = 1 - (equity / self.guard.peak_equity)
        headroom = max(0.0, 1 - dd_used / self.guard.max_drawdown_pct)
        # scale leverage down linearly as drawdown budget is consumed
        return max(1, int(self.max_leverage * headroom))


# ---------------------------------------------------------------------------
# Account / Exchange abstractions (paper-trading implementation)
# ---------------------------------------------------------------------------

class Account(ABC):
    @abstractmethod
    def balance(self) -> Decimal: ...
    @abstractmethod
    def get_position(self, sym: str) -> Optional[Position]: ...


class Exchange(ABC):
    @abstractmethod
    def market_order(self, sym: str, signed_qty: Decimal, price: Decimal,
                      leverage: int = 1) -> Trade: ...


class PaperAccount(Account):
    def __init__(self, starting_balance: Decimal):
        self._balance = starting_balance
        self._positions: dict[str, Position] = {}
        self.trade_log: list[Trade] = []

    def balance(self) -> Decimal:
        return self._balance

    def get_position(self, sym: str) -> Optional[Position]:
        return self._positions.get(sym)


class PaperExchange(Exchange):
    """
    Paper-trading simulator with maker/taker fees and liquidation
    checks. This is the execution counterpart to Backtester in
    backtester.py — same guardrail objects can be shared between them
    if you want paper-trading to enforce identical portfolio-level
    limits to what was backtested.
    """
    def __init__(self, account: PaperAccount, taker_fee: Decimal = Decimal("0.0003"),
                 maker_fee: Decimal = Decimal("0.0001"),
                 guard: Optional[PortfolioGuardrails] = None,
                 sizer: Optional[KellySizer] = None):
        self.account = account
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.guard = guard
        self.sizer = sizer

    def market_order(self, sym: str, signed_qty: Decimal, price: Decimal,
                      leverage: int = 1) -> Trade:
        existing = self.account._positions.pop(sym, None)
        fee = abs(signed_qty) * price * self.taker_fee
        pnl = Decimal(0)

        if existing is not None and _sign(existing.signed_qty) != _sign(signed_qty):
            # closing (or flipping through) an existing position
            pnl = (price - existing.entry_price) * existing.signed_qty
            self.account._balance += pnl - fee
            if self.sizer is not None and existing.entry_price != 0:
                risk = abs(existing.entry_price - self._implied_stop(existing)) * abs(existing.signed_qty)
                r_multiple = float(pnl / risk) if risk else 0.0
                self.sizer.record_trade(r_multiple)
            remaining_qty = existing.signed_qty + signed_qty
            if remaining_qty != 0:
                self.account._positions[sym] = Position(
                    sym, remaining_qty, price, leverage,
                    liquidation_price(price, leverage, 'long' if remaining_qty > 0 else 'short'))
        else:
            merged_qty = signed_qty + (existing.signed_qty if existing else Decimal(0))
            entry_price = price if existing is None else (
                (existing.entry_price * existing.signed_qty + price * signed_qty) / merged_qty)
            side = 'long' if merged_qty > 0 else 'short'
            self.account._positions[sym] = Position(
                sym, merged_qty, entry_price, leverage,
                liquidation_price(entry_price, leverage, side))
            self.account._balance -= fee

        trade = Trade(sym, signed_qty, price, pnl, fee)
        self.account.trade_log.append(trade)
        if self.guard is not None:
            self.guard.check_drawdown(float(self.account._balance))
        return trade

    def _implied_stop(self, position: Position) -> Decimal:
        # Fallback when a position was opened without an explicit stop
        # tracked elsewhere (e.g. flattened via close_order()).
        return position.entry_price

    def check_liquidations(self, sym: str, low: Decimal, high: Decimal) -> Optional[Trade]:
        """Call once per bar with that bar's low/high to check whether
        an open leveraged position would have been liquidated."""
        pos = self.account.get_position(sym)
        if pos is None or pos.liquidation_price is None or pos.leverage <= 1:
            return None
        hit = (low <= pos.liquidation_price if pos.is_long()
               else high >= pos.liquidation_price)
        if hit:
            return self.market_order(sym, -pos.signed_qty, pos.liquidation_price, pos.leverage)
        return None


# ---------------------------------------------------------------------------
# Strategy: wires StreamingSignalEngine -> KellySizer/PortfolioGuardrails
# -> Order, i.e. the exact same decision logic the backtester uses,
# now driven tick-by-tick for paper/live trading.
# ---------------------------------------------------------------------------

class RegimeAwareStrategy:
    def __init__(self, symbols: list[str], guard: PortfolioGuardrails,
                 sizer: KellySizer, leverage_mgr: Optional[LeverageManager] = None,
                 conf_threshold: float = 0.35, stop_atr_mult: float = 2.0,
                 lookback: int = 300):
        self.signal_engine = StreamingSignalEngine(lookback=lookback)
        self.symbols = symbols
        self.guard = guard
        self.sizer = sizer
        self.leverage_mgr = leverage_mgr or LeverageManager(max_leverage=1, guard=guard)
        self.conf_threshold = conf_threshold
        self.stop_atr_mult = stop_atr_mult
        self._atr_state: dict[str, deque] = {}
        self._stops: dict[str, Decimal] = {}

    def _update_atr(self, sym: str, bar: dict) -> float:
        window = self._atr_state.setdefault(sym, deque(maxlen=14))
        window.append(bar['high'] - bar['low'])
        return sum(window) / len(window)

    def on_tick(self, sym: str, bar_time, bar: dict, account: Account) -> list[Order]:
        sig = self.signal_engine.on_tick(sym, bar_time, bar)
        atr = self._update_atr(sym, bar)
        if sig is None:
            return []

        position = account.get_position(sym)
        comp, conf = sig.get('composite'), sig.get('confidence', 0)

        # manage an existing position's stop, if any
        if position is not None and sym in self._stops:
            stop = self._stops[sym]
            hit = (bar['low'] <= float(stop) if position.is_long()
                   else bar['high'] >= float(stop))
            if hit:
                del self._stops[sym]
                return [position.close_order()]

        if position is not None or comp is None or abs(comp) < self.conf_threshold or conf < 0.5:
            return []

        open_positions = {sym: True for sym in self.symbols
                           if account.get_position(sym) is not None}
        can, _reason = self.guard.can_enter(sym, open_positions, {})
        if not can or atr <= 0:
            return []

        equity = float(account.balance())
        risk_pct = self.sizer.risk_pct()
        side = 'long' if comp > 0 else 'short'
        entry_price = bar['close']
        stop_price = (entry_price - self.stop_atr_mult * atr if side == 'long'
                      else entry_price + self.stop_atr_mult * atr)
        try:
            qty = position_size(equity, entry_price, stop_price, risk_pct)
        except ValueError:
            return []

        leverage = self.leverage_mgr.allowed_leverage(equity)
        self._stops[sym] = Decimal(str(stop_price))
        signed_qty = Decimal(str(qty)) * (1 if side == 'long' else -1)
        return [Order(sym, signed_qty, reason=f"composite={comp:.2f} conf={conf:.2f} lev={leverage}x")]
