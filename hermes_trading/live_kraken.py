"""
live_kraken.py — Live trading execution on real Kraken account.

Manages real fund execution with strict risk management:
  - $14 starting capital
  - 0.5% risk per trade (max ~$0.07 loss per trade)
  - Max 2 concurrent positions
  - 3% daily loss limit
  - ICT strategy signals from the paper portfolio
  - Real order execution via Kraken API

This module executes REAL trades with REAL money.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import ccxt.async_support as ccxt
from rich.console import Console

console = Console()


class SmallCapitalMode:
    """
    Risk preset for sub-$100 accounts where big funds haven't arrived yet.

    Philosophy:
      - Use higher risk per trade to generate meaningful P&L on tiny capital.
      - Allow more concurrent positions to increase throughput.
      - Keep R:R ≥ 1.5x so occasional wins offset more frequent small losses.
      - Maintain a daily-loss circuit so a bad session doesn't zero the account.

    Parameters (adjustable in LiveKrakenTrader.risk_profile):
      risk_per_trade_pct : 2.0%  (~$0.35 per trade on $17)
      max_positions      : 5     (~$13 deployed at once on $17)
      max_daily_loss_pct : 5.0%  (~$0.87 daily loss ceiling)
      stop_loss_pct      : 3.0%
      take_profit_pct    : 6.0%  (R:R 1:2 after fees)
      position_pct       : 15%   of equity per position
    """


class LiveKrakenTrader:
    """
    Executes real trades on Kraken based on ICT strategy signals.
    Mirrors the ict_prop paper portfolio but executes real orders.
    """

    def __init__(self, state_dir: Path, enabled: bool = False):
        self.state_dir = state_dir
        self.state_file = state_dir / "live_kraken_state.json"

        self.api_key = os.environ.get("KRAKEN_API_KEY", "")
        self.api_secret = os.environ.get("KRAKEN_API_SECRET", "")
        self.exchange = None

        # Live execution is OFF unless explicitly enabled AND credentials exist.
        # This prevents paper-mode runs from placing real orders.
        self.enabled = bool(enabled and self.api_key and self.api_secret)
        if enabled and not (self.api_key and self.api_secret):
            console.print("  [live] [yellow]⚠ Live trading requested but "
                          "KRAKEN_API_KEY/SECRET missing — staying disabled[/]")

        # Small-capital aggressive risk profile
        # Override these via set_risk_profile() for different modes
        self.risk_per_trade_pct = 2.0      # 2% equity at risk per trade
        self.max_positions = 5             # up to 5 concurrent positions
        self.max_daily_loss_pct = 5.0      # 5% daily loss circuit
        self.max_drawdown_pct = 15.0       # 15% peak-to-trough kill switch
        self.stop_loss_pct = 3.0           # 3% stop
        self.take_profit_pct = 6.0         # 6% target = 1:2 R:R
        self.position_pct = 0.15           # 15% of equity per position
        self.max_position_bump_pct = 0.25  # hard cap when bumping to exchange min
        self.min_notional_usd = 1.0        # skip entries below this notional

        # State
        self.positions = {}  # asset -> {side, entry, stop, target, amount, order_id, cost}
        self.starting_equity = None
        self.peak_equity = None
        self.daily_start_equity = None
        self.current_date = None
        self.daily_trading_halted = False
        self.drawdown_halted = False
        self.trade_count = 0
        self._last_equity = None

        self._load_state()

    # ---------- risk helpers ----------

    def drawdown_pct(self) -> float:
        """Peak-to-trough drawdown in percent (positive number)."""
        if not self.peak_equity or self.peak_equity <= 0:
            return 0.0
        current = self._last_equity if self._last_equity is not None else self.peak_equity
        return max(0.0, (self.peak_equity - current) / self.peak_equity * 100)

    def open_position_count(self) -> int:
        return sum(1 for p in self.positions.values() if p is not None)

    def trading_halted(self) -> bool:
        """True when any risk circuit has tripped."""
        return self.daily_trading_halted or self.drawdown_halted

    def set_risk_profile(self, profile: str = "small_capital"):
        """
        Switch risk presets without restarting.
          small_capital → high-risk small-capital mode (default)
          eval_strict   → Starter Eval rules (0.5% risk, 2 positions, 3% DD)
        """
        if profile == "eval_strict":
            self.risk_per_trade_pct = 0.5
            self.max_positions = 2
            self.max_daily_loss_pct = 3.0
            self.max_drawdown_pct = 6.0
            self.stop_loss_pct = 2.0
            self.take_profit_pct = 4.0
            self.position_pct = 0.05
            self.max_position_bump_pct = 0.10
        elif profile == "small_capital":
            self.risk_per_trade_pct = 2.0
            self.max_positions = 5
            self.max_daily_loss_pct = 5.0
            self.max_drawdown_pct = 15.0
            self.stop_loss_pct = 3.0
            self.take_profit_pct = 6.0
            self.position_pct = 0.15
            self.max_position_bump_pct = 0.25
        else:
            raise ValueError(f"unknown profile: {profile}")

        self.profile = profile
        console.print(f"  [live] Risk profile → {profile}: "
                      f"risk={self.risk_per_trade_pct}% pos={self.max_positions} "
                      f"daily={self.max_daily_loss_pct}% maxDD={self.max_drawdown_pct}% "
                      f"stop={self.stop_loss_pct}% TP={self.take_profit_pct}%")
        self._save_state()

    def _load_state(self):
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.positions = data.get("positions", {})
            self.starting_equity = data.get("starting_equity")
            self.peak_equity = data.get("peak_equity")
            self.trade_count = data.get("trade_count", 0)
            self.daily_start_equity = data.get("daily_start_equity")
            self.current_date = data.get("current_date")
            self.daily_trading_halted = data.get("daily_trading_halted", False)
            self.drawdown_halted = data.get("drawdown_halted", False)
            self._last_equity = data.get("last_equity")

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({
            "positions": self.positions,
            "starting_equity": self.starting_equity,
            "peak_equity": self.peak_equity,
            "last_equity": self._last_equity,
            "trade_count": self.trade_count,
            "daily_start_equity": self.daily_start_equity,
            "current_date": self.current_date,
            "daily_trading_halted": self.daily_trading_halted,
            "drawdown_halted": self.drawdown_halted,
            "profile": getattr(self, "profile", "small_capital"),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    async def _get_exchange(self):
        if self.exchange is None:
            self.exchange = ccxt.kraken({
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
            })
        return self.exchange

    async def get_balance(self) -> dict:
        """Get current USD balance and all crypto holdings."""
        ex = await self._get_exchange()
        balance = await ex.fetch_balance()
        usd = float(balance.get("USD", {}).get("free", 0))
        holdings = {}
        for asset, vals in balance.get("total", {}).items():
            amt = float(vals)
            if amt > 0:
                holdings[asset] = amt
        return {"usd": usd, "holdings": holdings}

    async def get_equity(self) -> float:
        """Get total equity (USD + crypto valued in USD)."""
        bal = await self.get_balance()
        ex = await self._get_exchange()
        equity = bal["usd"]
        for asset, amount in bal["holdings"].items():
            if asset in ("USD", "ZUSD"):
                continue
            try:
                ticker = await ex.fetch_ticker(f"{asset}/USD")
                last = float(ticker.get("last") or 0)
                equity += amount * last
            except Exception:
                continue  # unpriceable asset (delisted/no USD pair) — ignore
        return equity

    async def tick(self, signals: dict):
        """
        Main live trading tick.
        signals: {asset: {"signal": "long"/"short"/"flat", "price": float}}

        1. Refresh equity, update peak, evaluate risk circuits
        2. Manage open positions (stops/targets)
        3. Enter new positions when circuits allow
        """
        if not self.enabled:
            return

        ts = datetime.now(timezone.utc).isoformat()
        ex = await self._get_exchange()

        # --- equity + peak tracking ---
        equity = await self.get_equity()
        self._last_equity = equity
        if self.starting_equity is None:
            self.starting_equity = equity
            console.print(f"  [live] Starting equity: ${equity:.2f}")
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity

        # --- daily reset ---
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.current_date:
            self.current_date = today
            self.daily_start_equity = equity
            self.daily_trading_halted = False
            console.print(f"  [live] New day — daily start: ${equity:.2f}")

        # --- risk circuit: max drawdown (hard kill switch) ---
        dd = self.drawdown_pct()
        if dd >= self.max_drawdown_pct:
            if not self.drawdown_halted:
                console.print(f"  [live] 🛑 MAX DRAWDOWN {dd:.2f}% >= "
                              f"{self.max_drawdown_pct}% — halting all new entries")
                self.drawdown_halted = True
                self._save_state()
        elif self.drawdown_halted and dd < self.max_drawdown_pct * 0.5:
            # recover only after drawdown halves (hysteresis, avoids flapping)
            console.print(f"  [live] ✅ Drawdown recovered to {dd:.2f}% — resuming")
            self.drawdown_halted = False

        # --- risk circuit: daily loss ---
        if self.daily_start_equity and self.daily_start_equity > 0:
            daily_pnl_pct = (equity - self.daily_start_equity) / self.daily_start_equity * 100
            if daily_pnl_pct <= -self.max_daily_loss_pct and not self.daily_trading_halted:
                console.print(f"  [live] 🛑 DAILY LOSS LIMIT {daily_pnl_pct:.2f}% — halting for today")
                self.daily_trading_halted = True
                self._save_state()

        # 1. Always manage open positions, even when halted (stops must still fire)
        await self._manage_positions(ex, ts)

        # 2. No new entries while any circuit is tripped
        if self.trading_halted():
            self._save_state()
            return

        open_count = self.open_position_count()
        if open_count >= self.max_positions:
            self._save_state()
            return

        # 3. Enter new positions
        for asset, sig in signals.items():
            if open_count >= self.max_positions:
                break
            if self.positions.get(asset) is not None:
                continue  # already holding

            signal = sig.get("signal", "flat")
            price = float(sig.get("price") or 0)
            if signal not in ("long", "short") or price <= 0:
                continue
            # Kraken spot cannot short — skip early rather than round-trip the API
            if signal == "short":
                continue

            entered = await self._enter_position(ex, asset, signal, price, ts)
            if entered:
                open_count += 1

        self._save_state()

    async def _manage_positions(self, ex, ts):
        """Check stops and targets on open positions."""
        for asset in list(self.positions.keys()):
            pos = self.positions[asset]
            if pos is None:
                continue

            symbol = f"{asset}/USD" if "/" not in asset else asset
            try:
                ticker = await ex.fetch_ticker(symbol)
                current_price = float(ticker.get("last") or 0)
            except Exception as e:
                console.print(f"  [live] ⚠ Price fetch failed for {symbol}: {e}")
                continue

            if current_price <= 0:
                continue

            side = pos["side"]
            stop = float(pos["stop"])
            target = float(pos["target"])

            hit_stop = (current_price <= stop) if side == "long" else (current_price >= stop)
            hit_target = (current_price >= target) if side == "long" else (current_price <= target)

            if hit_stop or hit_target:
                reason = "stop_loss" if hit_stop else "take_profit"
                await self._close_position(ex, asset, current_price, reason, ts)

    async def _enter_position(self, ex, asset, side, price, ts) -> bool:
        """Execute a real buy order. Returns True if a position was opened."""
        symbol = f"{asset}/USD" if "/" not in asset else asset

        bal = await self.get_balance()
        usd = bal["usd"]

        # Position sizing: position_pct of available USD
        position_value = usd * self.position_pct
        if position_value < self.min_notional_usd:
            console.print(f"  [live] ⚠ Insufficient USD for {symbol} "
                          f"(need ${self.min_notional_usd:.2f}, sized ${position_value:.2f})")
            return False

        amount = position_value / price

        # Respect Kraken market limits/precision
        try:
            market = ex.market(symbol)
            limits = market.get("limits", {})
            min_amount = (limits.get("amount", {}) or {}).get("min") or 0.0
            min_cost = (limits.get("cost", {}) or {}).get("min") or 0.0

            if min_amount and amount < min_amount:
                needed = min_amount * price
                # Exchange minimum exceeds our intended size. Only bump up to a
                # hard cap — otherwise a single min-size order silently becomes
                # a huge share of a small account (e.g. $5.31 on a $14 balance
                # = 38% risk when we asked for 15%).
                max_bump = usd * self.max_position_bump_pct
                if needed > max_bump:
                    console.print(f"  [live] ⚠ {symbol} min size needs ${needed:.2f} > "
                                  f"cap ${max_bump:.2f} ({self.max_position_bump_pct:.0%} of "
                                  f"${usd:.2f}) — skipping")
                    return False
                console.print(f"  [live] ↑ {symbol} bumped ${position_value:.2f} → "
                              f"${needed:.2f} to meet exchange minimum")
                amount = min_amount

            if min_cost and amount * price < min_cost:
                console.print(f"  [live] ⚠ {symbol} notional ${amount * price:.2f} "
                              f"below exchange min ${min_cost:.2f}")
                return False

            # Use ccxt's amount_to_precision: it handles both tick-size and
            # decimal-place precision modes correctly across exchanges.
            amount = float(ex.amount_to_precision(symbol, amount))
        except Exception as e:
            console.print(f"  [live] ⚠ Precision/limits lookup failed for {symbol}: {e}")
            return False

        if amount <= 0:
            console.print(f"  [live] ⚠ Amount rounds to zero for {symbol}")
            return False

        if side == "long":
            stop = price * (1 - self.stop_loss_pct / 100)
            target = price * (1 + self.take_profit_pct / 100)
        else:
            console.print(f"  [live] ⚠ Short {symbol} unsupported on Kraken spot — skipping")
            return False

        try:
            order = await ex.create_market_buy_order(symbol, amount)
        except Exception as e:
            console.print(f"  [live] ⚠ Order error {symbol}: {e}")
            return False

        filled = float(order.get("filled") or 0)
        cost = float(order.get("cost") or 0)
        if filled <= 0:
            console.print(f"  [live] ⚠ {symbol} order returned zero fill")
            return False

        avg_price = (cost / filled) if cost > 0 else price
        # Recompute levels off the ACTUAL fill, not the signal price
        stop = avg_price * (1 - self.stop_loss_pct / 100)
        target = avg_price * (1 + self.take_profit_pct / 100)

        self.positions[asset] = {
            "side": side,
            "entry": avg_price,
            "stop": stop,
            "target": target,
            "amount": filled,
            "cost": cost if cost > 0 else filled * avg_price,
            "opened_at": ts,
            "order_id": order.get("id", ""),
        }
        console.print(f"  [live] 🔺 ENTER {symbol} {side.upper()} {filled} @ ${avg_price:.4f} "
                      f"(cost ${cost:.2f}) stop=${stop:.4f} target=${target:.4f}")
        self._save_state()
        return True

    async def _close_position(self, ex, asset, exit_price, reason, ts):
        """Close a position with a real sell order, reconciled against actual balance."""
        pos = self.positions.get(asset)
        if pos is None:
            return

        symbol = f"{asset}/USD" if "/" not in asset else asset
        base = symbol.split("/")[0]
        amount = float(pos["amount"])
        side = pos["side"]
        entry = float(pos["entry"])

        # Reconcile against real holdings — never try to sell more than we own
        try:
            bal = await self.get_balance()
            held = float(bal["holdings"].get(base, 0) or 0)
            if held <= 0:
                console.print(f"  [live] ⚠ No {base} balance to close — clearing stale position")
                self.positions[asset] = None
                self._save_state()
                return
            if held < amount:
                console.print(f"  [live] ⚠ Holding {held} {base} < tracked {amount} — closing what we hold")
                amount = held
            amount = float(ex.amount_to_precision(symbol, amount))
        except Exception as e:
            console.print(f"  [live] ⚠ Balance reconcile failed for {symbol}: {e}")

        if amount <= 0:
            self.positions[asset] = None
            self._save_state()
            return

        try:
            order = await ex.create_market_sell_order(symbol, amount)
        except Exception as e:
            console.print(f"  [live] ⚠ Close error {symbol}: {e}")
            return

        filled = float(order.get("filled") or 0)
        proceeds = float(order.get("cost") or 0)
        actual_exit = (proceeds / filled) if filled > 0 and proceeds > 0 else exit_price

        # Fee-aware P&L: use actual cash in/out where the exchange reports it
        entry_cost = float(pos.get("cost") or (amount * entry))
        fee_paid = 0.0
        try:
            fee = order.get("fee") or {}
            fee_paid = float(fee.get("cost") or 0)
        except (TypeError, ValueError):
            fee_paid = 0.0

        pnl_dollar = (proceeds - entry_cost - fee_paid) if proceeds > 0 else 0.0
        pnl_pct = (pnl_dollar / entry_cost * 100) if entry_cost > 0 else 0.0

        self.trade_count += 1
        color = "green" if pnl_dollar >= 0 else "red"
        console.print(f"  [live] [{color}]🔻 CLOSE {symbol} {side} {reason} "
                      f"@ ${actual_exit:.4f} pnl={pnl_pct:+.2f}% (${pnl_dollar:+.2f})[/]")

        self.positions[asset] = None

        trade = {
            "timestamp": ts, "portfolio": "live_kraken", "asset": asset,
            "side": side, "entry_price": entry, "exit_price": actual_exit,
            "pnl_pct": round(pnl_pct, 4), "pnl_dollar": round(pnl_dollar, 4),
            "fee_paid": round(fee_paid, 6),
            "reason": reason, "amount": filled, "mode": "live",
        }
        try:
            with open(self.state_dir / "trades.jsonl", "a") as f:
                f.write(json.dumps(trade) + "\n")
        except OSError as e:
            console.print(f"  [live] ⚠ Trade log write failed: {e}")

        self._save_state()

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
