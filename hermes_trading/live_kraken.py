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
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from decimal import Decimal

import ccxt.async_support as ccxt
from rich.console import Console

console = Console()


class LiveKrakenTrader:
    """
    Executes real trades on Kraken based on ICT strategy signals.
    Mirrors the ict_prop paper portfolio but executes real orders.
    """

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_file = state_dir / "live_kraken_state.json"

        self.api_key = os.environ.get("KRAKEN_API_KEY", "")
        self.api_secret = os.environ.get("KRAKEN_API_SECRET", "")
        self.exchange = None

        # Risk parameters (matching Starter Eval)
        self.risk_per_trade_pct = 0.5
        self.max_positions = 2
        self.max_daily_loss_pct = 3.0
        self.stop_loss_pct = 2.0
        self.take_profit_pct = 4.0  # R:R 1:2

        # State
        self.positions = {}  # asset -> {side, entry, stop, target, amount, order_id}
        self.starting_equity = None
        self.daily_start_equity = None
        self.current_date = None
        self.daily_trading_halted = False
        self.trade_count = 0

        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.positions = data.get("positions", {})
            self.starting_equity = data.get("starting_equity")
            self.trade_count = data.get("trade_count", 0)
            self.daily_start_equity = data.get("daily_start_equity")
            self.current_date = data.get("current_date")
            self.daily_trading_halted = data.get("daily_trading_halted", False)

    def _save_state(self):
        self.state_file.write_text(json.dumps({
            "positions": self.positions,
            "starting_equity": self.starting_equity,
            "trade_count": self.trade_count,
            "daily_start_equity": self.daily_start_equity,
            "current_date": self.current_date,
            "daily_trading_halted": self.daily_trading_halted,
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
        """Get total equity (USD + crypto value in USD)."""
        bal = await self.get_balance()
        ex = await self._get_exchange()
        equity = bal["usd"]
        for asset, amount in bal["holdings"].items():
            if asset == "USD":
                continue
            try:
                ticker = await ex.fetch_ticker(f"{asset}/USD")
                equity += amount * float(ticker.get("last", 0))
            except:
                pass  # can't price this asset
        return equity

    async def tick(self, signals: dict):
        """
        Main live trading tick.
        signals: {asset: {"signal": "long"/"short"/"flat", "price": float}}
        from the paper portfolio's current signals.

        1. Manage open positions (check stops/targets)
        2. Check daily loss limit
        3. Enter new positions based on signals
        """
        ts = datetime.now(timezone.utc).isoformat()
        ex = await self._get_exchange()

        # Get current equity
        equity = await self.get_equity()
        if self.starting_equity is None:
            self.starting_equity = equity
            console.print(f"  [live] Starting equity: ${equity:.2f}")

        # Daily reset
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.current_date:
            self.current_date = today
            self.daily_start_equity = equity
            self.daily_trading_halted = False
            console.print(f"  [live] New day — daily start: ${equity:.2f}")

        # Daily loss check
        if self.daily_start_equity and self.daily_start_equity > 0:
            daily_pnl_pct = (equity - self.daily_start_equity) / self.daily_start_equity * 100
            if daily_pnl_pct <= -self.max_daily_loss_pct:
                if not self.daily_trading_halted:
                    console.print(f"  [live] 🛑 DAILY LOSS LIMIT: {daily_pnl_pct:.1f}% — halting")
                    self.daily_trading_halted = True
                    self._save_state()
                    return

        # 1. Manage open positions
        await self._manage_positions(ex, ts)

        # 2. No new entries if halted or max positions
        if self.daily_trading_halted:
            return

        open_count = sum(1 for p in self.positions.values() if p is not None)
        if open_count >= self.max_positions:
            return

        # 3. Enter new positions
        for asset, sig in signals.items():
            if open_count >= self.max_positions:
                break

            if asset in self.positions and self.positions[asset] is not None:
                continue  # already holding

            signal = sig.get("signal", "flat")
            price = sig.get("price", 0)
            if signal not in ("long", "short") or price <= 0:
                continue

            await self._enter_position(ex, asset, signal, price, ts)
            open_count += 1

        self._save_state()

    async def _manage_positions(self, ex, ts):
        """Check stops and targets on open positions."""
        for asset in list(self.positions.keys()):
            pos = self.positions[asset]
            if pos is None:
                continue

            try:
                ticker = await ex.fetch_ticker(f"{asset}/USD")
                current_price = float(ticker.get("last", 0))
            except:
                continue

            if current_price <= 0:
                continue

            side = pos["side"]
            entry = pos["entry"]
            stop = pos["stop"]
            target = pos["target"]

            hit_stop = (current_price <= stop) if side == "long" else (current_price >= stop)
            hit_target = (current_price >= target) if side == "long" else (current_price <= target)

            if hit_stop or hit_target:
                reason = "stop_loss" if hit_stop else "take_profit"
                await self._close_position(ex, asset, current_price, reason, ts)

    async def _enter_position(self, ex, asset, side, price, ts):
        """Execute a real buy/sell order."""
        bal = await self.get_balance()
        usd = bal["usd"]

        # Position sizing: risk_per_trade_pct of equity
        position_value = usd * (self.risk_per_trade_pct / 100) * 2  # use 1% of capital per trade
        position_value = min(position_value, usd * 0.25)  # max 25% of available USD

        if position_value < 1.0:
            console.print(f"  [live] ⚠ Not enough USD for {asset} (need min $1, have ${usd:.2f})")
            return

        # Calculate amount
        amount = position_value / price

        # Round to Kraken precision
        try:
            market = ex.market(f"{asset}/USD")
            min_amount = market.get("limits", {}).get("amount", {}).get("min", 0.001)
            amount = max(amount, min_amount)
            # Round to precision
            precision = market.get("precision", {}).get("amount", 0.001)
            amount = int(amount / precision) * precision
            if amount <= 0:
                console.print(f"  [live] ⚠ Amount too small for {asset} after rounding")
                return
        except:
            pass

        stop = price * (1 - self.stop_loss_pct / 100) if side == "long" else price * (1 + self.stop_loss_pct / 100)
        target = price * (1 + self.take_profit_pct / 100) if side == "long" else price * (1 - self.take_profit_pct / 100)

        # Execute order
        try:
            if side == "long":
                order = await ex.create_market_buy_order(f"{asset}/USD", amount)
            else:
                # For short: need to sell what we don't have — Kraken spot doesn't support this
                # Skip shorts on spot (no margin without upgrade)
                console.print(f"  [live] ⚠ Short {asset} not supported on Kraken spot (need margin)")
                return

            filled = float(order.get("filled", 0) or 0)
            cost = float(order.get("cost", 0) or 0)
            avg_price = cost / filled if filled > 0 else price

            if filled > 0:
                self.positions[asset] = {
                    "side": side,
                    "entry": avg_price,
                    "stop": stop,
                    "target": target,
                    "amount": filled,
                    "cost": cost,
                    "opened_at": ts,
                    "order_id": order.get("id", ""),
                }
                console.print(f"  [live] 🔺 ENTER {asset} {side.upper()} {filled} @ ${avg_price:.4f} "
                              f"(cost ${cost:.2f}) stop=${stop:.4f} target=${target:.4f}")
                console.print(f"  [live] ⚡ Execute on Kraken Pro: BUY {filled} {asset}")

        except Exception as e:
            console.print(f"  [live] ⚠ Order error {asset}: {e}")

    async def _close_position(self, ex, asset, exit_price, reason, ts):
        """Close a position with a real sell order."""
        pos = self.positions[asset]
        if pos is None:
            return

        amount = pos["amount"]
        side = pos["side"]
        entry = pos["entry"]

        try:
            order = await ex.create_market_sell_order(f"{asset}/USD", amount)
            filled = float(order.get("filled", 0) or 0)
            cost = float(order.get("cost", 0) or 0)
            actual_exit = cost / filled if filled > 0 else exit_price

            pnl_pct = (actual_exit - entry) / entry * 100 if side == "long" else (entry - actual_exit) / entry * 100
            pnl_dollar = pnl_pct / 100 * pos["cost"]

            self.trade_count += 1
            color = "green" if pnl_dollar >= 0 else "red"
            console.print(f"  [live] [{color}]🔻 CLOSE {asset} {side} {reason} "
                          f"@ ${actual_exit:.4f} pnl={pnl_pct:+.2f}% (${pnl_dollar:+.2f})[/]")

            self.positions[asset] = None

            # Log trade
            trade = {
                "timestamp": ts, "portfolio": "live_kraken", "asset": asset,
                "side": side, "entry_price": entry, "exit_price": actual_exit,
                "pnl_pct": round(pnl_pct, 4), "pnl_dollar": round(pnl_dollar, 4),
                "reason": reason, "amount": filled, "mode": "live",
            }
            with open(self.state_dir / "trades.jsonl", "a") as f:
                f.write(json.dumps(trade) + "\n")

        except Exception as e:
            console.print(f"  [live] ⚠ Close error {asset}: {e}")

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
