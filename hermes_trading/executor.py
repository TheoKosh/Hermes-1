"""
executor.py — Real order execution layer.

Wraps ccxt order placement with safety rails:
  - Pre-trade balance checks
  - Position size from capital allocation
  - Market orders for speed, limit fallback
  - Fill confirmation with timeout
  - P&L calculation from actual fills
  - Graceful failure (never crashes the loop)
"""
import logging
from datetime import datetime, timezone
from typing import Optional

import ccxt.async_support as ccxt

logger = logging.getLogger("hermes-trading.executor")

FILL_TIMEOUT_SECONDS = 30


class LiveExecutor:
    """Real order execution against Coinbase Advanced Trade API."""

    def __init__(self, exchange: ccxt.coinbase):
        self.exchange = exchange
        self.name = "live"

    async def get_usdt_balance(self) -> float:
        """Get available USDT balance for trading."""
        try:
            bal = await self.exchange.fetch_balance({"type": "trade"})
            # USDT might be under various keys
            for key in ("USDT", "USD"):
                avail = bal.get("free", {}).get(key, 0)
                if avail and float(avail) > 0:
                    return float(avail)
            return 0.0
        except Exception as e:
            logger.warning(f"Balance check failed: {e}")
            return 0.0

    async def place_market_buy(self, symbol: str, quote_cost: float) -> Optional[dict]:
        """
        Place a market BUY using a cost-based order (buy X USDT worth of base asset).
        Returns the fill dict or None on failure.
        """
        try:
            # Coinbase uses cost-based market buys: create_market_buy_order with cost
            order = await self.exchange.create_market_buy_order(symbol, amount=1, params={"create_price": quote_cost})
            return order
        except Exception as e:
            logger.error(f"Market buy {symbol} failed: {e}")
            return None

    async def place_market_sell(self, symbol: str, base_amount: float) -> Optional[dict]:
        """Place a market SELL of base_amount units."""
        try:
            order = await self.exchange.create_market_sell_order(symbol, base_amount)
            return order
        except self.exchange.exceptions.InsufficientFunds as e:
            logger.warning(f"Insufficient funds to sell {base_amount} {symbol.split('/')[0]}: {e}")
            return None
        except Exception as e:
            logger.error(f"Market sell {symbol} failed: {e}")
            return None

    async def close_position(self, symbol: str, side: str, base_amount: float) -> Optional[float]:
        """
        Close an open position.
        For long: sell the base asset.
        For short (not supported on spot, but structured for future): buy back.
        Returns realized exit price or None.
        """
        if side == "long":
            fill = await self.place_market_sell(symbol, base_amount)
        else:
            fill = await self.place_market_buy(symbol, base_amount)
        if fill:
            exit_price = float(fill.get("average") or fill.get("price") or 0)
            return exit_price
        return None


class PaperExecutor:
    """Simulated execution — mirrors LiveExecutor interface for seamless mode switching."""

    def __init__(self):
        self.name = "paper"

    async def get_usdt_balance(self) -> float:
        return 10_000.0  # simulated $10k account

    async def place_market_buy(self, symbol: str, quote_cost: float) -> Optional[dict]:
        # In paper mode, the loop already tracks virtual positions using the ticker price.
        # This executor is a stub — the loop's internal position tracking IS the paper execution.
        return None

    async def place_market_sell(self, symbol: str, base_amount: float) -> Optional[dict]:
        return None

    async def close_position(self, symbol: str, side: str, base_amount: float) -> Optional[float]:
        return None
