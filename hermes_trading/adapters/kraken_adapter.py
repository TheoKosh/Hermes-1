"""
kraken_adapter.py — Kraken exchange adapter for price data and (future) live trading.

Uses ccxt's Kraken integration. In paper mode, fetches OHLCV + ticker data
alongside Coinbase for cross-exchange price verification. In live mode
(future), executes real trades via Kraken's API.

Kraken-specific notes:
  - Pair format: BTC/USD (ccxt normalizes from XBT/USD)
  - Kraken has different fee tiers than Coinbase (0.26% taker, 0.16% maker
    for Tier 0/1, decreasing with volume)
  - Some Coinbase pairs aren't on Kraken and vice versa
  - Kraken supports more altcoins and some pairs Coinbase doesn't
"""
import os
from typing import Optional

import ccxt.async_support as ccxt


class KrakenAdapter:
    """
    Kraken exchange adapter. In paper mode, provides price data only.
    In live mode (future), executes trades via authenticated API.
    """

    def __init__(self, api_key: str = "", api_secret: str = "", mode: str = "paper"):
        self.mode = mode
        self._api_key = api_key or os.environ.get("KRAKEN_API_KEY", "")
        self._api_secret = api_secret or os.environ.get("KRAKEN_API_SECRET", "")
        self._exchange = None

    async def _get_exchange(self):
        if self._exchange is None:
            params = {"enableRateLimit": True}
            if self._api_key and self._api_secret:
                params["apiKey"] = self._api_key
                params["secret"] = self._api_secret
            self._exchange = ccxt.kraken(params)
        return self._exchange

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker (price, bid, ask, volume) for a symbol."""
        ex = await self._get_exchange()
        ticker = await ex.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "price": ticker.get("last", 0),
            "bid": ticker.get("bid", 0),
            "ask": ticker.get("ask", 0),
            "volume": ticker.get("baseVolume", 0),
            "timestamp": ticker.get("timestamp"),
        }

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 50) -> list:
        """Fetch OHLCV candles for a symbol."""
        ex = await self._get_exchange()
        return await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    async def fetch_markets(self) -> list:
        """Fetch all available trading pairs on Kraken."""
        ex = await self._get_exchange()
        await ex.load_markets()
        return list(ex.symbols)

    async def verify_price(self, symbol: str, coinbase_price: float) -> dict:
        """
        Cross-reference Coinbase price against Kraken.
        Returns verification result with deviation check.
        """
        try:
            ticker = await self.fetch_ticker(symbol)
            kraken_price = ticker["price"]
            if kraken_price <= 0 or coinbase_price <= 0:
                return {"verified": False, "reason": "zero price", "kraken": kraken_price}

            deviation_pct = abs(kraken_price - coinbase_price) / coinbase_price * 100

            return {
                "verified": deviation_pct < 1.0,  # <1% deviation = verified
                "kraken_price": kraken_price,
                "coinbase_price": coinbase_price,
                "deviation_pct": round(deviation_pct, 3),
                "symbol": symbol,
            }
        except Exception as e:
            return {"verified": False, "reason": str(e), "kraken": 0}

    async def close(self):
        if self._exchange:
            await self._exchange.close()
            self._exchange = None


class KrakenFeeModel:
    """
    Kraken fee schedule (Tier 0/1, < $50K 30d volume):
      Taker: 0.26%
      Maker: 0.16%

    Higher tiers decrease with volume. Cheaper than Coinbase Tier 1
    (0.60% taker) for the same volume bracket.
    """

    def __init__(self, taker_fee_pct: float = 0.26, maker_fee_pct: float = 0.16):
        self.taker_fee_pct = taker_fee_pct
        self.maker_fee_pct = maker_fee_pct
        self.tier_name = "Tier 0/1 (<$50K)"

    def round_trip_cost_pct(self, order_type: str = "taker") -> float:
        fee = self.taker_fee_pct if order_type == "taker" else self.maker_fee_pct
        return fee * 2

    def fee_per_side_pct(self, order_type: str = "taker") -> float:
        return self.taker_fee_pct if order_type == "taker" else self.maker_fee_pct

    def compute_fee_cost(self, position_value: float, order_type: str = "taker") -> float:
        """Compute dollar fee cost for a single order."""
        return position_value * self.fee_per_side_pct(order_type) / 100

    def effective_rr(self, stop_loss_pct: float, take_profit_pct: float,
                     order_type: str = "taker") -> dict:
        """Compute effective R:R after fees."""
        rtc = self.round_trip_cost_pct(order_type)
        net_target = take_profit_pct - rtc
        net_stop = stop_loss_pct + rtc
        gross_rr = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0
        net_rr = net_target / net_stop if net_stop > 0 else 0
        return {
            "gross_rr": round(gross_rr, 3), "net_rr": round(net_rr, 3),
            "net_target_pct": round(net_target, 3), "net_stop_pct": round(net_stop, 3),
            "round_trip_cost_pct": round(rtc, 3), "viable": net_rr >= 1.0,
        }

    def summary(self) -> str:
        return (f"Kraken fees {self.tier_name}: "
                f"taker={self.taker_fee_pct}%, maker={self.maker_fee_pct}%, "
                f"round-trip={self.round_trip_cost_pct()}%")
