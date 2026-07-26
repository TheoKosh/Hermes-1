"""
price.py — Price adapter.

Fetches current price and RSI(14) for the configured asset.
Free: Binance public API via ccxt.
Premium: authenticated exchange via EXCHANGE_API_KEY/SECRET in .env.

Schema:
{
    "schema_version": 1,
    "asset": "BTC/USDT",
    "price": 65000.0,
    "rsi": 35.2,
    "timestamp": "2024-..."
}
"""
import os
import time
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import numpy as np


SCHEMA_VERSION = 1


class SchemaError(Exception):
    """Raised when adapter output doesn't match expected schema."""
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PriceAdapter:
    def __init__(self, asset: str):
        self.asset = asset
        self.name = "price"
        self._exchange = None
        self._ohlcv_cache = []

    async def _get_exchange(self):
        if self._exchange is not None:
            return self._exchange

        api_key = os.environ.get("EXCHANGE_API_KEY", "")
        api_secret = os.environ.get("EXCHANGE_API_SECRET", "")

        self._exchange = ccxt.kraken({
            "apiKey": api_key or None,
            "secret": api_secret or None,
            "enableRateLimit": True,
        })
        return self._exchange

    def _compute_rsi(self, closes: list, period: int = 14) -> float:
        """Compute RSI from a list of close prices."""
        if len(closes) < period + 1:
            return 50.0  # neutral

        closes_arr = np.array(closes, dtype=float)
        deltas = np.diff(closes_arr)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi), 2)

    def _compute_atr(self, ohlcv: list, period: int = 14) -> float:
        """Compute Average True Range for volatility-adaptive stops."""
        if len(ohlcv) < period + 1:
            return 0.0

        trs = []
        for i in range(1, len(ohlcv)):
            high = float(ohlcv[i][2])
            low = float(ohlcv[i][3])
            prev_close = float(ohlcv[i-1][4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)

        atr = sum(trs[-period:]) / period
        return round(atr, 6)

    def _compute_ema(self, closes: list, period: int = 20) -> float:
        """Compute EMA for trend filter."""
        if len(closes) < period:
            return sum(closes) / max(1, len(closes))

        multiplier = 2 / (period + 1)
        ema = closes[0]
        for c in closes[1:]:
            ema = (float(c) - ema) * multiplier + ema
        return ema

    async def fetch(self) -> dict:
        exchange = await self._get_exchange()

        # fetch ticker — illiquid/untraded pairs can report last=None,
        # so fall back to close/bid-ask midpoint before giving up.
        ticker = await exchange.fetch_ticker(self.asset)
        raw_price = ticker.get("last") or ticker.get("close")
        if raw_price is None:
            bid, ask = ticker.get("bid"), ticker.get("ask")
            if bid and ask:
                raw_price = (float(bid) + float(ask)) / 2
        if raw_price is None:
            raise SchemaError(f"price adapter: {self.asset} has no usable price "
                              f"(last/close/bid-ask all empty — illiquid pair)")
        price = float(raw_price)

        # fetch OHLCV for RSI (15m candles, last 50)
        ohlcv = await exchange.fetch_ohlcv(self.asset, timeframe="15m", limit=50)
        # Drop candles with null closes rather than crashing on float(None)
        ohlcv = [c for c in ohlcv if c and len(c) > 4 and c[4] is not None]
        if not ohlcv:
            raise SchemaError(f"price adapter: {self.asset} returned no valid candles")

        closes = [float(c[4]) for c in ohlcv]
        rsi = self._compute_rsi(closes)
        atr = self._compute_atr(ohlcv)
        ema20 = self._compute_ema(closes, 20)

        result = {
            "schema_version": SCHEMA_VERSION,
            "asset": self.asset,
            "price": price,
            "rsi": rsi,
            "atr": atr,
            "ema20": ema20,
            "closes": closes,
            "volumes": [float(c[5]) for c in ohlcv
                        if len(c) > 5 and c[5] is not None],
            "above_ema": price > ema20,
            "timestamp": now_iso(),
        }

        self._validate(result)
        return result

    def _validate(self, data: dict):
        required = {"schema_version", "asset", "price", "rsi", "timestamp"}
        if not required.issubset(data.keys()):
            raise SchemaError(f"price adapter: missing keys {required - set(data.keys())}")
        if data["schema_version"] != SCHEMA_VERSION:
            raise SchemaError(f"price adapter: schema mismatch (got {data['schema_version']}, expected {SCHEMA_VERSION})")

    async def close(self):
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
