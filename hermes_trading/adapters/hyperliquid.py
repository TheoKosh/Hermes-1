"""
hyperliquid.py — Hyperliquid DEX market data adapter.

Fetches real-time perpetual futures data from Hyperliquid's public API
(no auth needed): funding rates, open interest, volume, 24h changes, mark prices.

This data powers smarter entry signals for the rapid portfolio:
  - Funding rates → contrarian signals (overlevered = reversal likely)
  - Open interest → market conviction levels
  - Volume → liquidity confirmation
  - 24h change → momentum direction
"""
import asyncio
from datetime import datetime, timezone

import httpx


HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"


class HyperliquidAdapter:
    """
    Fetches perpetual futures market data from Hyperliquid.
    All endpoints are public, no authentication required.
    """

    def __init__(self):
        self.name = "hyperliquid"
        self._cache = {}        # symbol → market data
        self._last_update = None

    async def fetch_all_markets(self) -> dict:
        """
        Fetch all perpetual market data in one call.
        Returns: {symbol: {price, funding, oi_notional, vol_24h, change_24h, ...}}
        """
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(HYPERLIQUID_API, json={
                    "type": "metaAndAssetCtxs"
                })

                if r.status_code != 200:
                    return {}

                data = r.json()
                meta = data[0]
                ctxs = data[1]

            markets = {}
            for i, asset in enumerate(meta.get("universe", [])):
                ctx = ctxs[i] if i < len(ctxs) else {}
                name = asset.get("name", "?")

                price = float(ctx.get("markPx", 0))
                prev = float(ctx.get("prevDayPx", 0))
                funding = float(ctx.get("funding", 0))
                vol = float(ctx.get("dayNtlVlm", 0))
                oi = float(ctx.get("openInterest", 0))
                oi_notional = oi * price

                change_24h = ((price - prev) / prev * 100) if prev > 0 else 0.0

                markets[name] = {
                    "symbol": name,
                    "price": price,
                    "funding": funding,            # 8-hour funding rate (decimal)
                    "funding_pct_8h": funding * 100,  # as percentage
                    "open_interest": oi,           # in base units
                    "oi_notional_usd": oi_notional, # in USD
                    "volume_24h_usd": vol,
                    "change_24h_pct": round(change_24h, 2),
                    "prev_day_px": prev,
                    "oracle_px": float(ctx.get("oraclePx", 0)),
                    "max_leverage": int(asset.get("maxLeverage", 1)),
                }

            self._cache = markets
            self._last_update = datetime.now(timezone.utc).isoformat()
            return markets

        except Exception as e:
            return {}

    async def get_market(self, symbol: str) -> dict | None:
        """Get data for a single symbol. Fetches all then filters."""
        if not self._cache or self._is_stale():
            await self.fetch_all_markets()
        return self._cache.get(symbol.upper())

    def _is_stale(self, max_age_seconds: int = 60) -> bool:
        if not self._last_update:
            return True
        try:
            dt = datetime.fromisoformat(self._last_update)
            age = (datetime.now(timezone.utc) - dt).total_seconds()
            return age > max_age_seconds
        except (TypeError, ValueError):
            return True

    def compute_signal(self, symbol: str, base_price: float) -> dict:
        """
        Compute a Hyperliquid-derived signal for a symbol.

        Combines:
          - Funding rate (contrarian): high positive funding = bearish signal
            (longs are squeezed), high negative funding = bullish signal
          - Momentum (24h change): directional confirmation
          - Volume: liquidity weighting

        Returns:
        {
            "hl_score": float,       # -1.0 (bearish) to +1.0 (bullish)
            "hl_confidence": float,  # 0.0 to 1.0
            "funding_signal": float, # -1 to +1
            "momentum_signal": float,# -1 to +1
            "reasoning": str,
        }
        """
        m = self._cache.get(symbol.upper())
        if not m:
            return {"hl_score": 0.0, "hl_confidence": 0.0,
                    "funding_signal": 0.0, "momentum_signal": 0.0,
                    "reasoning": "no HL data"}

        funding = m.get("funding", 0)
        change = m.get("change_24h_pct", 0)
        vol = m.get("volume_24h_usd", 0)
        oi_notional = m.get("oi_notional_usd", 0)

        # --- FUNDING SIGNAL (contrarian) ---
        # Positive funding (longs pay shorts) → overlevered longs → bearish
        # Negative funding (shorts pay longs) → overlevered shorts → bullish
        # Typical funding ~0.001% to 0.05% per 8h
        # Map: funding > +0.01% → bearish, funding < -0.01% → bullish
        if funding > 0.0005:   # > 0.05% per 8h = very hot
            funding_signal = -1.0
        elif funding > 0.0002:
            funding_signal = -0.5
        elif funding > 0.00005:
            funding_signal = -0.2
        elif funding < -0.0005:
            funding_signal = 1.0
        elif funding < -0.0002:
            funding_signal = 0.5
        elif funding < -0.00005:
            funding_signal = 0.2
        else:
            funding_signal = 0.0

        # --- MOMENTUM SIGNAL (trend-following, NOT contrarian) ---
        # Strong positive 24h change → bullish momentum
        # Strong negative 24h change → bearish momentum
        if change > 10:
            momentum_signal = 1.0
        elif change > 5:
            momentum_signal = 0.6
        elif change > 2:
            momentum_signal = 0.3
        elif change > -2:
            momentum_signal = 0.0
        elif change > -5:
            momentum_signal = -0.3
        elif change > -10:
            momentum_signal = -0.6
        else:
            momentum_signal = -1.0

        # --- CONFIDENCE: higher volume + OI = more meaningful ---
        # Normalize: $10M+ vol = full confidence, <$1M = low confidence
        vol_score = min(1.0, vol / 10_000_000) if vol > 0 else 0.0
        oi_score = min(1.0, oi_notional / 50_000_000) if oi_notional > 0 else 0.0
        confidence = (vol_score * 0.6 + oi_score * 0.4)

        # --- COMPOSITE HL SCORE ---
        # Weight: 40% funding (contrarian), 40% momentum, 20% from confidence
        raw = funding_signal * 0.4 + momentum_signal * 0.4
        # Scale by confidence so low-liquidity signals are dampened
        hl_score = raw * (0.5 + 0.5 * confidence)  # dampened by 50% at zero confidence

        # Clamp
        hl_score = max(-1.0, min(1.0, hl_score))
        confidence = max(0.0, min(1.0, confidence))

        # Reasoning string
        parts = []
        if abs(funding_signal) > 0.2:
            direction = "overlevered longs" if funding > 0 else "overlevered shorts"
            parts.append(f"fund={funding*100:.4f}%({direction})")
        if abs(momentum_signal) > 0.2:
            parts.append(f"24h={change:+.1f}%")
        if vol > 1_000_000:
            parts.append(f"vol=${vol/1e6:.1f}M")
        reasoning = " | ".join(parts) if parts else "neutral"

        return {
            "hl_score": round(hl_score, 3),
            "hl_confidence": round(confidence, 3),
            "funding_signal": round(funding_signal, 3),
            "momentum_signal": round(momentum_signal, 3),
            "funding_pct": round(funding * 100, 4),
            "change_24h": round(change, 2),
            "volume_m": round(vol / 1e6, 1),
            "reasoning": reasoning,
        }
