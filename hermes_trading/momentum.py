"""
momentum.py — Dynamic momentum scanner for lesser-known coins.

Uses CoinGecko's free API to:
  1. Find top performers (24h and 7d) across the entire crypto market
  2. Filter by tradability on Coinbase (so we can actually trade them)
  3. Return a ranked list for basket rotation

Also uses the Coinbase market list to cross-reference which high-momentum
coins are actually tradeable.

Usage:
    from hermes_trading.momentum import MomentumScanner
    scanner = MomentumScanner()
    top = await scanner.scan_top_performers(min_rank=50, limit=20)
"""
import asyncio
import os
from datetime import datetime, timezone

import httpx


class MomentumScanner:
    """Scans crypto market for top-performing lesser-known coins."""

    COINGECKO_BASE = "https://api.coingecko.com/api/v3"

    def __init__(self):
        self.name = "momentum"

    async def _get_coinbase_symbols(self, client: httpx.AsyncClient) -> set:
        """Get all tradable symbols on Coinbase via ccxt."""
        import ccxt.async_support as ccxt
        ex = ccxt.coinbase({"enableRateLimit": True})
        try:
            await ex.load_markets()
            symbols = set()
            for s in ex.symbols:
                base = s.split("/")[0]
                if base not in ("USD", "USDT", "USDC", "EUR", "BTC", "ETH"):
                    symbols.add(base)
            return symbols
        finally:
            await ex.close()

    async def scan_top_performers(
        self,
        min_rank: int = 50,      # skip top-50 (already in basket or too big)
        limit: int = 20,          # how many to return
        min_volume_m: float = 1,  # min 24h volume in millions USD
    ) -> dict:
        """
        Scan CoinGecko for top 24h + 7d performers outside the top-50.

        Returns:
        {
            "top_24h": [...],  # sorted by 24h return
            "top_7d": [...],   # sorted by 7d return
            "tradable": [...], # intersection with Coinbase
            "timestamp": "..."
        }
        """
        async with httpx.AsyncClient(timeout=30) as client:
            # Get top performers via CoinGecko markets endpoint
            # Fetch pages 1-5 (250 coins per page = 1250 coins)
            all_coins = []
            for page in range(1, 6):
                r = await client.get(
                    f"{self.COINGECKO_BASE}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": 250,
                        "page": page,
                        "sparkline": "false",
                        "price_change_percentage": "24h,7d",
                    },
                )
                if r.status_code == 200:
                    all_coins.extend(r.json())
                else:
                    break
                await asyncio.sleep(0.5)  # rate limit

            # Also get Coinbase tradable set
            coinbase_bases = await self._get_coinbase_symbols(client)

        # Filter and sort
        filtered = []
        for coin in all_coins:
            rank = coin.get("market_cap_rank", 9999)
            if rank and rank < min_rank:
                continue  # skip blue chips

            vol = coin.get("total_volume", 0) or 0
            if vol < min_volume_m * 1_000_000:
                continue

            symbol = (coin.get("symbol") or "").upper()
            change_24h = coin.get("price_change_percentage_24h_in_currency") or coin.get("price_change_percentage_24h") or 0
            change_7d = coin.get("price_change_percentage_7d_in_currency") or coin.get("price_change_percentage_7d") or 0

            filtered.append({
                "symbol": symbol,
                "name": coin.get("name", ""),
                "rank": rank,
                "price_usd": coin.get("current_price", 0),
                "volume_24h_usd": vol,
                "change_24h_pct": round(change_24h, 2),
                "change_7d_pct": round(change_7d, 2),
                "tradable_on_coinbase": symbol in coinbase_bases,
                "coin_id": coin.get("id", ""),
            })

        # Sort by 24h and 7d
        top_24h = sorted(filtered, key=lambda x: x["change_24h_pct"], reverse=True)[:limit]
        top_7d = sorted(filtered, key=lambda x: x["change_7d_pct"], reverse=True)[:limit]

        # Tradable top performers (on Coinbase)
        tradable_24h = [c for c in top_24h if c["tradable_on_coinbase"]]
        tradable_7d = [c for c in top_7d if c["tradable_on_coinbase"]]

        # Also find lesser-known coins with momentum that ARE on Coinbase
        # but not in our current basket
        current_basket = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK"}
        new_opportunities = [
            c for c in filtered
            if c["tradable_on_coinbase"]
            and c["symbol"] not in current_basket
            and (abs(c["change_24h_pct"]) > 5 or abs(c["change_7d_pct"]) > 10)
        ]
        new_opportunities.sort(key=lambda x: x["change_7d_pct"], reverse=True)

        return {
            "top_24h": top_24h[:15],
            "top_7d": top_7d[:15],
            "tradable_24h": tradable_24h[:10],
            "tradable_7d": tradable_7d[:10],
            "new_opportunities": new_opportunities[:10],
            "scanned_count": len(all_coins),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def scan_gainers_and_losers(self, limit: int = 10) -> dict:
        """Quick scan of biggest 24h gainers and losers (all coins)."""
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                f"{self.COINGECKO_BASE}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": 1,
                    "sparkline": "false",
                    "price_change_percentage": "24h",
                },
            )
            if r.status_code != 200:
                return {"error": f"CoinGecko returned {r.status_code}"}

            coins = r.json()

        gainers = sorted(coins, key=lambda c: c.get("price_change_percentage_24h", 0) or 0, reverse=True)[:limit]
        losers = sorted(coins, key=lambda c: c.get("price_change_percentage_24h", 0) or 0)[:limit]

        def fmt(c):
            return {
                "symbol": (c.get("symbol") or "").upper(),
                "name": c.get("name"),
                "price": c.get("current_price"),
                "change_24h": round(c.get("price_change_percentage_24h", 0) or 0, 2),
                "volume_m": round((c.get("total_volume") or 0) / 1e6, 1),
                "rank": c.get("market_cap_rank"),
            }

        return {
            "gainers": [fmt(c) for c in gainers],
            "losers": [fmt(c) for c in losers],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
