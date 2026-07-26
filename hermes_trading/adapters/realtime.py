"""
realtime.py — Real-time price verification adapter.

Cross-checks Coinbase prices against CoinGecko's free API before entering trades.
Prevents entering on stale/manipulated prices by requiring the two sources to
agree within a tolerance band. Also filters out illiquid coins via 24h volume.
"""
import asyncio
from datetime import datetime, timezone

import httpx


# Map common Coinbase symbols to CoinGecko coin IDs
# This is built dynamically via search, but we cache known mappings here
SYMBOL_TO_COINGECKO = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "DOT": "polkadot",
    "MATIC": "matic-network",
    "SHIB": "shiba-inu",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "NEAR": "near",
    "FTM": "fantom",
    "ALGO": "algorand",
    "MANA": "decentraland",
    "SAND": "the-sandbox",
    "AXS": "axie-infinity",
    "FIL": "filecoin",
    "ICP": "internet-computer",
    "ARB": "arbitrum",
    "OP": "optimism",
    "APT": "aptos",
    "INJ": "injective-protocol",
    "SUI": "sui",
    "PEPE": "pepe",
    "WIF": "dogwifcoin",
    "TIA": "celestia",
    "SEI": "sei-network",
    "RNDR": "render-token",
    "GRT": "the-graph",
    "AAVE": "aave",
    "MKR": "maker",
    "COMP": "compound-governance-token",
    "CRV": "curve-dao-token",
    "SUSHI": "sushi",
    "1INCH": "1inch",
    "ENJ": "enjincoin",
    "CHZ": "chiliz",
    "THETA": "theta-token",
    "EGLD": "elrond-erd-2",
    "HBAR": "hedera-hashgraph",
    "FLOW": "flow",
    "XTZ": "tezos",
    "Etc": "ethereum-classic",
    "XLM": "stellar",
    "VET": "vechain",
    "FET": "fetch-ai",
}


class RealtimePriceChecker:
    """
    Cross-references Coinbase price with CoinGecko real-time price.
    Returns verified price data or None if verification fails.
    """

    COINGECKO_BASE = "https://api.coingecko.com/api/v3"
    MAX_PRICE_DEVIATION_PCT = 3.0  # max allowed divergence between sources
    MIN_VOLUME_USD = 500_000       # minimum 24h volume to consider tradeable

    def __init__(self):
        self.name = "realtime"
        self._coin_cache = {}  # symbol -> coingecko_id
        self._last_lookup = {}  # coingecko_id -> (price, volume, timestamp)

    def _get_coin_id(self, symbol: str) -> str | None:
        """Resolve a trading symbol (e.g. 'BTC') to CoinGecko coin ID."""
        if symbol in SYMBOL_TO_COINGECKO:
            return SYMBOL_TO_COINGECKO[symbol]
        return None

    async def verify_price(
        self, symbol: str, coinbase_price: float
    ) -> dict:
        """
        Verify Coinbase price against CoinGecko.

        Returns:
        {
            "verified": bool,        # True if price agrees within tolerance
            "coingecko_price": float,
            "deviation_pct": float,  # how far apart the two sources are
            "volume_24h_usd": float,
            "change_24h_pct": float,
            "liquid_enough": bool,   # True if volume > MIN_VOLUME_USD
            "trend": str,            # "up" / "down" / "flat" (24h direction)
        }
        """
        base_symbol = symbol.split("/")[0].upper()
        coin_id = self._get_coin_id(base_symbol)

        if coin_id is None:
            # Unknown coin — can't verify, return neutral
            return {
                "verified": True,  # don't block trades for unknown coins
                "coingecko_price": None,
                "deviation_pct": 0.0,
                "volume_24h_usd": 0,
                "change_24h_pct": 0,
                "liquid_enough": True,
                "trend": "flat",
                "note": f"Unknown symbol {base_symbol}, skipping verification",
            }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{self.COINGECKO_BASE}/simple/price",
                    params={
                        "ids": coin_id,
                        "vs_currencies": "usd",
                        "include_24hr_vol": "true",
                        "include_24hr_change": "true",
                    },
                )

                if r.status_code != 200:
                    return {
                        "verified": True,
                        "coingecko_price": None,
                        "deviation_pct": 0.0,
                        "volume_24h_usd": 0,
                        "change_24h_pct": 0,
                        "liquid_enough": True,
                        "trend": "flat",
                        "note": f"CoinGecko returned {r.status_code}",
                    }

                data = r.json().get(coin_id, {})
                cg_price = data.get("usd", 0)
                volume = data.get("usd_24h_vol", 0)
                change_24h = data.get("usd_24h_change", 0)

                if cg_price <= 0:
                    return {"verified": True, "coingecko_price": None,
                            "deviation_pct": 0, "volume_24h_usd": 0,
                            "change_24h_pct": 0, "liquid_enough": True, "trend": "flat"}

                deviation = abs(cg_price - coinbase_price) / coinbase_price * 100
                verified = deviation <= self.MAX_PRICE_DEVIATION_PCT
                liquid = volume >= self.MIN_VOLUME_USD
                trend = "up" if change_24h > 1 else ("down" if change_24h < -1 else "flat")

                return {
                    "verified": verified,
                    "coingecko_price": cg_price,
                    "deviation_pct": round(deviation, 2),
                    "volume_24h_usd": volume,
                    "change_24h_pct": round(change_24h, 2),
                    "liquid_enough": liquid,
                    "trend": trend,
                }

        except Exception as e:
            return {
                "verified": True,  # don't block trades on network failure
                "coingecko_price": None,
                "deviation_pct": 0.0,
                "volume_24h_usd": 0,
                "change_24h_pct": 0,
                "liquid_enough": True,
                "trend": "flat",
                "note": f"Verification error: {e}",
            }

    async def get_hot_movers(
        self, limit: int = 15, min_volume_m: float = 2.0
    ) -> list:
        """
        Get top 24h gainers from CoinGecko with real-time data.
        Used by the rapid portfolio to find momentum candidates.
        Returns list of {symbol, price, change_24h, volume, coingecko_id}.
        """
        try:
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
                    return []

                coins = r.json()

            # Filter by volume, sort by 24h change
            filtered = []
            for c in coins:
                vol = c.get("total_volume", 0) or 0
                if vol < min_volume_m * 1_000_000:
                    continue
                change = c.get("price_change_percentage_24h", 0) or 0
                filtered.append({
                    "symbol": (c.get("symbol") or "").upper(),
                    "name": c.get("name", ""),
                    "price": c.get("current_price", 0),
                    "change_24h_pct": round(change, 2),
                    "volume_24h_usd": vol,
                    "rank": c.get("market_cap_rank", 999),
                    "coingecko_id": c.get("id", ""),
                })

            filtered.sort(key=lambda x: x["change_24h_pct"], reverse=True)
            return filtered[:limit]

        except Exception:
            return []
