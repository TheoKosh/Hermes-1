"""
xstocks_adapter.py — Kraken xStocks (tokenized equities) adapter.

Fetches price data for tokenized U.S. stocks and ETFs from Kraken's
Futures/Derivatives API. xStocks are flexible futures instruments that
represent tokenized equities (AAPLx, TSLAx, NVDAx, SPYx, QQQx, etc).

Available xStocks (as of 2026-07-25):
  AAPLx  - Apple
  TSLAx  - Tesla
  NVDAx  - Nvidia
  MSFTx  - Microsoft (futures only, may not be liquid)
  AMZNx  - Amazon
  METAx  - Meta Platforms
  GOOGLx - Alphabet/Google
  SPYx   - S&P 500 ETF
  QQQx   - Nasdaq 100 ETF
  GLDx   - Gold ETF
"""
import httpx
from typing import Optional


# Mapping from xStock symbol to display name
XSTOCK_NAMES = {
    "AAPLx": "Apple",
    "TSLAx": "Tesla",
    "NVDAx": "Nvidia",
    "MSFTx": "Microsoft",
    "AMZNx": "Amazon",
    "METAx": "Meta Platforms",
    "GOOGLx": "Alphabet/Google",
    "SPYx": "S&P 500 ETF",
    "QQQx": "Nasdaq 100 ETF",
    "GLDx": "Gold ETF",
}

# Kraken Futures API base
FUTURES_API = "https://futures.kraken.com/derivatives/api/v3"


class XStocksAdapter:
    """
    Fetches xStocks prices from Kraken Futures API.
    Provides bid/ask/volume for tokenized equities.
    """

    def __init__(self):
        self._instruments = None
        self._tickers = {}

    async def fetch_ticker(self, symbol: str) -> dict:
        """
        Fetch ticker for a single xStock.
        symbol: xStock ticker (e.g. 'AAPLx', 'TSLAx', 'SPYx')
        """
        # Kraken futures use PF_ prefix
        pair = f"PF_{symbol.upper()}USD"

        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{FUTURES_API}/tickers", params={"pair": pair})

        if r.status_code != 200:
            return {"symbol": symbol, "price": 0, "error": f"HTTP {r.status_code}"}

        tickers = r.json().get("tickers", [])
        if not tickers:
            return {"symbol": symbol, "price": 0, "error": "no ticker"}

        t = tickers[0]
        bid = float(t.get("bid", 0) or 0)
        ask = float(t.get("ask", 0) or 0)
        last = float(t.get("lastPrice", 0) or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last

        return {
            "symbol": symbol,
            "price": mid,
            "bid": bid,
            "ask": ask,
            "last": last,
            "volume": float(t.get("volumeQuote", 0) or 0),
            "name": XSTOCK_NAMES.get(symbol, symbol),
        }

    async def fetch_all_tickers(self) -> dict:
        """Fetch all available xStock tickers at once."""
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{FUTURES_API}/tickers")

        if r.status_code != 200:
            return {}

        tickers = r.json().get("tickers", [])
        result = {}

        for t in tickers:
            sym = t.get("symbol", "")
            # Filter for xStock symbols (PF_xxxXUSD pattern)
            if not sym.startswith("PF_"):
                continue

            # Extract the base symbol (e.g. AAPLx from PF_AAPLXUSD)
            base = sym[3:].replace("USD", "")  # Remove PF_ prefix and USD suffix

            # Check if it's a known xStock
            if base in XSTOCK_NAMES or base.upper() in [k.upper() for k in XSTOCK_NAMES]:
                bid = float(t.get("bid", 0) or 0)
                ask = float(t.get("ask", 0) or 0)
                last = float(t.get("lastPrice", 0) or 0)
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                vol = float(t.get("volumeQuote", 0) or 0)

                if mid > 0 and vol > 0:  # Only include liquid ones
                    result[base] = {
                        "price": mid,
                        "bid": bid,
                        "ask": ask,
                        "last": last,
                        "volume": vol,
                        "name": XSTOCK_NAMES.get(base, base),
                    }

        return result

    def get_tradable_symbols(self) -> list:
        """Return list of all known xStock symbols."""
        return list(XSTOCK_NAMES.keys())
