"""
macro.py — Macro data adapter.

Fetches macro indicators (DXY, S&P 500, VIX) via yfinance.
No API key required — Yahoo Finance is free.

Schema:
{
    "schema_version": 1,
    "dxy": 104.2,           # US Dollar Index
    "sp500": 5200.5,        # S&P 500
    "vix": 14.5,            # Volatility Index
    "timestamp": "2024-..."
}
"""
from datetime import datetime, timezone

import yfinance as yf


SCHEMA_VERSION = 1


class SchemaError(Exception):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MacroAdapter:
    def __init__(self):
        self.asset = "MACRO"
        self.name = "macro"

    async def fetch(self) -> dict:
        """Fetch macro data via yfinance."""
        try:
            # DXY (US Dollar Index)
            dxy = yf.Ticker("DX-Y.NYB")
            dxy_hist = dxy.history(period="2d")
            dxy_price = float(dxy_hist["Close"].iloc[-1]) if not dxy_hist.empty else 0.0

            # S&P 500
            sp = yf.Ticker("^GSPC")
            sp_hist = sp.history(period="2d")
            sp_price = float(sp_hist["Close"].iloc[-1]) if not sp_hist.empty else 0.0

            # VIX
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="2d")
            vix_price = float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else 0.0

        except Exception as e:
            # macro is non-critical — return zeros on failure
            dxy_price = 0.0
            sp_price = 0.0
            vix_price = 0.0

        result = {
            "schema_version": SCHEMA_VERSION,
            "dxy": round(dxy_price, 2),
            "sp500": round(sp_price, 2),
            "vix": round(vix_price, 2),
            "timestamp": now_iso(),
        }

        self._validate(result)
        return result

    def _validate(self, data: dict):
        required = {"schema_version", "dxy", "sp500", "vix", "timestamp"}
        if not required.issubset(data.keys()):
            raise SchemaError(f"macro adapter: missing keys {required - set(data.keys())}")
        if data["schema_version"] != SCHEMA_VERSION:
            raise SchemaError(f"macro adapter: schema mismatch")
