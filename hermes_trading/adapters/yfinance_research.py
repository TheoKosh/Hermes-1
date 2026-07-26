"""
yfinance_research.py — Comprehensive Yahoo Finance research database.

Provides multi-asset correlation intelligence for the trading bot:
  - Equity indices (SPY, QQQ, DIA, IWM)
  - Commodities (Gold, Oil, Silver)
  - Bonds/rates (TLT, HYG, 10Y yield)
  - Currencies (DXY)
  - Crypto (BTC-USD, ETH-USD)
  - Volatility (VIX)

Outputs a macro risk score that the signal engine uses to adjust
crypto trade entries: when traditional markets are selling off hard
(VIX spike, equity dump), the bot should be more cautious on longs.
When risk-on (VIX low, equities green), crypto longs get a boost.
"""
import asyncio
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf


# Asset universe for the research database
RESEARCH_UNIVERSE = {
    # Equity indices (ETFs, more reliable than ^GSPC on weekends)
    "SPY":   {"name": "S&P 500 ETF",       "category": "equity",   "weight": 0.20},
    "QQQ":   {"name": "Nasdaq 100 ETF",    "category": "equity",   "weight": 0.15},
    "DIA":   {"name": "Dow Jones ETF",     "category": "equity",   "weight": 0.05},
    "IWM":   {"name": "Russell 2000 ETF",  "category": "equity",   "weight": 0.05},

    # Commodities
    "GC=F":  {"name": "Gold Futures",      "category": "commodity", "weight": 0.10},
    "CL=F":  {"name": "Crude Oil Futures", "category": "commodity", "weight": 0.05},
    "SI=F":  {"name": "Silver Futures",    "category": "commodity", "weight": 0.05},

    # Rates & credit
    "TLT":   {"name": "20+ Year Treasury",  "category": "rates",    "weight": 0.05},
    "HYG":   {"name": "High Yield Bonds",  "category": "credit",   "weight": 0.05},
    "^TNX":  {"name": "10Y Treasury Yield", "category": "rates",    "weight": 0.05},

    # Currencies
    "DX-Y.NYB": {"name": "US Dollar Index", "category": "currency", "weight": 0.10},

    # Volatility
    "^VIX":  {"name": "Volatility Index",  "category": "volatility", "weight": 0.05},

    # Crypto benchmarks
    "BTC-USD": {"name": "Bitcoin",         "category": "crypto",    "weight": 0.05},
    "ETH-USD": {"name": "Ethereum",        "category": "crypto",    "weight": 0.05},
}


class YFinanceResearch:
    """
    Multi-asset Yahoo Finance research database.
    Fetches prices, computes cross-asset correlations, and derives
    a macro risk sentiment score for crypto trading signals.
    """

    def __init__(self):
        self.name = "yfinance_research"
        self._cache = {}
        self._last_update = None
        self._lookback_days = 30  # for correlation computation

    async def fetch_all(self) -> dict:
        """
        Fetch current prices + recent history for the entire research universe.
        Returns dict keyed by symbol with price, change, volume, and category.
        """
        results = {}

        for symbol, meta in RESEARCH_UNIVERSE.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{self._lookback_days}d")

                if hist.empty:
                    continue

                closes = hist["Close"].dropna()
                volumes = hist["Volume"].dropna() if "Volume" in hist else None

                if len(closes) < 2:
                    continue

                current = closes.iloc[-1]
                prev = closes.iloc[-2]

                # daily change
                change_1d = (current - prev) / prev * 100 if prev > 0 else 0.0

                # weekly change (5 trading days ago)
                if len(closes) >= 6:
                    week_ago = closes.iloc[-6]
                    change_1w = (current - week_ago) / week_ago * 100 if week_ago > 0 else 0.0
                else:
                    change_1w = change_1d

                # monthly change
                if len(closes) >= 2:
                    month_ago = closes.iloc[0]
                    change_1m = (current - month_ago) / month_ago * 100 if month_ago > 0 else 0.0
                else:
                    change_1m = change_1d

                # average volume
                avg_vol = float(volumes.tail(10).mean()) if volumes is not None and not volumes.empty else 0

                results[symbol] = {
                    "name": meta["name"],
                    "category": meta["category"],
                    "weight": meta["weight"],
                    "price": round(current, 4),
                    "change_1d_pct": round(change_1d, 2),
                    "change_1w_pct": round(change_1w, 2),
                    "change_1m_pct": round(change_1m, 2),
                    "avg_volume": avg_vol,
                    "history": closes.tolist(),
                }

            except Exception:
                continue

        self._cache = results
        self._last_update = datetime.now(timezone.utc).isoformat()
        return results

    def compute_risk_sentiment(self) -> dict:
        """
        Compute a macro risk sentiment score from the research universe.

        Returns:
        {
            "risk_score": float,    # -1.0 (risk-off) to +1.0 (risk-on)
            "confidence": float,    # 0 to 1
            "vix_level": float,
            "dxy_trend": str,       # "up", "down", "flat"
            "equity_trend": str,
            "commodity_trend": str,
            "correlation_note": str,
            "signals": list,        # human-readable signal descriptions
        }
        """
        if not self._cache:
            return {"risk_score": 0.0, "confidence": 0.0, "signals": ["no data"]}

        signals = []
        risk_components = []

        # --- VIX: the primary fear gauge ---
        vix = self._cache.get("^VIX", {})
        vix_price = vix.get("price", 15)
        vix_change = vix.get("change_1d_pct", 0)

        if vix_price > 30:
            vix_signal = -1.0
            signals.append(f"VIX={vix_price:.0f} (extreme fear → risk-off)")
        elif vix_price > 25:
            vix_signal = -0.5
            signals.append(f"VIX={vix_price:.0f} (elevated → cautious)")
        elif vix_price > 20:
            vix_signal = -0.2
            signals.append(f"VIX={vix_price:.0f} (slightly elevated)")
        elif vix_price < 12:
            vix_signal = 0.5
            signals.append(f"VIX={vix_price:.0f} (complacent → risk-on)")
        else:
            vix_signal = 0.2
            signals.append(f"VIX={vix_price:.0f} (normal)")

        # VIX spike adds urgency
        if vix_change > 10:
            vix_signal = max(-1.0, vix_signal - 0.3)
            signals.append(f"VIX spike +{vix_change:.0f}% → risk-off urgent")

        risk_components.append(("vix", vix_signal, 0.30))

        # --- DXY: strong dollar = crypto headwind ---
        dxy = self._cache.get("DX-Y.NYB", {})
        dxy_change = dxy.get("change_1d_pct", 0)
        dxy_trend = "up" if dxy_change > 0.2 else ("down" if dxy_change < -0.2 else "flat")

        if dxy_change > 0.5:
            dxy_signal = -0.5
            signals.append(f"DXY +{dxy_change:.2f}% (dollar strengthening → crypto headwind)")
        elif dxy_change < -0.5:
            dxy_signal = 0.5
            signals.append(f"DXY {dxy_change:.2f}% (dollar weakening → crypto tailwind)")
        else:
            dxy_signal = 0.0

        risk_components.append(("dxy", dxy_signal, 0.15))

        # --- Equity indices: risk-on/off proxy ---
        spy = self._cache.get("SPY", {})
        qqq = self._cache.get("QQQ", {})
        spy_change = spy.get("change_1d_pct", 0)
        qqq_change = qqq.get("change_1d_pct", 0)
        equity_avg = (spy_change + qqq_change) / 2

        if equity_avg > 1.5:
            equity_signal = 0.7
            signals.append(f"Equities strong (SPY {spy_change:+.1f}% QQQ {qqq_change:+.1f}%) → risk-on")
        elif equity_avg > 0.5:
            equity_signal = 0.3
        elif equity_avg < -1.5:
            equity_signal = -0.7
            signals.append(f"Equities selling off (SPY {spy_change:+.1f}% QQQ {qqq_change:+.1f}%) → risk-off")
        elif equity_avg < -0.5:
            equity_signal = -0.3
        else:
            equity_signal = 0.0

        equity_trend = "up" if equity_avg > 0.3 else ("down" if equity_avg < -0.3 else "flat")
        risk_components.append(("equity", equity_signal, 0.25))

        # --- Gold: safe haven flow ---
        gold = self._cache.get("GC=F", {})
        gold_change = gold.get("change_1d_pct", 0)
        gold_spy_divergence = gold_change - spy_change

        if gold_change > 1.0 and spy_change < 0:
            gold_signal = -0.3  # gold up + equities down = fear
            signals.append(f"Gold up {gold_change:+.1f}% while equities down → flight to safety")
        elif gold_change > 1.0:
            gold_signal = 0.1
        else:
            gold_signal = 0.0

        risk_components.append(("gold", gold_signal, 0.10))

        # --- Oil: economic activity proxy ---
        oil = self._cache.get("CL=F", {})
        oil_change = oil.get("change_1d_pct", 0)
        if oil_change < -3:
            oil_signal = -0.3
            signals.append(f"Oil -{abs(oil_change):.1f}% → demand concern")
        elif oil_change > 3:
            oil_signal = 0.2
        else:
            oil_signal = 0.0

        risk_components.append(("oil", oil_signal, 0.10))

        # --- Treasury yields ---
        tnx = self._cache.get("^TNX", {})
        tnx_change = tnx.get("change_1d_pct", 0)
        if tnx_change > 3:
            signals.append(f"10Y yield +{tnx_change:.1f}% → tightening pressure")

        # --- Composite risk score ---
        total_weight = sum(w for _, _, w in risk_components)
        risk_score = sum(s * w for _, s, w in risk_components) / total_weight if total_weight > 0 else 0
        risk_score = max(-1.0, min(1.0, risk_score))

        # Confidence: how many data points we have
        confidence = min(1.0, len([x for x in self._cache.values() if x.get("price", 0) > 0]) / len(RESEARCH_UNIVERSE))

        # Correlation note
        if spy_change < -1 and vix_change > 10:
            corr_note = "High stress: equities dumping + VIX spiking → crypto likely correlated selloff"
        elif spy_change > 1 and vix_price < 15:
            corr_note = "Risk-on regime: low vol + equities up → crypto tailwind"
        else:
            corr_note = "Mixed signals: no strong cross-asset direction"

        commodity_trend = "up" if gold_change > 0.5 and oil_change > 0.5 else \
                          ("down" if gold_change < -0.5 and oil_change < -0.5 else "flat")

        return {
            "risk_score": round(risk_score, 3),
            "confidence": round(confidence, 3),
            "vix_level": vix_price,
            "dxy_trend": dxy_trend,
            "equity_trend": equity_trend,
            "commodity_trend": commodity_trend,
            "correlation_note": corr_note,
            "signals": signals,
        }

    def get_snapshot(self) -> dict:
        """Return current cached data as a flat snapshot for logging/display."""
        snap = {}
        for sym, data in self._cache.items():
            snap[sym] = {
                "price": data.get("price", 0),
                "change_1d": data.get("change_1d_pct", 0),
                "category": data.get("category", ""),
            }
        return snap
