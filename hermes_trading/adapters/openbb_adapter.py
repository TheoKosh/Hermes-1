"""openbb_adapter.py — Lightweight macro intelligence overlay.

MIRRORS the data exposed by OpenBB Platform providers (SSRN-free, zero deps):
  - FRED provider  -> macro rates / yield curve (FRED public CSV endpoint)
  - Deribit provider -> crypto perp funding + open interest (Deribit public API)

OpenBB (github.com/OpenBB-finance/OpenBB) is a 182MB FastAPI platform with
30+ provider packages. Installing it into a lean crypto trading worker would
risk destabilizing its dependency tree. This adapter fetches the SAME public
endpoints OpenBB's providers wrap, via stdlib urllib — no new dependencies —
and exposes a composite `macro_risk_score` for strategy consolidation.

Intended use: called from the strategy consolidation / signal blending layer
to gate trade aggressiveness by macro regime (risk-on / risk-off).
"""

import json
import math
import urllib.request
from datetime import datetime, timezone

_CACHE: dict = {"ts": 0.0, "data": {}}
_CACHE_TTL = 600  # seconds — macro data changes slowly


def _get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


# ----------------------------------------------------------------------------
# FRED provider mirror — macro rates / yield curve
# ----------------------------------------------------------------------------

def fred_series(series_id: str, cosd: str = "2025-01-01") -> tuple[str, float]:
    """Fetch the latest value of a FRED series via the public CSV endpoint.
    No API key required. Returns (date_string, value)."""
    try:
        csv = _get(
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={cosd}"
        )
        lines = [l for l in csv.strip().split("\n") if l]
        if len(lines) < 2:
            return ("", 0.0)
        last = lines[-1].split(",")
        val = float(last[1]) if len(last) > 1 and last[1] else 0.0
        return (last[0], val)
    except Exception:
        return ("", 0.0)


def fetch_macro_rates() -> dict:
    """Pull key FRED series that OpenBB's fred provider exposes:
    FEDFUNDS (policy rate), DGS10/DGS2 (yield curve), T10YIE (breakeven
    inflation), VIXCLS (volatility)."""
    out = {}
    for sid, key in [
        ("FEDFUNDS", "fed_funds"),
        ("DGS10", "ust_10y"),
        ("DGS2", "ust_2y"),
        ("T10YIE", "breakeven_infl"),
        ("VIXCLS", "vix"),
        ("DXY", "dxy"),  # may be empty on FRED; handled gracefully
    ]:
        date, val = fred_series(sid)
        if val != 0.0:
            out[key] = {"value": val, "date": date}
    # derived: yield curve slope
    if "ust_10y" in out and "ust_2y" in out:
        out["yield_curve_slope"] = out["ust_10y"]["value"] - out["ust_2y"]["value"]
    return out


# ----------------------------------------------------------------------------
# Deribit provider mirror — crypto perp funding / OI
# ----------------------------------------------------------------------------

def deribit_ticker(instrument: str = "BTC-PERPETUAL") -> dict:
    """Fetch Deribit perpetual ticker (funding, OI, mark). Public, no key."""
    try:
        data = json.loads(
            _get(f"https://www.deribit.com/api/v2/public/ticker?instrument_name={instrument}")
        )
        r = data.get("result", {})
        return {
            "last": r.get("last_price", 0),
            "mark": r.get("mark_price", 0),
            "funding_8h": r.get("current_funding", 0),
            "open_interest_usd": r.get("open_interest", 0),
            "instrument": instrument,
        }
    except Exception:
        return {}


def fetch_perps_intelligence() -> dict:
    """Funding + OI for BTC and ETH perps. Extreme positive funding => long
    crowding (risk of long squeeze); negative => short crowding."""
    out = {}
    for sym in ["BTC-PERPETUAL", "ETH-PERPETUAL"]:
        t = deribit_ticker(sym)
        if t.get("last"):
            out[sym.split("-")[0]] = t
    return out


# ----------------------------------------------------------------------------
# Composite macro risk score (for strategy consolidation)
# ----------------------------------------------------------------------------

def macro_risk_score(macro: dict | None = None, perps: dict | None = None) -> dict:
    """Composite score in [-1, +1].
       +1 = max risk-on (low rates, flat/inverted curve, low VIX, neutral funding)
       -1 = max risk-off (spiking VIX, inverted/steep stress, extreme funding)

    Returns {score, regime, factors}."""
    macro = macro if macro is not None else fetch_macro_rates()
    perps = perps if perps is not None else fetch_perps_intelligence()

    score = 0.0
    factors = {}

    vix = macro.get("vix", {}).get("value", 0)
    if vix:
        # VIX < 15 = risk-on, > 30 = risk-off
        vix_contrib = max(-0.4, min(0.4, (20 - vix) / 25))
        score += vix_contrib
        factors["vix"] = round(vix_contrib, 3)

    slope = macro.get("yield_curve_slope")
    if slope is not None:
        # inverted curve (negative) = recession risk = risk-off
        slope_contrib = max(-0.3, min(0.3, slope / 2.0))
        score += slope_contrib
        factors["yield_slope"] = round(slope_contrib, 3)

    # funding crowding from perps
    funding_sum = 0.0
    n = 0
    for sym, t in perps.items():
        f8 = t.get("funding_8h", 0)
        if f8:
            funding_sum += f8
            n += 1
    if n:
        avg_funding = funding_sum / n
        # annualized: funding_8h * 3 * 365
        ann = avg_funding * 3 * 365
        # extreme positive (>50% ann) = long squeeze risk; extreme negative = short squeeze
        fund_contrib = max(-0.3, min(0.3, -ann / 100))
        score += fund_contrib
        factors["funding_ann"] = round(ann, 1)
        factors["funding_contrib"] = round(fund_contrib, 3)

    score = max(-1.0, min(1.0, score))
    if score > 0.2:
        regime = "risk-on"
    elif score < -0.2:
        regime = "risk-off"
    else:
        regime = "neutral"

    return {"score": round(score, 3), "regime": regime, "factors": factors}


# ----------------------------------------------------------------------------
# Cached fetch (TTL) — for use inside the trading loop
# ----------------------------------------------------------------------------

def fetch_all_cached() -> dict:
    """Single entry point for the loop: returns cached macro + perps + score,
    refreshing at most every CACHE_TTL seconds."""
    now = datetime.now(timezone.utc).timestamp()
    if now - _CACHE["ts"] > _CACHE_TTL:
        try:
            macro = fetch_macro_rates()
            perps = fetch_perps_intelligence()
            _CACHE["data"] = {
                "macro": macro,
                "perps": perps,
                "risk": macro_risk_score(macro, perps),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            _CACHE["ts"] = now
        except Exception as e:
            _CACHE["data"] = {"error": str(e)}
    return _CACHE["data"]
