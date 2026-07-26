"""VWAP Trend Trading strategy — crypto-adapted from Zarattini & Aziz (2023),
"Volume Weighted Average Price (VWAP): The Holy Grail for Day Trading Systems"
(SSRN 4631351). See research_db/vwap_paper.json for the source analysis.

Original paper: long when price > session VWAP, short when price < VWAP, on
1-min QQQ/TQQQ candles during RTH. 671% return / Sharpe 2.1 / MDD 9.4% on QQQ.

Crypto adaptations (paper has no market close, crypto is 24/7):
- Anchor VWAP to the UTC daily open (00:00) rather than the 9:30 RTH open.
- Exit on VWAP re-cross close OR a time-based session cap (default 24h).
- Wider stops via the VWAP cross itself; volatility handled by position_pct.
- Signal is emitted only when price is a minimum distance from VWAP to avoid
  chop around the line (the paper's biggest loss source: mean-reverting days).
"""

from datetime import datetime, timezone


def compute_vwap(candles: list) -> float:
    """Session VWAP from OHLCV candles anchored to the current UTC day.

    candles: list of [timestamp_ms, open, high, low, close, volume].
    Returns the volume-weighted average of (H+L+C)/3 for candles that fall on
    the same UTC calendar day as the most recent candle. Falls back to all
    candles if the anchored window has zero volume.
    """
    if not candles:
        return 0.0

    last_ts = candles[-1][0]
    anchor_day = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).date()

    def _accumulate(rows):
        num = den = 0.0
        for c in rows:
            if len(c) < 6 or c[4] is None or c[5] is None:
                continue
            hlc = (float(c[2]) + float(c[3]) + float(c[4])) / 3.0
            vol = float(c[5])
            num += hlc * vol
            den += vol
        return num, den

    same_day = [
        c for c in candles
        if datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc).date() == anchor_day
    ]
    num, den = _accumulate(same_day)
    if den <= 0:
        num, den = _accumulate(candles)
    if den <= 0:
        # No volume at all — degrade to simple close average.
        closes = [float(c[4]) for c in candles if len(c) > 4 and c[4] is not None]
        return sum(closes) / len(closes) if closes else 0.0
    return num / den


def generate_vwap_signal(candles: list, min_distance_pct: float = 1.5) -> dict:
    """VWAP trend signal from OHLCV candles.

    Paper rule: long above VWAP, short below. Crypto guard: require price to be
    at least `min_distance_pct` away from VWAP so we don't churn on every tick
    that hugs the line (the paper's mean-reverting-day loss source).

    NOTE: on crypto 15m data a naive 0.15% band loses money (fees + chop). A
    walk-forward sweep on Kraken (BTC/ETH/SOL/AVAX/LINK, 7d) showed only a wide
    1.5% band with a multi-bar hold is marginally profitable. Default raised to
    1.5% accordingly. See research_db/ssrn-4631351.json crypto_backtest.
    """
    if len(candles) < 2:
        return {"signal": "flat", "price": 0.0, "vwap": 0.0,
                "distance_pct": 0.0, "confidence": 0.0, "reason": "insufficient candles"}

    vwap = compute_vwap(candles)
    price = float(candles[-1][4])
    if vwap <= 0 or price <= 0:
        return {"signal": "flat", "price": price, "vwap": vwap,
                "distance_pct": 0.0, "confidence": 0.0, "reason": "no valid vwap/price"}

    distance_pct = (price - vwap) / vwap * 100.0
    abs_dist = abs(distance_pct)

    if abs_dist < min_distance_pct:
        return {"signal": "flat", "price": price, "vwap": vwap,
                "distance_pct": round(distance_pct, 3), "confidence": 0.0,
                "reason": f"price within {min_distance_pct}% of VWAP (chop zone)"}

    # Confidence scales with distance, capped at 1.0 (saturates at ~2% away).
    confidence = min(1.0, abs_dist / 2.0)
    signal = "long" if distance_pct > 0 else "short"
    return {"signal": signal, "price": price, "vwap": round(vwap, 6),
            "distance_pct": round(distance_pct, 3), "confidence": round(confidence, 3),
            "reason": f"price {'above' if signal == 'long' else 'below'} VWAP by {abs_dist:.2f}%"}


def check_vwap_exit(candles: list, position_side: str) -> bool:
    """Paper exit rule: close the position when a candle closes back through VWAP.

    Returns True if the current (last-closed) candle has crossed VWAP against
    the open position, signalling an exit.
    """
    if len(candles) < 2 or position_side not in ("long", "short"):
        return False
    vwap = compute_vwap(candles)
    price = float(candles[-1][4])
    if vwap <= 0:
        return False
    if position_side == "long":
        return price < vwap
    return price > vwap
