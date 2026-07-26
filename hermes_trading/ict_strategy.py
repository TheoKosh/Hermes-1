"""
ict_strategy.py — ICT (Inner Circle Trader) strategy adapted for crypto.

Based on the ICT mentorship YouTube playlist (PLOqlWG05MfW6T7mUjI2CK5BdQ6FrV3zEm):
  - Liquidity: old highs/lows as stop pools
  - Market Structure Shift (MSS): reversal after liquidity sweep
  - Fair Value Gap (FVG): price imbalance that gets filled
  - Premium/Discount: Fibonacci equilibrium for entry zones
  - 15m timeframe (matches ICT's teaching)

Adapted for crypto on Kraken with:
  - 3% max daily loss (prop account rule)
  - ATR-based stops
  - Volume confirmation (crypto-specific)
  - Cross-exchange price verification (Coinbase + Kraken)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ICTSignal:
    """ICT strategy signal output."""
    signal: str = "flat"          # "long", "short", "flat"
    composite: float = 0.0       # -1 to +1
    confidence: float = 0.0      # 0 to 1
    liquidity_swept: bool = False
    mss_detected: bool = False
    fvg_present: bool = False
    zone: str = "equilibrium"     # "premium", "discount", "equilibrium"
    reasoning: str = ""


class ICTStrategy:
    """
    ICT-based strategy for crypto trading.

    Detects:
    1. Liquidity pools at recent swing highs/lows
    2. Liquidity sweeps (price wicks beyond a pool then reverses)
    3. Market Structure Shift (break of recent high/low in opposite direction)
    4. Fair Value Gaps (3-bar imbalance)
    5. Premium/Discount zones via Fibonacci equilibrium
    """

    LOOKBACK_SWINGS = 20    # bars to look back for swing highs/lows
    LOOKBACK_FVG = 5        # bars to check for FVG
    MIN_SWING_SEPARATION = 3  # minimum bars between swings

    def __init__(self):
        self._swing_highs = {}
        self._swing_lows = {}

    def find_swings(self, highs: list, lows: list, closes: list,
                    lookback: int = 20) -> dict:
        """
        Find recent swing highs and lows.
        A swing high = local maximum in a window.
        A swing low = local minimum in a window.
        """
        n = len(closes)
        if n < 5:
            return {"highs": [], "lows": []}

        swing_highs = []
        swing_lows = []
        window = 3  # bars on each side

        for i in range(window, min(n, lookback + window) - window):
            # Swing high: highest point in window
            is_high = all(highs[i] >= highs[j] for j in range(i - window, i + window + 1) if j != i)
            if is_high:
                swing_highs.append({"index": i, "price": highs[i]})

            # Swing low: lowest point in window
            is_low = all(lows[i] <= lows[j] for j in range(i - window, i + window + 1) if j != i)
            if is_low:
                swing_lows.append({"index": i, "price": lows[i]})

        return {"highs": swing_highs, "lows": swing_lows}

    def detect_liquidity_sweep(self, highs: list, lows: list, closes: list,
                              swings: dict) -> dict:
        """
        Detect if the most recent bars swept a liquidity pool.
        A sweep = price goes beyond a swing high/low but closes back inside.
        """
        if len(closes) < 2 or not swings["highs"] or not swings["lows"]:
            return {"swept": False, "direction": None, "level": None}

        recent_high = highs[-1]
        recent_low = lows[-1]
        close = closes[-1]

        # Check for sweep of a swing high (price went above, then closed below)
        nearest_swing_high = max(s["price"] for s in swings["highs"])
        if recent_high > nearest_swing_high and close < nearest_swing_high:
            return {"swept": True, "direction": "high", "level": nearest_swing_high}

        # Check for sweep of a swing low (price went below, then closed above)
        nearest_swing_low = min(s["price"] for s in swings["lows"])
        if recent_low < nearest_swing_low and close > nearest_swing_low:
            return {"swept": True, "direction": "low", "level": nearest_swing_low}

        return {"swept": False, "direction": None, "level": None}

    def detect_mss(self, highs: list, lows: list, closes: list,
                   swings: dict, sweep: dict) -> bool:
        """
        Detect Market Structure Shift after a liquidity sweep.
        MSS = price breaks in the opposite direction of the sweep.
        """
        if not sweep["swept"] or len(closes) < 3:
            return False

        # If high was swept (bearish), MSS = break below recent swing low
        if sweep["direction"] == "high":
            if swings["lows"]:
                recent_low = min(s["price"] for s in swings["lows"])
                return closes[-1] < recent_low

        # If low was swept (bullish), MSS = break above recent swing high
        if sweep["direction"] == "low":
            if swings["highs"]:
                recent_high = max(s["price"] for s in swings["highs"])
                return closes[-1] > recent_high

        return False

    def detect_fvg(self, highs: list, lows: list) -> Optional[dict]:
        """
        Detect Fair Value Gap (3-bar pattern).
        Bullish FVG: bar[i-2].high < bar[i].low (gap up)
        Bearish FVG: bar[i-2].low > bar[i].high (gap down)
        """
        if len(highs) < 3 or len(lows) < 3:
            return None

        # Bullish FVG (last 3 bars)
        if lows[-1] > highs[-3]:
            return {"type": "bullish", "top": lows[-1], "bottom": highs[-3]}

        # Bearish FVG (last 3 bars)
        if highs[-1] < lows[-3]:
            return {"type": "bearish", "top": lows[-3], "bottom": highs[-1]}

        return None

    def compute_premium_discount(self, highs: list, lows: list) -> dict:
        """
        Compute Fibonacci premium/discount zones.
        Uses recent swing high to swing low.
        """
        if not highs or not lows:
            return {"zone": "equilibrium", "fib_50": 0}

        swing_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
        swing_low = min(lows[-20:]) if len(lows) >= 20 else min(lows)

        fib_50 = (swing_high + swing_low) / 2  # equilibrium
        current = (highs[-1] + lows[-1]) / 2  # current mid-price

        if current > fib_50 * 1.001:
            return {"zone": "premium", "fib_50": fib_50, "swing_high": swing_high, "swing_low": swing_low}
        elif current < fib_50 * 0.999:
            return {"zone": "discount", "fib_50": fib_50, "swing_high": swing_high, "swing_low": swing_low}
        else:
            return {"zone": "equilibrium", "fib_50": fib_50, "swing_high": swing_high, "swing_low": swing_low}

    def generate_signal(self, highs: list, lows: list, closes: list,
                        volumes: list = None) -> ICTSignal:
        """
        Generate ICT strategy signal.
        Entry conditions (all must align):
        1. Liquidity sweep at recent swing high/low
        2. Market Structure Shift confirms reversal
        3. Fair Value Gap present (imbalance to fill)
        4. Price in discount (for longs) or premium (for shorts)
        """
        sig = ICTSignal()

        if len(closes) < 20:
            sig.reasoning = "insufficient history"
            return sig

        # 1. Find swing highs/lows
        swings = self.find_swings(highs, lows, closes, self.LOOKBACK_SWINGS)
        if not swings["highs"] or not swings["lows"]:
            sig.reasoning = "no swings detected"
            return sig

        # 2. Detect liquidity sweep
        sweep = self.detect_liquidity_sweep(highs, lows, closes, swings)
        sig.liquidity_swept = sweep["swept"]

        # 3. Detect MSS
        mss = self.detect_mss(highs, lows, closes, swings, sweep)
        sig.mss_detected = mss

        # 4. Detect FVG
        fvg = self.detect_fvg(highs, lows)
        sig.fvg_present = fvg is not None

        # 5. Premium/Discount
        pd = self.compute_premium_discount(highs, lows)
        sig.zone = pd["zone"]

        # 6. Generate signal
        # Strong signal: sweep + MSS + FVG + correct zone
        # Medium signal: sweep + MSS
        # Weak signal: sweep only
        score = 0.0
        confidence = 0.0
        reasons = []

        if sweep["swept"]:
            score += 0.2
            confidence += 0.3
            reasons.append(f"liquidity sweep ({sweep['direction']})")

        if mss:
            score += 0.3
            confidence += 0.3
            reasons.append("MSS confirmed")

        if fvg:
            score += 0.15
            confidence += 0.2
            reasons.append(f"FVG ({fvg['type']})")

        # Zone alignment
        if sweep["direction"] == "high" and pd["zone"] == "premium":
            # Swept high in premium = short
            score -= 0.15
            reasons.append("premium zone (short bias)")
        elif sweep["direction"] == "low" and pd["zone"] == "discount":
            # Swept low in discount = long
            score += 0.15
            reasons.append("discount zone (long bias)")

        # Volume confirmation (if available)
        if volumes and len(volumes) >= 5:
            avg_vol = sum(volumes[-20:]) / max(1, len(volumes[-20:]))
            recent_vol = sum(volumes[-3:]) / 3
            if recent_vol > avg_vol * 1.5:
                confidence += 0.1
                reasons.append("volume confirmed")

        # Direction: if high was swept + MSS, short. If low swept + MSS, long.
        if sweep["swept"] and mss:
            if sweep["direction"] == "high":
                sig.signal = "short"
                score = -abs(score)  # flip to negative for short
            elif sweep["direction"] == "low":
                sig.signal = "long"
                score = abs(score)
        elif sweep["swept"] and fvg:
            # FVG gives direction even without MSS
            if fvg["type"] == "bullish" and sweep["direction"] == "low":
                sig.signal = "long"
            elif fvg["type"] == "bearish" and sweep["direction"] == "high":
                sig.signal = "short"

        sig.composite = max(-1.0, min(1.0, score))
        sig.confidence = min(1.0, confidence)
        sig.reasoning = " | ".join(reasons) if reasons else "no signal"

        return sig
