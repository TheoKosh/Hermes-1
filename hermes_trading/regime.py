"""
regime.py — Market regime classifier.

Classifies each asset into a regime BEFORE any signal is generated,
because strategies that work in one regime lose money in another.

Three orthogonal regime dimensions:
  1. Trend vs. range (price vs. moving average, ADX proxy)
  2. Volatility state (rolling realized vol percentile)
  3. Liquidity/volume context (volume vs. historical average)

Routes the asset to the appropriate sub-strategy:
  - Trending + volume-confirmed → momentum continuation
  - Range-bound + low-volume → mean-reversion
  - High-volatility → raise confidence bar, cut position size
"""
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Regime:
    """Detected market regime for a single asset."""
    trend: str = "range"           # "trend_up", "trend_down", "range"
    volatility: str = "normal"     # "calm", "normal", "elevated", "extreme"
    volume_context: str = "normal" # "thin", "normal", "confirmed"
    sub_strategy: str = "mean_reversion"  # "momentum", "mean_reversion", "skip"
    vol_percentile: float = 50.0   # 0-100, where current vol sits historically
    confidence_multiplier: float = 1.0  # multiplier on signal confidence (0.5-1.0)
    size_multiplier: float = 1.0   # multiplier on position size (0.25-1.0)
    reasoning: str = ""


class RegimeClassifier:
    """
    Classifies market regime from OHLCV history.
    Requires at least 50 candles of history for stable estimates.
    """

    MIN_HISTORY = 50
    LOOKBACK_VOL = 20      # 20-bar rolling realized vol
    LOOKBACK_TREND = 50    # 50-bar for trend direction
    LOOKBACK_VOLUME = 20   # 20-bar average volume

    def classify(
        self,
        closes: list,
        volumes: list = None,
        current_rsi: float = 50.0,
    ) -> Regime:
        """
        Classify regime from price + volume history.

        Args:
            closes: list of close prices (oldest first, most recent last)
            volumes: list of volumes (same length as closes, optional)
            current_rsi: current RSI value (used as auxiliary trend signal)

        Returns:
            Regime dataclass with all fields populated.
        """
        r = Regime()

        if len(closes) < self.MIN_HISTORY:
            r.reasoning = f"insufficient history ({len(closes)}/{self.MIN_HISTORY})"
            r.sub_strategy = "skip"
            return r

        # --- 1. TREND vs RANGE ---
        closes_arr = closes[-self.LOOKBACK_TREND:]
        current = closes_arr[-1]

        # Simple moving average (50-bar or available)
        sma = sum(closes_arr) / len(closes_arr)

        # ADX proxy: ratio of |price - SMA| to average true range
        # Higher = stronger trend, lower = more range-bound
        sma_deviation_pct = abs(current - sma) / sma * 100

        # Recent slope: linear regression of last 20 closes
        recent = closes[-self.LOOKBACK_VOL:]
        n = len(recent)
        if n >= 5:
            x_mean = (n - 1) / 2
            y_mean = sum(recent) / n
            numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator > 0 else 0
            slope_pct = (slope / current) * 100 if current > 0 else 0
        else:
            slope_pct = 0.0

        # Trend classification
        # sma_deviation > 3% AND |slope| > 0.05%/bar = trending
        if sma_deviation_pct > 3.0 and abs(slope_pct) > 0.02:
            if slope_pct > 0:
                r.trend = "trend_up"
            else:
                r.trend = "trend_down"
        else:
            r.trend = "range"

        # --- 2. VOLATILITY STATE ---
        # Rolling realized volatility: stdev of bar-to-bar returns
        returns = []
        for i in range(1, len(closes[-self.LOOKBACK_VOL:])):
            prev = closes[-self.LOOKBACK_VOL + i - 1]
            curr = closes[-self.LOOKBACK_VOL + i]
            if prev > 0:
                returns.append((curr - prev) / prev)

        if len(returns) >= 5:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            realized_vol = math.sqrt(variance) * 100  # as percentage
        else:
            realized_vol = 2.0  # default moderate

        # Classify volatility using rough crypto percentiles
        # Crypto ATR is typically 1-5% per bar; we use absolute thresholds
        if realized_vol < 1.0:
            r.volatility = "calm"
            r.vol_percentile = 15.0
        elif realized_vol < 2.5:
            r.volatility = "normal"
            r.vol_percentile = 40.0
        elif realized_vol < 5.0:
            r.volatility = "elevated"
            r.vol_percentile = 75.0
            r.confidence_multiplier = 0.7  # raise bar in high vol
            r.size_multiplier = 0.5       # cut size in high vol
        else:
            r.volatility = "extreme"
            r.vol_percentile = 95.0
            r.confidence_multiplier = 0.4
            r.size_multiplier = 0.25
            r.sub_strategy = "skip"  # don't trade in extreme vol

        # --- 3. VOLUME CONTEXT ---
        if volumes and len(volumes) >= self.LOOKBACK_VOLUME:
            recent_vol = volumes[-5:] if len(volumes) >= 5 else volumes
            avg_vol = sum(volumes[-self.LOOKBACK_VOLUME:]) / self.LOOKBACK_VOLUME
            current_vol = sum(recent_vol) / len(recent_vol) if recent_vol else 0

            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

            if vol_ratio > 1.5:
                r.volume_context = "confirmed"
            elif vol_ratio < 0.5:
                r.volume_context = "thin"
            else:
                r.volume_context = "normal"
        else:
            r.volume_context = "normal"

        # --- ROUTE TO SUB-STRATEGY ---
        if r.sub_strategy != "skip":
            if r.trend in ("trend_up", "trend_down") and r.volume_context == "confirmed":
                r.sub_strategy = "momentum"
            elif r.trend == "range":
                r.sub_strategy = "mean_reversion"
            elif r.trend in ("trend_up", "trend_down") and r.volume_context == "thin":
                # Thin volume trend = likely false breakout
                r.sub_strategy = "mean_reversion"
                r.confidence_multiplier *= 0.6
            else:
                r.sub_strategy = "mean_reversion"

        # --- BUILD REASONING STRING ---
        parts = []
        parts.append(f"trend={r.trend}")
        parts.append(f"vol={r.volatility}({r.vol_percentile:.0f}pct)")
        parts.append(f"volume={r.volume_context}")
        parts.append(f"strategy={r.sub_strategy}")
        if r.confidence_multiplier < 1.0:
            parts.append(f"conf_mult={r.confidence_multiplier:.1f}")
        if r.size_multiplier < 1.0:
            parts.append(f"size_mult={r.size_multiplier:.1f}")
        r.reasoning = " | ".join(parts)

        return r
