"""
signals.py — Multi-factor signal fusion engine.

Combines all available data into a single composite score [-1, +1]:
  - RSI (momentum) — mean-reversion or breakout
  - News sentiment (qualitative) — crowd positioning + news flow
  - Fear & Greed (contrarian) — extreme fear = long, extreme greed = short
  - On-chain health (fundamental) — network usage
  - Macro (correlation) — DXY/VIX headwinds/tailwinds for crypto
  - Trending/buzz (attention) — high buzz amplifies directional signals

Each factor outputs a score in [-1, +1] and a confidence [0, 1].
Composite = weighted average, where weights come from strategy.yaml.

OUTPUT:
  {
    "composite_score": 0.42,    # -1 (strong sell) to +1 (strong buy)
    "signal": "long",           # "long" | "short" | "flat"
    "confidence": 0.73,         # 0-1
    "factors": { ... per-factor breakdown ... },
    "reasoning": "RSI oversold (0.8) + sentiment bullish (0.3) → long"
  }
"""


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def rsi_score(rsi: float, direction: str = "both") -> tuple:
    """
    RSI factor: mean-reversion model.
    Returns (score, confidence) each in [-1,+1] and [0,1].

    Long: RSI < 35 → bullish (oversold), RSI > 70 → bearish
    Short: RSI > 65 → bearish (overbought), RSI < 30 → bullish
    Neutral zone: 40-60
    """
    if rsi is None:
        return 0.0, 0.0

    # Long-oriented score: lower RSI = more bullish
    if rsi <= 30:
        long_score = 1.0
    elif rsi <= 35:
        long_score = 0.7
    elif rsi <= 45:
        long_score = 0.3
    elif rsi <= 55:
        long_score = 0.0
    elif rsi <= 65:
        long_score = -0.3
    elif rsi <= 70:
        long_score = -0.7
    else:
        long_score = -1.0

    # For shorts, invert: high RSI = short signal
    short_score = -long_score

    # Confidence peaks at extremes
    confidence = abs(long_score)  # 0 in neutral, 1 at extremes

    if direction == "long":
        return _clamp(long_score), confidence
    elif direction == "short":
        return _clamp(short_score), confidence
    else:  # both — return whichever direction is actionable
        return _clamp(long_score), confidence


def sentiment_score(sentiment: float, buzz: float = 0.0, fear_greed: int = 50) -> tuple:
    """
    News sentiment factor.
    sentiment: -1 (bearish) to +1 (bullish)
    buzz: 0-1 (coverage volume)
    fear_greed: 0-100

    Contrarian overlay: extreme fear (<25) adds bullish bias, extreme greed (>75) adds bearish.
    """
    if sentiment is None:
        sentiment = 0.0

    # Base: news sentiment directly
    score = sentiment

    # Contrarian Fear & Greed overlay
    if fear_greed < 25:
        score += 0.3 * (1 - fear_greed / 25)  # extreme fear → contrarian long
    elif fear_greed > 75:
        score -= 0.3 * ((fear_greed - 75) / 25)  # extreme greed → contrarian short

    # Buzz amplification: high buzz + strong sentiment = higher confidence
    confidence = min(1.0, abs(sentiment) * (0.5 + buzz))

    return _clamp(score), confidence


def onchain_score(onchain_data: dict) -> tuple:
    """
    On-chain factor (BTC-specific, best-effort for alts).
    Uses hashrate trend + mempool as network health proxy.
    Returns (score, confidence).
    """
    if not onchain_data:
        return 0.0, 0.0

    hashrate = onchain_data.get("hashrate", 0)
    mempool = onchain_data.get("mempool_size", 0)

    # High mempool = network congestion = high demand = slightly bullish
    if mempool > 100_000:
        mempool_score = 0.3
    elif mempool > 50_000:
        mempool_score = 0.15
    elif mempool > 10_000:
        mempool_score = 0.05
    else:
        mempool_score = 0.0

    # Hashrate presence = network security = baseline bullish (BTC only)
    hashrate_score = 0.1 if hashrate > 0 else 0.0

    score = hashrate_score + mempool_score
    confidence = 0.3 if hashrate > 0 else 0.1  # low confidence for onchain

    return _clamp(score), confidence


def macro_score(macro_data: dict) -> tuple:
    """
    Macro factor: DXY + VIX impact on crypto.
    Strong dollar (high DXY) = headwind for crypto.
    High VIX = risk-off = headwind for crypto.
    Returns (score, confidence).
    """
    if not macro_data:
        return 0.0, 0.0

    dxy = macro_data.get("dxy", 0)
    vix = macro_data.get("vix", 0)

    score = 0.0
    confidence_parts = []

    # DXY: crypto tends to be inversely correlated with USD
    # DXY > 105 = strong dollar = bearish for crypto
    # DXY < 100 = weak dollar = bullish for crypto
    if dxy > 0:
        if dxy > 106:
            score -= 0.4
            confidence_parts.append(0.4)
        elif dxy > 104:
            score -= 0.2
            confidence_parts.append(0.2)
        elif dxy < 99:
            score += 0.3
            confidence_parts.append(0.3)
        elif dxy < 101:
            score += 0.15
            confidence_parts.append(0.15)

    # VIX: high volatility = risk-off = crypto sells off
    if vix > 0:
        if vix > 30:
            score -= 0.4
            confidence_parts.append(0.4)
        elif vix > 20:
            score -= 0.2
            confidence_parts.append(0.2)
        elif vix < 13:
            score += 0.15
            confidence_parts.append(0.15)

    confidence = min(1.0, sum(confidence_parts) / 2) if confidence_parts else 0.2

    return _clamp(score), confidence


def trending_boost(trending: bool, buzz: float) -> float:
    """
    Trending coins get amplified moves in both directions.
    Returns a multiplier [0.5, 1.5] to apply to the composite confidence.
    """
    boost = 1.0
    if trending:
        boost += 0.3
    boost += buzz * 0.2  # up to +0.2 from buzz
    return min(1.5, boost)


def composite_signal(
    rsi: float,
    direction: str,
    news_data: dict,
    onchain_data: dict,
    macro_data: dict,
    weights: dict,
    entry_threshold: float = 0.3,
) -> dict:
    """
    Fuse all factors into a composite signal.

    weights: dict from strategy.yaml, e.g.:
      {"rsi": 0.35, "sentiment": 0.30, "onchain": 0.10, "macro": 0.25}

    entry_threshold: minimum |composite_score| to trigger entry

    Returns decision dict.
    """
    # --- compute individual factor scores ---
    rsi_s, rsi_c = rsi_score(rsi, direction)
    sent_s, sent_c = sentiment_score(
        news_data.get("sentiment_score", 0) if news_data else 0,
        news_data.get("buzz_score", 0) if news_data else 0,
        news_data.get("fear_greed", 50) if news_data else 50,
    )
    chain_s, chain_c = onchain_score(onchain_data or {})
    macro_s, macro_c = macro_score(macro_data or {})

    # --- weighted composite ---
    w_rsi = weights.get("rsi", 0.35)
    w_sent = weights.get("sentiment", 0.30)
    w_chain = weights.get("onchain", 0.10)
    w_macro = weights.get("macro", 0.25)

    total_weight = w_rsi + w_sent + w_chain + w_macro
    if total_weight == 0:
        total_weight = 1.0

    composite = (
        rsi_s * w_rsi
        + sent_s * w_sent
        + chain_s * w_chain
        + macro_s * w_macro
    ) / total_weight

    # confidence: weighted average of individual confidences
    confidence = (
        rsi_c * w_rsi
        + sent_c * w_sent
        + chain_c * w_chain
        + macro_c * w_macro
    ) / total_weight

    # trending/buzz amplification
    trending = news_data.get("trending", False) if news_data else False
    buzz = news_data.get("buzz_score", 0) if news_data else 0
    confidence = min(1.0, confidence * trending_boost(trending, buzz))

    composite = _clamp(composite)
    confidence = _clamp(confidence, 0, 1)

    # --- determine signal ---
    signal = "flat"
    reasoning_parts = []

    if composite >= entry_threshold:
        signal = "long"
        reasoning_parts.append(f"composite {composite:+.2f} >= {entry_threshold}")
    elif composite <= -entry_threshold:
        signal = "short"
        reasoning_parts.append(f"composite {composite:+.2f} <= {-entry_threshold}")
    else:
        reasoning_parts.append(f"composite {composite:+.2f} in neutral zone [{-entry_threshold}, {entry_threshold}]")

    # Build readable reasoning
    factor_summary = []
    if abs(rsi_s) > 0.1:
        factor_summary.append(f"RSI={rsi_s:+.2f}({rsi:.0f})")
    if abs(sent_s) > 0.1:
        factor_summary.append(f"news={sent_s:+.2f}")
    if abs(chain_s) > 0.1:
        factor_summary.append(f"chain={chain_s:+.2f}")
    if abs(macro_s) > 0.1:
        factor_summary.append(f"macro={macro_s:+.2f}")

    reasoning = f"{signal.upper()} — {' | '.join(factor_summary)} → composite {composite:+.3f} conf {confidence:.2f}"

    return {
        "composite_score": round(composite, 4),
        "signal": signal,
        "confidence": round(confidence, 4),
        "factors": {
            "rsi": {"score": round(rsi_s, 3), "confidence": round(rsi_c, 3), "raw": rsi},
            "sentiment": {"score": round(sent_s, 3), "confidence": round(sent_c, 3)},
            "onchain": {"score": round(chain_s, 3), "confidence": round(chain_c, 3)},
            "macro": {"score": round(macro_s, 3), "confidence": round(macro_c, 3)},
        },
        "reasoning": reasoning,
    }


# ===========================================================================
# V2: Regime-aware structured decision engine (per the Market Analysis Prompt)
# ===========================================================================

def composite_signal_v2(
    rsi: float,
    direction: str,
    news_data: dict,
    onchain_data: dict,
    macro_data: dict,
    weights: dict,
    entry_threshold: float = 0.3,
    regime=None,
    hl_data: dict = None,
) -> dict:
    """
    Regime-aware signal engine. Requires a Regime object (from regime.py).
    Routes to momentum or mean-reversion sub-strategy based on regime.
    Outputs a structured decision with hypothesis, evidence, confidence,
    falsification condition, and sizing recommendation.

    All the economic rationale is stated inline — if we can't say WHY a
    signal should have edge, we don't use it.
    """
    from .regime import Regime

    if regime is None:
        regime = Regime()
        regime.reasoning = "no regime data — defaulting to mean_reversion"
        regime.sub_strategy = "mean_reversion"

    # --- ORTHOGONAL SIGNAL COMPONENTS ---

    # 1. MEAN-REVERSION (RSI distance from neutral)
    # Economic rationale: crypto RSI extremes tend to revert in range-bound
    # regimes because there's no trend to sustain the dislocation.
    rsi_s, rsi_c = rsi_score(rsi, direction)

    # 2. TREND/MOMENTUM (slope of recent closes)
    # Economic rationale: in trending regimes with volume confirmation,
    # price continuation is driven by herding + slow information diffusion.
    # This is empty here because we don't have closes; HL provides momentum.
    momentum_s = 0.0
    momentum_c = 0.0
    if hl_data and "momentum_signal" in hl_data:
        momentum_s = hl_data.get("momentum_signal", 0.0)
        momentum_c = 0.5  # HL momentum confidence is inherent in the score

    # 3. FUNDING (contrarian, from Hyperliquid)
    # Economic rationale: when longs pay high funding, the market is
    # overlevered on one side → squeeze risk → contrarian signal.
    funding_s = 0.0
    funding_c = 0.0
    if hl_data and "funding_signal" in hl_data:
        funding_s = hl_data.get("funding_signal", 0.0)
        funding_c = 0.5

    # 4. SENTIMENT (news flow + Fear/Greed)
    sent_s, sent_c = sentiment_score(
        news_data.get("sentiment_score", 0) if news_data else 0,
        news_data.get("buzz_score", 0) if news_data else 0,
        news_data.get("fear_greed", 50) if news_data else 50,
    )

    # 5. MACRO (DXY + VIX + risk sentiment)
    macro_s, macro_c = macro_score(macro_data or {})

    # --- ROUTE BY REGIME ---
    if regime.sub_strategy == "momentum":
        # In trending regime: weight momentum + funding (contrarian)
        # and de-emphasize RSI mean-reversion
        w_rsi = weights.get("rsi", 0.15) * 0.5  # halve mean-reversion
        w_mom = 0.30  # momentum gets high weight
        w_fund = 0.20  # funding contrarian
        w_sent = weights.get("sentiment", 0.15)
        w_macro = weights.get("macro", 0.20)
    elif regime.sub_strategy == "mean_reversion":
        # In range regime: weight RSI mean-reversion heavily, de-emphasize momentum
        w_rsi = weights.get("rsi", 0.40)
        w_mom = 0.05  # momentum gets low weight in range
        w_fund = 0.15
        w_sent = weights.get("sentiment", 0.20)
        w_macro = weights.get("macro", 0.20)
    else:  # skip or unknown
        w_rsi = weights.get("rsi", 0.35)
        w_mom = 0.10
        w_fund = 0.10
        w_sent = weights.get("sentiment", 0.20)
        w_macro = weights.get("macro", 0.25)

    total_w = w_rsi + w_mom + w_fund + w_sent + w_macro
    if total_w == 0:
        total_w = 1.0

    composite = (
        rsi_s * w_rsi
        + momentum_s * w_mom
        + funding_s * w_fund
        + sent_s * w_sent
        + macro_s * w_macro
    ) / total_w

    # Confidence: weighted agreement of components
    confidence = (
        rsi_c * w_rsi
        + momentum_c * w_mom
        + funding_c * w_fund
        + sent_c * w_sent
        + macro_c * w_macro
    ) / total_w

    # Apply regime confidence multiplier (reduces confidence in high-vol)
    confidence *= regime.confidence_multiplier
    composite *= regime.confidence_multiplier

    # Count agreeing orthogonal signals (not correlated variants)
    agreeing = 0
    total_signals = 0
    for s, c in [(rsi_s, rsi_c), (momentum_s, momentum_c),
                 (funding_s, funding_c), (sent_s, sent_c), (macro_s, macro_c)]:
        if c > 0.1:  # only count signals with meaningful confidence
            total_signals += 1
            if (s > 0 and composite > 0) or (s < 0 and composite < 0):
                agreeing += 1

    # Honest confidence: a composite of +0.35 from one weak input is NOT the
    # same as +0.35 from three agreeing orthogonal signals
    if total_signals >= 3 and agreeing >= 3:
        confidence_label = "high (3+ orthogonal signals agree)"
    elif total_signals >= 2 and agreeing >= 2:
        confidence_label = "moderate (2 signals agree)"
    elif total_signals >= 1:
        confidence_label = "low (single signal)"
    else:
        confidence_label = "very low (no significant signals)"

    composite = _clamp(composite)
    confidence = _clamp(confidence, 0, 1)

    # --- SIGNAL DETERMINATION ---
    signal = "flat"
    if regime.sub_strategy == "skip":
        signal = "skip"
        reasoning = f"SKIP — regime={regime.trend}/{regime.volatility} — {regime.reasoning}"
    elif composite >= entry_threshold:
        signal = "long"
    elif composite <= -entry_threshold:
        signal = "short"
    else:
        signal = "flat"

    # --- HYPOTHESIS + FALSIFICATION ---
    if signal == "long":
        if regime.sub_strategy == "momentum":
            hypothesis = f"Price continuation: {regime.trend} with volume confirmation; momentum traders will push further"
            falsification = f"Invalidated if volume drops (no confirmation) or if price crosses below 20-bar SMA"
        else:
            hypothesis = f"RSI oversold in range-bound regime; mean-reversion expected as {regime.trend} lacks momentum to sustain the dislocation"
            falsification = f"Invalidated if RSI drops below 20 (deeper oversold = trend may be forming) or if regime flips to trend_down"
    elif signal == "short":
        if regime.sub_strategy == "momentum":
            hypothesis = f"Downward momentum continuation: {regime.trend} with volume; panic selling drives further downside"
            falsification = f"Invalidated if volume dries up or price crosses above 20-bar SMA"
        else:
            hypothesis = f"RSI overbought in range-bound regime; mean-reversion expected to drag price back to mean"
            falsification = f"Invalidated if RSI exceeds 85 (extreme overbought in a breakout) or if regime flips to trend_up"
    else:
        hypothesis = ""
        falsification = ""

    # --- EVIDENCE STRING ---
    evidence_parts = []
    evidence_parts.append(f"RSI={rsi:.0f}({rsi_s:+.2f})")
    if abs(momentum_s) > 0.05:
        evidence_parts.append(f"momentum={momentum_s:+.2f}")
    if abs(funding_s) > 0.05:
        evidence_parts.append(f"funding={funding_s:+.2f}")
    if abs(sent_s) > 0.05:
        evidence_parts.append(f"sentiment={sent_s:+.2f}")
    if abs(macro_s) > 0.05:
        evidence_parts.append(f"macro={macro_s:+.2f}")
    evidence_parts.append(f"regime={regime.trend}/{regime.volatility}")
    evidence = " | ".join(evidence_parts)

    # --- BUILD STRUCTURED DECISION ---
    reasoning = f"{signal.upper()} — {evidence} → composite {composite:+.3f} conf {confidence:.2f} [{confidence_label}]"

    return {
        "composite_score": round(composite, 4),
        "signal": signal,
        "confidence": round(confidence, 4),
        "confidence_label": confidence_label,
        "regime": {
            "trend": regime.trend,
            "volatility": regime.volatility,
            "vol_percentile": regime.vol_percentile,
            "volume_context": regime.volume_context,
            "sub_strategy": regime.sub_strategy,
            "confidence_multiplier": regime.confidence_multiplier,
            "size_multiplier": regime.size_multiplier,
            "reasoning": regime.reasoning,
        },
        "factors": {
            "rsi": {"score": round(rsi_s, 3), "confidence": round(rsi_c, 3), "raw": rsi},
            "momentum": {"score": round(momentum_s, 3), "confidence": round(momentum_c, 3)},
            "funding": {"score": round(funding_s, 3), "confidence": round(funding_c, 3)},
            "sentiment": {"score": round(sent_s, 3), "confidence": round(sent_c, 3)},
            "macro": {"score": round(macro_s, 3), "confidence": round(macro_c, 3)},
        },
        "agreeing_signals": agreeing,
        "total_signals": total_signals,
        "hypothesis": hypothesis,
        "evidence": evidence,
        "falsification": falsification,
        "reasoning": reasoning,
        "size_multiplier": regime.size_multiplier,
    }
