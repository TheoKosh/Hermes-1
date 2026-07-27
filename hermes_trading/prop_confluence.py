"""prop_confluence.py — Deep multi-factor confluence scan for the Kraken prop eval.

Surface a trade ONLY when multiple independent factors align. This is the
opposite of trade-spam: most scans return flat. Designed to be run on demand
or by the 15-min alert cron.

Factors (each independently scored in [-1, +1]):
  1. ICT structure   — liquidity sweep + MSS + FVG (the prop strategy's core)
  2. VWAP position   — price clearly above/below session VWAP (trend confirm)
  3. RSI             — momentum: oversold bounce setup / overbought short
  4. EMA trend       — 20-EMA direction (pullback-with-trend)
  5. Macro regime    — OpenBB composite (risk-on helps longs, risk-off helps shorts)
  6. Funding (perps) — Deribit funding: extreme positive = avoid longs, etc.

Confluence = weighted sum. A signal fires only if:
  - |confluence| >= 0.55 (strong, multi-factor)
  - At least 3 of the 6 factors agree in direction
  - Macro is not strongly opposing the direction
"""
import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import ccxt.async_support as ccxt
from hermes_trading.ict_strategy import ICTStrategy
from hermes_trading.vwap_strategy import compute_vwap
from hermes_trading.adapters.openbb_adapter import fetch_all_cached


def _ema(closes: list, period: int = 20) -> float:
    if len(closes) < period:
        return sum(closes) / max(1, len(closes))
    k = 2 / (period + 1)
    e = closes[0]
    for c in closes[1:]:
        e = (float(c) - e) * k + e
    return e


def _rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0.0 for d in deltas[-period:]]
    ag = sum(gains) / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    rs = ag / al
    return 100.0 - (100.0 / (1.0 + rs))


async def scan_symbol(ex, symbol: str, ict: ICTStrategy, macro: dict, perps: dict) -> dict:
    """Return confluence analysis for one symbol."""
    ohlcv = await ex.fetch_ohlcv(symbol, timeframe="15m", limit=100)
    if not ohlcv or len(ohlcv) < 50:
        return {"symbol": symbol, "valid": False}
    highs = [float(c[2]) for c in ohlcv]
    lows = [float(c[3]) for c in ohlcv]
    closes = [float(c[4]) for c in ohlcv]
    vols = [float(c[5]) for c in ohlcv if len(c) > 5 and c[5] is not None]
    price = closes[-1]

    # 1. ICT structure
    ict_sig = ict.generate_signal(highs=highs, lows=lows, closes=closes, volumes=vols)
    ict_score = float(ict_sig.composite)  # already in [-1, 1]
    ict_dir = "long" if ict_score > 0.15 else ("short" if ict_score < -0.15 else "none")

    # 2. VWAP position
    vwap = compute_vwap(ohlcv)
    if vwap > 0:
        vwap_dist_pct = (price - vwap) / vwap * 100
        vwap_score = max(-1, min(1, vwap_dist_pct / 2.0))
    else:
        vwap_score = 0.0
    vwap_dir = "long" if vwap_score > 0.1 else ("short" if vwap_score < -0.1 else "none")

    # 3. RSI — oversold bounce (long) / overbought fade (short)
    rsi = _rsi(closes)
    if rsi < 30:
        rsi_score = (30 - rsi) / 30  # deeper oversold = stronger long
        rsi_dir = "long"
    elif rsi > 70:
        rsi_score = (rsi - 70) / 30  # higher overbought = stronger short
        rsi_dir = "short"
    else:
        rsi_score = 0.0
        rsi_dir = "none"
    rsi_score = max(-1, min(1, rsi_score))

    # 4. EMA trend (price vs 20-EMA)
    ema = _ema(closes, 20)
    ema_dist_pct = (price - ema) / ema * 100 if ema > 0 else 0
    ema_score = max(-1, min(1, ema_dist_pct / 2.0))
    ema_dir = "long" if ema_score > 0.1 else ("short" if ema_score < -0.1 else "none")

    # 5. Macro regime
    macro_score = macro.get("score", 0.0)  # risk-on positive, risk-off negative
    # macro only contributes if it's a directional bias (risk-on helps longs)
    macro_dir = "long" if macro_score > 0.15 else ("short" if macro_score < -0.15 else "none")

    # 6. Funding (only for BTC/ETH — the perps we track)
    base = symbol.split("/")[0]
    perp = perps.get(base, {})
    funding_score = 0.0
    if perp:
        ann = perp.get("funding_8h", 0) * 3 * 365  # annualized
        # extreme positive funding (>40% ann) = longs crowded = slight short bias
        # extreme negative (<-20% ann) = shorts crowded = slight long bias
        if ann > 40:
            funding_score = -min(0.5, (ann - 40) / 80)
        elif ann < -20:
            funding_score = min(0.5, (-20 - ann) / 80)
    funding_dir = "long" if funding_score > 0.05 else ("short" if funding_score < -0.05 else "none")

    # --- confluence ---
    weights = {"ict": 0.30, "vwap": 0.20, "rsi": 0.15, "ema": 0.10, "macro": 0.15, "funding": 0.10}
    conf_score = (ict_score * weights["ict"] + vwap_score * weights["vwap"] +
                  rsi_score * weights["rsi"] + ema_score * weights["ema"] +
                  macro_score * weights["macro"] + funding_score * weights["funding"])
    conf_score = max(-1, min(1, conf_score))

    factors = {"ict": (ict_dir, round(ict_score, 2)),
               "vwap": (vwap_dir, round(vwap_score, 2)),
               "rsi": (rsi_dir, round(rsi_score, 2)),
               "ema": (ema_dir, round(ema_score, 2)),
               "macro": (macro_dir, round(macro_score, 2)),
               "funding": (funding_dir, round(funding_score, 2))}

    # direction = whichever side has more agreeing factors
    long_n = sum(1 for d, _ in factors.values() if d == "long")
    short_n = sum(1 for d, _ in factors.values() if d == "short")
    if long_n > short_n:
        direction, agree = "long", long_n
    elif short_n > long_n:
        direction, agree = "short", short_n
    else:
        direction, agree = "none", 0

    # signal fires only on strong confluence
    fires = (abs(conf_score) >= 0.55 and agree >= 3 and
             not (direction == "long" and macro_score < -0.3) and
             not (direction == "short" and macro_score > 0.3))

    return {
        "symbol": symbol, "valid": True, "price": round(price, 4),
        "vwap": round(vwap, 4) if vwap > 0 else 0,
        "rsi": round(rsi, 1), "ict_reason": ict_sig.reasoning[:50],
        "confluence": round(conf_score, 3), "direction": direction,
        "agreeing_factors": agree, "factors": factors, "fires": fires,
    }


async def main():
    ex = ccxt.kraken({"enableRateLimit": True})
    await ex.load_markets()
    ict = ICTStrategy()
    obb = fetch_all_cached()
    macro = obb.get("risk", {})
    perps = obb.get("perps", {})
    print(f"Macro: {macro.get('regime','?')} score={macro.get('score',0):+.2f}")
    print("=" * 78)

    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
               "XRP/USD", "ADA/USD", "DOT/USD", "MATIC/USD", "ATOM/USD"]
    results = []
    for sym in symbols:
        try:
            r = await scan_symbol(ex, sym, ict, macro, perps)
            if r.get("valid"):
                results.append(r)
        except Exception as e:
            print(f"  {sym}: scan error {e}")
    await ex.close()

    results.sort(key=lambda x: abs(x["confluence"]), reverse=True)
    fires = [r for r in results if r["fires"]]
    print(f"\n{'SYM':<9}{'PRICE':>10}{'RSI':>6}{'CONF':>7}{'DIR':>6}{'AGREE':>6}  FACTORS")
    print("-" * 78)
    for r in results:
        mark = " ◀ SIGNAL" if r["fires"] else ""
        f_str = " ".join(f"{k}:{d[0][0].upper()}({d[1]:+.2f})" for k, d in r["factors"].items())
        print(f"{r['symbol']:<9}{r['price']:>10}{r['rsi']:>6}{r['confluence']:>+7.2f}"
              f"{r['direction']:>6}{r['agreeing_factors']:>6}  {f_str}{mark}")
    print("-" * 78)
    if fires:
        print(f"\n▶ {len(fires)} HIGH-CONFIDENCE SETUP(S):")
        for r in fires:
            print(f"  {r['symbol']} {r['direction'].upper()} @ ${r['price']} "
                  f"(conf {r['confluence']:+.2f}, {r['agreeing_factors']}/6 factors)")
    else:
        print("\n✗ No high-confidence setups — staying flat. (bar: |conf|≥0.55 AND ≥3 factors agree)")


if __name__ == "__main__":
    asyncio.run(main())
