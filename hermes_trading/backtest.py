"""
backtest.py — Historical backtest engine for strategy comparison.

Downloads 30 days of 15-minute OHLCV for each asset via Coinbase (ccxt),
replays candle-by-candle through each strategy version, and outputs a
comparison table: P&L %, Sharpe, max drawdown, win rate, trade count.

Usage:
    python -m hermes_trading.backtest
"""
import asyncio
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Ensure we can import the package
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hermes_trading.adapters.price import PriceAdapter


STRATEGIES = {
    "v01_rsi30_long": {
        "version": "01",
        "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
        "stop_loss_pct": 2.0,
        "take_profit_multiple": 1.5,
        "position_size_r": 0.5,
        "max_concurrent_positions": 1,
        "risk_per_trade_pct": 5.0,
    },
    "v02_rsi32_long": {
        "version": "02",
        "entry": {"indicator": "rsi", "threshold": 32, "direction": "long"},
        "stop_loss_pct": 2.0,
        "take_profit_multiple": 1.5,
        "position_size_r": 0.5,
        "max_concurrent_positions": 1,
        "risk_per_trade_pct": 5.0,
    },
    "v03_composite_both": {
        "version": "03",
        "entry": {"indicator": "composite", "threshold": 35, "direction": "both"},
        "entry_threshold": 0.3,
        "signal_weights": {"rsi": 0.35, "sentiment": 0.30, "onchain": 0.10, "macro": 0.25},
        "stop_loss_pct": 3.0,
        "take_profit_multiple": 2.0,
        "position_size_r": 0.15,
        "max_concurrent_positions": 5,
        "risk_per_trade_pct": 2.0,
    },
}

BASKET = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT"]
STARTING_EQUITY = 100.0  # 100 EUR paper account
DAYS = 30


async def fetch_historical(pair: str, days: int = 30) -> list:
    """Download historical 15m OHLCV candles via ccxt (Coinbase)."""
    import ccxt.async_support as ccxt

    exchange = ccxt.coinbase({"enableRateLimit": True})
    try:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        # 15m candles: 96 per day, 30 days = 2880 candles. Fetch in batches.
        all_candles = []
        limit = 300  # max per request
        current_since = since

        for _ in range(12):  # 12 batches × 300 = 3600 max
            ohlcv = await exchange.fetch_ohlcv(pair, timeframe="15m", since=current_since, limit=limit)
            if not ohlcv:
                break
            all_candles.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1  # next candle after last
            if len(ohlcv) < limit:
                break
            await asyncio.sleep(0.3)  # rate limit courtesy

        await exchange.close()

        # Deduplicate by timestamp
        seen = set()
        unique = []
        for c in all_candles:
            if c[0] not in seen:
                seen.add(c[0])
                unique.append(c)

        # Sort by timestamp
        unique.sort(key=lambda x: x[0])
        return unique

    except Exception as e:
        await exchange.close()
        print(f"  [!] Failed to fetch {pair}: {e}")
        return []


def compute_rsi_series(closes: list, period: int = 14) -> list:
    """Compute RSI for a full series of closes. Returns list of RSI values aligned to closes."""
    rsi_values = [50.0] * len(closes)
    if len(closes) < period + 1:
        return rsi_values

    closes_arr = np.array(closes, dtype=float)
    deltas = np.diff(closes_arr)

    for i in range(period, len(closes)):
        gains = np.where(deltas[i-period:i] > 0, deltas[i-period:i], 0)
        losses = np.where(deltas[i-period:i] < 0, -deltas[i-period:i], 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))

    return rsi_values


def rsi_signal_score(rsi: float, threshold: float, direction: str) -> str:
    """Simple RSI-only entry for v01/v02."""
    if direction in ("long", "both") and rsi <= threshold:
        return "long"
    if direction in ("short", "both") and rsi >= (100 - threshold):
        return "short"
    return "flat"


def composite_score_simple(rsi: float, threshold: float, direction: str, weights: dict, entry_threshold: float) -> str:
    """
    Simplified composite for backtesting (no live news/macro during replay).
    Uses RSI as primary + simulated sentiment from recent price action.
    """
    # RSI component (mean-reversion)
    if rsi <= 30:
        rsi_s = 1.0
    elif rsi <= threshold:
        rsi_s = 0.7
    elif rsi <= 45:
        rsi_s = 0.3
    elif rsi <= 55:
        rsi_s = 0.0
    elif rsi <= 65:
        rsi_s = -0.3
    elif rsi <= 70:
        rsi_s = -0.7
    else:
        rsi_s = -1.0

    # In backtest, sentiment/onchain/macro unavailable → RSI dominates
    # Apply weight: RSI gets its weight, rest contributes 0 (neutral)
    w_rsi = weights.get("rsi", 0.35)
    total_w = sum(weights.values())
    composite = (rsi_s * w_rsi) / total_w if total_w > 0 else 0

    if composite >= entry_threshold:
        return "long"
    elif composite <= -entry_threshold:
        return "short"
    return "flat"


def backtest_strategy(candles_by_pair: dict, strategy: dict, starting_equity: float) -> dict:
    """
    Replay historical data through a strategy.
    Returns performance metrics.
    """
    equity = starting_equity
    peak_equity = starting_equity
    max_dd = 0.0
    trades = []
    positions = {}  # pair -> {side, entry, stop, target, value, opened_idx}

    # Find the shortest candle series (they should all be ~same length)
    min_len = min(len(v) for v in candles_by_pair.values()) if candles_by_pair else 0
    if min_len == 0:
        return {"error": "no data"}

    # Precompute RSI for each pair
    rsi_by_pair = {}
    for pair, candles in candles_by_pair.items():
        closes = [c[4] for c in candles]  # close price
        rsi_by_pair[pair] = compute_rsi_series(closes)

    # Iterate candle by candle (all pairs in sync)
    max_concurrent = strategy.get("max_concurrent_positions", 5)
    stop_pct = strategy.get("stop_loss_pct", 3.0)
    tp_mult = strategy.get("take_profit_multiple", 2.0)
    risk_pct = strategy.get("risk_per_trade_pct", 2.0)
    size_r = strategy.get("position_size_r", 0.15)
    direction = strategy.get("entry", {}).get("direction", "long")
    threshold = strategy.get("entry", {}).get("threshold", 35)
    entry_threshold = strategy.get("entry_threshold", 0.3)
    signal_weights = strategy.get("signal_weights", {"rsi": 0.35})
    is_composite = strategy.get("entry", {}).get("indicator") == "composite"

    for i in range(1, min_len):
        for pair, candles in candles_by_pair.items():
            candle = candles[i]
            close = candle[4]
            rsi = rsi_by_pair[pair][i]

            # --- Manage open position (check SL/TP) ---
            if pair in positions:
                pos = positions[pair]
                high = candle[2]
                low = candle[3]

                hit_stop = False
                hit_tp = False

                if pos["side"] == "long":
                    if low <= pos["stop"]:
                        hit_stop = True
                    elif high >= pos["target"]:
                        hit_tp = True
                else:
                    if high >= pos["stop"]:
                        hit_stop = True
                    elif low <= pos["target"]:
                        hit_tp = True

                if hit_stop or hit_tp:
                    exit_price = pos["stop"] if hit_stop else pos["target"]
                    reason = "stop_loss" if hit_stop else "take_profit"

                    if pos["side"] == "long":
                        pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
                    else:
                        pnl_pct = (pos["entry"] - exit_price) / pos["entry"] * 100

                    pnl_dollar = pnl_pct / 100 * pos["value"]
                    equity += pnl_dollar

                    trades.append({
                        "pair": pair,
                        "side": pos["side"],
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "pnl_pct": round(pnl_pct, 2),
                        "pnl_dollar": round(pnl_dollar, 2),
                        "reason": reason,
                    })
                    del positions[pair]

                    # Update peak/DD
                    if equity > peak_equity:
                        peak_equity = equity
                    dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
                    if dd > max_dd:
                        max_dd = dd

                    # Drawdown circuit breaker
                    if dd >= 10.0:
                        # Stop trading this pair but keep recording equity
                        continue

                else:
                    continue  # still holding, skip entry

            # --- Check for new entry ---
            if len(positions) >= max_concurrent:
                continue

            if is_composite:
                signal = composite_score_simple(rsi, threshold, direction, signal_weights, entry_threshold)
            else:
                signal = rsi_signal_score(rsi, threshold, direction)

            if signal in ("long", "short"):
                # Position sizing
                stop_distance_pct = stop_pct
                position_value = (equity * risk_pct / 100) / (stop_distance_pct / 100)
                max_position = equity * size_r
                position_value = min(position_value, max_position)

                if position_value < 1:  # skip if too small
                    continue

                if signal == "long":
                    stop = close * (1 - stop_pct / 100)
                    target = close * (1 + stop_pct * tp_mult / 100)
                else:
                    stop = close * (1 + stop_pct / 100)
                    target = close * (1 - stop_pct * tp_mult / 100)

                positions[pair] = {
                    "side": signal,
                    "entry": close,
                    "stop": stop,
                    "target": target,
                    "value": position_value,
                    "opened_idx": i,
                }

    # Close any remaining positions at last close
    for pair in list(positions.keys()):
        pos = positions[pair]
        candles = candles_by_pair.get(pair, [])
        if candles:
            last_close = candles[-1][4]
            if pos["side"] == "long":
                pnl_pct = (last_close - pos["entry"]) / pos["entry"] * 100
            else:
                pnl_pct = (pos["entry"] - last_close) / pos["entry"] * 100
            pnl_dollar = pnl_pct / 100 * pos["value"]
            equity += pnl_dollar
            trades.append({
                "pair": pair,
                "side": pos["side"],
                "entry": pos["entry"],
                "exit": last_close,
                "pnl_pct": round(pnl_pct, 2),
                "pnl_dollar": round(pnl_dollar, 2),
                "reason": "end_of_backtest",
            })

    # --- Compute metrics ---
    total_return_pct = (equity - starting_equity) / starting_equity * 100

    if trades:
        pnls = [t["pnl_dollar"] for t in trades]
        wins = [t for t in trades if t["pnl_dollar"] > 0]
        losses = [t for t in trades if t["pnl_dollar"] <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t["pnl_dollar"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_dollar"] for t in losses]) if losses else 0

        # Sharpe: mean(pnl) / std(pnl) * sqrt(annualization)
        pnl_arr = np.array(pnls)
        if np.std(pnl_arr) > 0:
            # 15min candles, 96/day → annualize by sqrt(96*365)
            sharpe = np.mean(pnl_arr) / np.std(pnl_arr) * math.sqrt(96 * 365)
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = sum(t["pnl_dollar"] for t in wins)
        gross_loss = abs(sum(t["pnl_dollar"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        sharpe = 0
        profit_factor = 0

    # Per-pair breakdown
    pair_pnl = {}
    for t in trades:
        pair_pnl.setdefault(t["pair"], []).append(t["pnl_dollar"])
    pair_summary = {p: round(sum(v), 2) for p, v in pair_pnl.items()}

    return {
        "strategy_version": strategy.get("version"),
        "starting_equity": starting_equity,
        "final_equity": round(equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_trades": len(trades),
        "wins": len([t for t in trades if t["pnl_dollar"] > 0]),
        "losses": len([t for t in trades if t["pnl_dollar"] <= 0]),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_eur": round(avg_win, 2),
        "avg_loss_eur": round(avg_loss, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "pair_pnl": pair_summary,
        "trades_sample": trades[:10],  # first 10 for inspection
    }


async def main():
    print("=" * 65)
    print("  BACKTEST: 30-day historical comparison of 3 strategies")
    print(f"  Basket: {len(BASKET)} assets | Starting equity: EUR {STARTING_EQUITY}")
    print("=" * 65)

    # Download historical data
    print("\n[1/3] Downloading 30 days of 15m OHLCV data...")
    candles_by_pair = {}
    adapter = None
    for pair in BASKET:
        print(f"  → {pair}...", end=" ", flush=True)
        candles = await fetch_historical(pair, DAYS)
        candles_by_pair[pair] = candles
        print(f"{len(candles)} candles")

    # Check we got data
    total_candles = sum(len(v) for v in candles_by_pair.values())
    if total_candles == 0:
        print("\n[FATAL] No historical data downloaded. Check network/API.")
        return

    print(f"\n  Total: {total_candles} candles across {len(BASKET)} pairs")

    # Run backtests
    print("\n[2/3] Running backtests...")
    results = {}
    for name, strategy in STRATEGIES.items():
        print(f"  → {name}...", end=" ", flush=True)
        result = backtest_strategy(candles_by_pair, strategy, STARTING_EQUITY)
        results[name] = result
        if "error" in result:
            print("ERROR")
        else:
            print(f"return={result['total_return_pct']:+.1f}% trades={result['total_trades']} sharpe={result['sharpe']}")

    # Print comparison table
    print("\n" + "=" * 65)
    print("  RESULTS COMPARISON")
    print("=" * 65)
    header = f"{'Metric':<22} {'v01 (RSI30 long)':>18} {'v02 (RSI32 long)':>18} {'v03 (composite)':>18}"
    print(header)
    print("-" * 65)

    r1, r2, r3 = results.get("v01_rsi30_long", {}), results.get("v02_rsi32_long", {}), results.get("v03_composite_both", {})

    def fmt(v, suffix="", default="N/A"):
        if isinstance(v, dict) and "error" in v:
            return default
        return f"{v}{suffix}" if v is not None else default

    rows = [
        ("Return %", "total_return_pct", "%"),
        ("Final equity (EUR)", "final_equity", ""),
        ("Total trades", "total_trades", ""),
        ("Win rate", "win_rate_pct", "%"),
        ("Max drawdown", "max_drawdown_pct", "%"),
        ("Sharpe", "sharpe", ""),
        ("Profit factor", "profit_factor", ""),
        ("Avg win (EUR)", "avg_win_eur", ""),
        ("Avg loss (EUR)", "avg_loss_eur", ""),
    ]

    for label, key, suffix in rows:
        v1 = r1.get(key, "N/A") if isinstance(r1, dict) else "N/A"
        v2 = r2.get(key, "N/A") if isinstance(r2, dict) else "N/A"
        v3 = r3.get(key, "N/A") if isinstance(r3, dict) else "N/A"
        print(f"  {label:<20} {str(v1)+suffix:>18} {str(v2)+suffix:>18} {str(v3)+suffix:>18}")

    # Per-pair P&L for v03
    print(f"\n  Per-pair P&L (v03 composite):")
    for pair, pnl in sorted(r3.get("pair_pnl", {}).items(), key=lambda x: x[1], reverse=True):
        bar = "█" * max(1, int(abs(pnl) / 2))
        color_indicator = "+" if pnl >= 0 else "-"
        print(f"    {pair:12s} {pnl:+8.2f} EUR {color_indicator}{bar}")

    # Verdict
    print("\n" + "=" * 65)
    best = max(results.values(), key=lambda x: x.get("total_return_pct", -999) if isinstance(x, dict) and "total_return_pct" in x else -999)
    if isinstance(best, dict) and "total_return_pct" in best:
        print(f"  BEST STRATEGY: v{best['strategy_version']} — return {best['total_return_pct']:+.1f}%, "
              f"Sharpe {best['sharpe']}, max DD {best['max_drawdown_pct']}%")
        meets_target = best["total_return_pct"] >= 15.0
        meets_dd = best["max_drawdown_pct"] <= 10.0
        meets_sharpe = best["sharpe"] >= 1.2
        print(f"  Target +15%: {'✓ PASS' if meets_target else '✗ FAIL'}")
        print(f"  Max DD 10%: {'✓ PASS' if meets_dd else '✗ FAIL'}")
        print(f"  Sharpe 1.2: {'✓ PASS' if meets_sharpe else '✗ FAIL'}")
    print("=" * 65)

    # Save full results to file
    output_path = REPO / "state" / "backtest_results.json"
    # Clean for JSON (replace inf)
    clean_results = {}
    for name, r in results.items():
        clean_results[name] = {k: v for k, v in r.items() if k != "trades_sample"}
        clean_results[name]["timestamp"] = datetime.now(timezone.utc).isoformat()
    output_path.write_text(json.dumps(clean_results, indent=2, default=str))
    print(f"\n  Full results saved to: {output_path}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
