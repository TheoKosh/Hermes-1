"""
strategy_adapter.py — Map live trading strategies to backtester configs.

Each test strategy from strategies_test.py gets mapped to a Backtester
configuration that matches its live parameters. This lets us run
walk-forward backtests on all 8 strategies using the same framework.
"""
from .backtest.backtester import run_walkforward, check_survivorship


def strategy_to_backtester_config(strategy: dict) -> dict:
    """
    Convert a live strategy dict into a Backtester kwargs dict.

    The backtester uses: conf_threshold, stop_atr_mult, target_atr_mult,
    fee_bps, max_positions, max_drawdown_pct, kelly_fraction.
    """
    s = strategy

    # Map stop_loss_pct to ATR multiplier (approximate: stop_pct / 2 ≈ ATR mult)
    stop_atr_mult = s.get("stop_loss_pct", 2.5) / 2.0
    target_atr_mult = s.get("stop_loss_pct", 2.5) * s.get("take_profit_multiple", 2.5) / 2.0

    # Coinbase taker fee = 0.60% = 60 bps round-trip / 2 = 30 bps per side
    # Backtester applies fee_bps to closed trades (entry+exit)
    fee_bps = 12.0  # 0.12% per side → 24 bps round-trip (conservative)

    return {
        "conf_threshold": s.get("entry_threshold", 0.30),
        "stop_atr_mult": max(0.5, stop_atr_mult),
        "target_atr_mult": max(0.5, target_atr_mult),
        "fee_bps": fee_bps,
        "max_positions": s.get("max_concurrent_positions", 3),
        "max_drawdown_pct": 0.10,
        "kelly_fraction": 0.25,  # Quarter-Kelly
    }


def run_strategy_backtest(panel: dict, strategy: dict, n_trials: int = 8) -> dict:
    """
    Run walk-forward backtest for a single strategy.

    Args:
        panel: {symbol: OHLCV DataFrame}
        strategy: strategy dict from strategies_test.py
        n_trials: number of strategies tested (for Deflated Sharpe)

    Returns:
        Dict with IS/OOS metrics, overfitting flags, degradation.
    """
    config = strategy_to_backtester_config(strategy)

    result = run_walkforward(panel, split_frac=0.7, **config)

    is_m = result["in_sample"]["metrics"]
    oos_m = result["out_of_sample"]["metrics"]
    deg = result.get("sharpe_degradation_pct")

    # Add strategy identification
    result["strategy_id"] = strategy.get("version", "?")
    result["strategy_config"] = config

    return result


def run_all_strategy_backtests(panel: dict, strategies: list) -> list:
    """
    Run walk-forward backtests for all strategies.
    Returns list of result dicts sorted by OOS Sharpe.
    """
    results = []
    n_trials = len(strategies)

    for i, strat_def in enumerate(strategies):
        sid = strat_def["id"]
        strat = strat_def["strategy"]
        print(f"\n{'='*60}")
        print(f"  Backtesting [{i+1}/{len(strategies)}]: {sid}")
        print(f"{'='*60}")

        try:
            result = run_strategy_backtest(panel, strat, n_trials=n_trials)
            result["strategy_id"] = sid
            result["strategy_name"] = strat_def.get("name", sid)
            result["strategy_description"] = strat_def.get("description", "")

            is_m = result["in_sample"]["metrics"]
            oos_m = result["out_of_sample"]["metrics"]

            print(f"  IN-SAMPLE:  Sharpe={_fmt(is_m.get('sharpe'))}  Win={_fmt(is_m.get('win_rate'), True)}  "
                  f"Trades={is_m.get('n_trades', 0)}  DD={_fmt(is_m.get('max_drawdown'), True)}")
            print(f"  OUT-OF-SAMPLE: Sharpe={_fmt(oos_m.get('sharpe'))}  Win={_fmt(oos_m.get('win_rate'), True)}  "
                  f"Trades={oos_m.get('n_trades', 0)}  DD={_fmt(oos_m.get('max_drawdown'), True)}")
            print(f"  Sharpe degradation: {_fmt(result.get('sharpe_degradation_pct'), True)}")

            for flag in oos_m.get("overfitting_flags", []):
                print(f"    ⚠ {flag}")

            results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "strategy_id": sid,
                "error": str(e),
                "in_sample": {"metrics": {}},
                "out_of_sample": {"metrics": {}},
            })

    # Sort by OOS Sharpe (best first)
    results.sort(key=lambda r: _safe_sharpe(r.get("out_of_sample", {}).get("metrics", {})), reverse=True)

    return results


def _safe_sharpe(metrics: dict) -> float:
    """Extract Sharpe safely for sorting."""
    s = metrics.get("sharpe", 0)
    if s is None or (isinstance(s, float) and (s != s)):  # NaN check
        return -999
    return float(s)


def _fmt(x, pct=False) -> str:
    """Format a number for display."""
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and x != x:  # NaN
            return "n/a"
        if pct:
            return f"{x:.1%}"
        return f"{x:.2f}"
    except:
        return str(x)
