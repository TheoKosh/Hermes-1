"""
score.py — Composite score for trades against goal.

score(trades, goal) -> float in [-1, +1]

Composite of:
  - realised return vs target
  - drawdown vs max
  - Sharpe vs min
"""
import json
import math
from pathlib import Path

import numpy as np


def score(trades: list, goal: dict) -> float:
    """
    Compute a composite score in [-1, +1] from a list of trade dicts and a goal dict.

    trades: list of dicts with at least 'pnl_pct' (percentage, e.g. 1.5 = +1.5%)
    goal:   dict with target_return_30d, max_drawdown, min_sharpe
    """
    if not trades:
        return 0.0

    pnls = np.array([t["pnl_pct"] for t in trades], dtype=float)

    # --- realised return component ---
    total_return = float(np.sum(pnls))
    # express as fraction (pnl_pct is in percent; target_return_30d is a fraction)
    total_return_frac = total_return / 100.0
    target = goal.get("target_return_30d", 0.05)

    if target > 0:
        return_score = min(1.0, total_return_frac / target)
    else:
        return_score = 0.0

    # --- drawdown component ---
    # worst single-trade loss as a fraction
    worst = float(np.min(pnls)) / 100.0
    max_dd = goal.get("max_drawdown", 0.08)

    if worst < 0:
        # penalty grows as worst loss approaches/exceeds max_dd
        dd_ratio = abs(worst) / max_dd if max_dd > 0 else 1.0
        dd_score = max(-1.0, 1.0 - dd_ratio)
    else:
        dd_score = 1.0

    # --- Sharpe component ---
    if len(pnls) > 1 and np.std(pnls) > 0:
        # annualise crudely: assume ~1 trade/day, 365 days
        sharpe = float(np.mean(pnls) / np.std(pnls)) * math.sqrt(365)
    elif len(pnls) == 1 and pnls[0] != 0:
        sharpe = 1.0 if pnls[0] > 0 else -1.0
    else:
        sharpe = 0.0

    min_sharpe = goal.get("min_sharpe", 1.2)
    if min_sharpe > 0:
        sharpe_score = min(1.0, max(-1.0, sharpe / min_sharpe))
    else:
        sharpe_score = 0.0

    # --- composite (weighted average) ---
    # weights: 45% return, 30% drawdown, 25% Sharpe
    composite = 0.45 * return_score + 0.30 * dd_score + 0.25 * sharpe_score

    # apply floor
    failure_below = goal.get("failure_below", -0.04)
    if composite < failure_below:
        composite = failure_below

    # clamp
    composite = max(-1.0, min(1.0, composite))

    return round(composite, 6)


def score_from_files():
    """Convenience: score from state files."""
    import yaml

    STATE_DIR = Path(__file__).resolve().parent.parent / "state"
    goal_file = STATE_DIR / "goal.yaml"
    trades_file = STATE_DIR / "trades.jsonl"

    with open(goal_file) as f:
        goal = yaml.safe_load(f)

    trades = []
    if trades_file.exists():
        with open(trades_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))

    s = score(trades, goal)
    print(f"Score: {s:+.4f}  (from {len(trades)} trades)")
    return s


if __name__ == "__main__":
    score_from_files()
