"""
run.py — Entrypoint for the Hermes trading worker.

Trial phase: runs 8 strategies in parallel with EUR100 each to determine
which approach works best for short/medium-term gains.
"""
import asyncio
import os
import sys
import yaml
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
GOAL_FILE = STATE_DIR / "goal.yaml"


def load_goal() -> dict:
    if not GOAL_FILE.exists():
        print(f"[FATAL] goal.yaml not found at {GOAL_FILE}", file=sys.stderr)
        sys.exit(1)
    with open(GOAL_FILE) as f:
        return yaml.safe_load(f)


def main():
    goal = load_goal()
    mode = os.environ.get("HERMES_TRADING_MODE", "paper")
    accept_risk = os.environ.get("HERMES_TRADING_I_ACCEPT_RISK", "false").lower() == "true"
    budget = float(os.environ.get("HERMES_TEST_BUDGET", "100"))

    print("Booting hermes-trading multi-strategy test harness")
    print(f"  Mode: {mode}")
    if mode == "live" and not accept_risk:
        print("\n[FATAL] LIVE MODE requires HERMES_TRADING_I_ACCEPT_RISK=true")
        sys.exit(1)
    print(f"  Budget per strategy: EUR {budget:.0f}")
    print(f"  Target: +{goal['target_return_30d']*100:.1f}% / 30d")
    print(f"  Max DD: {goal['max_drawdown']*100:.1f}%")
    print()

    from .test_runner import MultiStrategyRunner

    runner = MultiStrategyRunner(goal=goal, mode=mode, budget_per_strategy=budget)
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Received interrupt, exiting gracefully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
