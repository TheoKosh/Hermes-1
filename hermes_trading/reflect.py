"""
reflect.py — Reflection cycle.

TWO modes:
  --fallback: deterministic rule (used before Hermes is installed).
  --hermes:   production mode (calls `hermes` subprocess).
"""
import argparse
import json
import subprocess
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
GOAL_FILE = STATE_DIR / "goal.yaml"
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
TRADES_FILE = STATE_DIR / "trades.jsonl"
HYPOTHESES_FILE = STATE_DIR / "hypotheses.jsonl"
HISTORY_DIR = STATE_DIR / "history"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_trades():
    trades = []
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
    return trades


def bump_version(version_str: str) -> str:
    """01 -> 02, 09 -> 10, 99 -> 100"""
    try:
        n = int(version_str)
        return str(n + 1).zfill(len(version_str))
    except ValueError:
        return str(int(version_str) + 1)


def save_prior(strategy: dict):
    """Save current strategy to history before mutating."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    version = strategy.get("version", "01")
    hist_path = HISTORY_DIR / f"v{version}.yaml"
    with open(hist_path, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)


def append_hypothesis(hyp: dict):
    """Append a hypothesis record to hypotheses.jsonl."""
    with open(HYPOTHESES_FILE, "a") as f:
        f.write(json.dumps(hyp) + "\n")


def write_strategy(strategy: dict):
    with open(STRATEGY_FILE, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)


def fallback_reflect():
    """
    Deterministic reflection:
      - If realised return < target → loosen entry.threshold by 2.
      - If drawdown > max → tighten stop_loss_pct by 0.2.
      - Always changes exactly ONE variable.
    """
    goal = load_yaml(GOAL_FILE)
    strategy = load_yaml(STRATEGY_FILE)
    trades = load_trades()

    if not trades:
        print("[REFLECT] No trades yet — nothing to reflect on.")
        sys.exit(0)

    # compute realised return from recent trades
    recent = trades[-25:]
    pnls = [t["pnl_pct"] for t in recent]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / len(pnls) if pnls else 0

    # crude drawdown: worst single-trade loss
    worst_loss = min(pnls) if pnls else 0
    worst_loss_dec = worst_loss / 100

    target = goal["target_return_30d"]
    max_dd = goal["max_drawdown"]

    variable_changed = None
    old_value = None
    new_value = None
    reasoning = ""

    # PRIORITY: if drawdown breached, tighten stop loss first
    if worst_loss_dec < -max_dd:
        old_sl = strategy.get("stop_loss_pct", 2.0)
        new_sl = round(max(0.5, old_sl - 0.2), 2)
        strategy["stop_loss_pct"] = new_sl
        variable_changed = "stop_loss_pct"
        old_value = old_sl
        new_value = new_sl
        reasoning = f"Worst single-trade loss ({worst_loss:.2f}%) exceeded max drawdown ({max_dd*100:.1f}%). Tightening stop loss from {old_sl} to {new_sl}."

    # Otherwise: if underperforming target, loosen entry threshold
    elif avg_pnl < target * 100:
        old_thresh = strategy.get("entry", {}).get("threshold", 30)
        new_thresh = old_thresh + 2
        strategy.setdefault("entry", {})["threshold"] = new_thresh
        variable_changed = "entry.threshold"
        old_value = old_thresh
        new_value = new_thresh
        reasoning = f"Avg realised P&L per trade ({avg_pnl:.2f}%) below target ({target*100:.1f}% per 30d). Loosening RSI entry threshold from {old_thresh} to {new_thresh} to capture more entries."

    else:
        print("[REFLECT] Strategy is performing within targets bounds. No change needed.")
        # still record a hypothesis
        hypothesis = {
            "timestamp": now_iso(),
            "mode": "fallback",
            "version_before": strategy.get("version", "01"),
            "version_after": strategy.get("version", "01"),
            "variable_changed": None,
            "old_value": None,
            "new_value": None,
            "reasoning": "Strategy within target bounds — no change applied.",
            "avg_pnl": round(avg_pnl, 4),
            "total_pnl": round(total_pnl, 4),
            "n_trades": len(recent),
        }
        append_hypothesis(hypothesis)
        print(json.dumps(hypothesis, indent=2))
        sys.exit(0)

    # save prior version
    save_prior(strategy)  # save BEFORE bumping version in the new file

    # actually we need to save the prior as-is, then mutate
    # reload prior from what we had
    prior_strategy = load_yaml(STRATEGY_FILE)
    save_prior(prior_strategy)

    # bump version
    old_version = strategy.get("version", "01")
    new_version = bump_version(old_version)
    strategy["version"] = new_version

    # write new strategy
    write_strategy(strategy)

    # record hypothesis
    hypothesis = {
        "timestamp": now_iso(),
        "mode": "fallback",
        "version_before": old_version,
        "version_after": new_version,
        "variable_changed": variable_changed,
        "old_value": old_value,
        "new_value": new_value,
        "reasoning": reasoning,
        "avg_pnl": round(avg_pnl, 4),
        "total_pnl": round(total_pnl, 4),
        "n_trades": len(recent),
    }
    append_hypothesis(hypothesis)

    print(f"[REFLECT] Strategy v{old_version} → v{new_version}")
    print(f"  Changed: {variable_changed}  {old_value} → {new_value}")
    print(f"  Reason:  {reasoning}")
    print(json.dumps(hypothesis, indent=2))


def hermes_reflect():
    """
    Production mode: read latest 25 trades + current strategy,
    format as prompt, call `hermes` subprocess, parse hypothesis, apply it.
    """
    goal = load_yaml(GOAL_FILE)
    strategy = load_yaml(STRATEGY_FILE)
    trades = load_trades()

    recent = trades[-25:]

    if len(recent) == 0:
        print("[REFLECT-HERMES] No trades yet — nothing to reflect on.")
        sys.exit(0)

    prompt = f"""You are the reflection engine for a self-improving trading agent.

## Current Strategy
{yaml.dump(strategy, default_flow_style=False)}

## Goal
{yaml.dump(goal, default_flow_style=False)}

## Recent Trades (last {len(recent)})
{json.dumps(recent, indent=2)}

## Your Task
Analyse the outcomes above. Propose exactly ONE change to a single variable in the strategy.

Rules:
- Change exactly ONE variable.
- Predict whether the score will go up or down.
- Give your confidence (0.0–1.0).

Respond in this exact YAML format:
```yaml
variable: <variable name>
old_value: <current value>
new_value: <new value>
prediction: <up|down>
confidence: <0.0-1.0>
reasoning: <one paragraph>
```
"""

    try:
        result = subprocess.run(
            ["hermes", "--no-interactive", "--prompt", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        print("[REFLECT-HERMES] `hermes` command not found. Falling back to deterministic mode.")
        fallback_reflect()
        return
    except subprocess.TimeoutExpired:
        print("[REFLECT-HERMES] Hermes timed out. Falling back to deterministic mode.")
        fallback_reflect()
        return

    output = result.stdout

    # parse YAML from hermes output
    try:
        # find yaml block
        if "```yaml" in output:
            yaml_block = output.split("```yaml")[1].split("```")[0]
        elif "```" in output:
            yaml_block = output.split("```")[1].split("```")[0]
        else:
            yaml_block = output

        hypothesis = yaml.safe_load(yaml_block)
    except Exception as e:
        print(f"[REFLECT-HERMES] Failed to parse Hermes response: {e}")
        print(f"Raw output:\n{output}")
        fallback_reflect()
        return

    # apply the change
    variable = hypothesis.get("variable")
    new_value = hypothesis.get("new_value")

    if not variable:
        print("[REFLECT-HERMES] No variable in hypothesis. Aborting.")
        sys.exit(1)

    # save prior
    save_prior(strategy)

    # navigate dotted path (e.g., "entry.threshold")
    parts = variable.split(".")
    obj = strategy
    for p in parts[:-1]:
        obj = obj.setdefault(p, {})
    old_value = obj.get(parts[-1])
    obj[parts[-1]] = new_value

    # bump version
    old_version = strategy.get("version", "01")
    strategy["version"] = bump_version(old_version)

    write_strategy(strategy)

    hypothesis_record = {
        "timestamp": now_iso(),
        "mode": "hermes",
        "version_before": old_version,
        "version_after": strategy["version"],
        "variable_changed": variable,
        "old_value": old_value,
        "new_value": new_value,
        "prediction": hypothesis.get("prediction"),
        "confidence": hypothesis.get("confidence"),
        "reasoning": hypothesis.get("reasoning"),
        "n_trades": len(recent),
    }
    append_hypothesis(hypothesis_record)

    print(f"[REFLECT-HERMES] Strategy v{old_version} → v{strategy['version']}")
    print(f"  Changed: {variable}  {old_value} → {new_value}")
    print(f"  Confidence: {hypothesis.get('confidence')}")
    print(f"  Reason: {hypothesis.get('reasoning')}")


def main():
    parser = argparse.ArgumentParser(description="Hermes Trading — Reflection Cycle")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fallback", action="store_true", help="Deterministic fallback reflection")
    mode.add_argument("--hermes", action="store_true", help="Hermes-powered reflection")
    args = parser.parse_args()

    if args.fallback:
        fallback_reflect()
    elif args.hermes:
        hermes_reflect()


if __name__ == "__main__":
    main()
