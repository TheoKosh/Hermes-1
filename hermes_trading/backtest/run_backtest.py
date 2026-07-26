"""
CLI entry point.

Usage:
    python run_backtest.py                       # synthetic demo panel
    python run_backtest.py --data /path/to/csvs   # your own OHLCV CSVs
    python run_backtest.py --n-trials 250         # deflate Sharpe for a
                                                   # search that tested
                                                   # 250 candidate configs
"""
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hermes_backtest.data import load_csv_panel, generate_synthetic_panel
from hermes_backtest.backtester import run_walkforward


def fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2%}"


def fmt_num(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def print_metrics_block(title, metrics):
    print(f"\n--- {title} ---")
    print(f"  Total return:       {fmt_pct(metrics['total_return'])}")
    print(f"  Annualized return:  {fmt_pct(metrics['annualized_return'])}")
    print(f"  Sharpe:             {fmt_num(metrics['sharpe'])}")
    print(f"  Sortino:            {fmt_num(metrics['sortino'])}")
    print(f"  Max drawdown:       {fmt_pct(metrics['max_drawdown'])}")
    print(f"  Trades:             {metrics['n_trades']}")
    print(f"  Win rate:           {fmt_pct(metrics['win_rate'])}")
    print(f"  Profit factor:      {fmt_num(metrics['profit_factor'])}")
    print(f"  Deflated Sharpe P(real edge > 0): {fmt_pct(metrics['deflated_sharpe_prob'])}")
    print("  Bias/overfitting checklist:")
    for flag in metrics['overfitting_flags']:
        print(f"    - {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default=None,
                     help="directory of per-symbol OHLCV CSVs; omit to use synthetic demo data")
    ap.add_argument('--split', type=float, default=0.7, help="in-sample fraction for walk-forward split")
    ap.add_argument('--n-trials', type=int, default=1,
                     help="number of candidate configs tested in this search run, "
                          "for Deflated Sharpe Ratio correction")
    ap.add_argument('--starting-equity', type=float, default=10_000)
    ap.add_argument('--max-positions', type=int, default=8)
    ap.add_argument('--max-drawdown', type=float, default=0.10)
    ap.add_argument('--kelly-fraction', type=float, default=0.25)
    ap.add_argument('--out', type=str, default='backtest_report')
    args = ap.parse_args()

    if args.data:
        panel = load_csv_panel(args.data)
        print(f"Loaded {len(panel)} symbols from {args.data}")
    else:
        panel = generate_synthetic_panel()
        print(f"No --data given: using {len(panel)}-symbol SYNTHETIC demo panel "
              f"(regime-switching, for engine verification only -- not a real strategy result).")

    result = run_walkforward(
        panel, split_frac=args.split,
        starting_equity=args.starting_equity,
        max_positions=args.max_positions,
        max_drawdown_pct=args.max_drawdown,
        kelly_fraction=args.kelly_fraction,
    )

    if result['survivorship_warning']:
        print(f"\n[!] {result['survivorship_warning']}")

    is_metrics = result['in_sample']['metrics']
    oos_metrics = result['out_of_sample']['metrics']
    is_metrics['deflated_sharpe_prob'] = __import__('hermes_backtest.metrics', fromlist=['deflated_sharpe_ratio']) \
        .deflated_sharpe_ratio(is_metrics['sharpe'], args.n_trials, result['in_sample']['equity_curve'].shape[0])

    print_metrics_block("IN-SAMPLE", is_metrics)
    print_metrics_block("OUT-OF-SAMPLE", oos_metrics)

    deg = result['sharpe_degradation_pct']
    print(f"\nSharpe degradation IS -> OOS: {fmt_pct(deg)}")
    if deg is not None and not np.isnan(deg) and deg > 0.5:
        print("  [!] OOS Sharpe is less than half of in-sample Sharpe -- classic overfitting "
              "signature. Do not deploy this configuration as-is.")
    elif deg is not None and not np.isnan(deg) and deg < 0:
        print("  Note: OOS Sharpe exceeded in-sample Sharpe. Not itself a problem, but with "
              "small OOS samples this can just be noise -- don't over-interpret a single run.")

    # equity curve chart
    fig, ax = plt.subplots(figsize=(10, 5))
    is_eq = result['in_sample']['equity_curve']
    oos_eq = result['out_of_sample']['equity_curve']
    if len(is_eq):
        ax.plot(is_eq.index, is_eq.values, label='In-sample', color='#2563eb')
    if len(oos_eq):
        ax.plot(oos_eq.index, oos_eq.values, label='Out-of-sample', color='#dc2626')
    ax.set_title('Hermes Backtest Equity Curve')
    ax.set_ylabel('Equity')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}_equity_curve.png", dpi=150)
    print(f"\nSaved equity curve chart -> {args.out}_equity_curve.png")

    # trade log
    all_trades = result['in_sample']['trades'] + result['out_of_sample']['trades']
    import pandas as pd
    if all_trades:
        pd.DataFrame(all_trades).to_csv(f"{args.out}_trades.csv", index=False)
        print(f"Saved trade log -> {args.out}_trades.csv")
    else:
        print("No trades were generated in this run.")

    with open(f"{args.out}_metrics.json", 'w') as f:
        json.dump({
            'in_sample': {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                          for k, v in is_metrics.items() if k != 'overfitting_flags'},
            'out_of_sample': {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                               for k, v in oos_metrics.items() if k != 'overfitting_flags'},
            'sharpe_degradation_pct': deg,
        }, f, indent=2, default=str)
    print(f"Saved metrics JSON -> {args.out}_metrics.json")


if __name__ == '__main__':
    main()
