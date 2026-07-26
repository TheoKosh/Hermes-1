"""
Demo: streams synthetic bars tick-by-tick through RegimeAwareStrategy /
PaperExchange, proving the live-execution layer (live_execution.py) runs
end-to-end and obeys the same guardrails as the vectorized backtester.

Run: python -m hermes_backtest.demo_live
"""
from decimal import Decimal
from hermes_backtest.data import generate_synthetic_panel
from hermes_backtest.sizing import KellySizer, PortfolioGuardrails
from hermes_backtest.live_execution import (
    RegimeAwareStrategy, PaperAccount, PaperExchange, LeverageManager,
)


def main():
    panel = generate_synthetic_panel(n_bars=400)
    symbols = list(panel.keys())

    guard = PortfolioGuardrails(max_drawdown_pct=0.10, max_positions=8)
    sizer = KellySizer(kelly_fraction=0.25)
    lev_mgr = LeverageManager(max_leverage=2, guard=guard)
    strategy = RegimeAwareStrategy(symbols, guard, sizer, leverage_mgr=lev_mgr)

    account = PaperAccount(Decimal(10_000))
    exchange = PaperExchange(account, guard=guard, sizer=sizer)

    all_times = sorted(set().union(*[df.index for df in panel.values()]))
    n_orders = 0
    for t in all_times:
        for sym in symbols:
            df = panel[sym]
            if t not in df.index:
                continue
            row = df.loc[t]
            bar = {'open': float(row['open']), 'high': float(row['high']),
                   'low': float(row['low']), 'close': float(row['close']),
                   'volume': float(row.get('volume', 0))}

            # check liquidation before evaluating new signals
            exchange.check_liquidations(sym, Decimal(str(bar['low'])), Decimal(str(bar['high'])))

            orders = strategy.on_tick(sym, t, bar, account)
            for order in orders:
                pos = account.get_position(sym)
                leverage = pos.leverage if pos else lev_mgr.allowed_leverage(float(account.balance()))
                exchange.market_order(order.sym, order.signed_qty, Decimal(str(bar['close'])), leverage)
                n_orders += 1

    print(f"Processed {len(all_times)} bars x {len(symbols)} symbols tick-by-tick.")
    print(f"Orders executed: {n_orders}")
    print(f"Final balance: {account.balance():.2f}")
    print(f"Trades recorded in KellySizer: {len(sizer.trade_history)}")
    print(f"Drawdown breaker tripped: {guard.halted}")
    open_syms = [s for s in symbols if account.get_position(s) is not None]
    print(f"Open positions at end: {open_syms}")


if __name__ == '__main__':
    main()
