# Hermes-1

Hermes autonomous crypto trading agent. Runs 24/7 on Railway, trades on Kraken, and self-improves via reflection cycles.

## What's inside

- **10 parallel strategies**: 8 original (trend-following, breakout, contrarian-funding, mean-reversion, bollinger, scalper, swing, composite) + 2 ICT (liquidity-sweep + prop eval)
- **ICT strategy**: Inner Circle Trader setup — liquidity sweeps, market-structure shifts, FVG fills
- **MCPT validation**: Monte Carlo Permutation Testing from neurotrader888/mcpt — signals curve-fit vs real edge
- **Kraken integration**: ccxt adapter, KrakenFeeModel (0.26% taker / 0.16% maker), 685-pair full basket
- **xStocks support**: AAPLx, TSLAx, NVDAx, SPYx, QQQx, GOOGLx, GLDx via Kraken Futures API
- **Live execution**: LiveKrakenTrader mirrors ICT prop signals with real orders; 0.5% risk/trade, 3% daily-loss circuit, 6% max drawdown
- **Macro overlay**: Yahoo Finance risk-score + Hyperliquid perps intelligence (OI, funding, volume)
- **Signal alerts**: Fires ENTER / CLOSED / RISK alerts for the Starter Eval account

## Architecture

- `hermes_trading/adapters/` — price, Kraken, Hyperliquid, Yahoo Finance, xStocks, realtime cross-check
- `hermes_trading/backtest/` — walk-forward engine, Kelly sizer, portfolio guardrails, live_execution parity
- `hermes_trading/mcpt_validation.py` — bar-permutation statistical test
- `hermes_trading/ict_strategy.py` — ICT liquidity-sweep entry logic
- `hermes_trading/live_kraken.py` — real-money execution on Kraken
- `hermes_trading/loop.py` — main trading loop (dual portfolio runner, ATR stops, EMA filter, trailing)
- `hermes_trading/signal_alerts.py` — alert system for Starter Eval signals
- `hermes_trading/strategies_test.py` — 10 strategy definitions
- `hermes_trading/test_runner.py` — MultiStrategyRunner harness
- `hermes_trading/run.py` — entry point

## Risk model (Kraken Starter Eval)

| Parameter | Value |
|---|---|
| Profit target | +10% |
| Max daily loss | 3% |
| Max drawdown | 6% |
| Risk per trade | 0.5% |
| Max concurrent positions | 2 |
| Stop loss | 2% ATR |
| Take profit | 4% (R:R 1:2) |
| Fee model | Kraken round-trip 0.52% |

## Deployment

Runs on Railway (`sweet-clarity`). Deploy with:

```bash
railway up --detach
```

State lives under `/app/state/`. Edit state remotely:

```bash
ssh -i ~/.ssh/id_ed25519_railway railway-sweet-clarity
```

## Setup

1. `git clone https://github.com/TheoKosh/Hermes-1.git`
2. `cd Hermes-1 && python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Set Kraken API keys in Railway environment (`KRAKEN_API_KEY`, `KRAKEN_API_SECRET`)

## Disclaimer

This software trades real cryptocurrency on real exchanges. It is provided for educational and research purposes. Trading involves significant risk of loss. Past performance does not guarantee future results. Use at your own risk.
