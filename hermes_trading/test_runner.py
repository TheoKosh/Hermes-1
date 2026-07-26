"""
test_runner.py — Multi-strategy parallel test harness.

Runs N strategies simultaneously with identical EUR100 budgets,
scanning the same asset basket. Each strategy gets its own SubPortfolio
with isolated equity, positions, and heartbeat. Results are collected
for comparison.

Output: heartbeat_test_{id}.json per strategy, shared trades.jsonl tagged
with portfolio=strategy_id.
"""
import asyncio
import hashlib
import json
import os
from rich.console import Console
from rich.table import Table

from .loop import SubPortfolio, STATE_DIR, TRADES_FILE, TICK_SECONDS, MAX_CONSECUTIVE_FAILURES
from .adapters.hyperliquid import HyperliquidAdapter
from .adapters.yfinance_research import YFinanceResearch

console = Console()


class MultiStrategyRunner:
    """
    Runs multiple strategy portfolios in parallel for A/B testing.
    Each gets identical starting capital and the same asset basket.
    """

    def __init__(self, goal: dict, mode: str = "paper", budget_per_strategy: float = 100.0):
        self.goal = goal
        self.mode = mode
        self.budget = budget_per_strategy
        self.consecutive_failures = 0
        self.portfolios = {}
        self._tick_count = 0

        # Shared adapters (market data is the same for all strategies)
        self.hyperliquid = HyperliquidAdapter()
        self.research = YFinanceResearch()
        self._hl_markets = {}
        self._risk_sentiment = {}

        # Kraken adapter for cross-exchange price verification
        from .adapters.kraken_adapter import KrakenAdapter
        self.kraken = KrakenAdapter(
            api_key=os.environ.get("KRAKEN_API_KEY", ""),
            api_secret=os.environ.get("KRAKEN_API_SECRET", ""),
            mode="paper",
        )

        # xStocks adapter for tokenized equities (AAPLx, TSLAx, NVDAx, etc.)
        from .adapters.xstocks_adapter import XStocksAdapter
        self.xstocks = XStocksAdapter()
        self._xstock_prices = {}

        # Signal alert system for Kraken Prop eval
        from .signal_alerts import SignalAlertSystem
        self.alerts = SignalAlertSystem(STATE_DIR)

        # Live Kraken trader — executes REAL trades with real money.
        # Only enabled when the worker runs in live mode AND the operator
        # has explicitly accepted risk. Paper mode never places real orders.
        from .live_kraken import LiveKrakenTrader
        live_enabled = (
            mode == "live"
            and os.environ.get("HERMES_TRADING_I_ACCEPT_RISK", "false").lower() == "true"
        )
        self.live_trader = LiveKrakenTrader(STATE_DIR, enabled=live_enabled)
        profile = os.environ.get("HERMES_LIVE_RISK_PROFILE", "small_capital")
        self.live_trader.set_risk_profile(profile)
        if self.live_trader.enabled:
            console.print(f"  [bold red]LIVE EXECUTION ARMED[/] — profile={profile}")
        else:
            console.print(f"  [dim]Live execution disabled (mode={mode}) — paper only[/]")
        self._live_tick_counter = 0

        # Load strategies
        from .strategies_test import TEST_STRATEGIES
        self.test_strategies = TEST_STRATEGIES

        # Load asset basket — uses ALL available Kraken pairs (688 pairs)
        basket_file = STATE_DIR / "full_basket.json"
        if basket_file.exists():
            self.basket = json.loads(basket_file.read_text())
            console.print(f"  Loaded full Kraken basket: {len(self.basket)} pairs")
        else:
            self.basket = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD",
                           "LINK/USD", "ADA/USD", "XRP/USD"]
            console.print(f"  [yellow]Warning: using fallback basket ({len(self.basket)} pairs)[/]")

        # Add xStocks (tokenized equities) to the basket
        from .adapters.xstocks_adapter import XSTOCK_NAMES
        self.xstock_symbols = list(XSTOCK_NAMES.keys())
        # xStocks use their own price adapter, not ccxt
        # They're added as supplementary tradable assets
        console.print(f"  xStocks available: {len(self.xstock_symbols)} ({', '.join(self.xstock_symbols[:5])}...)")

        console.print(f"[bold cyan]═══ Hermes Multi-Strategy Test Harness — {len(self.test_strategies)} strategies ═══[/]")
        console.print(f"  Budget per strategy: EUR {budget_per_strategy:.0f}")
        console.print(f"  Total test capital:  EUR {budget_per_strategy * len(self.test_strategies):.0f}")
        console.print(f"  Basket: {len(self.basket)} pairs")
        console.print(f"  Mode: {mode}")

    async def run(self):
        # Initialize all portfolios
        for ts_def in self.test_strategies:
            sid = ts_def["id"]
            strat = ts_def["strategy"]
            budget = ts_def.get("budget", self.budget)

            # Use a rotating subset of the basket for each strategy to avoid
            # rate limits (8 strategies x 429 pairs = too many API calls)
            # Each strategy gets 50 pairs, rotated
            portfolio_basket = self._get_strategy_basket(sid)

            p = SubPortfolio(
                name=sid,
                assets=portfolio_basket,
                strategy=strat,
                starting_equity=budget,
                mode=self.mode,
                goal=self.goal,
            )
            await p.init_adapters()
            p._load_positions()
            self.portfolios[sid] = p

            console.print(f"  [{sid}] {ts_def['name']} — {len(portfolio_basket)} pairs, EUR {budget:.0f}")

        console.print()

        # Main loop
        while True:
            try:
                # Refresh shared market data every 10 ticks
                if self._tick_count % 10 == 0:
                    await self._refresh_shared_data()

                # Run all portfolios in parallel
                tasks = [p.tick() for p in self.portfolios.values()]
                await asyncio.gather(*tasks, return_exceptions=True)

                self.consecutive_failures = 0
                self._tick_count += 1

                # Fire signal alerts for ict_prop portfolio (Kraken Starter Eval)
                prop = self.portfolios.get("ict_prop")
                if prop:
                    # Build heartbeat-like dict for alert system
                    prop_hb = {
                        "drawdown_pct": prop._drawdown_pct(),
                        "daily_loss_pct": prop._daily_loss_pct,
                        "equity": prop.current_equity,
                    }
                    self.alerts.check_and_alert("ict_prop", prop_hb, prop.positions.copy())

                    # Execute REAL trades mirroring ict_prop signals.
                    # No-op unless live execution is armed (see __init__).
                    self._live_tick_counter += 1
                    if self.live_trader.enabled and self._live_tick_counter % 2 == 0:
                        # Paper positions use "entry_price"; the live trader expects
                        # "price". Reading the wrong key silently produced price=0
                        # and every mirrored entry was skipped.
                        live_signals = {}
                        for asset, pos in prop.positions.items():
                            if not pos:
                                continue
                            entry_px = pos.get("entry_price") or pos.get("entry") or 0
                            if entry_px <= 0:
                                continue
                            live_signals[asset] = {
                                "signal": pos.get("side", "flat"),
                                "price": float(entry_px),
                            }
                        try:
                            await self.live_trader.tick(live_signals)
                        except Exception as e:
                            console.print(f"[red]✗ live trader error: {e}[/]")

                # Print comparison table every 20 ticks (~10 min at 30s)
                if self._tick_count % 20 == 0:
                    self._print_comparison()

            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.consecutive_failures += 1
                console.print(f"[red]✗ Multi-strategy tick error ({self.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}[/]")
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    console.print("[bold red]🛑 CIRCUIT BREAKER — halting.[/]")
                    break

            # Use the fastest tick from any strategy
            min_tick = min(
                p.strategy.get("tick_seconds", TICK_SECONDS)
                for p in self.portfolios.values()
            )
            await asyncio.sleep(min_tick)

    async def shutdown(self):
        """Close every exchange session so aiohttp connectors don't leak."""
        closers = [
            ("kraken", getattr(self, "kraken", None)),
            ("xstocks", getattr(self, "xstocks", None)),
            ("hyperliquid", getattr(self, "hyperliquid", None)),
            ("live_trader", getattr(self, "live_trader", None)),
        ]
        for name, obj in closers:
            close = getattr(obj, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                console.print(f"  [shutdown] {name}: {e}")

        for p in self.portfolios.values():
            close = getattr(getattr(p, "price_adapter", None), "close", None)
            if close is None:
                continue
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass
        console.print("  [shutdown] exchange sessions closed")

    def _get_strategy_basket(self, strategy_id: str) -> list:
        """Assign a rotating subset of the basket to each strategy.
        50 pairs per strategy, offset by strategy index to maximize coverage."""
        # NOTE: uses a stable hash — Python's builtin hash() is randomized per
        # process for str, so baskets would shuffle on every restart.
        idx = int(hashlib.md5(strategy_id.encode()).hexdigest(), 16) % max(1, len(self.basket))
        subset_size = min(50, len(self.basket))
        rotated = self.basket[idx:] + self.basket[:idx]
        return rotated[:subset_size]

    async def _refresh_shared_data(self):
        """Refresh Hyperliquid + Yahoo Finance + Kraken data, inject into all portfolios."""
        try:
            self._hl_markets = await self.hyperliquid.fetch_all_markets()
        except Exception:
            pass

        try:
            await self.research.fetch_all()
            sentiment = self.research.compute_risk_sentiment()
            self._risk_sentiment = sentiment
            score = sentiment.get("risk_score", 0)
            console.print(f"  [shared] YF risk={score:+.2f} VIX={sentiment.get('vix_level',0):.0f} "
                          f"| HL: {len(self._hl_markets)} markets")
        except Exception:
            pass

        # Kraken price verification for top assets (every 10 ticks)
        try:
            kraken_prices = {}
            for symbol in ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]:
                ticker = await self.kraken.fetch_ticker(symbol)
                kraken_prices[symbol] = ticker["price"]
            if kraken_prices:
                console.print(f"  [shared] Kraken: BTC=${kraken_prices.get('BTC/USD',0):.0f} "
                              f"ETH=${kraken_prices.get('ETH/USD',0):.0f} "
                              f"({len(kraken_prices)} pairs verified)")
            self._kraken_prices = kraken_prices
        except Exception as e:
            console.print(f"  [shared] Kraken: {e}")
            self._kraken_prices = {}

        # xStocks price refresh (tokenized equities)
        try:
            self._xstock_prices = await self.xstocks.fetch_all_tickers()
            if self._xstock_prices:
                aapl = self._xstock_prices.get("AAPLx", {}).get("price", 0)
                tsla = self._xstock_prices.get("TSLAx", {}).get("price", 0)
                spy = self._xstock_prices.get("SPYx", {}).get("price", 0)
                console.print(f"  [shared] xStocks: AAPL=${aapl:.0f} TSLA=${tsla:.0f} "
                              f"SPY=${spy:.0f} ({len(self._xstock_prices)} equities)")
        except Exception as e:
            console.print(f"  [shared] xStocks: {e}")

        # Inject into all portfolios
        for p in self.portfolios.values():
            p._last_macro = {**p._last_macro, "risk_sentiment": score} if self._risk_sentiment else p._last_macro

    def _print_comparison(self):
        """Print a comparison table of all strategy results."""
        table = Table(title=f"Strategy Comparison (tick {self._tick_count})", show_lines=True)
        table.add_column("Strategy", style="cyan")
        table.add_column("Equity", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("Win%", justify="right")
        table.add_column("Open", justify="right")
        table.add_column("DD%", justify="right")

        # Read trade stats from trades.jsonl
        all_trades = []
        try:
            with open(TRADES_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # skip corrupt line, keep the rest
        except (OSError, FileNotFoundError):
            all_trades = []

        for ts_def in self.test_strategies:
            sid = ts_def["id"]
            p = self.portfolios.get(sid)
            if not p:
                continue

            eq = p.current_equity
            ret = (eq - self.budget) / self.budget * 100
            tc = p.trade_count

            # Win rate from trades
            strat_trades = [t for t in all_trades if t.get("portfolio") == sid]
            wins = sum(1 for t in strat_trades if t.get("pnl_dollar", 0) > 0)
            wr = wins / max(1, len(strat_trades)) * 100
            open_pos = p._positions_open()
            dd = p._drawdown_pct()

            color = "green" if ret >= 0 else "red"
            table.add_row(
                sid,
                f"EUR {eq:.2f}",
                f"[{color}]{ret:+.1f}%[/]",
                str(tc),
                f"{wr:.0f}%",
                f"{open_pos}/{p.strategy.get('max_concurrent_positions', 3)}",
                f"{dd:.1f}%",
            )

        console.print(table)
