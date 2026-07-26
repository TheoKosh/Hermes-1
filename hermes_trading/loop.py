"""
loop.py — 24/7 async trading loop (dual-portfolio, multi-asset, live + paper).

Runs two independent sub-portfolios in parallel:

  STEADY (50 EUR): the existing v03 composite-RSI strategy. Scans the full
    429-pair Coinbase basket on rotation, uses multi-factor signals, 3% stop /
    6% target, max 5 concurrent positions. Conservative, slow, patient.

  RAPID (50 EUR): aggressive momentum strategy. Every few minutes, scans
    CoinGecko + Coinbase for the hottest movers, enters the top performers,
    uses tight 1.5% stops with 3% targets for rapid turnover. Max risk per
    trade, up to 8 concurrent positions. Fast in, fast out.

Both portfolios share the same trades.jsonl (tagged by portfolio) and each
writes its own heartbeat and positions file.

MODES:
  HERMES_TRADING_MODE=paper  → virtual positions, simulated P&L (default, safe)
  HERMES_TRADING_MODE=live   → real orders via ccxt, real fills, real money
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console

from .adapters.price import PriceAdapter
from .adapters.onchain import OnchainAdapter
from .adapters.news import NewsAdapter
from .adapters.macro import MacroAdapter

console = Console()
logger = logging.getLogger("hermes-trading.loop")

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
TRADES_FILE = STATE_DIR / "trades.jsonl"

TICK_SECONDS = 60
MAX_CONSECUTIVE_FAILURES = 5

DEFAULT_MAX_CONCURRENT_POSITIONS = 5
DEFAULT_RISK_PER_TRADE_PCT = 2.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_strategy() -> dict:
    if not STRATEGY_FILE.exists():
        return {
            "version": "01",
            "entry": {"indicator": "rsi", "threshold": 35, "direction": "both"},
            "stop_loss_pct": 2.0,
            "take_profit_multiple": 2.0,
            "position_size_r": 0.5,
            "max_concurrent_positions": DEFAULT_MAX_CONCURRENT_POSITIONS,
            "risk_per_trade_pct": DEFAULT_RISK_PER_TRADE_PCT,
        }
    with open(STRATEGY_FILE) as f:
        return yaml.safe_load(f)


async def fetch_with_retry(adapter, max_retries=3, base_delay=1.0):
    """Fetch with exponential backoff. Raises on final failure."""
    for attempt in range(max_retries):
        try:
            return await adapter.fetch()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)


class SubPortfolio:
    """
    A single independent trading portfolio with its own:
      - equity pool and tracking
      - positions dict (in-memory + persisted to disk)
      - strategy parameters
      - heartbeat and positions files
      - trade logging (shared trades.jsonl, tagged by portfolio name)
    """

    def __init__(
        self,
        name: str,
        assets: list,
        strategy: dict,
        starting_equity: float,
        mode: str = "paper",
        goal: dict = None,
    ):
        self.name = name
        self.assets = list(assets)
        self.strategy = strategy
        self.starting_equity = starting_equity
        self.current_equity = starting_equity
        self.peak_equity = starting_equity
        self.mode = mode
        self.goal = goal or {}
        self.trade_count = 0

        # FRAMEWORK: daily loss tracking (3% max daily loss)
        self._daily_start_equity = starting_equity
        self._daily_loss_pct = 0.0
        self._daily_trading_halted = False
        self._current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # FEE MODEL: Kraken transaction fees (cheaper than Coinbase)
        # taker=0.26%, maker=0.16%, round-trip=0.52%
        from .adapters.kraken_adapter import KrakenFeeModel
        self.fees = KrakenFeeModel()  # Tier 0/1: taker 0.26%, maker 0.16%

        # per-portfolio state files
        self.positions_file = STATE_DIR / f"positions_{name}.json"
        self.heartbeat_file = STATE_DIR / f"heartbeat_{name}.json"

        # positions: {asset: position_dict or None}
        self.positions = {a: None for a in self.assets}
        self._cap_logged_this_tick = False

        # adapters (price is essential; context is best-effort)
        self.price_adapters = {}
        self.news_adapters = {}
        self.onchain_adapters = {}
        self._last_macro = {}

        console.print(f"  [{name}] {len(self.assets)} assets, €{starting_equity:.0f} equity, "
                      f"max {strategy.get('max_concurrent_positions', 5)} positions")
        # Print fee model at init so it's visible in logs
        console.print(f"  [{name}] {self.fees.summary()}")

    def add_assets(self, new_assets: list):
        """Dynamically add assets to the basket (for momentum rotation)."""
        for a in new_assets:
            if a not in self.positions:
                self.positions[a] = None
                self.assets.append(a)
                self.price_adapters[a] = PriceAdapter(a)
                self.news_adapters[a] = NewsAdapter(a)
                self.onchain_adapters[a] = OnchainAdapter(a)

    async def init_adapters(self):
        """Create price/context adapters. Called once at startup."""
        for a in self.assets:
            self.price_adapters[a] = PriceAdapter(a)
            self.news_adapters[a] = NewsAdapter(a)
            self.onchain_adapters[a] = OnchainAdapter(a)
        self.macro_adapter = MacroAdapter()

    # ---- position persistence ----

    def _load_positions(self):
        if not self.positions_file.exists():
            return
        try:
            saved = json.loads(self.positions_file.read_text())
            count = 0
            for asset, pos in saved.items():
                if pos is not None:
                    if asset not in self.positions:
                        self.positions[asset] = None
                        self.assets.append(asset)
                        self.price_adapters[asset] = PriceAdapter(asset)
                    self.positions[asset] = pos
                    count += 1
            if count:
                console.print(f"  [{self.name}] ↻ Restored {count} open positions from disk")
        except Exception as e:
            console.print(f"  [{self.name}] ⚠ Could not load positions: {e}")

    def _save_positions(self):
        try:
            open_positions = {a: p for a, p in self.positions.items() if p is not None}
            self.positions_file.write_text(json.dumps(open_positions, indent=2))
        except Exception:
            pass

    # ---- portfolio accounting ----

    def _positions_open(self) -> int:
        return sum(1 for p in self.positions.values() if p is not None)

    def _drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity * 100)
    def _check_drawdown_circuit(self) -> bool:
        """Returns True if drawdown exceeds limit — kills the loop."""
        dd = self._drawdown_pct()
        # Use strategy-specific max_drawdown_pct (default 10%, prop account uses 6%)
        max_dd = self.strategy.get("max_drawdown_pct", None)
        if max_dd is None:
            max_dd = self.goal.get("max_drawdown", 0.10) * 100
        elif max_dd < 1:  # fraction like 0.06 → 6.0
            max_dd = max_dd * 100
        if dd >= max_dd:
            console.print(f"  [{self.name}] 🛑 DRAWDOWN CIRCUIT: {dd:.1f}% >= {max_dd:.1f}%")
            return True
        return False

    def _check_daily_loss_circuit(self) -> bool:
        """Returns True if daily loss exceeds the strategy's limit."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            # Reset for new day
            self._current_date = today
            self._daily_start_equity = self.current_equity
            self._daily_loss_pct = 0.0
            self._daily_trading_halted = False
            return False

        self._daily_loss_pct = ((self.current_equity - self._daily_start_equity)
                                 / max(self._daily_start_equity, 0.01)) * 100

        # Use strategy-specific max_daily_loss_pct (default 3%, prop account uses 3%)
        max_daily_loss = self.strategy.get("max_daily_loss_pct", 3.0)
        if abs(self._daily_loss_pct) >= max_daily_loss:
            self._daily_trading_halted = True
            console.print(f"  [{self.name}] 🛑 DAILY LOSS LIMIT: "
                          f"{self._daily_loss_pct:+.1f}% >= {max_daily_loss}% — halting new entries for today")
            return True
        return False

    def _log_open_positions(self):
        s = self.strategy
        open_count = self._positions_open()
        max_pos = s.get("max_concurrent_positions", DEFAULT_MAX_CONCURRENT_POSITIONS)
        console.print(f"  [{self.name}] ━━━ Open: {open_count}/{max_pos} | "
                      f"equity=€{self.current_equity:.2f} DD={self._drawdown_pct():.1f}% ━━━")
        for asset, pos in self.positions.items():
            if pos is not None:
                opened = pos.get("opened_at", "")
                age_str = ""
                try:
                    opened_dt = datetime.fromisoformat(opened.replace("Z", "+00:00"))
                    age = datetime.now(timezone.utc) - opened_dt
                    mins = int(age.total_seconds() / 60)
                    age_str = f"{mins}m" if mins < 60 else f"{mins // 60}h{mins % 60}m"
                except Exception:
                    age_str = "?"
                console.print(f"    [{self.name}] {asset:12s} {pos['side']:5s} "
                              f"entry={pos['entry_price']:.4f} age={age_str}")

    # ---- trading logic ----

    async def tick(self):
        """Run one evaluation cycle for this portfolio."""
        ts = now_iso()
        s = self.strategy
        max_pos = s.get("max_concurrent_positions", DEFAULT_MAX_CONCURRENT_POSITIONS)
        open_count = self._positions_open()
        self._cap_logged_this_tick = False

        console.print(f"\n  [{self.name}] {ts} — tick (positions={open_count}/{max_pos}, "
                      f"equity=€{self.current_equity:.2f}, DD={self._drawdown_pct():.1f}%)")

        if open_count > 0:
            self._log_open_positions()

        if self._check_drawdown_circuit():
            self._save_positions()
            return  # don't trade, just keep writing heartbeats

        # FRAMEWORK: daily loss circuit (3% max)
        if self._check_daily_loss_circuit():
            console.print(f"  [{self.name}] daily loss limit hit — managing existing positions only")
            # Still manage open positions (exits take priority) but no new entries
            # The _tick_asset method will skip entries when _daily_trading_halted is True

        # macro context (best-effort)
        try:
            self._last_macro = await fetch_with_retry(self.macro_adapter)
        except Exception:
            pass

        # determine scan set
        # If _scan_override is set (e.g. by RapidFirePortfolio), use it
        scan_override = getattr(self, '_scan_override', None)
        if scan_override is not None:
            assets_to_scan = scan_override
            self._scan_override = None  # consume
        elif len(self.assets) > 40:
            assets_with_positions = [a for a in self.assets if self.positions.get(a) is not None]
            batch_size = 35
            scan_candidates = [a for a in self.assets if a not in assets_with_positions]
            offset = getattr(self, '_rotation_offset', 0)
            scan_start = offset % max(1, len(scan_candidates))
            scan_batch = scan_candidates[scan_start:scan_start + batch_size]
            if len(scan_batch) < batch_size:
                scan_batch += scan_candidates[:batch_size - len(scan_batch)]
            self._rotation_offset = offset + batch_size
            assets_to_scan = assets_with_positions + scan_batch
        else:
            assets_to_scan = self.assets

        heartbeat_assets = []
        for asset in assets_to_scan:
            try:
                result = await self._tick_asset(asset, ts)
                heartbeat_assets.append(result)
            except Exception as e:
                console.print(f"  [{self.name}] ✗ {asset}: {e}")
                heartbeat_assets.append({"asset": asset, "error": str(e)})

        self._write_heartbeat(ts, heartbeat_assets)
        self._save_positions()

    async def _tick_asset(self, asset, ts):
        price_data = await fetch_with_retry(self.price_adapters[asset])
        price = price_data.get("price", 0)
        rsi = price_data.get("rsi", 50)

        # context (best-effort)
        news_data = {}
        onchain_data = {}
        try:
            news_data = await fetch_with_retry(self.news_adapters[asset])
        except Exception:
            pass
        try:
            onchain_data = await fetch_with_retry(self.onchain_adapters[asset])
        except Exception:
            pass

        # --- REGIME CLASSIFICATION ---
        # Classify regime before generating any signal — strategies that work
        # in one regime lose money in another.
        from .regime import RegimeClassifier
        classifier = getattr(self, '_regime_classifier', None) or RegimeClassifier()
        self._regime_classifier = classifier

        # --- EMA TREND FILTER DATA ---
        # Extract price data for ATR/EMA (applied after signal computation)
        above_ema = price_data.get("above_ema", None)
        ema20 = price_data.get("ema20", 0)
        atr = price_data.get("atr", 0)
        self._current_atr = atr  # store for _enter_trade

        # Store closes/volumes for regime classifier
        closes = price_data.get("closes", [])
        volumes = price_data.get("volumes", [])
        regime = classifier.classify(closes, volumes, rsi)

        # --- STRUCTURED SIGNAL (V2) ---
        from .signals import composite_signal_v2
        entry = self.strategy.get("entry", {})
        direction_cfg = entry.get("direction", "both")
        signal_weights = self.strategy.get("signal_weights", {})
        entry_threshold = self.strategy.get("entry_threshold", 0.3)

        # --- ICT STRATEGY OVERRIDE ---
        # If use_ict is set, use the ICT signal engine instead of composite RSI
        if self.strategy.get("use_ict"):
            from .ict_strategy import ICTStrategy
            if not hasattr(self, "_ict_engine"):
                self._ict_engine = ICTStrategy()

            # Extract OHLCV from price_data
            all_highs = [c[2] if isinstance(c, list) and len(c) > 4 else price
                         for c in price_data.get("closes", [])]
            all_lows = [c[3] if isinstance(c, list) and len(c) > 4 else price
                        for c in price_data.get("closes", [])]
            all_closes = price_data.get("closes", [price])

            # If we don't have OHLCV arrays, approximate from current price
            if len(all_closes) < 20:
                all_closes = [price] * 20
                all_highs = [price * 1.001] * 20
                all_lows = [price * 0.999] * 20

            ict_sig = self._ict_engine.generate_signal(
                highs=all_highs, lows=all_lows, closes=all_closes,
                volumes=price_data.get("volumes", []),
            )

            signal = ict_sig.signal
            composite = ict_sig.composite
            confidence = ict_sig.confidence

            # Override the decision with ICT output
            decision = {
                "composite_score": composite,
                "signal": signal,
                "confidence": confidence,
                "confidence_label": "ICT: " + ict_sig.reasoning[:40],
                "regime": {"trend": "n/a", "volatility": "n/a", "sub_strategy": "ict"},
                "factors": {"ict": {"sweep": ict_sig.liquidity_swept, "mss": ict_sig.mss_detected,
                                   "fvg": ict_sig.fvg_present, "zone": ict_sig.zone}},
                "hypothesis": f"ICT: {ict_sig.reasoning}",
                "evidence": f"sweep={ict_sig.liquidity_swept} mss={ict_sig.mss_detected} fvg={ict_sig.fvg_present} zone={ict_sig.zone}",
                "falsification": "Invalidated if MSS fails or FVG fills without reversal",
                "reasoning": f"ICT {signal}: {ict_sig.reasoning}",
                "size_multiplier": 1.0,
            }

            reg_str = f"ict/{ict_sig.zone}"
            conf_label = decision["confidence_label"]
        else:
            decision = composite_signal_v2(
                rsi=rsi, direction=direction_cfg, news_data=news_data,
                onchain_data=onchain_data, macro_data=self._last_macro or {},
                weights=signal_weights, entry_threshold=entry_threshold,
                regime=regime,
            )
            signal = decision["signal"]
            composite = decision["composite_score"]
            confidence = decision["confidence"]

            # Structured log line: regime + hypothesis + confidence label
            reg_str = f"{regime.trend}/{regime.volatility}/{regime.sub_strategy}"
            conf_label = decision.get("confidence_label", "")
        if self.strategy.get("use_trend_filter") and above_ema is not None:
            if signal == "long" and not above_ema:
                signal = "flat"
            elif signal == "short" and above_ema:
                signal = "flat"

        # Structured log line: regime + hypothesis + confidence label
        reg_str = f"{regime.trend}/{regime.volatility}/{regime.sub_strategy}"
        conf_label = decision.get("confidence_label", "")
        console.print(f"  [{self.name}] {asset:12s} ${price:.4f} rsi={rsi:.0f} "
                      f"comp={composite:+.2f} [{reg_str}] {conf_label[:25]}")

        # manage open position first (exits take priority, dedup guard)
        if self.positions.get(asset) is not None:
            pos = self.positions[asset]
            self._manage_position(asset, price, pos)
            if self.positions.get(asset) is not None:
                pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
                if pos["side"] == "short":
                    pnl_pct = -pnl_pct
                console.print(f"  [{self.name}]     holding {pos['side']} @ {pnl_pct:+.2f}%")
                return {"asset": asset, "price": price, "rsi": rsi, "in_position": True,
                        "side": pos["side"], "entry_price": pos["entry_price"],
                        "unrealized_pnl_pct": round(pnl_pct, 2),
                        "composite": composite, "signal": signal, "confidence": confidence}
            else:
                return {"asset": asset, "price": price, "rsi": rsi, "in_position": False,
                        "composite": composite, "signal": signal, "confidence": confidence}

        # check position cap
        max_pos = self.strategy.get("max_concurrent_positions", DEFAULT_MAX_CONCURRENT_POSITIONS)
        if self._positions_open() >= max_pos:
            if not self._cap_logged_this_tick:
                console.print(f"  [{self.name}]     (cap {self._positions_open()}/{max_pos})")
                self._cap_logged_this_tick = True
            return {"asset": asset, "price": price, "rsi": rsi, "in_position": False,
                    "composite": composite, "signal": signal}

        # FRAMEWORK: no new entries when daily loss limit hit
        if getattr(self, '_daily_trading_halted', False):
            console.print(f"  [{self.name}]     (daily loss halt — no new entries)")
            return {"asset": asset, "price": price, "rsi": rsi, "in_position": False,
                    "composite": composite, "signal": signal, "skipped": "daily_loss_halt"}

        # enter
        if signal in ("long", "short"):
            self._enter_trade(asset, signal, price, ts)
            # Store thesis + falsification in position for logging on close
            if self.positions.get(asset):
                self.positions[asset]["thesis"] = decision.get("hypothesis", "")
                self.positions[asset]["falsification"] = decision.get("falsification", "")
                self.positions[asset]["regime"] = reg_str
            console.print(f"  [{self.name}]     ▲ ENTER {signal} (comp {composite:+.2f} conf {confidence:.0%} [{conf_label[:30]}])")

        return {"asset": asset, "price": price, "rsi": rsi,
                "in_position": self.positions.get(asset) is not None,
                "composite": composite, "signal": signal, "confidence": confidence}

    def _enter_trade(self, asset, side, price, ts):
        s = self.strategy
        stop_loss_pct = s.get("stop_loss_pct", 2.0)
        tp_multiple = s.get("take_profit_multiple", 2.0)
        size_r = s.get("position_size_r", 0.5)
        risk_pct = s.get("risk_per_trade_pct", DEFAULT_RISK_PER_TRADE_PCT)

        # --- ATR-ADAPTIVE STOPS ---
        # Use ATR from price data if available, fall back to fixed %
        atr = getattr(self, '_current_atr', 0)
        if s.get("use_atr_stops") and atr > 0:
            atr_mult = s.get("atr_multiplier", 1.5)
            # ATR is in absolute price terms; convert to percentage
            stop_distance_pct = atr * atr_mult / price * 100 if price > 0 else stop_loss_pct
            # Clamp to reasonable range (1%-8%)
            stop_loss_pct = max(1.0, min(8.0, stop_distance_pct))
        else:
            atr_mult = s.get("atr_multiplier", 1.5)
            stop_distance_pct = stop_loss_pct

        # --- FEE-AWARE VIABILITY GATE ---
        take_profit_pct = stop_loss_pct * tp_multiple
        rr_check = self.fees.effective_rr(stop_loss_pct, take_profit_pct)
        if not rr_check["viable"]:
            console.print(f"  [{self.name}] ✗ REJECT {asset}: R:R after fees = "
                          f"{rr_check['net_rr']:.2f} < 1.0 (round-trip cost {rr_check['round_trip_cost_pct']:.1f}%)")
            return

        if side == "long":
            stop = price * (1 - stop_loss_pct / 100)
            target = price * (1 + stop_loss_pct * tp_multiple / 100)
        else:
            stop = price * (1 + stop_loss_pct / 100)
            target = price * (1 - stop_loss_pct * tp_multiple / 100)

        stop_distance_pct = abs(price - stop) / price * 100
        # Base position sizing: risk_pct of equity / stop distance
        position_value = (self.current_equity * risk_pct / 100) / (stop_distance_pct / 100) if stop_distance_pct > 0 else 0

        # --- FRACTIONAL KELLY SIZING ---
        # Kelly fraction: f* = (b*p - q) / b, where p = win rate, q = 1-p, b = R:R
        # Use Quarter-Kelly (f* / 4) for safety with correlated crypto positions.
        # Requires 50+ closed trades before trusting the estimate — otherwise
        # fall back to the fixed risk_pct sizing above.
        kelly_adjustment = 1.0  # default: no adjustment (use full risk_pct)
        kelly_note = ""

        closed_trades = self._get_closed_trade_stats()
        if closed_trades["n_trades"] >= 50:
            p = closed_trades["win_rate"]
            b = tp_multiple  # reward:risk ratio
            q = 1.0 - p
            if b > 0 and p > 0:
                kelly_f = (b * p - q) / b
                quarter_kelly = kelly_f / 4.0
                # Map Kelly fraction to a position size adjustment (0.25x to 1.0x)
                kelly_adjustment = max(0.25, min(1.0, quarter_kelly / 0.15))
                kelly_note = f" Kelly: p={p:.2f} b={b:.1f} f*={kelly_f:.3f} QK={quarter_kelly:.3f} size_adj={kelly_adjustment:.2f}"
            elif p == 0:
                kelly_adjustment = 0.25
                kelly_note = f" Kelly: 0% win rate → min size (0.25x)"
        else:
            kelly_note = f" Kelly: {closed_trades['n_trades']}/50 trades → using fixed risk_pct"

        position_value *= kelly_adjustment

        # Cap at size_r fraction of equity (no leverage beyond allocation)
        max_position = self.current_equity * size_r
        position_value = min(position_value, max_position)
        base_amount = position_value / price if price > 0 else 0

        self.positions[asset] = {
            "side": side, "entry_price": price, "stop_loss": stop,
            "take_profit": target, "size_r": size_r, "base_amount": base_amount,
            "position_value": position_value, "opened_at": ts,
            "kelly_adjustment": kelly_adjustment,
            "atr": atr,
            "initial_stop_pct": stop_loss_pct,
            "highest_price_since_entry": price if side == "long" else price,
            "lowest_price_since_entry": price if side == "short" else price,
        }

        console.print(f"  [{self.name}] ▲ ENTER {asset} {side} @ {price:.4f} "
                      f"stop={stop:.4f} target={target:.4f} size=EUR{position_value:.2f}{kelly_note}")
        self._save_positions()

    def _get_closed_trade_stats(self) -> dict:
        """Compute win rate and avg R:R from closed trades for Kelly sizing."""
        try:
            trades = []
            for line in open(TRADES_FILE):
                t = json.loads(line.strip())
                if t.get("portfolio") == self.name and "pnl_dollar" in t:
                    trades.append(t)

            if not trades:
                return {"n_trades": 0, "win_rate": 0.5, "avg_rr": 2.0}

            wins = [t for t in trades if t.get("pnl_dollar", 0) > 0]
            win_rate = len(wins) / len(trades)
            return {"n_trades": len(trades), "win_rate": win_rate, "avg_rr": 2.0}
        except Exception:
            return {"n_trades": 0, "win_rate": 0.5, "avg_rr": 2.0}

    def _manage_position(self, asset, price, pos):
        entry = pos["entry_price"]
        stop = pos["stop_loss"]
        target = pos["take_profit"]
        side = pos["side"]

        # --- TRAILING STOP UPDATE ---
        # Track highest/lowest price since entry and trail the stop
        if self.strategy.get("use_trailing_stop"):
            trail_mult = self.strategy.get("trail_atr_multiple", 1.0)
            atr = pos.get("atr", 0)
            activation_pct = pos.get("initial_stop_pct", 2.5) * 1.5

            if side == "long":
                pos["highest_price_since_entry"] = max(
                    pos.get("highest_price_since_entry", entry), price)
                pnl_pct = (price - entry) / entry * 100
                # Activate trailing after price moves +1.5x stop distance
                if pnl_pct >= activation_pct and atr > 0:
                    trail_stop = price - atr * trail_mult
                    # Only move stop up, never down
                    if trail_stop > stop:
                        old_stop = stop
                        stop = trail_stop
                        pos["stop_loss"] = stop
                        if old_stop != stop:
                            console.print(f"  [{self.name}]    ↗ trailing stop {asset}: "
                                          f"{old_stop:.4f} -> {stop:.4f}")
            elif side == "short":
                pos["lowest_price_since_entry"] = min(
                    pos.get("lowest_price_since_entry", entry), price)
                pnl_pct = (entry - price) / entry * 100
                if pnl_pct >= activation_pct and atr > 0:
                    trail_stop = price + atr * trail_mult
                    # Only move stop down, never up
                    if trail_stop < stop:
                        old_stop = stop
                        stop = trail_stop
                        pos["stop_loss"] = stop
                        if old_stop != stop:
                            console.print(f"  [{self.name}]    ↘ trailing stop {asset}: "
                                          f"{old_stop:.4f} -> {stop:.4f}")

        hit_stop = False
        hit_target = False

        if side == "long":
            if price <= stop:
                hit_stop = True
            elif price >= target:
                hit_target = True
        else:
            if price >= stop:
                hit_stop = True
            elif price <= target:
                hit_target = True

        if hit_stop:
            self._close_trade(asset, price, "stop_loss", pos)
            return
        if hit_target:
            self._close_trade(asset, price, "take_profit", pos)
            return

    def _close_trade(self, asset, exit_price, reason, pos):
        entry = pos["entry_price"]
        side = pos["side"]
        pnl_pct = ((exit_price - entry) / entry * 100) if side == "long" else ((entry - exit_price) / entry * 100)

        # FEE DEDUCTION: subtract round-trip Coinbase fees from realized P&L
        position_value = pos.get("position_value", 0)
        entry_fee = self.fees.compute_fee_cost(position_value, "taker")
        exit_fee = self.fees.compute_fee_cost(position_value, "taker")
        total_fees = entry_fee + exit_fee

        pnl_dollar_gross = pnl_pct / 100 * position_value
        pnl_dollar = pnl_dollar_gross - total_fees  # net of fees
        ts = now_iso()

        self.current_equity += pnl_dollar
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        trade = {
            "timestamp": ts, "portfolio": self.name, "asset": asset, "side": side,
            "entry_price": entry, "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 4), "pnl_dollar": round(pnl_dollar, 2),
            "reason": reason, "stop_loss": pos["stop_loss"],
            "take_profit": pos.get("take_profit"), "size_r": pos["size_r"],
            "base_amount": pos.get("base_amount", 0),
            "position_value": pos.get("position_value", 0),
            "opened_at": pos["opened_at"], "version": self.strategy.get("version", "?"),
            "mode": self.mode,
            "kelly_adjustment": pos.get("kelly_adjustment", 1.0),
            "thesis": pos.get("thesis", ""),
            "falsification": pos.get("falsification", ""),
            "regime": pos.get("regime", ""),
            "fees_paid": round(total_fees, 4),
            "pnl_gross_dollar": round(pnl_dollar_gross, 2),
            "pnl_net_dollar": round(pnl_dollar, 2),
        }

        with open(TRADES_FILE, "a") as f:
            f.write(json.dumps(trade) + "\n")

        color = "green" if pnl_dollar >= 0 else "red"
        console.print(f"  [{self.name}] [{color}]▼ CLOSE {asset} {side} {reason} "
                      f"@ {exit_price:.4f} pnl={pnl_pct:+.2f}% (gross €{pnl_dollar_gross:+.2f} "
                      f"- fees €{total_fees:.2f} = net €{pnl_dollar:+.2f})[/]")
        self.positions[asset] = None
        self.trade_count += 1
        self._save_positions()

    def _write_heartbeat(self, ts, heartbeat_assets):
        hb = {
            "timestamp": ts, "portfolio": self.name, "mode": self.mode,
            "basket": self.assets, "assets": heartbeat_assets,
            "trade_count": self.trade_count, "equity": round(self.current_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round(self._drawdown_pct(), 2),
        }
        with open(self.heartbeat_file, "w") as f:
            json.dump(hb, f, indent=2)

    async def close_adapters(self):
        for adapter in self.price_adapters.values():
            try:
                await adapter.close()
            except Exception:
                pass


class RapidFirePortfolio(SubPortfolio):
    """
    Aggressive momentum portfolio: scans CoinGecko for hottest movers,
    enters the top performers on Coinbase, uses tight stops for rapid turnover.
    """

    MOMENTUM_SCAN_INTERVAL = 3  # re-scan CoinGecko every 3 ticks (~3 min) — faster refresh
    MAX_MOMENTUM_ASSETS = 25    # keep basket trimmed to top movers

    async def init_adapters(self):
        await super().init_adapters()
        from .momentum import MomentumScanner
        from .adapters.realtime import RealtimePriceChecker
        from .adapters.hyperliquid import HyperliquidAdapter
        self.scanner = MomentumScanner()
        self.price_checker = RealtimePriceChecker()
        self.hyperliquid = HyperliquidAdapter()
        self._momentum_tick_counter = 0
        self._hot_assets = []
        self._hl_markets = {}  # cached HL market data

    async def _tick_asset(self, asset, ts):
        """Override: add real-time price verification and liquidity filter
        before allowing an entry. Existing positions are managed normally."""
        from .signals import composite_signal
        from .adapters.price import PriceAdapter

        price_data = await fetch_with_retry(self.price_adapters[asset])
        price = price_data.get("price", 0)
        rsi = price_data.get("rsi", 50)

        # context (best-effort)
        news_data = {}
        onchain_data = {}
        try:
            news_data = await fetch_with_retry(self.news_adapters[asset])
        except Exception:
            pass
        try:
            onchain_data = await fetch_with_retry(self.onchain_adapters[asset])
        except Exception:
            pass

        # signal
        entry = self.strategy.get("entry", {})
        direction_cfg = entry.get("direction", "both")
        signal_weights = self.strategy.get("signal_weights", {})
        entry_threshold = self.strategy.get("entry_threshold", 0.3)

        decision = composite_signal(
            rsi=rsi, direction=direction_cfg, news_data=news_data,
            onchain_data=onchain_data, macro_data=self._last_macro or {},
            weights=signal_weights, entry_threshold=entry_threshold,
        )
        signal = decision["signal"]
        composite = decision["composite_score"]
        confidence = decision["confidence"]

        # --- HYPERLIQUID SIGNAL BOOST ---
        # Blend RSI composite with Hyperliquid funding + momentum data.
        # HL provides real perps market intelligence: funding rates (contrarian),
        # open interest (conviction), and 24h momentum — all from the same DEX
        # that Invo runs on. Only boosts when HL confidence is meaningful.
        hl = self.hyperliquid.compute_signal(asset.split("/")[0], price)
        hl_score = hl.get("hl_score", 0)
        hl_conf = hl.get("hl_confidence", 0)
        blend_weight = 0.30 * hl_conf
        blended = composite * (1 - blend_weight) + hl_score * blend_weight

        if blended >= entry_threshold:
            signal = "long"
        elif blended <= -entry_threshold:
            signal = "short"
        else:
            signal = "flat"

        composite = round(blended, 3)  # update for downstream logging

        hl_reason = hl.get("reasoning", "")

        # manage open position FIRST (exits take priority, dedup guard)
        if self.positions.get(asset) is not None:
            pos = self.positions[asset]
            self._manage_position(asset, price, pos)
            if self.positions.get(asset) is not None:
                pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
                if pos["side"] == "short":
                    pnl_pct = -pnl_pct
                console.print(f"  [{self.name}]     holding {pos['side']} @ {pnl_pct:+.2f}%")
                return {"asset": asset, "price": price, "rsi": rsi, "in_position": True,
                        "side": pos["side"], "entry_price": pos["entry_price"],
                        "unrealized_pnl_pct": round(pnl_pct, 2),
                        "composite": composite, "signal": signal, "confidence": confidence}
            else:
                return {"asset": asset, "price": price, "rsi": rsi, "in_position": False,
                        "composite": composite, "signal": signal, "confidence": confidence}

        # check position cap
        max_pos = self.strategy.get("max_concurrent_positions", DEFAULT_MAX_CONCURRENT_POSITIONS)
        if self._positions_open() >= max_pos:
            if not self._cap_logged_this_tick:
                console.print(f"  [{self.name}]     (cap {self._positions_open()}/{max_pos})")
                self._cap_logged_this_tick = True
            return {"asset": asset, "price": price, "rsi": rsi, "in_position": False,
                    "composite": composite, "signal": signal}

        # FRAMEWORK: no new entries when daily loss limit hit
        if getattr(self, '_daily_trading_halted', False):
            console.print(f"  [{self.name}]     (daily loss halt — no new entries)")
            return {"asset": asset, "price": price, "rsi": rsi, "in_position": False,
                    "composite": composite, "signal": signal, "skipped": "daily_loss_halt"}

        # Only enter if signal is strong enough
        if signal not in ("long", "short"):
            console.print(f"  [{self.name}] {asset:12s} ${price:.4f} rsi={rsi:.0f} "
                          f"comp={composite:+.2f} hl={hl_score:+.2f}({hl_conf:.0%}) "
                          f"| hl:{hl_reason[:40]}")
            return {"asset": asset, "price": price, "rsi": rsi,
                    "in_position": self.positions.get(asset) is not None,
                    "composite": composite, "signal": signal, "confidence": confidence}

        # PRICE VERIFICATION GATE: check CoinGecko before entering
        verification = await self.price_checker.verify_price(asset, price)
        if not verification.get("verified", True):
            console.print(f"  [{self.name}] {asset:12s} ⚠ PRICE MISMATCH: "
                          f"Coinbase={price:.4f} CoinGecko={verification.get('coingecko_price'):.4f} "
                          f"(deviation {verification.get('deviation_pct', 0):.1f}%) — SKIP")
            return {"asset": asset, "price": price, "rsi": rsi, "in_position": False,
                    "composite": composite, "signal": signal, "skipped": "price_mismatch"}

        if not verification.get("liquid_enough", True):
            console.print(f"  [{self.name}] {asset:12s} ⚠ ILLIQUID: "
                          f"24h vol=${verification.get('volume_24h_usd', 0)/1e6:.1f}M — SKIP")
            return {"asset": asset, "price": price, "rsi": rsi, "in_position": False,
                    "composite": composite, "signal": signal, "skipped": "illiquid"}

        # Entry approved with verified price + sufficient liquidity
        trend = verification.get("trend", "flat")
        vol_m = verification.get("volume_24h_usd", 0) / 1e6
        console.print(f"  [{self.name}] {asset:12s} ${price:.4f} rsi={rsi:.0f} "
                      f"comp={composite:+.2f} hl={hl_score:+.2f}({hl_conf:.0%}) "
                      f"| CG vol=${vol_m:.0f}M trend={trend} | hl:{hl_reason[:30]}")
        self._enter_trade(asset, signal, price, ts)
        console.print(f"  [{self.name}]     ▲ ENTER {signal} (blend {composite:+.2f} conf {confidence:.0%})")

        return {"asset": asset, "price": price, "rsi": rsi,
                "in_position": self.positions.get(asset) is not None,
                "composite": composite, "signal": signal, "confidence": confidence}

    async def refresh_momentum(self):
        """Scan CoinGecko + Coinbase for hottest movers, update basket."""
        console.print(f"  [{self.name}] 🔥 Scanning CoinGecko for hot movers...")
        try:
            result = await self.scanner.scan_top_performers(min_rank=30, limit=self.MAX_MOMENTUM_ASSETS)
            tradable = result.get("tradable_24h", [])
            # Build Coinbase pair symbols
            new_assets = []
            for coin in tradable:
                sym = coin.get("symbol", "")
                pair = f"{sym}/USD"
                new_assets.append(pair)
            if new_assets:
                # Keep existing positions, add new hot assets, trim cold ones
                held = [a for a in self.assets if self.positions.get(a) is not None]
                fresh = [a for a in new_assets if a not in held]
                # New basket: held positions + fresh hot assets (trimmed)
                new_basket = held + fresh[:self.MAX_MOMENTUM_ASSETS - len(held)]
                # Add any new assets we don't have adapters for yet
                to_add = [a for a in new_basket if a not in self.positions]
                if to_add:
                    self.add_assets(to_add)
                console.print(f"  [{self.name}] 🔥 {len(new_basket)} hot assets ready "
                              f"(held={len(held)}, new={len(to_add)})")
                self._hot_assets = new_basket
        except Exception as e:
            console.print(f"  [{self.name}] ⚠ Momentum scan failed: {e}")

    async def tick(self):
        """Override: refresh momentum basket periodically, refresh Hyperliquid
        data each tick, then scan."""
        self._momentum_tick_counter += 1
        if self._momentum_tick_counter % self.MOMENTUM_SCAN_INTERVAL == 0 or not self._hot_assets:
            await self.refresh_momentum()

        # Refresh Hyperliquid market data each tick (one API call for all markets)
        try:
            self._hl_markets = await self.hyperliquid.fetch_all_markets()
        except Exception:
            pass  # keep last cached data

        # Only scan hot assets + held positions (not the full 429 list)
        held = [a for a in self.assets if self.positions.get(a) is not None]
        scan_list = held + [a for a in self._hot_assets if a not in held]
        if scan_list:
            # temporarily set assets to scan list for parent tick()
            # but keep all assets in the dict for position management
            original_assets = self.assets
            self._scan_override = scan_list
            await super().tick()
            self.assets = original_assets
        else:
            await super().tick()


class DualPortfolioRunner:
    """
    Runs two sub-portfolios (steady + rapid) in parallel, each with its own
    equity, strategy, positions, and heartbeat. Shares a single trades.jsonl.
    """

    def __init__(self, goal: dict, mode: str = "paper", total_equity: float = 100.0):
        self.goal = goal
        self.mode = mode
        self.total_equity = total_equity
        self.consecutive_failures = 0

        # Split: 50/50
        steady_equity = total_equity * 0.5
        rapid_equity = total_equity * 0.5

        # Load full Coinbase basket for steady portfolio
        basket_file = STATE_DIR / "full_basket.json"
        if basket_file.exists():
            full_basket = json.loads(basket_file.read_text())
        else:
            full_basket = ["BTC/USD", "ETH/USD", "SOL/USD"]

        # Steady strategy: EVIDENCE-BASED OPTIMIZED
        # - 0.35 threshold (fewer, better entries — steady had 100% win rate)
        # - ATR-adaptive stops (not fixed %)
        # - Trailing stop to lock gains
        # - EMA trend filter: only long above EMA, short below
        steady_strategy = {
            "version": "steady-v6-atr",
            "entry": {"indicator": "composite", "direction": "both"},
            "entry_threshold": 0.35,
            "signal_weights": {"rsi": 0.35, "sentiment": 0.25, "onchain": 0.10, "macro": 0.30},
            "stop_loss_pct": 2.5,          # fallback if ATR unavailable
            "use_atr_stops": True,         # ATR-adaptive stops
            "atr_multiplier": 1.5,         # stop = 1.5x ATR
            "take_profit_multiple": 2.5,   # target = 2.5x stop (R:R 2.5)
            "position_size_r": 0.05,
            "max_concurrent_positions": 3,
            "risk_per_trade_pct": 0.75,
            "use_trend_filter": True,      # EMA20 filter
            "use_trailing_stop": True,     # trail after +1.5x ATR
            "trail_atr_multiple": 1.0,     # trail at 1x ATR behind
        }

        # Rapid strategy: EVIDENCE-BASED OPTIMIZED
        # - Same ATR-adaptive stops as steady (proven to work)
        # - Faster tick (15s) for rapid reaction
        # - EMA trend filter to stop catching falling knives
        # - Trailing stop to let winners run
        rapid_strategy = {
            "version": "rapid-v6-atr",
            "entry": {"indicator": "composite", "direction": "both"},
            "entry_threshold": 0.35,
            "signal_weights": {"rsi": 0.35, "sentiment": 0.25, "onchain": 0.10, "macro": 0.30},
            "stop_loss_pct": 2.5,
            "use_atr_stops": True,
            "atr_multiplier": 1.5,
            "take_profit_multiple": 2.5,
            "position_size_r": 0.05,
            "max_concurrent_positions": 3,
            "risk_per_trade_pct": 0.75,
            "use_trend_filter": True,
            "use_trailing_stop": True,
            "trail_atr_multiple": 1.0,
            "tick_seconds": 15,
        }

        self.steady = SubPortfolio(
            name="steady", assets=full_basket, strategy=steady_strategy,
            starting_equity=steady_equity, mode=mode, goal=goal,
        )

        self.rapid = RapidFirePortfolio(
            name="rapid", assets=[], strategy=rapid_strategy,
            starting_equity=rapid_equity, mode=mode, goal=goal,
        )

    async def run(self):
        console.print("[bold cyan]═══ Hermes Dual Portfolio — started ═══[/]")
        console.print(f"  Total equity: €{self.total_equity:.0f} (steady=€{self.steady.starting_equity:.0f} "
                      f"+ rapid=€{self.rapid.starting_equity:.0f})")
        console.print(f"  Mode: {self.mode}")
        console.print()

        # Initialize adapters and load persisted positions
        await self.steady.init_adapters()
        self.steady._load_positions()
        await self.rapid.init_adapters()
        self.rapid._load_positions()

        # Initialize Yahoo Finance research database
        from .adapters.yfinance_research import YFinanceResearch
        self.research = YFinanceResearch()
        await self._refresh_research()

        # Migrate old positions if this is first boot of dual system
        await self._migrate_legacy_positions()

        self._tick_count = 0

        while True:
            try:
                # Refresh Yahoo Finance research every 10 ticks (~2.5 min at 15s ticks)
                if self._tick_count % 10 == 0:
                    await self._refresh_research()

                # Run both portfolios in parallel each tick
                await asyncio.gather(
                    self.steady.tick(),
                    self.rapid.tick(),
                )
                self.consecutive_failures = 0
                self._tick_count += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.consecutive_failures += 1
                console.print(f"[red]✗ Dual tick error ({self.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}[/]")
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    console.print("[bold red]🛑 CIRCUIT BREAKER — halting.[/]")
                    break
            # Use the rapid portfolio's faster tick if configured
            tick = self.rapid.strategy.get("tick_seconds", TICK_SECONDS)
            await asyncio.sleep(tick)

    async def _migrate_legacy_positions(self):
        """On first boot of dual system, migrate positions from old positions.json to steady."""
        legacy = STATE_DIR / "positions.json"
        steady_pos = STATE_DIR / "positions_steady.json"
        if legacy.exists() and not steady_pos.exists():
            console.print("  [steady] 📦 Migrating legacy positions.json → positions_steady.json")
            import shutil
            shutil.copy2(legacy, steady_pos)
            self.steady._load_positions()

    async def _refresh_research(self):
        """Refresh Yahoo Finance data and compute macro risk sentiment.
        Shared across both portfolios as macro context."""
        try:
            await self.research.fetch_all()
            sentiment = self.research.compute_risk_sentiment()
            self._risk_sentiment = sentiment

            score = sentiment.get("risk_score", 0)
            vix = sentiment.get("vix_level", 0)
            trend = sentiment.get("equity_trend", "?")
            dxy = sentiment.get("dxy_trend", "?")
            note = sentiment.get("correlation_note", "")

            console.print(f"  [research] Yahoo Finance: risk_score={score:+.2f} "
                          f"VIX={vix:.0f} equities={trend} DXY={dxy} | {note[:50]}")

            # Inject risk sentiment into both portfolios' macro context
            # Positive risk_score → boost longs, negative → boost shorts
            self.steady._last_macro = {**self.steady._last_macro, "risk_sentiment": score}
            self.rapid._last_macro = {**self.rapid._last_macro, "risk_sentiment": score}

        except Exception as e:
            console.print(f"  [research] ⚠ YFinance refresh failed: {e}")

    async def close_all(self):
        await self.steady.close_adapters()
        await self.rapid.close_adapters()


# Backward compat: keep TradingLoop as alias for DualPortfolioRunner
# so old run.py imports still work
TradingLoop = DualPortfolioRunner
