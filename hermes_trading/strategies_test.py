"""
strategies_test.py — Strategy definitions for the trial phase.

Each strategy is a distinct, well-documented approach. All run in parallel
with identical starting capital so results are directly comparable.

Strategy IDs:
  1. mean_reversion    — RSI extremes in range-bound markets
  2. trend_following   — EMA20 direction + momentum
  3. breakout          — New highs/lows with volume confirmation
  4. bollinger         — Band extremes mean-reversion
  5. contrarian_funding — Fade extreme Hyperliquid funding rates
  6. composite_conservative — Multi-factor (current proven approach)
  7. scalper           — Tight stops, high frequency
  8. swing             — Wide stops, patient targets
"""

# All strategies share these framework-compliant defaults:
# - max_concurrent_positions: 3
# - risk_per_trade_pct: 0.75
# - position_size_r: 0.05
# - use_atr_stops: True (where applicable)
# - use_trend_filter: True (where applicable)
# - use_trailing_stop: True (where applicable)

BASE = {
    "max_concurrent_positions": 3,
    "risk_per_trade_pct": 0.75,
    "position_size_r": 0.05,
    "entry": {"indicator": "composite", "direction": "both"},
}

TEST_STRATEGIES = [
    {
        "id": "mean_reversion",
        "name": "Mean Reversion (RSI) — LIVE TRACKING",
        "description": "Classic RSI oversold/overbought. OOS Sharpe=-2.50. Reduced budget for live tracking.",
        "budget": 25.0,
        "strategy": {
            **BASE,
            "version": "test-mr-v1",
            "entry_threshold": 0.30,
            "signal_weights": {"rsi": 0.80, "sentiment": 0.10, "onchain": 0.0, "macro": 0.10},
            "stop_loss_pct": 2.5,
            "take_profit_multiple": 2.0,
            "use_atr_stops": False,
            "use_trend_filter": False,
            "use_trailing_stop": False,
            "tick_seconds": 30,
        },
    },
    {
        "id": "trend_following",
        "name": "Trend Following (EMA) ⭐ PROMISING",
        "description": "Follows EMA20 direction. OOS Sharpe=1.76, Win=40%. Most reliable promising strategy (20 OOS trades). Full budget.",
        "budget": 25.0,
        "strategy": {
            **BASE,
            "version": "test-tf-v1",
            "entry_threshold": 0.20,
            "signal_weights": {"rsi": 0.30, "sentiment": 0.10, "onchain": 0.0, "macro": 0.60},
            "stop_loss_pct": 3.0,
            "take_profit_multiple": 3.0,
            "use_atr_stops": True,
            "atr_multiplier": 2.0,
            "use_trend_filter": True,
            "use_trailing_stop": True,
            "trail_atr_multiple": 1.5,
            "tick_seconds": 30,
        },
    },
    {
        "id": "breakout",
        "name": "Breakout (Vol + Price) ⭐ PROMISING",
        "description": "Enters on momentum breakouts. OOS Sharpe=2.08, Win=45.5%. Full budget. Only 11 OOS trades — provisional.",
        "budget": 25.0,
        "strategy": {
            **BASE,
            "version": "test-bo-v1",
            "entry_threshold": 0.35,
            "signal_weights": {"rsi": 0.20, "sentiment": 0.30, "onchain": 0.0, "macro": 0.50},
            "stop_loss_pct": 3.5,
            "take_profit_multiple": 3.0,
            "use_atr_stops": True,
            "atr_multiplier": 1.5,
            "use_trend_filter": True,
            "use_trailing_stop": True,
            "trail_atr_multiple": 1.0,
            "tick_seconds": 30,
        },
    },
    {
        "id": "bollinger",
        "name": "Bollinger Band Reversion — REDUCED BUDGET",
        "description": "Mean-reversion at Bollinger extremes. OOS Sharpe=-1.08. Reduced budget.",
        "budget": 25.0,
        "strategy": {
            **BASE,
            "version": "test-bb-v1",
            "entry_threshold": 0.25,
            "signal_weights": {"rsi": 0.60, "sentiment": 0.20, "onchain": 0.0, "macro": 0.20},
            "stop_loss_pct": 2.0,
            "take_profit_multiple": 1.5,
            "use_atr_stops": False,
            "use_trend_filter": False,
            "use_trailing_stop": False,
            "tick_seconds": 30,
        },
    },
    {
        "id": "contrarian_funding",
        "name": "Contrarian (Funding Rate) ⭐ PROMISING",
        "description": "Fades extreme Hyperliquid funding rates. OOS Sharpe=4.62 (flagged suspicious >3), Win=63.6%. Full budget but only 11 trades — treat with caution.",
        "budget": 25.0,
        "strategy": {
            **BASE,
            "version": "test-cf-v1",
            "entry_threshold": 0.15,
            "signal_weights": {"rsi": 0.20, "sentiment": 0.10, "onchain": 0.0, "macro": 0.70},
            "stop_loss_pct": 4.0,
            "take_profit_multiple": 2.5,
            "use_atr_stops": True,
            "atr_multiplier": 2.0,
            "use_trend_filter": False,
            "use_trailing_stop": True,
            "trail_atr_multiple": 1.5,
            "tick_seconds": 30,
        },
    },
    {
        "id": "composite_conservative",
        "name": "Composite Conservative — REDUCED BUDGET",
        "description": "The steady strategy. OOS Sharpe=-0.86 in backtest. Reduced budget — live results were better than backtest.",
        "budget": 25.0,
        "strategy": {
            **BASE,
            "version": "test-cc-v1",
            "entry_threshold": 0.35,
            "signal_weights": {"rsi": 0.35, "sentiment": 0.25, "onchain": 0.10, "macro": 0.30},
            "stop_loss_pct": 2.5,
            "take_profit_multiple": 2.5,
            "use_atr_stops": True,
            "atr_multiplier": 1.5,
            "use_trend_filter": True,
            "use_trailing_stop": True,
            "trail_atr_multiple": 1.0,
            "tick_seconds": 30,
        },
    },
    {
        "id": "scalper",
        "name": "Scalper (Tight) — REDUCED BUDGET",
        "description": "Fast in/out with tight stops. OOS Sharpe=-0.50. Reduced budget.",
        "budget": 25.0,
        "strategy": {
            **BASE,
            "version": "test-sc-v1",
            "entry_threshold": 0.25,
            "signal_weights": {"rsi": 0.50, "sentiment": 0.20, "onchain": 0.0, "macro": 0.30},
            "stop_loss_pct": 1.5,
            "take_profit_multiple": 2.0,
            "use_atr_stops": False,
            "use_trend_filter": True,
            "use_trailing_stop": False,
            "tick_seconds": 15,
        },
    },
    {
        "id": "swing",
        "name": "Swing (Patient) — REDUCED BUDGET",
        "description": "Wide stops, patient targets. OOS Sharpe=-2.08. Reduced budget.",
        "budget": 25.0,
        "strategy": {
            **BASE,
            "version": "test-sw-v1",
            "entry_threshold": 0.40,
            "signal_weights": {"rsi": 0.30, "sentiment": 0.30, "onchain": 0.10, "macro": 0.30},
            "stop_loss_pct": 5.0,
            "take_profit_multiple": 3.0,
            "use_atr_stops": True,
            "atr_multiplier": 2.5,
            "use_trend_filter": True,
            "use_trailing_stop": True,
            "trail_atr_multiple": 2.0,
            "tick_seconds": 60,
        },
    },
    {
        "id": "ict_liquidity",
        "name": "ICT Liquidity Sweep ⭐ NEW",
        "description": "ICT strategy: liquidity sweep + MSS + FVG + premium/discount. 15m timeframe. Adapted from YouTube mentorship playlist.",
        "budget": 10000.0,
        "strategy": {
            **BASE,
            "version": "ict-v1",
            "entry_threshold": 0.20,
            "signal_weights": {"rsi": 0.15, "sentiment": 0.15, "onchain": 0.0, "macro": 0.70},
            "stop_loss_pct": 2.5,
            "take_profit_multiple": 2.5,
            "use_atr_stops": True,
            "atr_multiplier": 1.5,
            "use_trend_filter": False,
            "use_trailing_stop": True,
            "trail_atr_multiple": 1.0,
            "tick_seconds": 30,
            "use_ict": True,
            "max_daily_loss_pct": 3.0,
            "max_drawdown_pct": 6.0,
        },
    },
    {
        "id": "ict_prop",
        "name": "ICT Prop Account (3% DD) ⭐ NEW",
        "description": "ICT strategy configured for Kraken prop account. Max 3% daily loss, 6% max drawdown, conservative sizing. Built for prop firm evaluation.",
        "budget": 10000.0,
        "strategy": {
            **BASE,
            "version": "ict-prop-v1",
            "entry_threshold": 0.25,
            "signal_weights": {"rsi": 0.10, "sentiment": 0.10, "onchain": 0.0, "macro": 0.80},
            "stop_loss_pct": 1.5,
            "take_profit_multiple": 2.0,
            "use_atr_stops": True,
            "atr_multiplier": 1.0,
            "use_trend_filter": False,
            "use_trailing_stop": True,
            "trail_atr_multiple": 0.5,
            "tick_seconds": 30,
            "use_ict": True,
            "max_daily_loss_pct": 3.0,
            "max_drawdown_pct": 6.0,
            "risk_per_trade_pct": 0.5,
            "position_size_r": 0.03,
            "max_concurrent_positions": 2,
        },
    },
]
