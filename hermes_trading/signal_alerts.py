"""
signal_alerts.py — Real-time trade signal alert system.

Monitors the ict_prop portfolio and fires alerts when:
  1. A new ENTER signal is generated (execute on Kraken Pro)
  2. A position closes (stop-loss or take-profit hit)
  3. Daily loss limit is approaching
  4. Drawdown is approaching the 6% limit

Alerts are written to:
  - /app/state/signal_alerts.jsonl (append log)
  - Console (visible in Railway logs)
  - Environment webhook URL (if ALERT_WEBHOOK_URL is set)

Usage: runs as a background task inside the main loop, checking every tick.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

console = Console()


class SignalAlertSystem:
    """
    Monitors portfolio state and fires actionable alerts.
    Designed for the Kraken Starter Prop eval (3% daily loss, 6% max DD).
    """

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.alerts_file = state_dir / "signal_alerts.jsonl"
        self.webhook_url = os.environ.get("ALERT_WEBHOOK_URL", "")
        self._last_alerted_signals = {}  # asset -> last signal to avoid duplicates
        self._last_positions = {}  # asset -> position dict (to detect closes)
        self._daily_loss_warned = False
        self._drawdown_warned_5pct = False

    def check_and_alert(self, portfolio_name: str, heartbeat: dict, positions: dict):
        """
        Called every tick with current portfolio state.
        Fires alerts for new signals, position closes, and risk warnings.

        Args:
            portfolio_name: e.g. "ict_prop"
            heartbeat: the heartbeat dict (equity, drawdown, trade_count)
            positions: {asset: position_dict or None}
        """
        ts = datetime.now(timezone.utc).isoformat()
        alerts_fired = []

        # 1. Check for new ENTER signals (position opened)
        for asset, pos in positions.items():
            if pos is not None and self._last_positions.get(asset) is None:
                # New position opened
                alert = self._make_alert(
                    ts, portfolio_name, "ENTER", asset,
                    side=pos.get("side", ""),
                    entry_price=pos.get("entry_price", 0),
                    stop_loss=pos.get("stop_loss", 0),
                    take_profit=pos.get("take_profit", 0),
                    position_value=pos.get("position_value", 0),
                    message=f"🔺 ENTER {asset} {pos.get('side','').upper()} @ ${pos.get('entry_price',0):.4f}\n"
                            f"   Stop: ${pos.get('stop_loss',0):.4f}\n"
                            f"   Target: ${pos.get('take_profit',0):.4f}\n"
                            f"   Size: ${pos.get('position_value',0):.2f}\n"
                            f"   ⚡ Execute on Kraken Pro NOW",
                )
                alerts_fired.append(alert)

        # 2. Check for position closes (position was open, now None)
        for asset, pos in positions.items():
            if pos is None and self._last_positions.get(asset) is not None:
                old_pos = self._last_positions[asset]
                alert = self._make_alert(
                    ts, portfolio_name, "CLOSED", asset,
                    side=old_pos.get("side", ""),
                    message=f"🔻 CLOSED {asset} {old_pos.get('side','').upper()}\n"
                            f"   Position closed — check P&L",
                )
                alerts_fired.append(alert)

        # 3. Risk warnings: daily loss approaching 3%
        daily_loss = heartbeat.get("daily_loss_pct", 0)
        if abs(daily_loss) >= 2.0 and not self._daily_loss_warned:
            alert = self._make_alert(
                ts, portfolio_name, "RISK", "",
                message=f"⚠️ DAILY LOSS WARNING: {daily_loss:+.1f}%\n"
                        f"   Approaching 3% daily limit (Starter Eval)\n"
                        f"   Only {3.0 - abs(daily_loss):.1f}% headroom left",
            )
            alerts_fired.append(alert)
            self._daily_loss_warned = True

        # 4. Risk warnings: drawdown approaching 6%
        drawdown = heartbeat.get("drawdown_pct", 0)
        if drawdown >= 5.0 and not self._drawdown_warned_5pct:
            alert = self._make_alert(
                ts, portfolio_name, "RISK", "",
                message=f"⚠️ DRAWDOWN WARNING: {drawdown:.1f}%\n"
                        f"   Approaching 6% max drawdown (Starter Eval)\n"
                        f"   Only {6.0 - drawdown:.1f}% headroom left",
            )
            alerts_fired.append(alert)
            self._drawdown_warned_5pct = True

        # Reset daily warning at new day
        if daily_loss > -0.5:
            self._daily_loss_warned = False

        # 5. Update tracking
        self._last_positions = dict(positions)

        # Fire all alerts
        for alert in alerts_fired:
            self._fire_alert(alert)

        return alerts_fired

    def _make_alert(self, ts, portfolio, alert_type, asset, **kwargs):
        """Create an alert dict."""
        return {
            "timestamp": ts,
            "portfolio": portfolio,
            "type": alert_type,
            "asset": asset,
            **kwargs,
        }

    def _fire_alert(self, alert: dict):
        """Deliver alert to all channels."""
        msg = alert.get("message", "")

        # Console (Railway logs)
        console.print(f"\n  [bold yellow]{'─' * 50}[/]")
        console.print(f"  [bold yellow]🚨 SIGNAL ALERT[/]")
        for line in msg.split("\n"):
            console.print(f"  [yellow]{line}[/]")
        console.print(f"  [bold yellow]{'─' * 50}[/]\n")

        # Append to alerts log
        with open(self.alerts_file, "a") as f:
            f.write(json.dumps(alert) + "\n")

        # Webhook (if configured)
        if self.webhook_url:
            try:
                import httpx
                httpx.post(self.webhook_url, json=alert, timeout=5)
            except Exception:
                pass  # don't let webhook failure crash the bot
