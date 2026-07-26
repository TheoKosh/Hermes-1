"""openbb_mcp_adapter.py — Bridge to the OpenBB Workspace MCP server.

Connects to the user's OpenBB MCP backend (https://backend.openbb.co/mcp)
via streamable HTTP with the configured Bearer token, and exposes a small,
typed interface over the widgets that are useful for the trading system:

  - eod_price(symbol, start, end)        -> daily OHLCV + VWAP
  - company_profile(symbol)              -> sector / industry / market cap
  - key_metrics(symbol)                  -> P/E, P/B, ROE, etc.
  - valuation_multiples(symbol)          -> valuation snapshot
  - price_target(symbol)                 -> analyst targets
  - pyth_price_feeds(symbol)             -> crypto / tokenized price feeds

This is a thin wrapper over `get_widget_data(origin, widget_id, data_args)`.
It requires the openbb MCP server to be reachable and the token valid; all
calls degrade gracefully (return {} / []) on failure so the trading loop
never blocks on market-data outages.

NOT auto-wired into the live loop — call explicitly from research/analysis
flows (e.g. xStocks screening, earnings-driven re-ratings). Crypto execution
data still flows through adapters/kraken_adapter.py and adapters/price.py.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

# Lazily imported — mcp is an optional dependency
try:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    _MCP_AVAILABLE = True
except Exception:
    _MCP_AVAILABLE = False

# Connection config (read from env so secrets stay out of source)
_DEFAULT_URL = "https://backend.openbb.co/mcp"
_DEFAULT_ORIGIN = "OpenBB Sandbox"

# Module-level session cache — the MCP session is expensive to set up,
# so we reuse it across calls within a process. Keyed by the running event
# loop because streamablehttp_client's anyio task group is loop-bound and
# cannot be reused from a different loop (the common case is a long-lived
# worker loop; the per-call asyncio.run case reconnects each time, which is
# acceptable for short-lived scripts).
_SESSION: dict = {"by_loop": {}}


def _token() -> str:
    return os.environ.get("OPENBB_MCP_TOKEN", "")


def _url() -> str:
    return os.environ.get("OPENBB_MCP_URL", _DEFAULT_URL)


async def _get_session() -> "ClientSession":
    """Lazily connect and cache the MCP session, keyed by the current event
    loop. A long-running worker reuses one session; short-lived scripts that
    call via asyncio.run get a fresh session each run."""
    if not _MCP_AVAILABLE:
        raise RuntimeError("mcp package not installed (pip install mcp)")
    loop = asyncio.get_running_loop()
    cached = _SESSION["by_loop"].get(id(loop))
    if cached is not None:
        return cached
    token = _token()
    if not token:
        raise RuntimeError("OPENBB_MCP_TOKEN env var not set")
    headers = {"Authorization": f"Bearer {token}"}
    ctx = streamablehttp_client(_url(), headers=headers)
    read, write, _ = await ctx.__aenter__()
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()
    _SESSION["by_loop"][id(loop)] = session
    # keep ctx alive with the session so the transport isn't GC'd
    setattr(session, "_openbb_ctx", ctx)
    return session


async def _call_widget(widget_id: str, data_args: dict | None = None) -> list:
    """Call get_widget_data and return the unwrapped content rows.
    Defensive: only keeps dict rows (some widgets return strings/None)."""
    session = await _get_session()
    r = await session.call_tool("get_widget_data", {
        "origin": _DEFAULT_ORIGIN,
        "widget_id": widget_id,
        "data_args": data_args or {},
    })
    for block in (r.content or []):
        txt = getattr(block, "text", "") or ""
        if not txt:
            continue
        try:
            payload = json.loads(txt)
        except json.JSONDecodeError:
            continue
        if not payload.get("ok"):
            continue
        # Shape: data[].items[].content[]
        data = payload.get("data") or []
        rows: list = []
        for group in data:
            if not isinstance(group, dict):
                continue
            for item in (group.get("items") or []):
                if not isinstance(item, dict):
                    continue
                for row in (item.get("content") or []):
                    if isinstance(row, dict):
                        rows.append(row)
        return rows
    return []


# ----------------------------------------------------------------------------
# Public, typed accessors over the most useful widgets
# ----------------------------------------------------------------------------

async def eod_price(symbol: str, start_date: str, end_date: str | None = None) -> list[dict]:
    """Daily OHLCV + VWAP for an equity or ETF symbol (e.g. AAPL, SPY, QQQ)."""
    args = {"symbol": symbol, "start_date": start_date}
    if end_date:
        args["end_date"] = end_date
    try:
        return await _call_widget("eod_price", args)
    except Exception:
        return []


async def company_profile(symbol: str) -> dict:
    """Sector, industry, market cap, IPO date, etc."""
    try:
        rows = await _call_widget("company_profile", {"symbol": symbol})
        return rows[0] if rows else {}
    except Exception:
        return {}


async def key_metrics(symbol: str) -> list[dict]:
    """P/E, P/B, ROE, debt/equity, etc."""
    try:
        return await _call_widget("key_metrics", {"symbol": symbol})
    except Exception:
        return []


async def valuation_multiples(symbol: str) -> list[dict]:
    try:
        return await _call_widget("valuation_multiples", {"symbol": symbol})
    except Exception:
        return []


async def price_target(symbol: str) -> list[dict]:
    """Analyst price targets (mean/median/high/low, consensus)."""
    try:
        return await _call_widget("price_target", {"symbol": symbol})
    except Exception:
        return []


async def list_widgets() -> list[dict]:
    """Full widget catalog — useful for discovery."""
    try:
        session = await _get_session()
        r = await session.call_tool("list_available_widgets", {})
        for b in (r.content or []):
            txt = getattr(b, "text", "") or ""
            if txt:
                payload = json.loads(txt)
                return (payload.get("data") or {}).get("widgets", [])
    except Exception:
        pass
    return []


async def close():
    """No-op kept for API symmetry.

    The MCP streamable_http transport uses an anyio task group that cannot be
    exited from a different task than it was entered in — so we deliberately
    do NOT tear down the session here. In a long-running worker the session
    lives for the process lifetime (the connection is reused). The OS reaps
    everything on shutdown.

    Call this only if you are certain you're on the same task that created
    the session (rare); otherwise leave it.
    """
    return
