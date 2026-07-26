"""
historical_data.py — Fetch historical OHLCV data from Coinbase for backtesting.

Pulls 15m candles for a basket of assets using ccxt, saves to CSV per symbol,
compatible with the backtest framework's load_csv_panel().
"""
from pathlib import Path

import pandas as pd


async def fetch_ohlcv_panel(symbols: list, timeframe: str = "15m",
                            limit: int = 1000, data_dir: str = None) -> dict:
    """
    Fetch OHLCV data for multiple symbols from Coinbase.
    Returns {symbol: DataFrame} panel dict.

    Args:
        symbols: list of trading pairs (e.g. ["BTC/USD", "ETH/USD"])
        timeframe: candle interval (15m, 1h, 4h, 1d)
        limit: number of candles to fetch (max 1000 per call on Coinbase)
        data_dir: if set, cache CSVs here for reuse
    """
    import ccxt.async_support as ccxt

    panel = {}
    exchange = ccxt.coinbase({"enableRateLimit": True})

    try:
        for symbol in symbols:
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if not ohlcv:
                    print(f"  {symbol}: no data")
                    continue

                df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df = df.set_index("time").sort_index()

                # Cache to CSV
                if data_dir:
                    Path(data_dir).mkdir(parents=True, exist_ok=True)
                    safe_name = symbol.replace("/", "_")
                    df.to_csv(f"{data_dir}/{safe_name}.csv")

                panel[symbol] = df
                print(f"  {symbol}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

            except Exception as e:
                print(f"  {symbol}: {e}")

    finally:
        await exchange.close()

    return panel


def load_cached_panel(data_dir: str) -> dict:
    """Load cached CSVs from a directory into a panel dict."""
    from .backtest.data import load_csv_panel
    return load_csv_panel(data_dir)
