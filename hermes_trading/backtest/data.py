"""
Data loading.

load_csv_panel(): point at a directory of per-symbol OHLCV CSVs
    (columns: time,open,high,low,close,volume) and get back the panel
    dict this engine expects. This is how you plug in real exchange
    data (point-in-time, survivorship-bias-free if you sourced it that
    way -- see the strategy-search prompt for data-source guidance).

generate_synthetic_panel(): produces a multi-symbol panel with regime
    switches (trending / ranging / high-vol) baked in, purely so this
    engine can be demoed and unit-tested without a live data feed. This
    is NOT a substitute for backtesting on real market data -- treat any
    numbers from it as a plumbing check, not a strategy result.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd


def load_csv_panel(directory: str) -> dict[str, pd.DataFrame]:
    panel = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith('.csv'):
            continue
        symbol = os.path.splitext(fname)[0]
        df = pd.read_csv(os.path.join(directory, fname))
        time_col = 'time' if 'time' in df.columns else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()
        required = {'open', 'high', 'low', 'close'}
        missing = required - set(c.lower() for c in df.columns)
        if missing:
            raise ValueError(f"{fname} missing required columns: {missing}")
        panel[symbol] = df
    return panel


def generate_synthetic_panel(symbols: list[str] | None = None, n_bars: int = 900,
                              seed: int = 7, freq: str = 'h') -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    symbols = symbols or [f"SYN{i}/USD" for i in range(1, 9)]
    idx = pd.date_range('2024-01-01', periods=n_bars, freq=freq)
    panel = {}
    for j, sym in enumerate(symbols):
        # stitch together regime segments: trend up, range, trend down, high-vol chop
        segments = []
        remaining = n_bars
        seg_id = 0
        while remaining > 0:
            length = min(remaining, rng.integers(80, 180))
            kind = ['trend_up', 'range', 'trend_down', 'high_vol'][(seg_id + j) % 4]
            segments.append((kind, length))
            remaining -= length
            seg_id += 1

        prices = [100.0 * (1 + 0.05 * j)]
        vols = []
        for kind, length in segments:
            base_vol = {'trend_up': 0.006, 'trend_down': 0.006,
                        'range': 0.004, 'high_vol': 0.02}[kind]
            drift = {'trend_up': 0.0015, 'trend_down': -0.0015,
                     'range': 0.0, 'high_vol': 0.0}[kind]
            for _ in range(length):
                shock = rng.normal(drift, base_vol)
                prices.append(max(prices[-1] * (1 + shock), 0.0001))
                vols.append(base_vol)
        prices = prices[1:n_bars + 1]
        vols = (vols + [vols[-1]] * n_bars)[:n_bars]

        close = np.array(prices[:n_bars])
        high = close * (1 + np.abs(rng.normal(0, 0.002, n_bars)))
        low = close * (1 - np.abs(rng.normal(0, 0.002, n_bars)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = rng.lognormal(mean=8, sigma=0.5, size=n_bars) * (1 + np.array(vols) * 20)

        df = pd.DataFrame({'open': open_, 'high': high, 'low': low,
                            'close': close, 'volume': volume}, index=idx)
        panel[sym] = df
    return panel
