"""
Historical Data Provider & Cache
Downloads multi-year daily history for symbols and VIX,
computes technical indicators and volatility metrics, and caches locally to parquet/csv.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf


class HistoricalDataFeed:
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_historical_dataset(
        self,
        symbol: str,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieves processed historical DataFrame with:
        ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'hv_30', 'hv_60',
         'iv_proxy', 'iv_rank', 'ema_20', 'ema_50', 'ema_200', 'atr_14']
        """
        cache_file = self.cache_dir / f"{symbol.upper()}_daily.csv"

        if not force_refresh and cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                return df

        # Fetch underlying and VIX
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No historical data found for {symbol}")

        # Ensure datetime index is tz-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Log returns
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Realized Historical Volatility (annualized)
        df['hv_30'] = df['returns'].rolling(30).std() * np.sqrt(252)
        df['hv_60'] = df['returns'].rolling(60).std() * np.sqrt(252)

        # Fetch VIX to blend into IV proxy
        try:
            vix = yf.Ticker("^VIX").history(start=start_date, end=end_date)
            if vix.index.tz is not None:
                vix.index = vix.index.tz_localize(None)
            df['vix'] = vix['Close'].reindex(df.index).ffill().bfill()
        except Exception:
            df['vix'] = 18.0

        # Beta / Vol multiplier relative to SPY
        is_etf = symbol.upper() in ("SPY", "QQQ", "IWM", "DIA")
        vol_multiplier = 1.0 if is_etf else 1.35

        # IV Proxy: Blends VIX with stock's historical volatility (accounting for Volatility Risk Premium)
        df['iv_proxy'] = np.maximum(df['hv_30'] * 1.15, (df['vix'] / 100.0) * vol_multiplier)
        df['iv_proxy'] = df['iv_proxy'].ffill().bfill()

        # Rolling 252-day IV Rank
        roll_min = df['iv_proxy'].rolling(252).min()
        roll_max = df['iv_proxy'].rolling(252).max()
        df['iv_rank'] = np.clip((df['iv_proxy'] - roll_min) / (roll_max - roll_min + 1e-6) * 100.0, 0.0, 100.0)
        df['iv_rank'] = df['iv_rank'].fillna(50.0)

        # Moving Averages
        df['ema_20'] = df['Close'].ewm(span=20).mean()
        df['ema_50'] = df['Close'].ewm(span=50).mean()
        df['ema_200'] = df['Close'].ewm(span=200).mean()

        # ATR 14
        tr = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                (df['High'] - df['Close'].shift(1)).abs(),
                (df['Low'] - df['Close'].shift(1)).abs()
            )
        )
        df['atr_14'] = tr.rolling(14).mean()

        df = df.dropna(subset=['hv_30', 'ema_50']).copy()
        df.to_csv(cache_file)
        return df