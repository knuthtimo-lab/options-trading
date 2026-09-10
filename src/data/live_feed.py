"""
Live Options Data Feed
Pulls live/delayed market data and full options chains via yfinance.
Cleans data, computes IV Rank / IV Percentile, and computes realized volatility.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class TickerOverview:
    symbol: str
    spot_price: float
    historical_vol_30d: float
    historical_vol_60d: float
    current_iv_estimate: float
    iv_rank_1y: float
    iv_percentile_1y: float
    vix_level: float
    expirations: List[str]


class LiveDataFeed:
    @staticmethod
    def get_ticker_overview(symbol: str) -> TickerOverview:
        """Fetches underlying stock price, volatility metrics, and IV Rank."""
        ticker = yf.Ticker(symbol)
        
        # Spot Price
        hist = ticker.history(period="1y")
        if hist.empty:
            raise ValueError(f"Unable to fetch history for symbol {symbol}")
        
        spot_price = float(hist['Close'].iloc[-1])
        
        # Historical Volatility (30d and 60d annualized)
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        hv_30 = float(log_returns.tail(30).std() * np.sqrt(252))
        hv_60 = float(log_returns.tail(60).std() * np.sqrt(252))
        
        # VIX level as macro volatility benchmark
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(period="5d")
            vix_level = float(vix_hist['Close'].iloc[-1]) if not vix_hist.empty else 18.0
        except Exception:
            vix_level = 18.0
            
        # Get expirations
        expirations = list(ticker.options) if ticker.options else []
        
        # Estimate current ATM IV from options chain (nearest monthly 30-45 DTE)
        current_iv = hv_30  # fallback
        iv_rank = 50.0      # fallback
        iv_percentile = 50.0
        
        if expirations:
            today = date.today()
            # Find an expiration close to 30-45 DTE
            target_exp = None
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if 20 <= dte <= 50:
                    target_exp = exp_str
                    break
            if not target_exp and len(expirations) > 0:
                target_exp = expirations[min(2, len(expirations) - 1)]
                
            try:
                chain = ticker.option_chain(target_exp)
                calls = chain.calls
                # find ATM call
                calls['dist'] = (calls['strike'] - spot_price).abs()
                atm_call = calls.sort_values('dist').iloc[0]
                atm_iv = float(atm_call['impliedVolatility'])
                if 0.05 < atm_iv < 2.0:
                    current_iv = atm_iv
            except Exception:
                pass

        # Rolling 30d annualized realized volatility series over 1 year to compute IV/HV Rank
        rolling_hv = log_returns.rolling(30).std() * np.sqrt(252)
        rolling_hv = rolling_hv.dropna()
        if len(rolling_hv) > 50:
            min_v = float(rolling_hv.min())
            max_v = float(rolling_hv.max())
            if max_v > min_v:
                iv_rank = float(np.clip((current_iv - min_v) / (max_v - min_v) * 100.0, 0.0, 100.0))
                iv_percentile = float((rolling_hv < current_iv).mean() * 100.0)

        return TickerOverview(
            symbol=symbol.upper(),
            spot_price=spot_price,
            historical_vol_30d=hv_30,
            historical_vol_60d=hv_60,
            current_iv_estimate=current_iv,
            iv_rank_1y=iv_rank,
            iv_percentile_1y=iv_percentile,
            vix_level=vix_level,
            expirations=expirations,
        )

    @staticmethod
    def get_options_chain_for_dte_range(
        symbol: str,
        min_dte: int = 15,
        max_dte: int = 60,
    ) -> tuple[float, pd.DataFrame]:
        """
        Fetches full options chain filtered by DTE window.
        Returns: (spot_price, cleaned_dataframe)
        """
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            raise ValueError(f"No price data found for {symbol}")
        spot_price = float(hist['Close'].iloc[-1])
        
        expirations = ticker.options
        if not expirations:
            return spot_price, pd.DataFrame()

        today = date.today()
        target_expirations = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                target_expirations.append((exp_str, dte))

        all_rows = []
        for exp_str, dte in target_expirations:
            try:
                chain = ticker.option_chain(exp_str)
            except Exception:
                continue

            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    strike = float(row.get('strike', 0))
                    bid = float(row.get('bid', 0.0))
                    ask = float(row.get('ask', 0.0))
                    last = float(row.get('lastPrice', 0.0))
                    iv = float(row.get('impliedVolatility', 0.0))
                    oi = float(row.get('openInterest', 0.0))
                    volume = float(row.get('volume', 0.0))

                    if strike <= 0:
                        continue

                    # Realistic mid price
                    if bid > 0 and ask > 0 and ask >= bid:
                        mid = (bid + ask) / 2.0
                    elif last > 0:
                        mid = last
                    else:
                        continue

                    # Filter out zero IV anomalies
                    if iv <= 0.01:
                        continue

                    all_rows.append({
                        'symbol': symbol.upper(),
                        'expiration': exp_str,
                        'dte': dte,
                        'strike': strike,
                        'option_type': opt_type,
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'last': last,
                        'implied_volatility': iv,
                        'open_interest': oi,
                        'volume': volume,
                    })

        return spot_price, pd.DataFrame(all_rows)

    @staticmethod
    def get_sp500_constituents(use_cache: bool = True) -> List[str]:
        """Returns all constituent symbols in the S&P 500 index."""
        from src.data.sp500 import SP500ConstituentProvider
        return SP500ConstituentProvider.get_constituents(use_cache=use_cache)

    @staticmethod
    def is_sp500_member(symbol: str) -> bool:
        """Checks if a ticker belongs to the S&P 500 universe."""
        from src.data.sp500 import SP500ConstituentProvider
        return SP500ConstituentProvider.is_member(symbol)