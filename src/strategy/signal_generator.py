"""
Signal Generator Engine
Integrates Live Data, Dealer Greeks, Regime Detection, and Spread Selection
into a unified signal pipeline.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd
import yfinance as yf
from src.data.live_feed import LiveDataFeed, TickerOverview
from src.engine.dealer_greeks import DealerGreeksEngine, GammaProfile
from src.strategy.regime_detector import RegimeDetector, RegimeDecision
from src.strategy.spread_selector import SpreadSelector, TradeRecommendation


@dataclass
class MarketSignal:
    symbol: str
    overview: TickerOverview
    gamma_profile: GammaProfile
    regime: RegimeDecision
    trade: Optional[TradeRecommendation]


class SignalGenerator:
    @staticmethod
    def analyze_ticker(symbol: str) -> Optional[MarketSignal]:
        """Runs the complete quantitative options analysis pipeline on a single ticker."""
        try:
            overview = LiveDataFeed.get_ticker_overview(symbol)
            spot_price, chain_df = LiveDataFeed.get_options_chain_for_dte_range(symbol, min_dte=20, max_dte=60)
            if chain_df.empty:
                return None

            # Calculate Dealer Greeks (Net GEX, VEX, Walls)
            gamma_profile = DealerGreeksEngine.analyze_options_chain(chain_df, spot_price)

            # Trend metrics (EMA 20 and EMA 50)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            if len(hist) >= 50:
                ema_20 = float(hist['Close'].ewm(span=20).mean().iloc[-1])
                ema_50 = float(hist['Close'].ewm(span=50).mean().iloc[-1])
            else:
                ema_20 = spot_price
                ema_50 = spot_price

            # Regime Detection: SELL OPTIONS vs BUY OPTIONS
            regime = RegimeDetector.evaluate(
                iv_rank=overview.iv_rank_1y,
                iv_current=overview.current_iv_estimate,
                hv_30d=overview.historical_vol_30d,
                net_gex_dollar=gamma_profile.net_gex_dollar_1pct,
                net_vex=gamma_profile.net_vex,
                spot_price=spot_price,
                ema_20=ema_20,
                ema_50=ema_50,
            )

            # Spread & Contract Selection
            trade = SpreadSelector.select_best_trade(
                symbol=symbol,
                chain_df=chain_df,
                spot_price=spot_price,
                strategy_type=regime.strategy_type,
            )

            return MarketSignal(
                symbol=symbol,
                overview=overview,
                gamma_profile=gamma_profile,
                regime=regime,
                trade=trade,
            )
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None

    @classmethod
    def scan_universe(cls, symbols: List[str]) -> List[MarketSignal]:
        """Scans a list of liquid tickers and produces ranked signals."""
        results = []
        for sym in symbols:
            sig = cls.analyze_ticker(sym)
            if sig and sig.trade:
                results.append(sig)
        return results