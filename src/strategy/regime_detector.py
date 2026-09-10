"""
Market Regime Detector
Determines whether market conditions favor OPTION SELLING (Credit Spreads/Iron Condors)
or OPTION BUYING (Leveraged Long Calls/Puts/Debit Spreads) based on:
1. Implied Volatility Rank (IVR) & Volatility Risk Premium (IV vs RV)
2. Dealer Gamma Exposure (Net GEX: Positive vs Negative Gamma)
3. Dealer Vanna Exposure (VEX: Vol-compression vs Vol-expansion)
4. Macro Trend (EMA20 vs EMA50)
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd


@dataclass
class RegimeDecision:
    action: Literal["SELL_OPTIONS", "BUY_OPTIONS", "WAIT_NEUTRAL"]
    strategy_type: str
    confidence: float  # 0.0 to 1.0
    iv_rank: float
    gex_regime: str
    vanna_bias: str
    trend_bias: str
    reasoning: str


class RegimeDetector:
    @staticmethod
    def evaluate(
        iv_rank: float,
        iv_current: float,
        hv_30d: float,
        net_gex_dollar: float,
        net_vex: float,
        spot_price: float,
        ema_20: float,
        ema_50: float,
    ) -> RegimeDecision:
        """
        Evaluates quantitative indicators to output a clear BUY vs SELL recommendation.
        """
        # Trend
        if spot_price > ema_20 and ema_20 > ema_50:
            trend = "BULLISH_UPTREND"
        elif spot_price < ema_20 and ema_20 < ema_50:
            trend = "BEARISH_DOWNTREND"
        else:
            trend = "CHOPPY_SIDEWAYS"

        # Volatility Risk Premium (VRP): Is IV higher than realized vol?
        vrp_spread = (iv_current - hv_30d) * 100.0  # in vol points
        gex_regime = "POSITIVE_GAMMA" if net_gex_dollar >= 0 else "NEGATIVE_GAMMA"
        vanna_bias = "VOL_COMPRESSION" if net_vex >= 0 else "VOL_EXPANSION"

        # SCENARIO 1: High IV Rank (> 40%) OR high VRP + Positive Gamma -> EXCELLENT FOR SELLING OPTIONS
        if (iv_rank >= 40.0 or vrp_spread >= 2.5) and gex_regime == "POSITIVE_GAMMA":
            if trend == "BULLISH_UPTREND":
                strategy = "BULL_PUT_SPREAD"
                reasoning = (
                    f"IV Rank is elevated ({iv_rank:.1f}%), IV exceeds Realized Vol by {vrp_spread:+.1f}pts (rich premium). "
                    f"Market is in POSITIVE GAMMA, meaning Market Makers cushion pullbacks and suppress volatility. "
                    f"Uptrend intact -> Sell Bull Put Credit Spreads for high-probability theta harvest."
                )
            elif trend == "BEARISH_DOWNTREND":
                strategy = "BEAR_CALL_SPREAD"
                reasoning = (
                    f"IV Rank is elevated ({iv_rank:.1f}%) in a downtrend with positive dealer stability. "
                    f"Sell Bear Call Credit Spreads above resistance to collect overpriced call premium."
                )
            else:
                strategy = "IRON_CONDOR"
                reasoning = (
                    f"IV Rank is high ({iv_rank:.1f}%) and market is range-bound in Positive Gamma. "
                    f"Both call and put premiums are inflated. Sell Iron Condor to capture decay from both sides."
                )
            return RegimeDecision(
                action="SELL_OPTIONS",
                strategy_type=strategy,
                confidence=0.88,
                iv_rank=iv_rank,
                gex_regime=gex_regime,
                vanna_bias=vanna_bias,
                trend_bias=trend,
                reasoning=reasoning,
            )

        # SCENARIO 2: Low IV Rank (< 30%) + Negative Gamma OR Strong Trend Breakout -> EXCELLENT FOR BUYING OPTIONS (LEVERAGE)
        elif iv_rank < 30.0 or gex_regime == "NEGATIVE_GAMMA":
            if trend == "BULLISH_UPTREND":
                strategy = "LEVERAGED_LONG_CALL"
                reasoning = (
                    f"Options are CHEAP (IV Rank only {iv_rank:.1f}%). "
                    f"With {gex_regime}, dealer hedging accelerates upside moves (Gamma squeeze potential). "
                    f"Buying calls gives asymmetric 5x-15x leverage with strictly capped risk (max loss = premium paid)."
                )
                confidence = 0.85
            elif trend == "BEARISH_DOWNTREND":
                strategy = "LEVERAGED_LONG_PUT"
                reasoning = (
                    f"Volatility is underpriced (IV Rank {iv_rank:.1f}%) while market is breaking down in {gex_regime}. "
                    f"In Negative Gamma, dealers must sell into market drops, accelerating cascade risks. "
                    f"Buying puts provides massive downward convexity and crash protection."
                )
                confidence = 0.85
            else:
                strategy = "LONG_STRADDLE_OR_DEBIT_SPREAD"
                reasoning = (
                    f"Options are cheap (IV Rank {iv_rank:.1f}%), but trend is sideways. "
                    f"Use tight Debit Spreads or await clean breakout before buying outright leverage."
                )
                confidence = 0.65

            return RegimeDecision(
                action="BUY_OPTIONS",
                strategy_type=strategy,
                confidence=confidence,
                iv_rank=iv_rank,
                gex_regime=gex_regime,
                vanna_bias=vanna_bias,
                trend_bias=trend,
                reasoning=reasoning,
            )

        # SCENARIO 3: Intermediate zone
        else:
            # Default to Credit Spreads in positive gamma, Long Spreads in negative gamma
            if gex_regime == "POSITIVE_GAMMA":
                strategy = "BULL_PUT_SPREAD" if "BULLISH" in trend else "IRON_CONDOR"
                return RegimeDecision(
                    action="SELL_OPTIONS",
                    strategy_type=strategy,
                    confidence=0.72,
                    iv_rank=iv_rank,
                    gex_regime=gex_regime,
                    vanna_bias=vanna_bias,
                    trend_bias=trend,
                    reasoning=f"Moderate IV Rank ({iv_rank:.1f}%) with positive dealer gamma stabilization. Favorable for Defined-Risk Credit Spreads.",
                )
            else:
                strategy = "BULL_CALL_DEBIT_SPREAD" if "BULLISH" in trend else "BEAR_PUT_DEBIT_SPREAD"
                return RegimeDecision(
                    action="BUY_OPTIONS",
                    strategy_type=strategy,
                    confidence=0.70,
                    iv_rank=iv_rank,
                    gex_regime=gex_regime,
                    vanna_bias=vanna_bias,
                    trend_bias=trend,
                    reasoning=f"Moderate IV Rank ({iv_rank:.1f}%) in Negative Gamma environment. Favorable for Leveraged Debit Spreads.",
                )