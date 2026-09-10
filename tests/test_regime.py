"""
Unit tests for RegimeDetector & SpreadSelector
"""

import pytest
import pandas as pd
from src.strategy.regime_detector import RegimeDetector
from src.strategy.spread_selector import SpreadSelector


def test_regime_decision_selling():
    # High IV rank (55%), Positive Gamma ($20M) -> Should recommend SELL_OPTIONS
    res = RegimeDetector.evaluate(
        iv_rank=55.0,
        iv_current=0.25,
        hv_30d=0.18,
        net_gex_dollar=20000000.0,
        net_vex=500000.0,
        spot_price=500.0,
        ema_20=505.0,
        ema_50=495.0,
    )
    assert res.action == "SELL_OPTIONS"
    assert "BULL_PUT" in res.strategy_type or "IRON_CONDOR" in res.strategy_type


def test_regime_decision_buying():
    # Low IV rank (15%), Negative Gamma (-$15M), Bullish momentum -> Should recommend BUY_OPTIONS
    res = RegimeDetector.evaluate(
        iv_rank=15.0,
        iv_current=0.12,
        hv_30d=0.14,
        net_gex_dollar=-15000000.0,
        net_vex=-200000.0,
        spot_price=510.0,
        ema_20=505.0,
        ema_50=495.0,
    )
    assert res.action == "BUY_OPTIONS"
    assert "CALL" in res.strategy_type or "DEBIT" in res.strategy_type


def test_spread_selector_defined_risk():
    # Mock options chain
    data = []
    spot = 500.0
    for k in range(450, 550, 5):
        data.append({
            "expiration": "2026-10-16",
            "dte": 30,
            "strike": float(k),
            "option_type": "put",
            "bid": 2.0,
            "ask": 2.2,
            "mid": 2.1,
            "last": 2.1,
            "implied_volatility": 0.20,
            "open_interest": 1000,
            "volume": 500,
        })
        data.append({
            "expiration": "2026-10-16",
            "dte": 30,
            "strike": float(k),
            "option_type": "call",
            "bid": 2.5,
            "ask": 2.7,
            "mid": 2.6,
            "last": 2.6,
            "implied_volatility": 0.20,
            "open_interest": 1000,
            "volume": 500,
        })
    df = pd.DataFrame(data)

    trade = SpreadSelector.select_best_trade(
        symbol="SPY",
        chain_df=df,
        spot_price=spot,
        strategy_type="BULL_PUT_SPREAD",
    )
    assert trade is not None
    assert trade.action == "SELL (CREDIT)"
    assert trade.max_loss_dollar > 0
    assert trade.probability_of_profit_pct > 65.0