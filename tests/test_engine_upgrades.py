"""
Tests for quantitative options engine fixes and upgrades:
1. SpreadSelector handlers for LEVERAGED_LONG_PUT and BEAR_PUT_DEBIT_SPREAD, and safe bearish fallback.
2. ConfidenceEngine 6 sub-scores summing to exactly 100 max, and ConfidenceReport.to_dict().
3. DealerGreeksEngine Zero Gamma grid root interpolation.
4. App endpoints /api/scan and /api/asymmetric/setups.
"""

import json
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.engine.black_scholes import BlackScholesEngine
from src.strategy.spread_selector import SpreadSelector
from src.strategy.confidence_engine import ConfidenceEngine, ConfidenceReport
from src.engine.dealer_greeks import DealerGreeksEngine
from src.web.app import app, calculate_technical_indicators


@pytest.fixture
def mock_chain():
    """Generates a synthetic options chain around spot 100.0."""
    spot = 100.0
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    dte = 30
    T = dte / 365.0
    rows = []
    for k in strikes:
        p_call = BlackScholesEngine.price("call", spot, k, T, 0.045, 0.25)
        p_put = BlackScholesEngine.price("put", spot, k, T, 0.045, 0.25)
        rows.append({
            "expiration": "2026-10-10",
            "dte": dte,
            "strike": float(k),
            "option_type": "call",
            "bid": max(0.05, p_call - 0.10),
            "ask": p_call + 0.10,
            "mid": p_call,
            "last": p_call,
            "implied_volatility": 0.25,
            "open_interest": 1500,
            "volume": 300,
        })
        rows.append({
            "expiration": "2026-10-10",
            "dte": dte,
            "strike": float(k),
            "option_type": "put",
            "bid": max(0.05, p_put - 0.10),
            "ask": p_put + 0.10,
            "mid": p_put,
            "last": p_put,
            "implied_volatility": 0.25,
            "open_interest": 1500,
            "volume": 300,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 1. SpreadSelector Tests
# ---------------------------------------------------------------------
def test_spread_selector_leveraged_long_put(mock_chain):
    rec = SpreadSelector.select_best_trade(
        symbol="TEST",
        chain_df=mock_chain,
        spot_price=100.0,
        strategy_type="LEVERAGED_LONG_PUT",
        target_dte_min=20,
        target_dte_max=50,
    )
    assert rec is not None
    assert rec.action == "BUY (DEBIT/LEVERAGE)"
    assert "Long Put" in rec.strategy_name
    assert rec.short_strike is None
    assert rec.long_strike is not None
    # Delta per contract (greeks.delta * 100) should be in -40 to -50 range
    assert -55.0 <= rec.net_delta <= -35.0
    assert rec.max_loss_dollar > 0
    assert rec.max_profit_dollar > 0


def test_spread_selector_bear_put_debit_spread(mock_chain):
    rec = SpreadSelector.select_best_trade(
        symbol="TEST",
        chain_df=mock_chain,
        spot_price=100.0,
        strategy_type="BEAR_PUT_DEBIT_SPREAD",
        target_dte_min=20,
        target_dte_max=50,
    )
    assert rec is not None
    assert rec.action == "BUY (DEBIT/LEVERAGE)"
    assert "Bear Put Debit Spread" in rec.strategy_name
    assert rec.long_strike is not None
    assert rec.short_strike is not None
    # Long put strike must be higher than short put strike
    assert rec.long_strike > rec.short_strike
    assert rec.entry_limit_price > 0
    assert rec.max_loss_dollar > 0
    assert rec.max_profit_dollar > 0


def test_spread_selector_bearish_never_falls_back_to_bull_put(mock_chain):
    # Pass an unknown bearish strategy type
    rec = SpreadSelector.select_best_trade(
        symbol="TEST",
        chain_df=mock_chain,
        spot_price=100.0,
        strategy_type="BEARISH_CRASH_BREAKDOWN",
        target_dte_min=20,
        target_dte_max=50,
    )
    assert rec is not None
    assert "Bull Put" not in rec.strategy_name
    assert "BEAR" in rec.strategy_name.upper() or "PUT" in rec.strategy_name.upper()


# ---------------------------------------------------------------------
# 2. ConfidenceEngine Tests
# ---------------------------------------------------------------------
def test_confidence_subscores_max_sum_100():
    # Maximum theoretical inputs
    report = ConfidenceEngine.evaluate_trade(
        action="SELL (CREDIT)",
        strategy_name="Bull Put Credit Spread (Defined Risk)",
        spot_price=100.0,
        iv_current=0.35,
        hv_30d=0.25,        # vrp_spread = +10.0 -> max 25
        iv_rank=60.0,
        net_gex_dollar_m=20.0,  # positive gamma -> 15 + 10 = 25
        put_wall=95.0,
        call_wall=110.0,
        pop_pct=88.0,       # pop >= 85 -> max 15
        roc_pct=35.0,
        entry_price=0.80,   # > 0.40 -> max 10
        short_strike=90.0,  # <= put_wall -> max gex bonus
        long_strike=85.0,
        vix_level=18.0,     # ideal macro -> max 5
        ema_20=98.0,
        ema_50=95.0,
        ema_200=90.0,       # perfect uptrend -> max 20
    )

    assert report.vrp_score == 25.0
    assert report.gex_alignment_score == 25.0
    assert report.trend_score == 20.0
    assert report.pop_score == 15.0
    assert report.liquidity_score == 10.0
    assert report.macro_buffer_score == 5.0
    
    subscores_sum = (
        report.vrp_score +
        report.gex_alignment_score +
        report.trend_score +
        report.pop_score +
        report.liquidity_score +
        report.macro_buffer_score
    )
    assert subscores_sum == 100.0
    assert report.total_score <= 100.0


def test_confidence_report_to_dict_and_json_serialization():
    report = ConfidenceEngine.evaluate_trade(
        action="BUY (DEBIT/LEVERAGE)",
        strategy_name="Leveraged Long Put",
        spot_price=100.0,
        iv_current=0.20,
        hv_30d=0.25,
        iv_rank=15.0,
        net_gex_dollar_m=-5.0,
        put_wall=98.0,
        call_wall=105.0,
        pop_pct=50.0,
        roc_pct=100.0,
        entry_price=2.50,
        short_strike=None,
        long_strike=95.0,
        vix_level=22.0,
        ema_20=102.0,
        ema_50=105.0,
        ema_200=110.0,
    )
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "vrp_score" in d
    assert "gex_alignment_score" in d
    assert "trend_score" in d
    assert "pop_score" in d
    assert "liquidity_score" in d
    assert "macro_buffer_score" in d
    assert "verdict" in d

    # Ensure JSON serializable
    json_str = json.dumps(d)
    assert len(json_str) > 0


# ---------------------------------------------------------------------
# 3. DealerGreeksEngine Zero Gamma Test
# ---------------------------------------------------------------------
def test_zero_gamma_grid_interpolation():
    spot = 100.0
    # Create chain where puts dominate at 90-95 and calls dominate at 105-110
    rows = [
        {"strike": 90.0, "option_type": "put", "dte": 30, "implied_volatility": 0.25, "open_interest": 10000},
        {"strike": 95.0, "option_type": "put", "dte": 30, "implied_volatility": 0.25, "open_interest": 8000},
        {"strike": 105.0, "option_type": "call", "dte": 30, "implied_volatility": 0.25, "open_interest": 8000},
        {"strike": 110.0, "option_type": "call", "dte": 30, "implied_volatility": 0.25, "open_interest": 10000},
    ]
    df = pd.DataFrame(rows)
    gp = DealerGreeksEngine.analyze_options_chain(df, spot)
    assert gp.zero_gamma_strike is not None
    # Because of symmetry around 100, zero gamma should be around 100.0
    assert 90.0 <= gp.zero_gamma_strike <= 110.0


# ---------------------------------------------------------------------
# 4. App Technical Indicators & /api/scan /api/asymmetric tests
# ---------------------------------------------------------------------
def test_calculate_technical_indicators():
    # Helper should return real numeric dict even if network/ticker fails or succeeds
    tech = calculate_technical_indicators("SPY", 500.0)
    assert "rsi" in tech
    assert "ema_20" in tech
    assert "ema_50" in tech
    assert "ema_200" in tech
    assert 0.0 <= tech["rsi"] <= 100.0
    assert tech["ema_20"] > 0
    assert tech["ema_50"] > 0
    assert tech["ema_200"] > 0


def test_api_scan_trade_schema_and_subscores():
    client = TestClient(app)
    r = client.get("/api/scan?symbols=SPY&max_dte=65")
    assert r.status_code == 200
    data = r.json()
    assert "trades" in data
    assert "count" in data
    if data["trades"]:
        t = data["trades"][0]
        # Technical indicators
        assert "rsi" in t
        assert "ema_20" in t
        assert "ema_50" in t
        assert "ema_200" in t
        assert "vrp_spread" in t
        # 6 sub-scores
        assert "vrp_score" in t
        assert "gex_alignment_score" in t
        assert "trend_score" in t
        assert "pop_score" in t
        assert "liquidity_score" in t
        assert "macro_buffer_score" in t
        assert "confidence_verdict" in t


def test_api_asymmetric_setups_no_attribute_error():
    client = TestClient(app)
    r = client.get("/api/asymmetric/setups?symbols=SPY")
    assert r.status_code == 200
    data = r.json()
    assert "asymmetric_longs" in data
    assert "leaps_pmcc" in data
