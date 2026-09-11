"""
Unit tests for S&P 500 Asymmetric Qualification Engine,
strict filtering rules (low IV, momentum, LEAPS PMCC),
and FastAPI endpoints.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.strategy.asymmetric_engine import (
    AsymmetricEngine,
    AsymmetricLongSetup,
    LeapsPmccSetup,
)
from src.engine.black_scholes import BlackScholesEngine
from src.web.app import app


@pytest.fixture
def synthetic_sp500_chain():
    """Builds a realistic test options chain with 35 DTE and 300 DTE options."""
    spot = 150.0
    rows = []

    # 1. 35 DTE (Near-term for 45 DTE Long Calls/Puts)
    dte_near = 35
    T_near = dte_near / 365.0
    exp_near = "2026-10-16"
    for k in [135, 140, 145, 150, 155, 160, 165]:
        p_c = BlackScholesEngine.price('call', spot, k, T_near, 0.045, 0.22)
        p_p = BlackScholesEngine.price('put', spot, k, T_near, 0.045, 0.22)
        rows.append({
            'symbol': 'SPTEST',
            'expiration': exp_near,
            'dte': dte_near,
            'strike': float(k),
            'option_type': 'call',
            'mid': p_c,
            'implied_volatility': 0.22,
            'open_interest': 1000,
            'volume': 200,
        })
        rows.append({
            'symbol': 'SPTEST',
            'expiration': exp_near,
            'dte': dte_near,
            'strike': float(k),
            'option_type': 'put',
            'mid': p_p,
            'implied_volatility': 0.22,
            'open_interest': 1000,
            'volume': 200,
        })

    # 2. 300 DTE (For Deep ITM LEAPS)
    dte_far = 300
    T_far = dte_far / 365.0
    exp_far = "2027-07-16"
    for k in [110, 120, 130, 140, 150, 160, 170, 180]:
        p_c = BlackScholesEngine.price('call', spot, k, T_far, 0.045, 0.22)
        rows.append({
            'symbol': 'SPTEST',
            'expiration': exp_far,
            'dte': dte_far,
            'strike': float(k),
            'option_type': 'call',
            'mid': p_c,
            'implied_volatility': 0.22,
            'open_interest': 500,
            'volume': 50,
        })

    return pd.DataFrame(rows)


def test_rejects_high_iv_for_long_call(synthetic_sp500_chain):
    """High IV stocks (IVR > 25%) must be rejected to prevent IV crush."""
    tech = {"rsi": 55.0, "ema_20": 145.0, "ema_50": 140.0, "ema_200": 130.0}
    setups = AsymmetricEngine.qualify_asymmetric_long(
        symbol="SPTEST",
        spot_price=150.0,
        chain_df=synthetic_sp500_chain,
        tech=tech,
        iv_rank=65.0,
    )
    calls = [s for s in setups if s.option_type == "CALL"]
    assert len(calls) == 0


def test_rejects_downtrend_for_long_call(synthetic_sp500_chain):
    """Stocks trading below 20 EMA must be rejected for Long Calls."""
    tech = {"rsi": 40.0, "ema_20": 158.0, "ema_50": 160.0, "ema_200": 165.0}
    setups = AsymmetricEngine.qualify_asymmetric_long(
        symbol="SPTEST",
        spot_price=150.0,
        chain_df=synthetic_sp500_chain,
        tech=tech,
        iv_rank=18.0,
    )
    calls = [s for s in setups if s.option_type == "CALL"]
    assert len(calls) == 0


def test_qualifies_bull_call_with_favorable_metrics(synthetic_sp500_chain):
    """Stocks with cheap IV (IVR <= 25%) and bullish momentum must be qualified."""
    tech = {"rsi": 56.0, "ema_20": 146.0, "ema_50": 142.0, "ema_200": 135.0}
    setups = AsymmetricEngine.qualify_asymmetric_long(
        symbol="SPTEST",
        spot_price=150.0,
        chain_df=synthetic_sp500_chain,
        tech=tech,
        iv_rank=16.5,
    )
    calls = [s for s in setups if s.option_type == "CALL"]
    assert len(calls) >= 1
    call = calls[0]
    assert call.is_sp500_qualified is True
    assert "S&P 500 QUALIFIZIERT" in call.qualification_reason
    assert call.target_1_price > call.entry_price * 2.5
    assert call.stop_loss_price < call.entry_price


def test_qualifies_bear_put_on_breakdown(synthetic_sp500_chain):
    """Stocks breaking below 20 EMA with moderate IV must be qualified for Bear Puts."""
    tech = {"rsi": 42.0, "ema_20": 155.0, "ema_50": 156.0, "ema_200": 158.0}
    setups = AsymmetricEngine.qualify_asymmetric_long(
        symbol="SPTEST",
        spot_price=150.0,
        chain_df=synthetic_sp500_chain,
        tech=tech,
        iv_rank=22.0,
    )
    puts = [s for s in setups if s.option_type == "PUT"]
    assert len(puts) >= 1
    put = puts[0]
    assert put.is_sp500_qualified is True
    assert "S&P 500 QUALIFIZIERT" in put.qualification_reason
    assert put.delta < 0


def test_qualifies_leaps_pmcc(synthetic_sp500_chain):
    """Secular uptrend stock with moderate IV must generate qualified LEAPS PMCC."""
    tech = {"rsi": 54.0, "ema_20": 148.0, "ema_50": 144.0, "ema_200": 132.0}
    pmcc = AsymmetricEngine.qualify_leaps_pmcc(
        symbol="SPTEST",
        spot_price=150.0,
        chain_df=synthetic_sp500_chain,
        tech=tech,
        iv_rank=20.0,
    )
    assert pmcc is not None
    assert pmcc.is_sp500_qualified is True
    assert pmcc.leaps_strike < 150.0  # Deep ITM
    assert pmcc.short_strike > 150.0  # OTM Short Call
    assert pmcc.effective_leverage >= 1.8
    assert pmcc.monthly_yield_pct >= 1.4


def test_rejects_leaps_pmcc_in_downtrend(synthetic_sp500_chain):
    """Stock trading below 200 EMA must be rejected for LEAPS PMCC."""
    tech = {"rsi": 35.0, "ema_20": 160.0, "ema_50": 165.0, "ema_200": 175.0}
    pmcc = AsymmetricEngine.qualify_leaps_pmcc(
        symbol="SPTEST",
        spot_price=150.0,
        chain_df=synthetic_sp500_chain,
        tech=tech,
        iv_rank=20.0,
    )
    assert pmcc is None


def test_api_asymmetric_sp500_endpoint():
    """Verifies GET /api/asymmetric/sp500 response structure and filters."""
    client = TestClient(app)
    response = client.get("/api/asymmetric/sp500?limit=10")
    assert response.status_code == 200
    data = response.json()

    assert "universe" in data
    assert "S&P 500" in data["universe"]
    assert "asymmetric_longs" in data
    assert "leaps_pmcc" in data
    assert "screening_rules" in data

    # Verify filtering by type
    resp_call = client.get("/api/asymmetric/sp500?filter_type=CALL&limit=10")
    assert resp_call.status_code == 200
    d_call = resp_call.json()
    assert d_call["total_leaps_pmcc"] == 0
    for s in d_call.get("asymmetric_longs", []):
        assert s["option_type"] == "CALL"

    resp_pmcc = client.get("/api/asymmetric/sp500?filter_type=PMCC&limit=10")
    assert resp_pmcc.status_code == 200
    d_pmcc = resp_pmcc.json()
    assert d_pmcc["total_asymmetric_longs"] == 0


def test_api_asymmetric_setups_defaults_to_sp500():
    """GET /api/asymmetric/setups must default to S&P 500 qualified engine."""
    client = TestClient(app)
    response = client.get("/api/asymmetric/setups")
    assert response.status_code == 200
    data = response.json()
    assert "S&P 500" in data.get("universe", "")