"""
Unit tests for Asymmetric 45 DTE Longs & LEAPS / Barbell Engine
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.strategy.asymmetric_engine import AsymmetricEngine, AsymmetricLongSetup, LeapsPmccSetup
from src.engine.black_scholes import BlackScholesEngine
from src.web.app import app


@pytest.fixture
def synthetic_chain():
    """Generates a synthetic options chain covering 30 to 400 DTE."""
    spot = 100.0
    rows = []
    expirations = [
        ("2026-10-25", 45),
        ("2027-09-15", 370),
    ]
    strikes = [70, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130]

    for exp, dte in expirations:
        T = dte / 365.0
        for k in strikes:
            # Calls
            p_call = BlackScholesEngine.price('call', spot, k, T, 0.045, 0.25)
            rows.append({
                'symbol': 'TEST',
                'expiration': exp,
                'dte': dte,
                'strike': float(k),
                'option_type': 'call',
                'bid': max(0.05, p_call - 0.10),
                'ask': p_call + 0.10,
                'mid': p_call,
                'last': p_call,
                'implied_volatility': 0.25,
                'open_interest': 1000,
                'volume': 200,
            })
            # Puts
            p_put = BlackScholesEngine.price('put', spot, k, T, 0.045, 0.25)
            rows.append({
                'symbol': 'TEST',
                'expiration': exp,
                'dte': dte,
                'strike': float(k),
                'option_type': 'put',
                'bid': max(0.05, p_put - 0.10),
                'ask': p_put + 0.10,
                'mid': p_put,
                'last': p_put,
                'implied_volatility': 0.25,
                'open_interest': 1000,
                'volume': 200,
            })

    return pd.DataFrame(rows)


def test_asymmetric_45dte_call_selection(synthetic_chain):
    spot = 100.0
    longs = AsymmetricEngine.scan_asymmetric_longs(
        symbol='TEST',
        spot_price=spot,
        chain_df=synthetic_chain,
        iv_rank=16.0,
        ema_20=98.0,
        ema_50=95.0,
        rsi=58.0,
    )

    assert len(longs) > 0
    call_setup = next((l for l in longs if l.option_type == "CALL"), None)
    assert call_setup is not None
    assert call_setup.dte == 45
    assert call_setup.strategy_type == "ASYMMETRIC_45DTE_CALL"
    assert 0.20 <= call_setup.delta <= 0.55
    # Verify profit ladder targets
    assert call_setup.target_1_price == round(call_setup.entry_price * 3.0, 2)  # +200%
    assert call_setup.target_2_price == round(call_setup.entry_price * 5.0, 2)  # +400%
    assert call_setup.target_3_price == round(call_setup.entry_price * 9.0, 2)  # +800%
    assert call_setup.stop_loss_price == round(call_setup.entry_price * 0.50, 2)  # -50%
    assert call_setup.breakeven_price > call_setup.strike


def test_leaps_pmcc_selection(synthetic_chain):
    spot = 100.0
    pmcc = AsymmetricEngine.scan_leaps_pmcc(
        symbol='TEST',
        spot_price=spot,
        chain_df=synthetic_chain,
    )

    assert pmcc is not None
    assert pmcc.symbol == 'TEST'
    assert pmcc.leaps_dte >= 200
    assert pmcc.leaps_strike < spot  # Deep In-The-Money
    assert pmcc.leaps_delta >= 0.70  # ~0.80 Delta target
    assert pmcc.short_strike > spot  # Out-of-The-Money Short Call
    assert pmcc.effective_leverage > 1.5
    assert pmcc.monthly_yield_pct > 0.0
    assert pmcc.net_debit_dollar > 0.0


def test_asymmetric_endpoints():
    client = TestClient(app)
    
    # 1. Test Backtest results endpoint
    r_bt = client.get("/api/backtest/asymmetric")
    assert r_bt.status_code == 200
    data_bt = r_bt.json()
    assert "strategies" in data_bt
    strategies = data_bt["strategies"]
    assert "asymmetric_45dte_longs" in strategies
    assert "barbell_portfolio_80_20" in strategies
    assert "leaps_pmcc_vs_benchmark" in strategies

    # Verify metrics presence
    assert strategies["asymmetric_45dte_longs"]["cagr_pct"] > 10.0
    assert strategies["barbell_portfolio_80_20"]["win_rate_pct"] > 60.0
    assert strategies["leaps_pmcc_vs_benchmark"]["leaps_pmcc"]["cagr_pct"] > 50.0

    # 2. Test Setups endpoint schema
    r_setups = client.get("/api/asymmetric/setups?symbols=SPY")
    assert r_setups.status_code == 200
    data_setups = r_setups.json()
    assert "total_asymmetric_longs" in data_setups
    assert "total_leaps_pmcc" in data_setups
    assert "asymmetric_longs" in data_setups
    assert "leaps_pmcc" in data_setups
