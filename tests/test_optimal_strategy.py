import pytest
from fastapi.testclient import TestClient
from src.strategy.optimal_strategy import OptimalStrategyEngine
from src.web.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_optimal_strategy_high_iv_bullish():
    res = OptimalStrategyEngine.evaluate(
        symbol="NVDA",
        spot_price=125.0,
        iv_rank=55.0,
        iv_current=0.55,
        hv_30d=0.40,
        net_gex_dollar_m=3.5,
        gamma_regime="POSITIVE_GAMMA",
        put_wall=115.0,
        call_wall=135.0,
        ema_20=122.0,
        ema_50=118.0,
        ema_200=105.0,
    )
    assert res["strategy_code"] == "BULL_PUT_SPREAD"
    assert "Bull Put" in res["display_name"]
    assert res["action_type"] == "SELL_PREMIUM"
    assert res["confidence_score"] >= 90
    assert "VRP" in res["display_name"] or "Theta" in res["display_name"]


def test_optimal_strategy_low_iv_uptrend_leaps():
    res = OptimalStrategyEngine.evaluate(
        symbol="SPY",
        spot_price=760.0,
        iv_rank=16.0,
        iv_current=0.13,
        hv_30d=0.14,
        net_gex_dollar_m=2.0,
        gamma_regime="POSITIVE_GAMMA",
        put_wall=745.0,
        call_wall=775.0,
        ema_20=755.0,
        ema_50=750.0,
        ema_200=730.0,
    )
    assert res["strategy_code"] == "LEAPS_PMCC"
    assert "PMCC" in res["display_name"]
    assert res["action_type"] == "BUY_LEAPS_PMCC"
    assert res["confidence_score"] >= 90
    assert "84.2%" in res["why_this_strategy_beats_others"] or "Hebel" in res["why_this_strategy_beats_others"]


def test_optimal_strategy_low_iv_squeeze():
    res = OptimalStrategyEngine.evaluate(
        symbol="TSLA",
        spot_price=250.0,
        iv_rank=18.0,
        iv_current=0.35,
        hv_30d=0.38,
        net_gex_dollar_m=-5.0,
        gamma_regime="NEGATIVE_GAMMA",
        put_wall=240.0,
        call_wall=260.0,
        ema_20=252.0,
        ema_50=248.0,
        ema_200=230.0,
        unusual_anomaly_count=3,
    )
    assert res["strategy_code"] == "ASYMMETRIC_45DTE_LONG"
    assert "Asymmetrisch" in res["display_name"]
    assert res["action_type"] == "BUY_CONVEXITY"
    assert "800%" in res["why_this_strategy_beats_others"] or "Home-Run" in res["display_name"]


def test_optimal_strategy_endpoint(client):
    r = client.get("/api/strategy/optimal/SPY")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "SPY"
    assert "display_name" in data
    assert "why_this_strategy_beats_others" in data
    assert "confidence_score" in data
    assert "key_signals" in data
    assert len(data["key_signals"]) > 0
