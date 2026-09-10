import pytest
import pandas as pd
import numpy as np
from src.strategy.confidence_engine import ConfidenceEngine
from src.engine.payoff_visualizer import PayoffVisualizer
from src.strategy.paper_portfolio import PositionSizer, PaperPortfolio
from src.backtest.monte_carlo import MonteCarloSimulator

def test_confidence_engine_scoring():
    result = ConfidenceEngine.evaluate_trade(
        action="SELL_OPTIONS",
        strategy_name="BULL_PUT_SPREAD",
        spot_price=500.0,
        iv_current=0.22,
        hv_30d=0.15,
        iv_rank=45.0,
        net_gex_dollar_m=250.0,
        put_wall=485.0,
        call_wall=515.0,
        pop_pct=78.5,
        roc_pct=24.0,
        entry_price=1.85,
        short_strike=490.0,
        long_strike=480.0,
        vix_level=16.5,
        ema_20=498.0,
        ema_50=492.0,
        ema_200=475.0,
    )

    assert 0.0 <= result.total_score <= 100.0
    assert result.grade in ["A+", "A", "B+", "B", "C", "D"]
    assert result.vrp_score >= 0.0
    assert result.gex_alignment_score >= 0.0
    assert isinstance(result.strengths, list)
    assert isinstance(result.risks, list)
    assert result.total_score >= 70.0

def test_payoff_visualizer_credit_spread():
    payoff = PayoffVisualizer.generate_payoff_data(
        strategy_name="BULL_PUT_SPREAD",
        spot_price=500.0,
        entry_price=2.00,
        short_strike=490.0,
        long_strike=480.0,
        dte=30,
        iv=0.20,
        contracts=1,
    )

    assert payoff["max_profit"] == 200.0
    assert payoff["max_loss"] == 800.0
    assert len(payoff["breakevens"]) == 1
    assert abs(payoff["breakevens"][0] - 488.0) < 1e-4

    assert len(payoff["prices"]) > 0
    assert len(payoff["pnl_expiration"]) == len(payoff["prices"])
    assert len(payoff["pnl_today"]) == len(payoff["prices"])
    assert len(payoff["pnl_midway"]) == len(payoff["prices"])

    # At high underlying price (e.g. 520), pnl at expiration should equal max_profit
    idx_high = np.argmax(np.array(payoff["prices"]) >= 520.0)
    assert abs(payoff["pnl_expiration"][idx_high] - 200.0) < 1.0

def test_position_sizer_kelly():
    sizing = PositionSizer.calculate_sizing(
        account_equity=25000.0,
        risk_mode="HALF_KELLY",
        max_loss_per_contract=800.0,
        max_profit_per_contract=200.0,
        win_prob_pct=80.0,
    )

    assert sizing["recommended_contracts"] >= 1
    assert sizing["total_risk_dollar"] <= 25000.0
    assert sizing["portfolio_risk_pct"] <= 25.0
    assert sizing["expected_profit_dollar"] > 0.0

def test_monte_carlo_simulation():
    np.random.seed(42)
    pnls = np.random.choice([250.0, -400.0, 300.0, 150.0], size=100, p=[0.5, 0.25, 0.15, 0.1])
    df_trades = pd.DataFrame({
        "trade_id": range(1, 101),
        "pnl_dollar": pnls,
        "holding_days": np.random.randint(3, 20, size=100),
    })

    res = MonteCarloSimulator.run_simulation(df_trades, initial_capital=25000.0, num_simulations=100)

    assert res["num_simulations"] == 100
    assert "cagr_percentiles" in res
    cagr_p = res["cagr_percentiles"]
    assert cagr_p["p95"] >= cagr_p["p50_median"] >= cagr_p["p5"]
    assert len(res["chart_paths"]["steps"]) > 0
