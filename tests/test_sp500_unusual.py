"""
Unit tests for S&P 500 constituent retrieval, Unusual Greeks anomaly detection,
and 85/15 Barbell Strategy calculations and API responses.
"""

import pytest
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.data.sp500 import SP500ConstituentProvider, get_sp500_constituents, is_sp500_member
from src.data.sp500_constituents import get_sp500_symbols, get_symbol_sector
from src.data.live_feed import LiveDataFeed
from src.engine.unusual_greeks import UnusualGreeksEngine, GreekAnomaly, GreekStructureReport
from src.strategy.asymmetric_engine import AsymmetricEngine, AsymmetricLongSetup
from src.engine.black_scholes import BlackScholesEngine
from src.backtest.barbell_strategy import (
    BarbellStrategyEngine,
    run_85_15_barbell_backtest,
    get_cached_barbell_results,
)
from src.web.app import app


@pytest.fixture
def synthetic_flow_chain():
    """
    Generates a synthetic options chain with both normal contracts
    and massive institutional Vol/OI whale sweeps (>2.5x) with positive Gamma/Vanna surge.
    """
    spot = 150.0
    rows = []
    dte = 45
    T = dte / 365.0
    exp = "2026-10-25"
    strikes = [135, 140, 145, 150, 155, 160, 165]

    for k in strikes:
        p_call = BlackScholesEngine.price('call', spot, k, T, 0.045, 0.22)
        p_put = BlackScholesEngine.price('put', spot, k, T, 0.045, 0.22)

        # Strike 155 Call has massive whale sweep: 3,500 volume vs 1,000 OI = 3.5x Vol/OI ratio
        call_vol = 3500 if k == 155 else 200
        call_oi = 1000

        # Strike 145 Put has whale sweep: 2,800 volume vs 1,000 OI = 2.8x Vol/OI ratio
        put_vol = 2800 if k == 145 else 200
        put_oi = 1000

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
            'implied_volatility': 0.22,
            'open_interest': call_oi,
            'volume': call_vol,
            'vol_oi_ratio': round(call_vol / call_oi, 2),
        })
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
            'implied_volatility': 0.22,
            'open_interest': put_oi,
            'volume': put_vol,
            'vol_oi_ratio': round(put_vol / put_oi, 2),
        })

    return pd.DataFrame(rows)


# =============================================================================
# 1. S&P 500 CONSTITUENT RETRIEVAL TESTS
# =============================================================================
class TestSP500ConstituentRetrieval:
    def test_sp500_constituents_total_and_types(self):
        """Verifies S&P 500 constituents retrieval returns 500+ symbols."""
        syms_1 = get_sp500_constituents()
        syms_2 = SP500ConstituentProvider.get_constituents()
        syms_3 = LiveDataFeed.get_sp500_constituents()

        assert len(syms_1) >= 500
        assert len(syms_2) >= 500
        assert len(syms_3) >= 500

        # Consistent output across providers
        assert set(syms_1) == set(syms_2)

    def test_sp500_core_bellwethers_present(self):
        """Verifies mega-cap S&P 500 bellwethers are present in constituent set."""
        syms = set(get_sp500_constituents())
        core_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "LLY", "AVGO"]
        for ticker in core_tickers:
            assert ticker in syms, f"Expected {ticker} to be in S&P 500 constituents"

    def test_sp500_membership_check(self):
        """Verifies is_sp500_member accuracy for members and non-members."""
        assert is_sp500_member("AAPL") is True
        assert is_sp500_member("MSFT") is True
        assert is_sp500_member("SPY") is True  # Allowed as benchmark ETF proxy
        assert is_sp500_member("FAKE_TICKER_XYZ_123") is False

    def test_sp500_sector_groupings(self):
        """Verifies 11 official GICS sectors are mapped and populated."""
        sectors = SP500ConstituentProvider.get_sectors()
        assert len(sectors) >= 10  # GICS defines 11 sectors
        assert "Information Technology" in sectors or "Technology" in sectors
        assert "Financials" in sectors
        assert "Health Care" in sectors or "Healthcare" in sectors

        # Sector retrieval helper
        tech_symbols = SP500ConstituentProvider.get_by_sector("Technology") or SP500ConstituentProvider.get_by_sector("Information Technology")
        assert len(tech_symbols) > 10
        assert "AAPL" in tech_symbols or "MSFT" in tech_symbols or "NVDA" in tech_symbols


# =============================================================================
# 2. UNUSUAL GREEKS ANOMALY DETECTION TESTS
# =============================================================================
class TestUnusualGreeksAnomalyDetection:
    def test_flow_anomaly_detection_helper(self):
        """Verifies AsymmetricEngine.detect_flow_anomaly logic and scoring."""
        # Case A: Massive Vol/OI (> 2.5x) with positive Gamma/Vanna surge
        is_alert, desc, score = AsymmetricEngine.detect_flow_anomaly(
            vol_oi_ratio=3.2,
            gamma=0.015,
            vanna=0.045,
            threshold=2.5,
        )
        assert is_alert is True
        assert score >= 88.0
        assert "UNUSUAL_GAMMA_VANNA_SURGE" in desc or "Vol/OI" in desc

        # Case B: Standard institutional sweep (Vol/OI >= 2.5x without Greek surge)
        is_alert_sweep, desc_sweep, score_sweep = AsymmetricEngine.detect_flow_anomaly(
            vol_oi_ratio=2.6,
            gamma=0.001,
            vanna=0.001,
            threshold=2.5,
        )
        assert is_alert_sweep is True
        assert score_sweep >= 80.0

        # Case C: Normal market conditions (Vol/OI < 2.5x)
        is_alert_norm, desc_norm, score_norm = AsymmetricEngine.detect_flow_anomaly(
            vol_oi_ratio=1.1,
            gamma=0.005,
            vanna=0.01,
            threshold=2.5,
        )
        assert is_alert_norm is False
        assert score_norm <= 75.0

    def test_asymmetric_engine_flow_squeeze_alert(self, synthetic_flow_chain):
        """Verifies flow_squeeze_alert and 300%-800% profit potential on whale sweeps."""
        spot = 150.0
        longs = AsymmetricEngine.scan_asymmetric_longs(
            symbol="TEST",
            spot_price=spot,
            chain_df=synthetic_flow_chain,
            iv_rank=18.0,
            ema_20=148.0,
            ema_50=144.0,
            rsi=60.0,
        )

        assert len(longs) > 0
        call_setup = next((l for l in longs if l.option_type == "CALL"), None)
        assert call_setup is not None

        # Verify high-conviction flow_squeeze_alert flag
        assert call_setup.flow_squeeze_alert is True
        assert call_setup.vol_oi_ratio >= 2.5
        assert "300% - 800% ROI" in call_setup.target_profit_potential
        assert call_setup.strategy_score >= 88.0
        assert "FLOW SQUEEZE ALERT" in call_setup.catalyst_reason

        # Verify profit targets
        assert call_setup.target_1_price >= round(call_setup.entry_price * 3.0, 2)
        assert call_setup.target_3_price == round(call_setup.entry_price * 9.0, 2)  # +800%
        assert call_setup.stop_loss_price == round(call_setup.entry_price * 0.50, 2)  # -50%

    def test_unusual_greeks_engine_structure_report(self):
        """Verifies UnusualGreeksEngine structure and anomaly analysis."""
        report = UnusualGreeksEngine.analyze_ticker_anomalies("SPY", min_volume=1)
        assert report is not None
        assert isinstance(report, GreekStructureReport)
        assert report.symbol == "SPY"
        assert report.spot_price > 0.0
        assert report.primary_magnet_strike > 0.0
        assert report.magnet_pull_force in ["STRONG_PINNING", "MODERATE", "NEUTRAL"]
        assert report.total_gamma_volume_m >= 0.0


# =============================================================================
# 3. BARBELL STRATEGY CALCULATIONS & API RESPONSES TESTS
# =============================================================================
class TestBarbellStrategyAndAPI:
    def test_barbell_strategy_simulation_and_metrics(self):
        """
        Verifies the 85/15 Barbell Strategy meets target metrics:
        - CAGR > 70%
        - Max Drawdown < 20%
        - Sharpe Ratio >= 1.80
        - Win Rate >= 70%
        - Saves to data_cache/barbell_high_yield_backtest.json
        """
        engine = BarbellStrategyEngine(cache_dir="data_cache")
        res = engine.run_simulation(initial_capital=25000.0)

        assert "metrics" in res
        metrics = res["metrics"]

        # Verified Targets
        assert metrics["cagr_pct"] > 70.0, f"Expected CAGR > 70%, got {metrics['cagr_pct']}%"
        assert metrics["max_drawdown_pct"] < 20.0, f"Expected Max Drawdown < 20%, got {metrics['max_drawdown_pct']}%"
        assert metrics["sharpe_ratio"] >= 1.80, f"Expected Sharpe >= 1.80, got {metrics['sharpe_ratio']}"
        assert metrics["win_rate_pct"] >= 70.0, f"Expected Win Rate >= 70%, got {metrics['win_rate_pct']}%"
        assert metrics["profit_factor"] >= 2.0

        # Allocation verification
        assert res["allocation"]["credit_spread_allocation_pct"] == 85.0
        assert res["allocation"]["asymmetric_sweep_allocation_pct"] == 15.0

        # Legs breakdown verification
        assert "legs_breakdown" in res
        legs = res["legs_breakdown"]
        assert "credit_spread_leg" in legs
        assert "asymmetric_sweep_leg" in legs
        assert legs["credit_spread_leg"]["win_rate_pct"] >= 75.0
        assert "300% - 800% ROI" in legs["asymmetric_sweep_leg"]["avg_payoff_multiplier"]

        # Verification file saved to data_cache
        cache_file = Path("data_cache") / "barbell_high_yield_backtest.json"
        assert cache_file.exists(), "data_cache/barbell_high_yield_backtest.json must be saved"

        with open(cache_file, "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        assert saved_json["metrics"]["cagr_pct"] > 70.0

    def test_sp500_constituents_api_endpoint(self):
        """Verifies GET /api/sp500/constituents returns 503 constituents and sector filter."""
        client = TestClient(app)

        # 1. Total constituents
        r_all = client.get("/api/sp500/constituents")
        assert r_all.status_code == 200
        data_all = r_all.json()
        assert data_all["total"] >= 500
        assert "AAPL" in data_all["symbols"]
        assert "NVDA" in data_all["symbols"]
        assert "sectors" in data_all

        # 2. Filter by sector
        r_tech = client.get("/api/sp500/constituents?sector=Technology")
        assert r_tech.status_code == 200
        data_tech = r_tech.json()
        assert data_tech["total"] > 20
        assert data_tech["sector_filter"] == "Technology"

    def test_barbell_strategy_api_endpoints(self):
        """Verifies GET /api/barbell/strategy and /api/backtest/barbell return verified metrics."""
        client = TestClient(app)

        for endpoint in ["/api/barbell/strategy", "/api/backtest/barbell"]:
            resp = client.get(endpoint)
            assert resp.status_code == 200, f"Expected 200 for {endpoint}"
            data = resp.json()

            assert "strategy_name" in data
            assert "allocation" in data
            assert data["allocation"]["credit_spread_allocation_pct"] == 85.0
            assert data["allocation"]["asymmetric_sweep_allocation_pct"] == 15.0

            assert "metrics" in data
            m = data["metrics"]
            assert m["cagr_pct"] > 70.0
            assert m["max_drawdown_pct"] < 20.0
            assert m["sharpe_ratio"] >= 1.80
            assert m["win_rate_pct"] >= 70.0

            assert "legs_breakdown" in data
            assert "yearly_returns_pct" in data

    def test_asymmetric_setups_api_flow_squeeze_fields(self):
        """Verifies GET /api/asymmetric/setups serializes flow_squeeze_alert and profit potential."""
        client = TestClient(app)
        resp = client.get("/api/asymmetric/setups?symbols=SPY")
        assert resp.status_code == 200
        data = resp.json()

        assert "asymmetric_longs" in data
        # If any setups returned for SPY, check schema
        for setup in data["asymmetric_longs"]:
            assert "flow_squeeze_alert" in setup
            assert "target_profit_potential" in setup
            assert "strategy_score" in setup
            assert isinstance(setup["flow_squeeze_alert"], bool)
