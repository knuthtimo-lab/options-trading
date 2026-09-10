import pytest
import pandas as pd
import numpy as np
from src.engine.unusual_greeks import UnusualGreeksEngine, GreekAnomaly, GreekStructureReport
from src.backtest.magnet_backtest import MagnetBacktestEngine


def test_unusual_greeks_engine_structure():
    """Verify that UnusualGreeksEngine properly parses and calculates structures on a liquid symbol."""
    report = UnusualGreeksEngine.analyze_ticker_anomalies("SPY", min_volume=1)
    
    # SPY always has options data available
    assert report is not None
    assert isinstance(report, GreekStructureReport)
    assert report.spot_price > 0.0
    assert report.primary_magnet_strike > 0.0
    assert report.magnet_pull_force in ["STRONG_PINNING", "MODERATE", "NEUTRAL"]
    assert report.total_gamma_volume_m >= 0.0
    assert report.call_resistance_strike > 0.0
    assert report.put_support_strike > 0.0


def test_magnet_backtest_execution():
    """Verify that MagnetBacktestEngine computes pinning stats and mean-reversion returns."""
    # Test on short window for speed
    res = MagnetBacktestEngine.run_magnet_backtest(
        symbols=["SPY"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        initial_capital=25000.0,
    )

    assert "initial_capital" in res
    assert "ending_equity" in res
    assert "pinning_stats" in res
    pin_stats = res["pinning_stats"]
    assert "total_cycles_tested" in pin_stats
    assert pin_stats["total_cycles_tested"] > 0
    assert "pin_accuracy_1pct_rate" in pin_stats
    assert 0.0 <= pin_stats["pin_accuracy_1pct_rate"] <= 100.0
