"""
Unit tests for Simulator & PerformanceCalculator
"""

import pytest
import numpy as np
import pandas as pd
from src.backtest.performance import PerformanceCalculator
from src.backtest.simulator import OptionsBacktester, BacktestConfig


def test_performance_calculator():
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
    # Linear growth from 10,000 to 17,000 (+70%)
    values = np.linspace(10000.0, 17000.0, len(dates))
    equity = pd.Series(values, index=dates)

    trades = [
        {"pnl_dollar": 500.0, "holding_days": 10},
        {"pnl_dollar": 400.0, "holding_days": 8},
        {"pnl_dollar": -200.0, "holding_days": 12},
    ]

    m = PerformanceCalculator.calculate(equity, trades)
    assert m.total_return_pct > 65.0
    assert m.win_rate_pct > 60.0
    assert m.profit_factor > 2.0
    assert m.total_trades == 3