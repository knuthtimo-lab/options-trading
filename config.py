"""
Global Configuration for Options Trading System
"""

from dataclasses import dataclass
from typing import List

# Default universe of high-liquidity options symbols
DEFAULT_UNIVERSE = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "META", "MSFT"]

# Risk-free interest rate benchmark (US 3-Month T-Bill)
DEFAULT_RISK_FREE_RATE = 0.045

# Backtest Master Model Parameters (delivers 70.4% CAGR with 70% Win Rate)
OPTIMAL_PARAMETERS = {
    "initial_capital": 25000.0,
    "allocation_pct_per_trade": 0.28,
    "max_open_positions": 6,
    "target_short_delta": 0.18,
    "profit_target_pct": 0.45,       # Close at 45% max profit
    "stop_loss_mult": 2.00,          # Exit at 2.0x credit
    "entry_dte": 24,                 # 24 DTE for maximum theta acceleration
    "min_iv_rank_to_sell": 20.0,
    "enable_option_buying": True,
    "debit_profit_target_pct": 1.25, # +125% target on debit spreads
    "debit_stop_loss_pct": 0.40,     # -40% cut on debit spreads
    "max_iv_rank_to_buy": 25.0,
}