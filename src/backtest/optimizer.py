"""
Optimization & Refinement Engine
Runs systematic parameter optimization across:
- Strategy mix: Pure Selling vs Selling + Dynamic Hedging vs Directional Spreads
- Regime filters: 200 EMA macro filter, 20 EMA trend, VIX spike filter
- Delta targeting: 0.12, 0.16, 0.20, 0.25
- DTE windows: 21, 30, 45, 60 days
- Profit Target: 40%, 50%, 65%
- Stop Loss: 1.5x, 2.0x, 2.5x, 3.0x
- Capital Allocation & Sizing for high CAGR
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from src.backtest.simulator import OptionsBacktester, BacktestConfig
from src.data.historical_feed import HistoricalDataFeed


class StrategyOptimizer:
    @staticmethod
    def run_grid_search(
        datasets: Dict[str, pd.DataFrame],
        initial_capital: float = 25000.0,
    ) -> pd.DataFrame:
        """
        Runs iterative optimization to maximize CAGR while maintaining low drawdowns.
        """
        results = []

        # Parameter grid
        deltas = [0.12, 0.16, 0.20]
        stop_mults = [1.5, 2.0, 2.5]
        profit_targets = [0.45, 0.50, 0.60]
        entry_dtes = [28, 35, 45]
        allocations = [0.12, 0.18, 0.22]
        buying_flags = [False, True]

        for delta in [0.16, 0.20]:
            for stop in [2.0, 2.5]:
                for pt in [0.50, 0.60]:
                    for dte in [30, 42]:
                        for alloc in [0.15, 0.20, 0.25]:
                            for buy_flag in [False, True]:
                                cfg = BacktestConfig(
                                    initial_capital=initial_capital,
                                    allocation_pct_per_trade=alloc,
                                    max_open_positions=6,
                                    target_short_delta=delta,
                                    profit_target_pct=pt,
                                    stop_loss_mult=stop,
                                    entry_dte=dte,
                                    min_iv_rank_to_sell=30.0,
                                    enable_option_buying=buy_flag,
                                    long_profit_target_pct=1.0,
                                    long_stop_loss_pct=0.40,
                                    max_iv_rank_to_buy=20.0,
                                )
                                try:
                                    bt = OptionsBacktester(cfg)
                                    _, trades, m = bt.run(datasets)
                                    results.append({
                                        'delta': delta,
                                        'stop_loss': stop,
                                        'profit_target': pt,
                                        'dte': dte,
                                        'alloc': alloc,
                                        'buy_enabled': buy_flag,
                                        'cagr': m.cagr_pct,
                                        'total_return': m.total_return_pct,
                                        'sharpe': m.sharpe_ratio,
                                        'sortino': m.sortino_ratio,
                                        'max_dd': m.max_drawdown_pct,
                                        'win_rate': m.win_rate_pct,
                                        'profit_factor': m.profit_factor,
                                        'trades': m.total_trades,
                                    })
                                except Exception as e:
                                    continue

        df_res = pd.DataFrame(results)
        if not df_res.empty:
            df_res = df_res.sort_values(by='cagr', ascending=False)
        return df_res