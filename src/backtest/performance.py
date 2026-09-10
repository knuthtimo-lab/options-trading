"""
Performance & Risk Metrics Calculator
Calculates CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, Profit Factor,
and generate summary reports.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win_dollar: float
    avg_loss_dollar: float
    win_loss_ratio: float
    avg_trade_days: float
    annual_turnover: float


class PerformanceCalculator:
    @staticmethod
    def calculate(
        daily_equity: pd.Series,
        closed_trades: List[Dict[str, Any]],
        risk_free_rate: float = 0.045,
    ) -> BacktestMetrics:
        """Computes comprehensive performance statistics from daily equity series and trade logs."""
        if daily_equity.empty or len(daily_equity) < 2:
            return BacktestMetrics(
                total_return_pct=0.0, cagr_pct=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown_pct=0.0, win_rate_pct=0.0, profit_factor=0.0, total_trades=0,
                winning_trades=0, losing_trades=0, avg_win_dollar=0.0, avg_loss_dollar=0.0,
                win_loss_ratio=0.0, avg_trade_days=0.0, annual_turnover=0.0
            )

        start_val = float(daily_equity.iloc[0])
        end_val = float(daily_equity.iloc[-1])
        total_return_pct = ((end_val - start_val) / start_val) * 100.0

        # Time span in years
        num_days = (daily_equity.index[-1] - daily_equity.index[0]).days
        years = max(0.1, num_days / 365.25)
        cagr_pct = ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0

        # Daily returns
        daily_returns = daily_equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess_daily = daily_returns - (risk_free_rate / 252.0)
            sharpe = float(np.sqrt(252.0) * excess_daily.mean() / daily_returns.std())
            
            # Sortino: downside deviation
            downside = daily_returns[daily_returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino = float(np.sqrt(252.0) * excess_daily.mean() / downside.std())
            else:
                sortino = sharpe
        else:
            sharpe = 0.0
            sortino = 0.0

        # Max Drawdown
        cum_max = daily_equity.cummax()
        drawdowns = (daily_equity - cum_max) / cum_max
        max_drawdown_pct = float(abs(drawdowns.min()) * 100.0)

        # Trade metrics
        total_trades = len(closed_trades)
        if total_trades > 0:
            df_trades = pd.DataFrame(closed_trades)
            wins = df_trades[df_trades['pnl_dollar'] > 0]
            losses = df_trades[df_trades['pnl_dollar'] <= 0]
            num_wins = len(wins)
            num_losses = len(losses)
            win_rate = (num_wins / total_trades) * 100.0

            gross_profit = float(wins['pnl_dollar'].sum()) if not wins.empty else 0.0
            gross_loss = float(abs(losses['pnl_dollar'].sum())) if not losses.empty else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)

            avg_win = float(wins['pnl_dollar'].mean()) if not wins.empty else 0.0
            avg_loss = float(abs(losses['pnl_dollar'].mean())) if not losses.empty else 0.0
            win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

            avg_days = float(df_trades['holding_days'].mean()) if 'holding_days' in df_trades else 15.0
            annual_turnover = total_trades / years
        else:
            win_rate = 0.0
            num_wins = 0
            num_losses = 0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            win_loss_ratio = 0.0
            avg_days = 0.0
            annual_turnover = 0.0

        return BacktestMetrics(
            total_return_pct=round(total_return_pct, 2),
            cagr_pct=round(cagr_pct, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            win_rate_pct=round(win_rate, 1),
            profit_factor=round(profit_factor, 2),
            total_trades=total_trades,
            winning_trades=num_wins,
            losing_trades=num_losses,
            avg_win_dollar=round(avg_win, 2),
            avg_loss_dollar=round(avg_loss, 2),
            win_loss_ratio=round(win_loss_ratio, 2),
            avg_trade_days=round(avg_days, 1),
            annual_turnover=round(annual_turnover, 1),
        )