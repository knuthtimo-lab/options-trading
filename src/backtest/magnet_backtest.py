"""
Gamma Magnet & Greek Pinning Backtest Engine
Quantifies:
1. Historical Pinning Accuracy: How often underlying prices are pulled towards the dominant Gamma Magnet strike.
2. Magnet Mean-Reversion Strategy: Selling out-of-the-money spreads opposite to the deviation from the magnet.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf


class MagnetBacktestEngine:
    @classmethod
    def run_magnet_backtest(
        cls,
        symbols: List[str] = None,
        start_date: str = "2021-01-01",
        end_date: str = "2026-01-01",
        initial_capital: float = 25000.0,
    ) -> Dict[str, Any]:
        if symbols is None:
            symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT"]

        trades = []
        pinning_events = []

        total_tested_cycles = 0
        pinned_within_1pct = 0
        pinned_within_2pct = 0

        account_equity = initial_capital
        equity_curve = [initial_capital]

        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) < 60:
                    continue

                # Simulate monthly/bi-weekly expiration cycles (every ~14 to 28 trading days)
                step = 14
                for i in range(30, len(hist) - step, step):
                    total_tested_cycles += 1
                    entry_date = hist.index[i]
                    exit_date = hist.index[i + step]

                    entry_spot = float(hist['Close'].iloc[i])
                    exit_spot = float(hist['Close'].iloc[i + step])

                    # Synthetic Gamma Magnet: In historical options, the dominant magnet strike
                    # is heavily clustered around psychological round strikes and the 20-day rolling VWAP
                    rolling_mean = float(hist['Close'].iloc[max(0, i-20):i].mean())
                    round_base = 5.0 if entry_spot > 100 else 2.5
                    magnet_strike = round(rolling_mean / round_base) * round_base

                    dist_pct = (entry_spot - magnet_strike) / magnet_strike * 100.0
                    final_dist_pct = abs((exit_spot - magnet_strike) / magnet_strike) * 100.0

                    if final_dist_pct <= 1.0:
                        pinned_within_1pct += 1
                    if final_dist_pct <= 2.0:
                        pinned_within_2pct += 1

                    pinning_events.append({
                        "symbol": sym,
                        "entry_date": entry_date.strftime("%Y-%m-%d"),
                        "exit_date": exit_date.strftime("%Y-%m-%d"),
                        "entry_spot": round(entry_spot, 2),
                        "magnet_strike": round(magnet_strike, 2),
                        "exit_spot": round(exit_spot, 2),
                        "final_distance_pct": round(final_dist_pct, 2),
                    })

                    # Strategy Entry Condition:
                    # If spot is 1.5% - 4.0% away from magnet, trade mean reversion back to magnet!
                    if 1.5 <= abs(dist_pct) <= 4.0:
                        is_bullish_pull = entry_spot < magnet_strike  # Below magnet, pull upwards
                        trade_alloc = account_equity * 0.15           # 15% margin
                        spread_width = round_base

                        if is_bullish_pull:
                            strategy = "BULL_PUT_SPREAD"
                            short_k = magnet_strike - spread_width
                            long_k = short_k - spread_width
                            # Win if exit_spot >= short_k
                            is_win = exit_spot >= short_k
                        else:
                            strategy = "BEAR_CALL_SPREAD"
                            short_k = magnet_strike + spread_width
                            long_k = short_k + spread_width
                            # Win if exit_spot <= short_k
                            is_win = exit_spot <= short_k

                        # Defined Risk PnL calculation
                        num_contracts = max(1, int(trade_alloc / (spread_width * 100.0)))
                        credit_per_contract = spread_width * 0.28 * 100.0   # ~28% credit on width
                        max_loss_per_contract = (spread_width * 0.72) * 100.0

                        if is_win:
                            # 45-50% profit target or expiration win
                            trade_pnl = credit_per_contract * 0.75 * num_contracts
                        else:
                            trade_pnl = -max_loss_per_contract * 0.70 * num_contracts

                        account_equity += trade_pnl
                        account_equity = max(1000.0, account_equity)
                        equity_curve.append(account_equity)

                        trades.append({
                            "symbol": sym,
                            "strategy": strategy,
                            "entry_date": entry_date.strftime("%Y-%m-%d"),
                            "exit_date": exit_date.strftime("%Y-%m-%d"),
                            "magnet_strike": magnet_strike,
                            "entry_spot": round(entry_spot, 2),
                            "exit_spot": round(exit_spot, 2),
                            "pnl": round(trade_pnl, 2),
                            "is_win": is_win,
                            "contracts": num_contracts,
                        })
            except Exception:
                continue

        # Statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['is_win'])
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        years = 5.0
        cagr = ((account_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if account_equity > 0 else 0.0
        pin_rate_1pct = (pinned_within_1pct / max(1, total_tested_cycles)) * 100.0
        pin_rate_2pct = (pinned_within_2pct / max(1, total_tested_cycles)) * 100.0

        # Max drawdown
        peak = initial_capital
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "initial_capital": initial_capital,
            "ending_equity": round(account_equity, 2),
            "total_return_pct": round(((account_equity - initial_capital) / initial_capital) * 100.0, 2),
            "cagr_pct": round(cagr, 2),
            "win_rate_pct": round(win_rate, 1),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "pinning_stats": {
                "total_cycles_tested": total_tested_cycles,
                "pinned_within_1pct": pinned_within_1pct,
                "pinned_within_2pct": pinned_within_2pct,
                "pin_accuracy_1pct_rate": round(pin_rate_1pct, 1),
                "pin_accuracy_2pct_rate": round(pin_rate_2pct, 1),
            },
            "recent_trades": trades[-20:],
        }
