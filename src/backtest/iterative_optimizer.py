"""
Automated 20+ Iteration Backtesting Campaign and Strategy Benchmark Engine
Executes 20+ structured iterations across Delta, DTE, Profit Targets, Stop Loss Multipliers,
Convexity Hedging, IV Rank thresholds, and Position Sizing.
Outputs exhaustive metrics (CAGR, Sharpe, Sortino, Win Rate, Max DD, Profit Factor, Trade Count).
"""

import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import List, Dict, Any
import json
import time
import pandas as pd
import numpy as np

from src.backtest.simulator import OptionsBacktester, BacktestConfig
from src.data.historical_feed import HistoricalDataFeed


def run_20_iteration_campaign(export_json_path: str = "data_cache/backtest_20_iterations.json") -> List[Dict[str, Any]]:
    print("=" * 65)
    print("STARTING 20-ITERATION QUANTITATIVE OPTIONS BACKTEST CAMPAIGN")
    print("=" * 65)

    feed = HistoricalDataFeed(cache_dir="data_cache")
    symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "META", "MSFT"]
    datasets = {}
    for sym in symbols:
        try:
            df = feed.get_historical_dataset(sym, start_date="2021-01-01", end_date="2026-01-01")
            datasets[sym] = df
            print(f"Loaded {sym}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            print(f"Error loading {sym}: {e}")

    iterations_configs = [
        {"name": "Iter 01: Baseline Tastytrade (45 DTE, 50% PT, 2.0x SL)", "delta": 0.16, "dte": 45, "pt": 0.50, "sl": 2.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "Standard 45 DTE tastytrade benchmark mechanics"},
        {"name": "Iter 02: Quick Scalp Take-Profit (30% PT)", "delta": 0.16, "dte": 45, "pt": 0.30, "sl": 2.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "Locking early 30% credit reduces exposure time"},
        {"name": "Iter 03: Moderate Take-Profit (40% PT)", "delta": 0.16, "dte": 45, "pt": 0.40, "sl": 2.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "40% target captures rapid initial theta burn"},
        {"name": "Iter 04: Patient Extinction (60% PT)", "delta": 0.16, "dte": 45, "pt": 0.60, "sl": 2.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "Holding for 60% profit increases dollar return per trade"},
        {"name": "Iter 05: Maximum Decay Squeeze (75% PT)", "delta": 0.16, "dte": 45, "pt": 0.75, "sl": 2.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "Holding deep into expiration tests gamma risk vs premium"},
        {"name": "Iter 06: Aggressive Stop Loss (1.5x Credit)", "delta": 0.16, "dte": 45, "pt": 0.50, "sl": 1.5, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "Cutting losses at 1.5x prevents large single tail loss"},
        {"name": "Iter 07: Wide Stop Loss Buffer (2.5x Credit)", "delta": 0.16, "dte": 45, "pt": 0.50, "sl": 2.5, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "Giving trades breathing room to avoid whipsaws"},
        {"name": "Iter 08: Ultra-Wide Tolerance (3.0x Credit)", "delta": 0.16, "dte": 45, "pt": 0.50, "sl": 3.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "3.0x stop loss tests win rate preservation vs tail pain"},
        {"name": "Iter 09: Optimal Gamma/Theta Window (28 DTE)", "delta": 0.16, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "28 DTE enters the steepest curvature of theta decay"},
        {"name": "Iter 10: High-Gamma Sprint (21 DTE)", "delta": 0.16, "dte": 21, "pt": 0.50, "sl": 2.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "21 DTE maximizes turnover with higher gamma risk"},
        {"name": "Iter 11: Low-Gamma Float (60 DTE)", "delta": 0.16, "dte": 60, "pt": 0.50, "sl": 2.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "60 DTE minimizes gamma volatility but slows portfolio velocity"},
        {"name": "Iter 12: High-Margin Ultra-OTM (Delta 0.12)", "delta": 0.12, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "Delta 0.12 provides >88% theoretical probability buffer"},
        {"name": "Iter 13: High-Yield Near-the-Money (Delta 0.22)", "delta": 0.22, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.15, "buy": False, "ivr_sell": 20.0, "hypothesis": "Delta 0.22 extracts higher absolute dollar credits"},
        {"name": "Iter 14: Hybrid Selling + Convex Debit Spreads", "delta": 0.16, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.15, "buy": True, "ivr_sell": 20.0, "hypothesis": "Adds long debit spreads when IV rank is low to capture convexity"},
        {"name": "Iter 15: High-IV Threshold Sieve (IVR >= 28%)", "delta": 0.16, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.15, "buy": True, "ivr_sell": 28.0, "hypothesis": "Only sells premium when volatility is rich (>28th percentile)"},
        {"name": "Iter 16: Defensive Sizing (10% Allocation / Trade)", "delta": 0.16, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.10, "buy": True, "ivr_sell": 20.0, "hypothesis": "Quarter-Kelly conservative capital deployment"},
        {"name": "Iter 17: Balanced Sizing (18% Allocation / Trade)", "delta": 0.16, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.18, "buy": True, "ivr_sell": 20.0, "hypothesis": "Half-Kelly allocation to compound capital steadily"},
        {"name": "Iter 18: Aggressive Growth Sizing (25% Allocation / Trade)", "delta": 0.16, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.25, "buy": True, "ivr_sell": 20.0, "hypothesis": "Higher capital utilization for maximum CAGR"},
        {"name": "Iter 19: High-Conviction Sieve (IVR >= 25% + 20% Alloc)", "delta": 0.16, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.20, "buy": True, "ivr_sell": 25.0, "hypothesis": "Combined sweet-spot of IV filter and balanced position sizing"},
        {"name": "Iter 20: Institutional Master Model (Champion 22% Alloc)", "delta": 0.16, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.22, "buy": True, "ivr_sell": 20.0, "hypothesis": "Full institutional production setup with optimal velocity and risk parity"},
        {"name": "Iter 21: Asymmetric Long Target Extension (120% Long PT)", "delta": 0.16, "dte": 28, "pt": 0.50, "sl": 2.0, "alloc": 0.22, "buy": True, "ivr_sell": 20.0, "hypothesis": "Extending long debit profit target to 120% on breakout regimes"},
    ]

    results = []
    t0_all = time.time()

    for idx, item in enumerate(iterations_configs, 1):
        t0 = time.time()
        cfg = BacktestConfig(
            initial_capital=25000.0,
            allocation_pct_per_trade=item["alloc"],
            max_open_positions=6,
            target_short_delta=item["delta"],
            profit_target_pct=item["pt"],
            stop_loss_mult=item["sl"],
            entry_dte=item["dte"],
            min_iv_rank_to_sell=item["ivr_sell"],
            enable_option_buying=item["buy"],
            debit_profit_target_pct=1.20 if idx == 21 else 1.00,
            debit_stop_loss_pct=0.40,
            max_iv_rank_to_buy=24.0,
            slippage_per_contract=1.50,
        )

        bt = OptionsBacktester(cfg)
        equity_series, trades, m = bt.run(datasets)
        elapsed = time.time() - t0

        row = {
            "iteration": idx,
            "name": item["name"],
            "hypothesis": item["hypothesis"],
            "delta": item["delta"],
            "dte": item["dte"],
            "profit_target_pct": int(item["pt"] * 100),
            "stop_loss_mult": item["sl"],
            "allocation_pct": int(item["alloc"] * 100),
            "buying_enabled": item["buy"],
            "cagr_pct": round(m.cagr_pct, 2),
            "total_return_pct": round(m.total_return_pct, 1),
            "sharpe_ratio": round(m.sharpe_ratio, 2),
            "sortino_ratio": round(m.sortino_ratio, 2),
            "max_drawdown_pct": round(m.max_drawdown_pct, 1),
            "win_rate_pct": round(m.win_rate_pct, 1),
            "profit_factor": round(m.profit_factor, 2),
            "total_trades": m.total_trades,
            "final_equity": round(float(equity_series.iloc[-1]) if not equity_series.empty else cfg.initial_capital, 2),
            "duration_sec": round(elapsed, 2),
        }
        results.append(row)
        print(f"[{idx:02d}/21] {item['name'][:42]:<42} | CAGR: {row['cagr_pct']:>5}% | Sharpe: {row['sharpe_ratio']:>4} | Win: {row['win_rate_pct']:>5}% | MaxDD: {row['max_drawdown_pct']:>5}% | PF: {row['profit_factor']:>4} ({elapsed:.1f}s)")

    total_time = time.time() - t0_all
    print(f"\nAll 21 iterations completed in {total_time:.2f} seconds.")

    out_file = Path(export_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_iterations": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iterations": results,
        }, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv(out_file.with_suffix(".csv"), index=False)
    print(f"Saved benchmark data to {out_file} and {out_file.with_suffix('.csv')}")
    return results


if __name__ == "__main__":
    run_20_iteration_campaign()
