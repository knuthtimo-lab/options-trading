from src.data.historical_feed import HistoricalDataFeed
from src.backtest.simulator import OptionsBacktester, BacktestConfig

feed = HistoricalDataFeed(cache_dir="data_cache")

symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "META"]
datasets = {}
for s in symbols:
    print(f"Loading {s}...")
    datasets[s] = feed.get_historical_dataset(s, start_date="2020-01-01")

print("\n--- PUSHING PERFORMANCE TO 70%+ CAGR ---")

configs = [
    # (alloc, max_pos, dte, pt, stop, debit_pt, iv_sell, desc)
    (0.24, 6, 28, 0.45, 2.0, 1.00, 20.0, "Run 1: 24% Alloc, 45% PT Fast Flip, 28 DTE"),
    (0.25, 6, 25, 0.45, 1.8, 1.10, 20.0, "Run 2: 25% Alloc, 25 DTE, 110% Debit PT"),
    (0.26, 6, 28, 0.50, 2.0, 1.00, 20.0, "Run 3: 26% Alloc, 28 DTE, 50% PT"),
    (0.25, 7, 28, 0.48, 2.0, 1.00, 20.0, "Run 4: 25% Alloc, 7 Max Pos, 28 DTE"),
    (0.27, 6, 26, 0.45, 2.0, 1.10, 20.0, "Run 5: 27% Alloc, 26 DTE Fast Scalp"),
]

for alloc, max_pos, dte, pt, stop, debit_pt, iv_sell, desc in configs:
    cfg = BacktestConfig(
        initial_capital=25000.0,
        allocation_pct_per_trade=alloc,
        max_open_positions=max_pos,
        target_short_delta=0.16,
        profit_target_pct=pt,
        stop_loss_mult=stop,
        entry_dte=dte,
        min_iv_rank_to_sell=iv_sell,
        enable_option_buying=True,
        debit_profit_target_pct=debit_pt,
        debit_stop_loss_pct=0.40,
        max_iv_rank_to_buy=25.0,
    )
    bt = OptionsBacktester(cfg)
    eq, tr, m = bt.run(datasets)
    print(f"\n{desc}:")
    print(f"  CAGR:          {m.cagr_pct:+.2f}% p.a.  (Total: {m.total_return_pct:+.1f}%)")
    print(f"  Final Equity:  ${eq.iloc[-1]:,.2f}")
    print(f"  Sharpe Ratio:  {m.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:  {m.max_drawdown_pct:.2f}%")
    print(f"  Win Rate:      {m.win_rate_pct:.1f}% ({m.winning_trades}/{m.total_trades})")
    print(f"  Profit Factor: {m.profit_factor:.2f}")
    print(f"  Avg Duration:  {m.avg_trade_days:.1f} days")