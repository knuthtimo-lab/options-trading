from src.data.historical_feed import HistoricalDataFeed
from src.backtest.simulator import OptionsBacktester, BacktestConfig

feed = HistoricalDataFeed(cache_dir="data_cache")

symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD"]
datasets = {}
for s in symbols:
    print(f"Loading {s}...")
    datasets[s] = feed.get_historical_dataset(s, start_date="2020-01-01")

print("\n--- RUNNING PARAMETER EXPLORATION FOR 70%+ CAGR ---")

configs_to_test = [
    # (alloc, max_pos, dte, pt, stop, debit_pt, iv_sell, desc)
    (0.20, 6, 28, 0.50, 2.0, 0.85, 25.0, "Config A: 28 DTE, 20% Alloc, 50% PT"),
    (0.22, 6, 25, 0.50, 1.8, 0.90, 25.0, "Config B: 25 DTE Fast Theta, 22% Alloc"),
    (0.20, 6, 30, 0.55, 2.2, 0.90, 22.0, "Config C: 30 DTE, 55% PT, 2.2 Stop"),
    (0.24, 5, 25, 0.45, 1.8, 0.80, 25.0, "Config D: 25 DTE Fast Flip 45% PT, 24% Alloc"),
    (0.22, 6, 28, 0.50, 2.2, 1.00, 20.0, "Config E: 28 DTE, Low IV Buy Trigger, 100% Debit PT"),
]

for alloc, max_pos, dte, pt, stop, debit_pt, iv_sell, desc in configs_to_test:
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