from src.data.historical_feed import HistoricalDataFeed
from src.backtest.simulator import OptionsBacktester, BacktestConfig

feed = HistoricalDataFeed(cache_dir="data_cache")
symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "META", "MSFT"]
datasets = {s: feed.get_historical_dataset(s, start_date="2020-01-01") for s in symbols}

print("\n--- FINAL OPTIMIZATION: COMBINING VAR D & F FOR ULTIMATE METRICS ---")

candidates = [
    # (alloc, dte, pt, stop, debit_pt, delta, desc)
    (0.27, 24, 0.45, 1.95, 1.25, 0.18, "Master Model 1: Alloc 27%, Delta 0.18, Stop 1.95x, PT 45%"),
    (0.275, 24, 0.44, 1.90, 1.25, 0.18, "Master Model 2: Alloc 27.5%, Delta 0.18, Stop 1.90x, PT 44%"),
    (0.28, 24, 0.45, 2.00, 1.25, 0.18, "Master Model 3: Alloc 28%, Delta 0.18, Stop 2.00x, PT 45%"),
]

for alloc, dte, pt, stop, debit_pt, delta, desc in candidates:
    cfg = BacktestConfig(
        initial_capital=25000.0,
        allocation_pct_per_trade=alloc,
        max_open_positions=6,
        target_short_delta=delta,
        profit_target_pct=pt,
        stop_loss_mult=stop,
        entry_dte=dte,
        min_iv_rank_to_sell=20.0,
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