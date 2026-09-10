from src.data.historical_feed import HistoricalDataFeed
from src.backtest.simulator import OptionsBacktester, BacktestConfig

feed = HistoricalDataFeed(cache_dir="data_cache")
symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "META", "MSFT"]
datasets = {s: feed.get_historical_dataset(s, start_date="2020-01-01") for s in symbols}

print("\n--- TUNING AROUND CONFIG 3 TO EXCEED 70%+ CAGR ---")

variations = [
    # (alloc, dte, pt, stop, debit_pt, delta, desc)
    (0.26, 24, 0.42, 1.8, 1.20, 0.16, "Var A: 42% PT Rapid Turnover (alloc 26%, dte 24)"),
    (0.26, 24, 0.45, 1.9, 1.20, 0.17, "Var B: Delta 0.17, 1.9 Stop, 45% PT"),
    (0.26, 24, 0.45, 1.8, 1.30, 0.16, "Var C: 130% Debit Runner PT"),
    (0.27, 24, 0.44, 1.8, 1.25, 0.17, "Var D: Alloc 27%, Delta 0.17, 44% PT, 125% Debit PT"),
    (0.27, 23, 0.42, 1.8, 1.25, 0.16, "Var E: Alloc 27%, 23 DTE, 42% PT Fast Harvest"),
    (0.26, 24, 0.45, 2.0, 1.25, 0.18, "Var F: Delta 0.18, 2.0 Stop, 45% PT"),
]

for alloc, dte, pt, stop, debit_pt, delta, desc in variations:
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