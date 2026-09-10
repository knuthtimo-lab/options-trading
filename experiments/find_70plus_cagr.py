from src.data.historical_feed import HistoricalDataFeed
from src.backtest.simulator import OptionsBacktester, BacktestConfig

feed = HistoricalDataFeed(cache_dir="data_cache")
symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "META", "MSFT"]
datasets = {}
for s in symbols:
    datasets[s] = feed.get_historical_dataset(s, start_date="2020-01-01")

print(f"Loaded {len(datasets)} universe symbols.")

runs = [
    # (alloc, max_pos, dte, pt, stop, debit_pt, iv_rank_buy, desc)
    (0.24, 6, 25, 0.45, 1.8, 1.10, 24.0, "Config 1: 24% Alloc, 25 DTE, 45% PT, 110% Debit PT"),
    (0.25, 6, 25, 0.48, 2.0, 1.15, 24.0, "Config 2: 25% Alloc, 25 DTE, 48% PT, 115% Debit PT"),
    (0.26, 6, 24, 0.45, 1.8, 1.20, 25.0, "Config 3: 26% Alloc, 24 DTE, 45% PT, 120% Debit PT"),
    (0.27, 6, 25, 0.50, 2.0, 1.25, 25.0, "Config 4: 27% Alloc, 25 DTE, 50% PT, 125% Debit PT"),
    (0.28, 6, 22, 0.45, 1.8, 1.20, 24.0, "Config 5: 28% Alloc, 22 DTE Ultra-Fast Theta"),
    (0.30, 5, 24, 0.45, 1.8, 1.20, 25.0, "Config 6: 30% Alloc, 5 Pos, 24 DTE Concentrated"),
]

for alloc, max_pos, dte, pt, stop, debit_pt, iv_buy, desc in runs:
    cfg = BacktestConfig(
        initial_capital=25000.0,
        allocation_pct_per_trade=alloc,
        max_open_positions=max_pos,
        target_short_delta=0.16,
        profit_target_pct=pt,
        stop_loss_mult=stop,
        entry_dte=dte,
        min_iv_rank_to_sell=20.0,
        enable_option_buying=True,
        debit_profit_target_pct=debit_pt,
        debit_stop_loss_pct=0.40,
        max_iv_rank_to_buy=iv_buy,
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