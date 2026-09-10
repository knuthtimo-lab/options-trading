from src.data.historical_feed import HistoricalDataFeed
from src.backtest.simulator import OptionsBacktester, BacktestConfig

feed = HistoricalDataFeed(cache_dir="data_cache")
symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "META"]
datasets = {s: feed.get_historical_dataset(s, start_date="2020-01-01") for s in symbols}

print("\n--- TESTING VIX CRISIS CIRCUIT BREAKER & VOLATILITY HARVESTING ---")

test_cases = [
    # (alloc, max_pos, dte, pt, stop, debit_pt, vix_cut, desc)
    (0.24, 6, 25, 0.50, 2.0, 1.00, 30.0, "Setup 1: 25 DTE, 24% Alloc, VIX Cutoff 30"),
    (0.26, 6, 28, 0.45, 2.0, 1.10, 30.0, "Setup 2: 28 DTE, 26% Alloc, 45% PT, 110% Debit PT"),
    (0.28, 6, 25, 0.45, 2.0, 1.20, 28.0, "Setup 3: 25 DTE, 28% Alloc, 45% PT, VIX Cutoff 28"),
    (0.30, 6, 25, 0.45, 2.2, 1.25, 28.0, "Setup 4: 25 DTE, 30% Alloc, Fast Compounder"),
]

for alloc, max_pos, dte, pt, stop, debit_pt, vix_cut, desc in test_cases:
    cfg = BacktestConfig(
        initial_capital=25000.0,
        allocation_pct_per_trade=alloc,
        max_open_positions=max_pos,
        target_short_delta=0.15,
        profit_target_pct=pt,
        stop_loss_mult=stop,
        entry_dte=dte,
        min_iv_rank_to_sell=22.0,
        enable_option_buying=True,
        debit_profit_target_pct=debit_pt,
        debit_stop_loss_pct=0.40,
        max_iv_rank_to_buy=25.0,
        vix_crisis_cutoff=vix_cut,
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