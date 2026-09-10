from src.data.historical_feed import HistoricalDataFeed
from src.backtest.simulator import OptionsBacktester, BacktestConfig

print("Fetching historical datasets for SPY and QQQ...")
feed = HistoricalDataFeed(cache_dir="data_cache")
spy_df = feed.get_historical_dataset("SPY", start_date="2020-01-01")
qqq_df = feed.get_historical_dataset("QQQ", start_date="2020-01-01")

print(f"SPY rows: {len(spy_df)} ({spy_df.index[0].date()} to {spy_df.index[-1].date()})")
print(f"QQQ rows: {len(qqq_df)} ({qqq_df.index[0].date()} to {qqq_df.index[-1].date()})")

config = BacktestConfig(
    initial_capital=25000.0,
    allocation_pct_per_trade=0.15,
    max_open_positions=6,
    profit_target_pct=0.50,
    stop_loss_mult=1.75,
    entry_dte=35,
    min_iv_rank_to_sell=35.0,
    enable_option_buying=True,
    max_iv_rank_to_buy=25.0,
)

backtester = OptionsBacktester(config)
equity, trades, metrics = backtester.run({"SPY": spy_df, "QQQ": qqq_df})

print("\n" + "=" * 60)
print("BASELINE BACKTEST RESULTS (2020-2026)")
print("=" * 60)
print(f"Total Return:       {metrics.total_return_pct:+.2f}%")
print(f"CAGR (Annualized):  {metrics.cagr_pct:+.2f}% p.a.")
print(f"Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
print(f"Sortino Ratio:      {metrics.sortino_ratio:.2f}")
print(f"Max Drawdown:       {metrics.max_drawdown_pct:.2f}%")
print(f"Total Trades:       {metrics.total_trades}")
print(f"Win Rate:           {metrics.win_rate_pct:.1f}%")
print(f"Profit Factor:      {metrics.profit_factor:.2f}")
print(f"Avg Trade Days:     {metrics.avg_trade_days:.1f} days")
print(f"Annual Turnover:    {metrics.annual_turnover:.1f} trades/yr")
print("=" * 60)