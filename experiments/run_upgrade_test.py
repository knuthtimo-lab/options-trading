from src.data.historical_feed import HistoricalDataFeed
from src.backtest.simulator import OptionsBacktester, BacktestConfig

feed = HistoricalDataFeed(cache_dir="data_cache")
spy_df = feed.get_historical_dataset("SPY", start_date="2020-01-01")
qqq_df = feed.get_historical_dataset("QQQ", start_date="2020-01-01")
aapl_df = feed.get_historical_dataset("AAPL", start_date="2020-01-01")
nvda_df = feed.get_historical_dataset("NVDA", start_date="2020-01-01")

datasets = {
    "SPY": spy_df,
    "QQQ": qqq_df,
    "AAPL": aapl_df,
    "NVDA": nvda_df,
}

config = BacktestConfig(
    initial_capital=25000.0,
    allocation_pct_per_trade=0.18,
    max_open_positions=6,
    target_short_delta=0.16,
    profit_target_pct=0.50,
    stop_loss_mult=2.00,
    entry_dte=32,
    min_iv_rank_to_sell=25.0,
    enable_option_buying=True,
    debit_profit_target_pct=0.75,
    debit_stop_loss_pct=0.40,
    max_iv_rank_to_buy=25.0,
)

bt = OptionsBacktester(config)
equity, trades, m = bt.run(datasets)

print("=" * 60)
print("MULTI-ASSET OPTIONS PORTFOLIO BACKTEST (2020-2026)")
print("Assets: SPY, QQQ, AAPL, NVDA (Diversified US Tech & Index)")
print("=" * 60)
print(f"Initial Capital:    ${config.initial_capital:,.2f}")
print(f"Final Equity:       ${equity.iloc[-1]:,.2f}")
print(f"Total Return:       {m.total_return_pct:+.2f}%")
print(f"CAGR (Annualized):  {m.cagr_pct:+.2f}% p.a.")
print(f"Sharpe Ratio:       {m.sharpe_ratio:.2f}")
print(f"Sortino Ratio:      {m.sortino_ratio:.2f}")
print(f"Max Drawdown:       {m.max_drawdown_pct:.2f}%")
print(f"Total Trades:       {m.total_trades}")
print(f"Winning Trades:     {m.winning_trades} ({m.win_rate_pct:.1f}%)")
print(f"Losing Trades:      {m.losing_trades}")
print(f"Profit Factor:      {m.profit_factor:.2f}")
print(f"Avg Trade Days:     {m.avg_trade_days:.1f} days")
print(f"Annual Turnover:    {m.annual_turnover:.1f} trades/yr")
print("=" * 60)