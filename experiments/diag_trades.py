import pandas as pd
from test_backtest_run import trades

df = pd.DataFrame(trades)
print(df.groupby('strategy')[['pnl_dollar', 'contracts']].agg(['count', 'sum', 'mean']))
print("\nClose reasons breakdown:")
print(df.groupby(['strategy', 'close_reason'])['pnl_dollar'].agg(['count', 'sum', 'mean']))