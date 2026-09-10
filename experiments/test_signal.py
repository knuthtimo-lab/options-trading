from src.strategy.signal_generator import SignalGenerator

sig = SignalGenerator.analyze_ticker("SPY")
if sig and sig.trade:
    t = sig.trade
    print("=" * 65)
    print(f"TICKER:   {sig.symbol} (Spot: ${t.spot_price:.2f})")
    print(f"ACTION:   {t.action}")
    print(f"STRATEGY: {t.strategy_name}")
    print(f"LEGS:     {t.legs_summary}")
    print(f"EXPIRY:   {t.expiration} ({t.dte} DTE)")
    print(f"ENTRY:    ${t.entry_limit_price:.2f} (Limit)")
    print(f"TARGET:   ${t.target_exit_price:.2f} (50% Profit Take)")
    print(f"STOP:     ${t.stop_loss_price:.2f} (Risk Cut)")
    print(f"MAX PROF: ${t.max_profit_dollar:.0f} | MAX LOSS: ${t.max_loss_dollar:.0f}")
    print(f"POP:      {t.probability_of_profit_pct:.1f}% | ROC: {t.return_on_capital_pct:.1f}%")
    print(f"THETA:    +${t.net_theta_daily_dollar:.2f}/day decay harvest")
    print(f"VANNA:    {t.net_vanna:.2f}")
    print(f"WHY:      {t.reasoning}")
    print("=" * 65)
else:
    print("No trade found or error:", sig)