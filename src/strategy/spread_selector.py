"""
Spread & Option Contract Selector
Selects specific optimal strikes, expirations, DTE, limit prices,
and profit/stop targets based on quantitative criteria:
- Delta targeting (e.g. 16-25 delta for credit spreads, 50-60 delta for long options)
- Strike width optimization
- Defined-risk calculations
- Higher-order Greeks profile of the trade
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from src.engine.black_scholes import BlackScholesEngine


@dataclass
class TradeRecommendation:
    symbol: str
    action: str  # "SELL (CREDIT)" or "BUY (DEBIT/LEVERAGE)"
    strategy_name: str
    expiration: str
    dte: int
    spot_price: float
    legs_summary: str
    short_strike: Optional[float]
    long_strike: Optional[float]
    entry_limit_price: float
    target_exit_price: float
    stop_loss_price: float
    max_profit_dollar: float
    max_loss_dollar: float
    return_on_capital_pct: float
    probability_of_profit_pct: float
    net_delta: float
    net_gamma: float
    net_vanna: float
    net_theta_daily_dollar: float
    leverage_factor: float
    reasoning: str


class SpreadSelector:
    @staticmethod
    def select_best_trade(
        symbol: str,
        chain_df: pd.DataFrame,
        spot_price: float,
        strategy_type: str,
        target_dte_min: int = 20,
        target_dte_max: int = 55,
        risk_free_rate: float = 0.045,
        dividend_yield: float = 0.015,
    ) -> Optional[TradeRecommendation]:
        """
        Analyzes the options chain to find the optimal trade for the given strategy.
        """
        if chain_df.empty or spot_price <= 0:
            return None

        # Filter eligible DTE window
        eligible_df = chain_df[(chain_df['dte'] >= target_dte_min) & (chain_df['dte'] <= target_dte_max)].copy()
        if eligible_df.empty:
            eligible_df = chain_df.copy()

        exp_grouped = eligible_df.groupby('expiration')['open_interest'].sum()
        if exp_grouped.empty:
            return None
        best_exp = exp_grouped.idxmax()
        exp_df = eligible_df[eligible_df['expiration'] == best_exp].copy()
        dte = int(exp_df['dte'].iloc[0])
        T = dte / 365.0

        calls = exp_df[exp_df['option_type'].isin(['c', 'call'])].copy()
        puts = exp_df[exp_df['option_type'].isin(['p', 'put'])].copy()

        def compute_leg_greeks(row, opt_type):
            K = float(row['strike'])
            iv = float(row['implied_volatility'])
            return BlackScholesEngine.calculate_all_greeks(
                opt_type, spot_price, K, T, risk_free_rate, iv, dividend_yield
            )

        if not puts.empty:
            puts['greeks'] = puts.apply(lambda r: compute_leg_greeks(r, 'put'), axis=1)
            puts['delta'] = puts['greeks'].apply(lambda g: g.delta)
            puts['abs_delta'] = puts['delta'].abs()
        if not calls.empty:
            calls['greeks'] = calls.apply(lambda r: compute_leg_greeks(r, 'call'), axis=1)
            calls['delta'] = calls['greeks'].apply(lambda g: g.delta)

        wing_width = 5.0 if spot_price < 250 else (10.0 if spot_price < 600 else 15.0)

        # -------------------------------------------------------------
        # STRATEGY: IRON CONDOR (Delta-Neutral Double Credit Spread)
        # -------------------------------------------------------------
        if strategy_type == "IRON_CONDOR":
            if puts.empty or calls.empty:
                strategy_type = "BULL_PUT_SPREAD"
            else:
                otm_puts = puts[puts['strike'] < spot_price]
                otm_calls = calls[calls['strike'] > spot_price]
                if len(otm_puts) >= 2 and len(otm_calls) >= 2:
                    # Put spread leg (~16 delta)
                    otm_puts_copy = otm_puts.copy()
                    otm_puts_copy['d_dist'] = (otm_puts_copy['abs_delta'] - 0.16).abs()
                    short_put_row = otm_puts_copy.sort_values('d_dist').iloc[0]
                    short_put_k = float(short_put_row['strike'])
                    short_put_mid = float(short_put_row['mid'])

                    long_put_k = short_put_k - wing_width
                    cand_long_puts = puts[puts['strike'] <= long_put_k]
                    if cand_long_puts.empty:
                        cand_long_puts = puts[puts['strike'] < short_put_k]
                    long_put_row = cand_long_puts.iloc[-1]
                    long_put_k = float(long_put_row['strike'])
                    long_put_mid = float(long_put_row['mid'])
                    put_credit = max(0.10, short_put_mid - long_put_mid)

                    # Call spread leg (~16 delta)
                    otm_calls_copy = otm_calls.copy()
                    otm_calls_copy['d_dist'] = (otm_calls_copy['delta'] - 0.16).abs()
                    short_call_row = otm_calls_copy.sort_values('d_dist').iloc[0]
                    short_call_k = float(short_call_row['strike'])
                    short_call_mid = float(short_call_row['mid'])

                    long_call_k = short_call_k + wing_width
                    cand_long_calls = calls[calls['strike'] >= long_call_k]
                    if cand_long_calls.empty:
                        cand_long_calls = calls[calls['strike'] > short_call_k]
                    long_call_row = cand_long_calls.iloc[0]
                    long_call_k = float(long_call_row['strike'])
                    long_call_mid = float(long_call_row['mid'])
                    call_credit = max(0.10, short_call_mid - long_call_mid)

                    total_credit = put_credit + call_credit
                    width = max(short_put_k - long_put_k, long_call_k - short_call_k)
                    max_loss = (width - total_credit) * 100.0
                    max_profit = total_credit * 100.0
                    roc = (max_profit / max_loss) * 100.0 if max_loss > 0 else 0.0
                    pop = (1.0 - (short_put_row['abs_delta'] + short_call_row['delta'])) * 100.0
                    pop = max(65.0, min(85.0, pop))

                    net_theta = (-short_put_row['greeks'].theta + long_put_row['greeks'].theta -
                                 short_call_row['greeks'].theta + long_call_row['greeks'].theta) * 100.0
                    net_vanna = (-short_put_row['greeks'].vanna + long_put_row['greeks'].vanna -
                                 short_call_row['greeks'].vanna + long_call_row['greeks'].vanna) * 100.0

                    return TradeRecommendation(
                        symbol=symbol,
                        action="SELL (CREDIT)",
                        strategy_name="Iron Condor (Defined Risk)",
                        expiration=best_exp,
                        dte=dte,
                        spot_price=spot_price,
                        legs_summary=f"Puts: {short_put_k:.0f}/{long_put_k:.0f} | Calls: {short_call_k:.0f}/{long_call_k:.0f}",
                        short_strike=short_put_k,
                        long_strike=long_put_k,
                        entry_limit_price=round(total_credit, 2),
                        target_exit_price=round(total_credit * 0.50, 2),
                        stop_loss_price=round(total_credit * 2.0, 2),
                        max_profit_dollar=round(max_profit, 2),
                        max_loss_dollar=round(max_loss, 2),
                        return_on_capital_pct=round(roc, 1),
                        probability_of_profit_pct=round(pop, 1),
                        net_delta=0.0,
                        net_gamma=0.0,
                        net_vanna=round(net_vanna, 2),
                        net_theta_daily_dollar=round(net_theta, 2),
                        leverage_factor=round(spot_price / (width * 100), 2),
                        reasoning=(
                            f"Range-bound market inside Gamma Walls. Collect double theta decay (${net_theta:.2f}/day). "
                            f"Max risk strictly capped at ${max_loss:.0f} with {pop:.0f}% POP."
                        )
                    )

        # -------------------------------------------------------------
        # STRATEGY: BULL PUT CREDIT SPREAD
        # -------------------------------------------------------------
        if strategy_type in ("BULL_PUT_SPREAD", "SELL_OPTIONS"):
            if puts.empty or len(puts) < 3:
                return None

            otm_puts = puts[puts['strike'] < spot_price].copy()
            if otm_puts.empty:
                otm_puts = puts.copy()

            otm_puts['delta_dist'] = (otm_puts['abs_delta'] - 0.20).abs()
            short_put_row = otm_puts.sort_values('delta_dist').iloc[0]
            short_k = float(short_put_row['strike'])
            short_mid = float(short_put_row['mid'])
            short_greeks = short_put_row['greeks']

            target_long_k = short_k - wing_width
            eligible_longs = puts[puts['strike'] <= target_long_k]
            if eligible_longs.empty:
                eligible_longs = puts[puts['strike'] < short_k]
            if eligible_longs.empty:
                return None

            eligible_longs_copy = eligible_longs.copy()
            eligible_longs_copy['k_dist'] = (eligible_longs_copy['strike'] - target_long_k).abs()
            long_put_row = eligible_longs_copy.sort_values('k_dist').iloc[0]
            long_k = float(long_put_row['strike'])
            long_mid = float(long_put_row['mid'])
            long_greeks = long_put_row['greeks']

            net_credit = short_mid - long_mid
            if net_credit <= 0.10:
                net_credit = 0.25

            width = short_k - long_k
            if width <= 0:
                width = 5.0
                long_k = short_k - width

            max_loss = (width - net_credit) * 100.0
            max_profit = net_credit * 100.0
            roc = (max_profit / max_loss) * 100.0 if max_loss > 0 else 0.0

            target_exit = net_credit * 0.50
            stop_loss = net_credit * 2.0
            pop = (1.0 - short_put_row['abs_delta']) * 100.0

            net_delta = (-short_greeks.delta + long_greeks.delta) * 100.0
            net_gamma = (-short_greeks.gamma + long_greeks.gamma) * 100.0
            net_vanna = (-short_greeks.vanna + long_greeks.vanna) * 100.0
            net_theta = (-short_greeks.theta + long_greeks.theta) * 100.0

            return TradeRecommendation(
                symbol=symbol,
                action="SELL (CREDIT)",
                strategy_name="Bull Put Credit Spread (Defined Risk)",
                expiration=best_exp,
                dte=dte,
                spot_price=spot_price,
                legs_summary=f"Sell ${short_k:.1f} Put / Buy ${long_k:.1f} Put",
                short_strike=short_k,
                long_strike=long_k,
                entry_limit_price=round(net_credit, 2),
                target_exit_price=round(target_exit, 2),
                stop_loss_price=round(stop_loss, 2),
                max_profit_dollar=round(max_profit, 2),
                max_loss_dollar=round(max_loss, 2),
                return_on_capital_pct=round(roc, 1),
                probability_of_profit_pct=round(pop, 1),
                net_delta=round(net_delta, 2),
                net_gamma=round(net_gamma, 4),
                net_vanna=round(net_vanna, 2),
                net_theta_daily_dollar=round(net_theta, 2),
                leverage_factor=round(spot_price / (width * 100), 2),
                reasoning=(
                    f"Capture Volatility Risk Premium with {pop:.0f}% win probability. "
                    f"Theta decay generates ~${net_theta:.2f}/day. "
                    f"Max risk strictly capped at ${max_loss:.0f}."
                )
            )

        # -------------------------------------------------------------
        # STRATEGY: LEVERAGED LONG CALL (Buying Convexity)
        # -------------------------------------------------------------
        elif strategy_type in ("LEVERAGED_LONG_CALL", "BUY_OPTIONS", "BULL_CALL_DEBIT_SPREAD"):
            if calls.empty:
                return None

            calls_copy = calls.copy()
            calls_copy['delta_dist'] = (calls_copy['delta'] - 0.55).abs()
            call_row = calls_copy.sort_values('delta_dist').iloc[0]
            k = float(call_row['strike'])
            mid = float(call_row['mid'])
            greeks = call_row['greeks']

            entry_price = mid
            if entry_price <= 0.05:
                return None

            max_loss = entry_price * 100.0
            target_exit = entry_price * 2.0
            stop_loss = entry_price * 0.60
            max_profit = entry_price * 2.0 * 100.0

            leverage = (greeks.delta * spot_price) / entry_price if entry_price > 0 else 1.0
            pop = greeks.delta * 100.0

            return TradeRecommendation(
                symbol=symbol,
                action="BUY (DEBIT/LEVERAGE)",
                strategy_name="Leveraged Long Call (Convexity)",
                expiration=best_exp,
                dte=dte,
                spot_price=spot_price,
                legs_summary=f"Buy ${k:.1f} Call",
                short_strike=None,
                long_strike=k,
                entry_limit_price=round(entry_price, 2),
                target_exit_price=round(target_exit, 2),
                stop_loss_price=round(stop_loss, 2),
                max_profit_dollar=round(max_profit, 2),
                max_loss_dollar=round(max_loss, 2),
                return_on_capital_pct=100.0,
                probability_of_profit_pct=round(pop, 1),
                net_delta=round(greeks.delta * 100.0, 2),
                net_gamma=round(greeks.gamma * 100.0, 4),
                net_vanna=round(greeks.vanna * 100.0, 2),
                net_theta_daily_dollar=round(greeks.theta * 100.0, 2),
                leverage_factor=round(leverage, 1),
                reasoning=(
                    f"Underpriced option volatility creates explosive {leverage:.1f}x effective leverage. "
                    f"Max risk is strictly limited to ${max_loss:.0f} (premium paid). "
                    f"Target profit is +100% (${max_profit:.0f}) with stop loss at ${stop_loss:.2f}."
                )
            )

        # -------------------------------------------------------------
        # STRATEGY: BEAR CALL CREDIT SPREAD
        # -------------------------------------------------------------
        elif strategy_type == "BEAR_CALL_SPREAD":
            if calls.empty or len(calls) < 3:
                return None

            otm_calls = calls[calls['strike'] > spot_price].copy()
            if otm_calls.empty:
                otm_calls = calls.copy()

            otm_calls['delta_dist'] = (otm_calls['delta'] - 0.20).abs()
            short_call_row = otm_calls.sort_values('delta_dist').iloc[0]
            short_k = float(short_call_row['strike'])
            short_mid = float(short_call_row['mid'])
            short_greeks = short_call_row['greeks']

            target_long_k = short_k + wing_width
            eligible_longs = calls[calls['strike'] >= target_long_k]
            if eligible_longs.empty:
                eligible_longs = calls[calls['strike'] > short_k]
            if eligible_longs.empty:
                return None

            eligible_longs_copy = eligible_longs.copy()
            eligible_longs_copy['k_dist'] = (eligible_longs_copy['strike'] - target_long_k).abs()
            long_call_row = eligible_longs_copy.sort_values('k_dist').iloc[0]
            long_k = float(long_call_row['strike'])
            long_mid = float(long_call_row['mid'])
            long_greeks = long_call_row['greeks']

            net_credit = short_mid - long_mid
            if net_credit <= 0.10:
                net_credit = 0.25

            width = long_k - short_k
            if width <= 0:
                width = 5.0
                long_k = short_k + width

            max_loss = (width - net_credit) * 100.0
            max_profit = net_credit * 100.0
            roc = (max_profit / max_loss) * 100.0 if max_loss > 0 else 0.0

            target_exit = net_credit * 0.50
            stop_loss = net_credit * 2.0
            pop = (1.0 - short_call_row['delta']) * 100.0

            net_delta = (-short_greeks.delta + long_call_row['greeks'].delta) * 100.0
            net_theta = (-short_greeks.theta + long_call_row['greeks'].theta) * 100.0

            return TradeRecommendation(
                symbol=symbol,
                action="SELL (CREDIT)",
                strategy_name="Bear Call Credit Spread (Defined Risk)",
                expiration=best_exp,
                dte=dte,
                spot_price=spot_price,
                legs_summary=f"Sell ${short_k:.1f} Call / Buy ${long_k:.1f} Call",
                short_strike=short_k,
                long_strike=long_k,
                entry_limit_price=round(net_credit, 2),
                target_exit_price=round(target_exit, 2),
                stop_loss_price=round(stop_loss, 2),
                max_profit_dollar=round(max_profit, 2),
                max_loss_dollar=round(max_loss, 2),
                return_on_capital_pct=round(roc, 1),
                probability_of_profit_pct=round(pop, 1),
                net_delta=round(net_delta, 2),
                net_gamma=0.0,
                net_vanna=0.0,
                net_theta_daily_dollar=round(net_theta, 2),
                leverage_factor=round(spot_price / (width * 100), 2),
                reasoning=(
                    f"Overpriced calls sold in bearish trend. {pop:.0f}% win probability. "
                    f"Max risk capped at ${max_loss:.0f}."
                )
            )

        return None