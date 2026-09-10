"""
Interactive Options Payoff & Greek Surface Visualizer
Computes payoff curves:
1. Payoff at Expiration (T = 0)
2. Mark-to-Market curve Today (T = DTE)
3. Mark-to-Market curve at Midpoint (T = DTE / 2)
Computes Breakeven Points, Max Profit, Max Loss, and Greek evolutions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from src.engine.black_scholes import BlackScholesEngine


class PayoffVisualizer:
    @classmethod
    def generate_payoff_data(
        cls,
        strategy_name: str,
        spot_price: float,
        entry_price: float,
        short_strike: Optional[float] = None,
        long_strike: Optional[float] = None,
        dte: int = 30,
        iv: float = 0.20,
        risk_free_rate: float = 0.045,
        contracts: int = 1,
        # For Iron Condor (4 strikes)
        short_call_k: Optional[float] = None,
        long_call_k: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generates 40 evaluation points across +-20% of spot price
        showing PnL at Expiration, Today, and Half-Time.
        """
        strat_upper = strategy_name.upper()
        min_p = spot_price * 0.82
        max_p = spot_price * 1.18
        num_points = 50
        price_range = np.linspace(min_p, max_p, num_points).tolist()

        t_now = dte / 365.0
        t_mid = (dte / 2.0) / 365.0

        pnl_expiration = []
        pnl_today = []
        pnl_midway = []
        deltas_today = []

        is_credit_spread = "CREDIT" in strat_upper or "PUT_SPREAD" in strat_upper or "CALL_SPREAD" in strat_upper or "CONDOR" in strat_upper
        is_iron_condor = "CONDOR" in strat_upper

        # Determine breakevens & limits
        breakevens = []
        max_profit = 0.0
        max_loss = 0.0

        if "BULL_PUT" in strat_upper and short_strike and long_strike:
            # Short Put at short_strike, Long Put at long_strike
            # Entry: Credit received
            credit = entry_price
            width = short_strike - long_strike
            max_profit = credit * 100.0 * contracts
            max_loss = (width - credit) * 100.0 * contracts
            breakevens.append(round(short_strike - credit, 2))

            for s in price_range:
                # Expiration
                short_payoff = -max(0.0, short_strike - s)
                long_payoff = max(0.0, long_strike - s)
                pnl_exp = (short_payoff + long_payoff + credit) * 100.0 * contracts
                pnl_expiration.append(round(pnl_exp, 2))

                # Today
                p_s_now = BlackScholesEngine.price("put", s, short_strike, t_now, risk_free_rate, iv)
                p_l_now = BlackScholesEngine.price("put", s, long_strike, t_now, risk_free_rate, iv)
                val_now = (credit - (p_s_now - p_l_now)) * 100.0 * contracts
                pnl_today.append(round(val_now, 2))

                # Mid
                p_s_mid = BlackScholesEngine.price("put", s, short_strike, t_mid, risk_free_rate, iv)
                p_l_mid = BlackScholesEngine.price("put", s, long_strike, t_mid, risk_free_rate, iv)
                val_mid = (credit - (p_s_mid - p_l_mid)) * 100.0 * contracts
                pnl_midway.append(round(val_mid, 2))

        elif "BEAR_CALL" in strat_upper and short_strike and long_strike:
            # Short Call at short_strike, Long Call at long_strike
            credit = entry_price
            width = long_strike - short_strike
            max_profit = credit * 100.0 * contracts
            max_loss = (width - credit) * 100.0 * contracts
            breakevens.append(round(short_strike + credit, 2))

            for s in price_range:
                short_payoff = -max(0.0, s - short_strike)
                long_payoff = max(0.0, s - long_strike)
                pnl_exp = (short_payoff + long_payoff + credit) * 100.0 * contracts
                pnl_expiration.append(round(pnl_exp, 2))

                p_s_now = BlackScholesEngine.price("call", s, short_strike, t_now, risk_free_rate, iv)
                p_l_now = BlackScholesEngine.price("call", s, long_strike, t_now, risk_free_rate, iv)
                val_now = (credit - (p_s_now - p_l_now)) * 100.0 * contracts
                pnl_today.append(round(val_now, 2))

                p_s_mid = BlackScholesEngine.price("call", s, short_strike, t_mid, risk_free_rate, iv)
                p_l_mid = BlackScholesEngine.price("call", s, long_strike, t_mid, risk_free_rate, iv)
                val_mid = (credit - (p_s_mid - p_l_mid)) * 100.0 * contracts
                pnl_midway.append(round(val_mid, 2))

        elif is_iron_condor and short_strike and long_strike:
            # Iron Condor: Puts: short_strike / long_strike; Calls: short_call_k / long_call_k
            credit = entry_price
            sk_put = short_strike
            lk_put = long_strike
            sk_call = short_call_k or (spot_price + (spot_price - sk_put))
            lk_call = long_call_k or (sk_call + (sk_put - lk_put))

            width = sk_put - lk_put
            max_profit = credit * 100.0 * contracts
            max_loss = (width - credit) * 100.0 * contracts
            breakevens.append(round(sk_put - credit, 2))
            breakevens.append(round(sk_call + credit, 2))

            for s in price_range:
                put_exp = -max(0.0, sk_put - s) + max(0.0, lk_put - s)
                call_exp = -max(0.0, s - sk_call) + max(0.0, s - lk_call)
                pnl_exp = (put_exp + call_exp + credit) * 100.0 * contracts
                pnl_expiration.append(round(pnl_exp, 2))

                p_sp_now = BlackScholesEngine.price("put", s, sk_put, t_now, risk_free_rate, iv)
                p_lp_now = BlackScholesEngine.price("put", s, lk_put, t_now, risk_free_rate, iv)
                p_sc_now = BlackScholesEngine.price("call", s, sk_call, t_now, risk_free_rate, iv)
                p_lc_now = BlackScholesEngine.price("call", s, lk_call, t_now, risk_free_rate, iv)
                curr_cost = (p_sp_now - p_lp_now) + (p_sc_now - p_lc_now)
                pnl_today.append(round((credit - curr_cost) * 100.0 * contracts, 2))

                p_sp_mid = BlackScholesEngine.price("put", s, sk_put, t_mid, risk_free_rate, iv)
                p_lp_mid = BlackScholesEngine.price("put", s, lk_put, t_mid, risk_free_rate, iv)
                p_sc_mid = BlackScholesEngine.price("call", s, sk_call, t_mid, risk_free_rate, iv)
                p_lc_mid = BlackScholesEngine.price("call", s, lk_call, t_mid, risk_free_rate, iv)
                mid_cost = (p_sp_mid - p_lp_mid) + (p_sc_mid - p_lc_mid)
                pnl_midway.append(round((credit - mid_cost) * 100.0 * contracts, 2))

        elif "CALL" in strat_upper:
            # Long Call or Bull Call Spread
            debit = entry_price
            k_long = long_strike or spot_price
            k_short = short_strike

            if k_short:  # Bull Call Debit Spread
                width = k_short - k_long
                max_profit = (width - debit) * 100.0 * contracts
                max_loss = debit * 100.0 * contracts
                breakevens.append(round(k_long + debit, 2))

                for s in price_range:
                    long_payoff = max(0.0, s - k_long)
                    short_payoff = -max(0.0, s - k_short)
                    pnl_expiration.append(round((long_payoff + short_payoff - debit) * 100.0 * contracts, 2))

                    p_l_now = BlackScholesEngine.price("call", s, k_long, t_now, risk_free_rate, iv)
                    p_s_now = BlackScholesEngine.price("call", s, k_short, t_now, risk_free_rate, iv)
                    pnl_today.append(round((p_l_now - p_s_now - debit) * 100.0 * contracts, 2))

                    p_l_mid = BlackScholesEngine.price("call", s, k_long, t_mid, risk_free_rate, iv)
                    p_s_mid = BlackScholesEngine.price("call", s, k_short, t_mid, risk_free_rate, iv)
                    pnl_midway.append(round((p_l_mid - p_s_mid - debit) * 100.0 * contracts, 2))
            else:  # Outright Long Call
                max_loss = debit * 100.0 * contracts
                max_profit = 99999.0
                breakevens.append(round(k_long + debit, 2))

                for s in price_range:
                    long_payoff = max(0.0, s - k_long)
                    pnl_expiration.append(round((long_payoff - debit) * 100.0 * contracts, 2))
                    p_now = BlackScholesEngine.price("call", s, k_long, t_now, risk_free_rate, iv)
                    pnl_today.append(round((p_now - debit) * 100.0 * contracts, 2))
                    p_mid = BlackScholesEngine.price("call", s, k_long, t_mid, risk_free_rate, iv)
                    pnl_midway.append(round((p_mid - debit) * 100.0 * contracts, 2))

        else:  # Bear Put Spread or Long Put
            debit = entry_price
            k_long = long_strike or spot_price
            k_short = short_strike

            if k_short:  # Bear Put Debit Spread
                width = k_long - k_short
                max_profit = (width - debit) * 100.0 * contracts
                max_loss = debit * 100.0 * contracts
                breakevens.append(round(k_long - debit, 2))

                for s in price_range:
                    long_payoff = max(0.0, k_long - s)
                    short_payoff = -max(0.0, k_short - s)
                    pnl_expiration.append(round((long_payoff + short_payoff - debit) * 100.0 * contracts, 2))

                    p_l_now = BlackScholesEngine.price("put", s, k_long, t_now, risk_free_rate, iv)
                    p_s_now = BlackScholesEngine.price("put", s, k_short, t_now, risk_free_rate, iv)
                    pnl_today.append(round((p_l_now - p_s_now - debit) * 100.0 * contracts, 2))

                    p_l_mid = BlackScholesEngine.price("put", s, k_long, t_mid, risk_free_rate, iv)
                    p_s_mid = BlackScholesEngine.price("put", s, k_short, t_mid, risk_free_rate, iv)
                    pnl_midway.append(round((p_l_mid - p_s_mid - debit) * 100.0 * contracts, 2))
            else:
                max_loss = debit * 100.0 * contracts
                max_profit = (k_long - debit) * 100.0 * contracts
                breakevens.append(round(k_long - debit, 2))

                for s in price_range:
                    long_payoff = max(0.0, k_long - s)
                    pnl_expiration.append(round((long_payoff - debit) * 100.0 * contracts, 2))
                    p_now = BlackScholesEngine.price("put", s, k_long, t_now, risk_free_rate, iv)
                    pnl_today.append(round((p_now - debit) * 100.0 * contracts, 2))
                    p_mid = BlackScholesEngine.price("put", s, k_long, t_mid, risk_free_rate, iv)
                    pnl_midway.append(round((p_mid - debit) * 100.0 * contracts, 2))

        return {
            "spot_price": round(spot_price, 2),
            "prices": [round(p, 2) for p in price_range],
            "pnl_expiration": pnl_expiration,
            "pnl_today": pnl_today,
            "pnl_midway": pnl_midway,
            "breakevens": breakevens,
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
            "risk_reward_ratio": round(max_profit / max_loss, 2) if max_loss > 0 else 0.0,
        }