"""
Dealer Greeks & Market Maker Exposure Engine
Calculates:
- Net Gamma Exposure (GEX) in Dollar per 1% move
- Net Vanna Exposure (VEX)
- Zero Gamma Level (Gamma Flip Point)
- Key Gamma Walls (Call Wall = Resistance, Put Wall = Support)
- Market Regime Classification: Long Gamma (Stable/Mean-Reverting) vs. Short Gamma (Volatile/Trending)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.engine.black_scholes import BlackScholesEngine


@dataclass
class GammaProfile:
    spot_price: float
    net_gex_dollar_1pct: float
    gamma_regime: str  # "POSITIVE_GAMMA" (low vol/mean reversion) or "NEGATIVE_GAMMA" (high vol/trend)
    zero_gamma_strike: Optional[float]
    call_wall_strike: float
    put_wall_strike: float
    total_call_gex: float
    total_put_gex: float
    net_vex: float
    charm_bias: str  # "BULLISH_DECAY" or "BEARISH_DECAY"
    summary: str


class DealerGreeksEngine:
    @staticmethod
    def analyze_options_chain(
        chain_df: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.045,
        dividend_yield: float = 0.015,
    ) -> GammaProfile:
        """
        Analyzes an options chain containing columns:
        ['strike', 'option_type', 'dte', 'implied_volatility', 'open_interest']
        """
        if chain_df.empty or spot_price <= 0:
            return GammaProfile(
                spot_price=spot_price,
                net_gex_dollar_1pct=0.0,
                gamma_regime="NEUTRAL",
                zero_gamma_strike=None,
                call_wall_strike=spot_price,
                put_wall_strike=spot_price,
                total_call_gex=0.0,
                total_put_gex=0.0,
                net_vex=0.0,
                charm_bias="NEUTRAL",
                summary="Empty options chain data.",
            )

        gex_list = []
        vex_list = []
        charm_list = []

        for _, row in chain_df.iterrows():
            strike = float(row['strike'])
            opt_type = str(row['option_type']).lower()
            dte = max(1.0, float(row.get('dte', 30)))
            T = dte / 365.0
            sigma = float(row.get('implied_volatility', 0.20))
            if sigma <= 0.01:
                sigma = 0.20
            oi = float(row.get('open_interest', 0))

            greeks = BlackScholesEngine.calculate_all_greeks(
                opt_type, spot_price, strike, T, risk_free_rate, sigma, dividend_yield
            )

            # Dollar Gamma per 1% move:
            # Dealer convention: Customers are Net Long Calls and Net Long Puts.
            # Thus, Dealers are Short Calls (negative gamma) and Short Puts (positive gamma when market rises,
            # or standard SqueezeMetrics convention: Call GEX is positive to market, Put GEX is negative to market).
            # Call GEX = +Gamma * OI * 100 * Spot^2 * 0.01
            # Put GEX  = -Gamma * OI * 100 * Spot^2 * 0.01
            dollar_gamma_1pct = greeks.gamma * 100.0 * (spot_price ** 2) * 0.01

            if opt_type in ('c', 'call'):
                call_gex = dollar_gamma_1pct * oi
                put_gex = 0.0
                vex = greeks.vanna * 100.0 * spot_price * oi
                charm_flow = -greeks.charm * 100.0 * oi
            else:
                call_gex = 0.0
                put_gex = -dollar_gamma_1pct * oi
                vex = -greeks.vanna * 100.0 * spot_price * oi
                charm_flow = greeks.charm * 100.0 * oi

            net_gex = call_gex + put_gex

            gex_list.append({
                'strike': strike,
                'option_type': opt_type,
                'oi': oi,
                'T': T,
                'sigma': sigma,
                'call_gex': call_gex,
                'put_gex': put_gex,
                'net_gex': net_gex,
                'vex': vex,
                'charm_flow': charm_flow,
            })

        df_metrics = pd.DataFrame(gex_list)
        if df_metrics.empty:
            return GammaProfile(
                spot_price=spot_price,
                net_gex_dollar_1pct=0.0,
                gamma_regime="NEUTRAL",
                zero_gamma_strike=None,
                call_wall_strike=spot_price,
                put_wall_strike=spot_price,
                total_call_gex=0.0,
                total_put_gex=0.0,
                net_vex=0.0,
                charm_bias="NEUTRAL",
                summary="Insufficient options chain data.",
            )

        total_call_gex = float(df_metrics['call_gex'].sum())
        total_put_gex = float(df_metrics['put_gex'].sum())
        net_gex_dollar = total_call_gex + total_put_gex
        net_vex = float(df_metrics['vex'].sum())
        total_charm = float(df_metrics['charm_flow'].sum())

        # Call Wall (strike with highest Call GEX or Call OI)
        calls = df_metrics[df_metrics['option_type'].isin(['c', 'call'])]
        call_wall = float(calls.loc[calls['call_gex'].idxmax()]['strike']) if not calls.empty else spot_price

        # Put Wall (strike with largest magnitude of Put GEX)
        puts = df_metrics[df_metrics['option_type'].isin(['p', 'put'])]
        put_wall = float(puts.loc[puts['put_gex'].abs().idxmax()]['strike']) if not puts.empty else spot_price

        # Zero Gamma Flip Point (spot price root where net GEX crosses 0)
        # Evaluates Net GEX across a grid of spot prices around spot_price * [0.7, 1.3]
        zero_gamma_strike = None
        try:
            grid_spots = np.linspace(spot_price * 0.7, spot_price * 1.3, 121)
            S_grid = grid_spots[:, np.newaxis]

            K_arr = df_metrics['strike'].to_numpy(dtype=float)[np.newaxis, :]
            T_arr = df_metrics['T'].to_numpy(dtype=float)[np.newaxis, :]
            sigma_arr = df_metrics['sigma'].to_numpy(dtype=float)[np.newaxis, :]
            oi_arr = df_metrics['oi'].to_numpy(dtype=float)[np.newaxis, :]
            sign_arr = np.where(df_metrics['option_type'].isin(['c', 'call']), 1.0, -1.0)[np.newaxis, :]

            sigma_sqrt_T = sigma_arr * np.sqrt(T_arr)
            d1 = (np.log(S_grid / K_arr) + (risk_free_rate - dividend_yield + 0.5 * sigma_arr**2) * T_arr) / sigma_sqrt_T
            phi_d1 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1**2)
            dollar_gamma_contract = S_grid * np.exp(-dividend_yield * T_arr) * phi_d1 / sigma_sqrt_T
            net_gex_grid = np.sum(sign_arr * oi_arr * dollar_gamma_contract, axis=1)

            zero_crossings = []
            for i in range(len(grid_spots) - 1):
                g1 = net_gex_grid[i]
                g2 = net_gex_grid[i + 1]
                if g1 == 0:
                    zero_crossings.append(float(grid_spots[i]))
                elif g1 * g2 < 0:
                    s1, s2 = grid_spots[i], grid_spots[i + 1]
                    s_zero = s1 - g1 * (s2 - s1) / (g2 - g1)
                    zero_crossings.append(float(s_zero))

            if zero_crossings:
                zero_gamma_strike = float(min(zero_crossings, key=lambda z: abs(z - spot_price)))
        except Exception:
            zero_gamma_strike = None

        if net_gex_dollar > 0:
            regime = "POSITIVE_GAMMA"
            summary = (
                f"Market in POSITIVE GAMMA (${net_gex_dollar/1e6:.1f}M GEX). "
                f"Volatility is dampened; mean-reversion dominant between Put Wall (${put_wall:.0f}) "
                f"and Call Wall (${call_wall:.0f}). Best suited for OPTION SELLING (Credit Spreads/Iron Condors)."
            )
        else:
            regime = "NEGATIVE_GAMMA"
            summary = (
                f"Market in NEGATIVE GAMMA (${net_gex_dollar/1e6:.1f}M GEX). "
                f"Market Makers accelerate moves. High volatility, explosive trend breakout potential. "
                f"Best suited for OPTION BUYING (Leveraged Long Calls/Puts / Debit Spreads)."
            )

        charm_bias = "BULLISH_DECAY" if total_charm > 0 else "BEARISH_DECAY"

        return GammaProfile(
            spot_price=spot_price,
            net_gex_dollar_1pct=net_gex_dollar,
            gamma_regime=regime,
            zero_gamma_strike=zero_gamma_strike,
            call_wall_strike=call_wall,
            put_wall_strike=put_wall,
            total_call_gex=total_call_gex,
            total_put_gex=total_put_gex,
            net_vex=net_vex,
            charm_bias=charm_bias,
            summary=summary,
        )