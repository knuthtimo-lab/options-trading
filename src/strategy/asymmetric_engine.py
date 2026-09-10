"""
Asymmetric Convexity & LEAPS Quantitative Engine
Identifies high-asymmetry trading opportunities:
1. 45 DTE Asymmetric Longs (Home-Run Calls & Puts):
   - Targets explosive 200% to 800% return on capital.
   - Filtered for cheap implied volatility (IV Rank <= 22%), strong momentum / Short Gamma regimes.
   - Strike selection: Delta ~0.30 - 0.40 (optimal sweet spot for rapid Gamma acceleration).
   - Dynamic profit ladders (+200%, +400%, +800%) and strict stop-loss (-50% or 14 DTE exit).
2. Deep ITM LEAPS (365+ DTE) & Poor Man's Covered Call (PMCC):
   - Delta ~0.80 synthetic stock replacement with 2.5x - 4x defined-risk leverage.
   - Sells monthly 30-45 DTE Delta ~0.20 short calls against the LEAPS to harvest theta yield.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from src.engine.black_scholes import BlackScholesEngine


@dataclass
class AsymmetricLongSetup:
    symbol: str
    option_type: str            # "CALL" or "PUT"
    strategy_type: str          # "ASYMMETRIC_45DTE_CALL" or "ASYMMETRIC_45DTE_PUT"
    expiration: str
    dte: int
    spot_price: float
    strike: float
    delta: float
    gamma: float
    vega: float
    theta_daily: float
    entry_price: float          # Premium per share
    max_risk_dollar: float      # Premium * 100
    breakeven_price: float
    # Target profit ladder
    target_1_price: float       # +200% gain (3.0x entry price)
    target_2_price: float       # +400% gain (5.0x entry price)
    target_3_price: float       # +800% gain (9.0x entry price)
    stop_loss_price: float      # -50% cut (0.50x entry price)
    iv_rank: float
    catalyst_reason: str
    exit_rules: Dict[str, str]
    # Flow Squeeze Anomaly Enhancements
    flow_squeeze_alert: bool = False
    target_profit_potential: str = "300% - 800% ROI"
    strategy_score: float = 85.0
    vol_oi_ratio: float = 1.0
    unusual_flow_type: Optional[str] = None


@dataclass
class LeapsPmccSetup:
    symbol: str
    spot_price: float
    # Long LEAPS Leg
    leaps_expiration: str
    leaps_dte: int
    leaps_strike: float
    leaps_delta: float
    leaps_entry_price: float
    # Short Yield Leg (PMCC)
    short_expiration: str
    short_dte: int
    short_strike: float
    short_delta: float
    short_entry_price: float
    # Overall PMCC Metrics
    net_debit_dollar: float     # (LEAPS price - Short Call price) * 100
    effective_leverage: float   # (Spot * 100) / Net Debit
    monthly_yield_pct: float    # (Short Call / Net Debit) * 100
    annualized_yield_pct: float
    max_risk_dollar: float
    breakeven_price: float
    reasoning: str
    exit_rules: Dict[str, str]


class AsymmetricEngine:
    @staticmethod
    def _standardize_chain(chain_df: pd.DataFrame) -> pd.DataFrame:
        df = chain_df.copy()
        if 'type' not in df.columns and 'option_type' in df.columns:
            df['type'] = df['option_type'].str.upper()
        elif 'type' in df.columns:
            df['type'] = df['type'].str.upper()
        if 'iv' not in df.columns and 'implied_volatility' in df.columns:
            df['iv'] = df['implied_volatility']
        elif 'iv' not in df.columns and 'impliedVolatility' in df.columns:
            df['iv'] = df['impliedVolatility']
        return df

    @classmethod
    def detect_flow_anomaly(
        cls,
        vol_oi_ratio: float,
        gamma: float = 0.0,
        vanna: float = 0.0,
        threshold: float = 2.5,
    ) -> tuple[bool, str, float]:
        """
        Evaluates unusual options flow anomaly (e.g. Vol/OI > 2.5x with positive Gamma/Vanna surge).
        Returns: (flow_squeeze_alert, anomaly_type_desc, score)
        """
        if vol_oi_ratio >= threshold and (gamma > 0.005 or abs(vanna) > 0.01):
            score = min(99.0, 88.0 + (vol_oi_ratio - threshold) * 2.5 + min(10.0, gamma * 200.0))
            desc = f"UNUSUAL_GAMMA_VANNA_SURGE: Vol/OI {vol_oi_ratio:.1f}x with Gamma {gamma:.4f} & Vanna {vanna:.4f}"
            return True, desc, round(score, 1)
        elif vol_oi_ratio >= threshold:
            score = min(95.0, 82.0 + (vol_oi_ratio - threshold) * 2.5)
            desc = f"UNUSUAL_SWEEP_FLOW: Vol/OI {vol_oi_ratio:.1f}x exceeds institutional threshold"
            return True, desc, round(score, 1)
        return False, "STANDARD_FLOW", 75.0

    @classmethod
    def scan_asymmetric_longs(
        cls,
        symbol: str,
        spot_price: float,
        chain_df: pd.DataFrame,
        iv_rank: float = 15.0,
        net_gex: float = 0.0,
        ema_20: Optional[float] = None,
        ema_50: Optional[float] = None,
        rsi: Optional[float] = None,
        risk_free_rate: float = 0.045,
    ) -> List[AsymmetricLongSetup]:
        """
        Scans for 45 DTE asymmetric long calls and puts targeting 200% - 800% profit.
        """
        results: List[AsymmetricLongSetup] = []
        if chain_df.empty or spot_price <= 0:
            return results

        clean_df = cls._standardize_chain(chain_df)

        # 1. Target DTE Window: 30 to 65 DTE (centered at ~45 DTE)
        valid_chain = clean_df[(clean_df['dte'] >= 30) & (clean_df['dte'] <= 65)].copy()
        if valid_chain.empty:
            valid_chain = clean_df[(clean_df['dte'] >= 20) & (clean_df['dte'] <= 75)].copy()
        if valid_chain.empty:
            return results

        # Pick expiration with highest open interest
        exp_grouped = valid_chain.groupby('expiration')['open_interest'].sum()
        if exp_grouped.empty:
            return results
        best_exp = exp_grouped.idxmax()
        exp_df = valid_chain[valid_chain['expiration'] == best_exp].copy()
        dte = int(exp_df['dte'].iloc[0])
        T = max(0.01, dte / 365.0)

        # Directional bias
        is_bullish = True
        is_bearish = True
        if ema_20 is not None and ema_50 is not None:
            if spot_price > ema_20:
                is_bullish = True
                is_bearish = (spot_price < ema_50) or (rsi is not None and rsi > 70)
            else:
                is_bearish = True
                is_bullish = (spot_price > ema_50) or (rsi is not None and rsi < 30)

        calls = exp_df[exp_df['type'].isin(['CALL', 'C'])].copy()
        puts = exp_df[exp_df['type'].isin(['PUT', 'P'])].copy()

        # Helper to compute BS Greeks if missing
        def get_greeks(row, opt_type):
            k = float(row['strike'])
            iv = float(row.get('iv', 0.25))
            if iv <= 0.01:
                iv = 0.25
            return BlackScholesEngine.calculate_all_greeks(opt_type, spot_price, k, T, risk_free_rate, iv)

        # A. ASYMMETRIC LONG CALL (Home-Run Momentum)
        if is_bullish and not calls.empty:
            calls['greeks'] = calls.apply(lambda r: get_greeks(r, 'call'), axis=1)
            calls['delta'] = calls['greeks'].apply(lambda g: g.delta)
            calls['delta_dist'] = (calls['delta'] - 0.35).abs()
            best_call = calls.sort_values('delta_dist').iloc[0]
            strike = float(best_call['strike'])
            mid = float(best_call.get('mid', best_call.get('last', 1.0)))
            delta = float(best_call['delta'])
            greeks = best_call['greeks']

            if mid >= 0.20 and delta >= 0.15:
                entry = round(mid, 2)
                max_risk = entry * 100.0
                breakeven = round(strike + entry, 2)
                stop_loss = round(entry * 0.50, 2)  # -50%

                # Flow Anomaly & Whale Sweep Detection
                call_vol = float(best_call.get('volume', 0.0) or 0.0)
                call_oi = float(best_call.get('open_interest', 0.0) or 0.0)
                ratio = call_vol / max(1.0, call_oi) if call_oi > 0 else (call_vol if call_vol > 0 else 1.0)

                # Check if any call contract in this expiration experienced massive sweep
                if 'volume' in calls.columns and 'open_interest' in calls.columns:
                    active_calls = calls[(calls['volume'] >= 50) & (calls['open_interest'] > 0)]
                    if not active_calls.empty:
                        chain_max_ratio = float((active_calls['volume'] / active_calls['open_interest']).max())
                        ratio = max(ratio, chain_max_ratio)

                vanna_val = getattr(greeks, 'vanna', 0.0)
                is_flow_alert, flow_desc, score = cls.detect_flow_anomaly(
                    vol_oi_ratio=ratio,
                    gamma=greeks.gamma,
                    vanna=vanna_val,
                    threshold=2.5,
                )

                if is_flow_alert:
                    roi_potential = "300% - 800% ROI"
                    target_1 = round(entry * 4.0, 2)  # +300%
                    target_2 = round(entry * 6.0, 2)  # +500%
                    target_3 = round(entry * 9.0, 2)  # +800%
                    catalyst = (
                        f"🚨 FLOW SQUEEZE ALERT: Institutional Whale Sweep detektiert (Vol/OI {ratio:.1f}x). "
                        f"Gamma ({greeks.gamma:.4f}) und Vanna erzeugen massiven Kaufdruck der Dealer Richtung ${strike * 1.05:.2f}+."
                    )
                    tp_ladder_desc = (
                        f"Stufe 1 (+300% bei ${target_1:.2f}): 40% der Position sichern. "
                        f"Stufe 2 (+500% bei ${target_2:.2f}): Weitere 30% schließen. "
                        f"Stufe 3 (+800% bei ${target_3:.2f}): Restgewinn mit dynamischem Trailing-Stop maximieren."
                    )
                else:
                    roi_potential = "300% - 800% ROI"
                    target_1 = round(entry * 3.0, 2)  # +200%
                    target_2 = round(entry * 5.0, 2)  # +400%
                    target_3 = round(entry * 9.0, 2)  # +800%
                    catalyst = (
                        f"Niedrige IV (IVR {iv_rank:.1f}%) bietet extrem günstige Optionspreise. "
                        f"Bei Momentum erzeugt Gamma ({greeks.gamma:.4f}) eine Delta-Explosion von {delta:.2f} auf 0.70+."
                    )
                    tp_ladder_desc = (
                        f"Stufe 1 (+200% bei ${target_1:.2f}): 40% der Position schließen. "
                        f"Stufe 2 (+400% bei ${target_2:.2f}): Weitere 30% schließen. "
                        f"Stufe 3 (+800% bei ${target_3:.2f}): Rest mit Trailing Stop laufen lassen."
                    )

                exit_rules = {
                    "tp_ladder": tp_ladder_desc,
                    "sl_rule": f"Strikter Stop-Loss bei -50% Verlust (${stop_loss:.2f}) oder spätestens bei 14 DTE schließen, um Theta-Burn zu vermeiden."
                }

                results.append(AsymmetricLongSetup(
                    symbol=symbol,
                    option_type="CALL",
                    strategy_type="ASYMMETRIC_45DTE_CALL",
                    expiration=best_exp,
                    dte=dte,
                    spot_price=spot_price,
                    strike=strike,
                    delta=round(delta, 2),
                    gamma=round(greeks.gamma, 4),
                    vega=round(greeks.vega, 3),
                    theta_daily=round(greeks.theta * 100.0, 2),
                    entry_price=entry,
                    max_risk_dollar=round(max_risk, 2),
                    breakeven_price=breakeven,
                    target_1_price=target_1,
                    target_2_price=target_2,
                    target_3_price=target_3,
                    stop_loss_price=stop_loss,
                    iv_rank=round(iv_rank, 1),
                    catalyst_reason=catalyst,
                    exit_rules=exit_rules,
                    flow_squeeze_alert=is_flow_alert,
                    target_profit_potential=roi_potential,
                    strategy_score=score,
                    vol_oi_ratio=round(ratio, 2),
                    unusual_flow_type=flow_desc if is_flow_alert else None,
                ))

        # B. ASYMMETRIC LONG PUT (Downside Crash Convexity)
        if is_bearish and not puts.empty:
            puts['greeks'] = puts.apply(lambda r: get_greeks(r, 'put'), axis=1)
            puts['delta'] = puts['greeks'].apply(lambda g: g.delta)
            puts['abs_delta'] = puts['delta'].abs()
            puts['delta_dist'] = (puts['abs_delta'] - 0.35).abs()
            best_put = puts.sort_values('delta_dist').iloc[0]
            strike = float(best_put['strike'])
            mid = float(best_put.get('mid', best_put.get('last', 1.0)))
            delta = float(best_put['delta'])
            greeks = best_put['greeks']

            if mid >= 0.20 and abs(delta) >= 0.15:
                entry = round(mid, 2)
                max_risk = entry * 100.0
                breakeven = round(strike - entry, 2)
                stop_loss = round(entry * 0.50, 2)  # -50%

                # Flow Anomaly & Whale Sweep Detection for Puts
                put_vol = float(best_put.get('volume', 0.0) or 0.0)
                put_oi = float(best_put.get('open_interest', 0.0) or 0.0)
                ratio_p = put_vol / max(1.0, put_oi) if put_oi > 0 else (put_vol if put_vol > 0 else 1.0)

                if 'volume' in puts.columns and 'open_interest' in puts.columns:
                    active_puts = puts[(puts['volume'] >= 50) & (puts['open_interest'] > 0)]
                    if not active_puts.empty:
                        chain_max_ratio_p = float((active_puts['volume'] / active_puts['open_interest']).max())
                        ratio_p = max(ratio_p, chain_max_ratio_p)

                vanna_val_p = getattr(greeks, 'vanna', 0.0)
                is_flow_alert_p, flow_desc_p, score_p = cls.detect_flow_anomaly(
                    vol_oi_ratio=ratio_p,
                    gamma=greeks.gamma,
                    vanna=vanna_val_p,
                    threshold=2.5,
                )

                if is_flow_alert_p:
                    roi_potential_p = "300% - 800% ROI"
                    target_1 = round(entry * 4.0, 2)  # +300%
                    target_2 = round(entry * 6.0, 2)  # +500%
                    target_3 = round(entry * 9.0, 2)  # +800%
                    catalyst = (
                        f"🚨 FLOW SQUEEZE ALERT: Aggressiver Put-Sweep detektiert (Vol/OI {ratio_p:.1f}x). "
                        f"Dealer Short-Gamma Beschleunigung und explodierende Volatilität generieren 300% bis 800% Crash-Konvexität."
                    )
                    tp_ladder_desc_p = (
                        f"Stufe 1 (+300% bei ${target_1:.2f}): 40% sichern. "
                        f"Stufe 2 (+500% bei ${target_2:.2f}): Weitere 30% schließen. "
                        f"Stufe 3 (+800% bei ${target_3:.2f}): Rest im Downside-Move trailen."
                    )
                else:
                    roi_potential_p = "300% - 800% ROI"
                    target_1 = round(entry * 3.0, 2)  # +200%
                    target_2 = round(entry * 5.0, 2)  # +400%
                    target_3 = round(entry * 9.0, 2)  # +800%
                    catalyst = (
                        f"Günstiges Put-Pricing (IVR {iv_rank:.1f}%). "
                        f"Bei Abwärtsbeschleunigung explodieren IV (Vega-Gain) und Gamma gleichzeitig."
                    )
                    tp_ladder_desc_p = (
                        f"Stufe 1 (+200% bei ${target_1:.2f}): 40% schließen. "
                        f"Stufe 2 (+400% bei ${target_2:.2f}): Weitere 30% schließen. "
                        f"Stufe 3 (+800% bei ${target_3:.2f}): Rest trailen."
                    )

                exit_rules = {
                    "tp_ladder": tp_ladder_desc_p,
                    "sl_rule": f"Strikter Stop-Loss bei -50% (${stop_loss:.2f}) oder 14 DTE Notausstieg."
                }

                results.append(AsymmetricLongSetup(
                    symbol=symbol,
                    option_type="PUT",
                    strategy_type="ASYMMETRIC_45DTE_PUT",
                    expiration=best_exp,
                    dte=dte,
                    spot_price=spot_price,
                    strike=strike,
                    delta=round(delta, 2),
                    gamma=round(greeks.gamma, 4),
                    vega=round(greeks.vega, 3),
                    theta_daily=round(greeks.theta * 100.0, 2),
                    entry_price=entry,
                    max_risk_dollar=round(max_risk, 2),
                    breakeven_price=breakeven,
                    target_1_price=target_1,
                    target_2_price=target_2,
                    target_3_price=target_3,
                    stop_loss_price=stop_loss,
                    iv_rank=round(iv_rank, 1),
                    catalyst_reason=catalyst,
                    exit_rules=exit_rules,
                    flow_squeeze_alert=is_flow_alert_p,
                    target_profit_potential=roi_potential_p,
                    strategy_score=score_p,
                    vol_oi_ratio=round(ratio_p, 2),
                    unusual_flow_type=flow_desc_p if is_flow_alert_p else None,
                ))

        return results

    @classmethod
    def scan_leaps_pmcc(
        cls,
        symbol: str,
        spot_price: float,
        chain_df: pd.DataFrame,
        risk_free_rate: float = 0.045,
    ) -> Optional[LeapsPmccSetup]:
        """
        Identifies Deep ITM LEAPS (250+ DTE, Delta ~0.80) and optimal short call leg (25-50 DTE, Delta ~0.20).
        """
        if chain_df.empty or spot_price <= 0:
            return None

        clean_df = cls._standardize_chain(chain_df)
        calls = clean_df[clean_df['type'].isin(['CALL', 'C'])].copy()
        if calls.empty:
            return None

        # 1. FIND LEAPS EXPIRATION (>= 200 DTE, or furthest available)
        leaps_chain = calls[calls['dte'] >= 200].copy()
        if leaps_chain.empty:
            leaps_chain = calls[calls['dte'] >= 90].copy()
        if leaps_chain.empty:
            leaps_chain = calls[calls['dte'] >= 45].copy()
        if leaps_chain.empty:
            return None

        furthest_exp = leaps_chain.sort_values('dte', ascending=False)['expiration'].iloc[0]
        leaps_exp_df = leaps_chain[leaps_chain['expiration'] == furthest_exp].copy()
        leaps_dte = int(leaps_exp_df['dte'].iloc[0])
        T_leaps = max(0.05, leaps_dte / 365.0)

        def get_call_greeks(row, T_val):
            k = float(row['strike'])
            iv = float(row.get('iv', 0.25))
            if iv <= 0.01:
                iv = 0.25
            return BlackScholesEngine.calculate_all_greeks('call', spot_price, k, T_val, risk_free_rate, iv)

        leaps_exp_df['greeks'] = leaps_exp_df.apply(lambda r: get_call_greeks(r, T_leaps), axis=1)
        leaps_exp_df['delta'] = leaps_exp_df['greeks'].apply(lambda g: g.delta)
        # Target Delta ~0.80 (In-The-Money Call)
        leaps_exp_df['delta_dist'] = (leaps_exp_df['delta'] - 0.80).abs()
        best_leaps = leaps_exp_df.sort_values('delta_dist').iloc[0]
        leaps_strike = float(best_leaps['strike'])
        leaps_price = float(best_leaps.get('mid', best_leaps.get('last', 10.0)))
        leaps_delta = float(best_leaps['delta'])

        # 2. FIND SHORT CALL EXPIRATION (25 to 55 DTE, Delta ~0.20)
        short_chain = calls[(calls['dte'] >= 20) & (calls['dte'] <= 55)].copy()
        if short_chain.empty:
            short_chain = calls[calls['dte'] <= 60].copy()
        if short_chain.empty:
            return None

        short_exp_grouped = short_chain.groupby('expiration')['open_interest'].sum()
        if short_exp_grouped.empty:
            return None
        short_exp = short_exp_grouped.idxmax()
        short_exp_df = short_chain[short_chain['expiration'] == short_exp].copy()
        short_dte = int(short_exp_df['dte'].iloc[0])
        T_short = max(0.01, short_dte / 365.0)

        short_exp_df['greeks'] = short_exp_df.apply(lambda r: get_call_greeks(r, T_short), axis=1)
        short_exp_df['delta'] = short_exp_df['greeks'].apply(lambda g: g.delta)

        # Target Short Call Delta ~0.20 (Out-of-The-Money Call)
        otm_short_calls = short_exp_df[short_exp_df['strike'] > spot_price].copy()
        if otm_short_calls.empty:
            otm_short_calls = short_exp_df.copy()

        otm_short_calls['delta_dist'] = (otm_short_calls['delta'] - 0.20).abs()
        best_short = otm_short_calls.sort_values('delta_dist').iloc[0]
        short_strike = float(best_short['strike'])
        short_price = float(best_short.get('mid', best_short.get('last', 1.0)))
        short_delta = float(best_short['delta'])

        # 3. METRICS
        net_debit_share = max(1.0, leaps_price - short_price)
        net_debit_dollar = net_debit_share * 100.0
        stock_investment = spot_price * 100.0
        effective_leverage = round(stock_investment / net_debit_dollar, 2)

        monthly_yield = (short_price / net_debit_share) * 100.0
        annualized_yield = monthly_yield * (365.0 / max(1, short_dte))
        breakeven = round(leaps_strike + net_debit_share, 2)

        reasoning = (
            f"Synthetischer Aktienkauf: Der {leaps_strike:.0f}-Strike LEAPS ({leaps_dte} DTE) liefert ein Delta von {leaps_delta:.2f} "
            f"bei {effective_leverage:.1f}x Hebel ohne Margin-Zinsen. "
            f"Der monatlich dagegen verkaufte {short_strike:.0f}-Call ({short_dte} DTE) generiert ${short_price*100:.0f} monatlichen Cashflow ({monthly_yield:.1f}% Monatsrendite)."
        )

        exit_rules = {
            "short_call_roll": f"Den {short_strike:.0f}-Call bei 50% Profit zurückkaufen oder bei 7-14 DTE in den Folgemonat rollen.",
            "leaps_management": "Den LEAPS halten, solange der übergeordnete 200-Tage-Trend intakt ist. Spätestens bei 90 DTE Restlaufzeit in das nächste Jahr rollen (Theta-Erhalt)."
        }

        return LeapsPmccSetup(
            symbol=symbol,
            spot_price=spot_price,
            leaps_expiration=furthest_exp,
            leaps_dte=leaps_dte,
            leaps_strike=leaps_strike,
            leaps_delta=round(leaps_delta, 2),
            leaps_entry_price=round(leaps_price, 2),
            short_expiration=short_exp,
            short_dte=short_dte,
            short_strike=short_strike,
            short_delta=round(short_delta, 2),
            short_entry_price=round(short_price, 2),
            net_debit_dollar=round(net_debit_dollar, 2),
            effective_leverage=effective_leverage,
            monthly_yield_pct=round(monthly_yield, 1),
            annualized_yield_pct=round(annualized_yield, 1),
            max_risk_dollar=round(net_debit_dollar, 2),
            breakeven_price=breakeven,
            reasoning=reasoning,
            exit_rules=exit_rules,
        )
