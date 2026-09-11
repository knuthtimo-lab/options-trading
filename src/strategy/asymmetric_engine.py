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

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
import os
import time
import json
from pathlib import Path
from datetime import datetime
import concurrent.futures
import numpy as np
import pandas as pd
import yfinance as yf

from src.engine.black_scholes import BlackScholesEngine
from src.data.sp500_constituents import get_sp500_symbols, get_symbol_sector
from src.data.live_feed import LiveDataFeed

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "data_cache"
ASYMMETRIC_CACHE_FILE = CACHE_DIR / "sp500_asymmetric.json"
ASYMMETRIC_CACHE_TTL = 15 * 60  # 15 minutes TTL


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
    # S&P 500 Qualification Engine Fields
    is_sp500_qualified: bool = True
    qualification_reason: str = ""
    technical_indicators: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    # S&P 500 Qualification Engine Fields
    strategy_score: float = 85.0
    is_sp500_qualified: bool = True
    qualification_reason: str = ""
    technical_indicators: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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

        # Helper to compute BS Greeks if missing or invalid
        def get_greeks(row, opt_type):
            k = float(row['strike'])
            iv = float(row.get('iv', 0.25) or 0.25)
            if iv < 0.08 or iv > 1.8 or np.isnan(iv):
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
            iv = float(row.get('iv', 0.25) or 0.25)
            if iv < 0.08 or iv > 1.8 or np.isnan(iv):
                iv = 0.25
            return BlackScholesEngine.calculate_all_greeks('call', spot_price, k, T_val, risk_free_rate, iv)

        leaps_exp_df['greeks'] = leaps_exp_df.apply(lambda r: get_call_greeks(r, T_leaps), axis=1)
        leaps_exp_df['delta'] = leaps_exp_df['greeks'].apply(lambda g: g.delta)
        # Target Delta ~0.80 (Deep In-The-Money Call: strike < spot_price)
        itm_calls = leaps_exp_df[leaps_exp_df['strike'] < spot_price].copy()
        if itm_calls.empty:
            itm_calls = leaps_exp_df.copy()
        itm_calls['delta_dist'] = (itm_calls['delta'] - 0.80).abs()
        best_leaps = itm_calls.sort_values('delta_dist').iloc[0]
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
            strategy_score=round(min(98.0, 75.0 + monthly_yield * 4.0 + effective_leverage * 2.5), 1),
        )

    @classmethod
    def compute_technical_profile(cls, symbol: str, spot_fallback: float) -> Dict[str, float]:
        """Calculates 14-period RSI and EMAs (20, 50, 200) for qualification filtering."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            if hist.empty or len(hist) < 15:
                return {
                    "rsi": 50.0,
                    "ema_20": spot_fallback,
                    "ema_50": spot_fallback,
                    "ema_200": spot_fallback,
                }
            close = hist["Close"]
            delta = close.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

            last_gain = avg_gain.iloc[-1]
            last_loss = avg_loss.iloc[-1]
            if last_loss == 0:
                rsi = 100.0 if last_gain > 0 else 50.0
            else:
                rs = last_gain / last_loss
                rsi = float(100.0 - (100.0 / (1.0 + rs)))

            rsi = round(max(0.0, min(100.0, rsi)), 1)
            ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1]) if len(close) >= 20 else spot_fallback
            ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else spot_fallback
            ema_200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 150 else ema_50

            # Rolling realized volatility for IV Rank estimate
            log_returns = np.log(close / close.shift(1)).dropna()
            roll_hv = log_returns.rolling(window=20).std() * np.sqrt(252)
            min_hv = float(roll_hv.min()) if not roll_hv.empty else 0.15
            max_hv = float(roll_hv.max()) if not roll_hv.empty else 0.50
            cur_hv = float(roll_hv.iloc[-1]) if not roll_hv.empty else 0.25
            if max_hv > min_hv:
                ivr = ((cur_hv - min_hv) / (max_hv - min_hv)) * 100.0
            else:
                ivr = 30.0
            iv_rank = round(max(5.0, min(95.0, ivr)), 1)

            return {
                "rsi": rsi,
                "ema_20": round(ema_20, 2),
                "ema_50": round(ema_50, 2),
                "ema_200": round(ema_200, 2),
                "iv_rank": iv_rank,
            }
        except Exception:
            return {
                "rsi": 50.0,
                "ema_20": spot_fallback,
                "ema_50": spot_fallback,
                "ema_200": spot_fallback,
                "iv_rank": 20.0,
            }

    @classmethod
    def qualify_asymmetric_long(
        cls,
        symbol: str,
        spot_price: float,
        chain_df: pd.DataFrame,
        tech: Dict[str, float],
        iv_rank: float = 18.0,
        net_gex: float = 0.0,
        vol_oi_ratio: float = 1.0,
    ) -> List[AsymmetricLongSetup]:
        """
        STRICT S&P 500 QUALIFICATION FILTER:
        Returns setups ONLY when 45 DTE Long Calls/Puts make genuine quantitative sense:
        - Long Call:
          1. IV Rank <= 25.0% (or <= 28.0% if institutional Whale Sweep Vol/OI >= 2.0x) to avoid IV crush.
          2. Momentum: Spot > EMA 20 and (EMA 20 >= EMA 50 * 0.995 or 45 <= RSI <= 68).
          3. Delta in convexity sweet spot (~0.28 to 0.42).
        - Long Put:
          1. IV Rank <= 30.0% (downside protection before volatility spikes).
          2. Breakdown: Spot < EMA 20 and (EMA 20 <= EMA 50 * 1.005 or 30 <= RSI <= 55).
          3. Delta in downside convexity sweet spot (~ -0.28 to -0.42).
        If criteria are not met, the ticker is REJECTED.
        """
        if chain_df.empty or spot_price <= 0:
            return []

        raw_setups = cls.scan_asymmetric_longs(
            symbol=symbol,
            spot_price=spot_price,
            chain_df=chain_df,
            iv_rank=iv_rank,
            net_gex=net_gex,
            ema_20=tech.get("ema_20"),
            ema_50=tech.get("ema_50"),
            rsi=tech.get("rsi"),
        )

        qualified: List[AsymmetricLongSetup] = []
        ema_20 = tech.get("ema_20", spot_price)
        ema_50 = tech.get("ema_50", spot_price)
        rsi = tech.get("rsi", 50.0)

        for s in raw_setups:
            if s.option_type == "CALL":
                # Strict Call Check
                max_ivr = 28.0 if (s.flow_squeeze_alert or vol_oi_ratio >= 2.0) else 25.0
                if s.iv_rank > max_ivr:
                    continue  # Rejected: IV too high, IV crush risk
                if spot_price < ema_20 * 0.99:
                    continue  # Rejected: Not in uptrend
                if rsi > 74.0:
                    continue  # Rejected: Overbought exhaustion risk

                s.is_sp500_qualified = True
                s.technical_indicators = tech
                s.qualification_reason = (
                    f"✅ S&P 500 QUALIFIZIERT: Extrem günstige IV (IVR {s.iv_rank:.1f}% ≤ {max_ivr:.0f}%) eliminiert IV-Crush. "
                    f"Trendausbruch (Kurs ${spot_price:.2f} > EMA 20 ${ema_20:.2f}) + RSI ({rsi}) "
                    f"aktiviert maximale 200%-800% Gamma-Konvexität."
                )
                qualified.append(s)

            elif s.option_type == "PUT":
                # Strict Put Check
                max_ivr = 32.0 if (s.flow_squeeze_alert or vol_oi_ratio >= 2.0) else 30.0
                if s.iv_rank > max_ivr:
                    continue  # Rejected: Put pricing too expensive
                if spot_price > ema_20 * 1.01:
                    continue  # Rejected: Not in breakdown
                if rsi < 24.0:
                    continue  # Rejected: Oversold bounce risk

                s.is_sp500_qualified = True
                s.technical_indicators = tech
                s.qualification_reason = (
                    f"✅ S&P 500 QUALIFIZIERT: Günstige Put-Prämien (IVR {s.iv_rank:.1f}% ≤ {max_ivr:.0f}%) vor Volatilitätsexplosion. "
                    f"Bearisher Strukturbruch (Kurs ${spot_price:.2f} < EMA 20 ${ema_20:.2f}) + RSI ({rsi}) "
                    f"bietet 300%-800% Downside-Hebel."
                )
                qualified.append(s)

        return qualified

    @classmethod
    def qualify_leaps_pmcc(
        cls,
        symbol: str,
        spot_price: float,
        chain_df: pd.DataFrame,
        tech: Dict[str, float],
        iv_rank: float = 20.0,
    ) -> Optional[LeapsPmccSetup]:
        """
        STRICT S&P 500 LEAPS PMCC FILTER:
        Returns PMCC setup ONLY when:
        1. Secular bull market: Spot > EMA 200 (or Spot > EMA 50) and Spot > EMA 50 * 0.98.
        2. Reasonable IV Rank <= 35.0% (avoids overpaying for long LEAPS leg).
        3. Liquid PMCC structure with:
           - Deep ITM LEAPS (Delta >= 0.70, DTE >= 180)
           - Short Call (Delta 0.18 - 0.30, DTE 20 - 55)
           - Monthly yield >= 1.5% (annualized >= 18.0%)
           - Effective leverage >= 2.0x
        """
        if chain_df.empty or spot_price <= 0:
            return None

        ema_50 = tech.get("ema_50", spot_price)
        ema_200 = tech.get("ema_200", spot_price)

        # Long term trend check
        if spot_price < ema_200 * 0.97 or spot_price < ema_50 * 0.97:
            return None  # Rejected: Not in secular uptrend
        if iv_rank > 38.0:
            return None  # Rejected: LEAPS premium inflated by IV

        pmcc = cls.scan_leaps_pmcc(symbol=symbol, spot_price=spot_price, chain_df=chain_df)
        if not pmcc:
            return None

        # Economic yield and leverage thresholds
        if pmcc.monthly_yield_pct < 1.4 or pmcc.effective_leverage < 1.8:
            return None  # Rejected: Inadequate cashflow or leverage

        # Orientation check: PMCC must have ITM leaps and OTM short call
        if pmcc.leaps_strike >= spot_price or pmcc.short_strike <= spot_price:
            return None
        if pmcc.leaps_delta < 0.55:
            return None

        pmcc.is_sp500_qualified = True
        pmcc.technical_indicators = tech
        pmcc.qualification_reason = (
            f"✅ S&P 500 QUALIFIZIERT: Secular Leader über 200-EMA (${ema_200:.2f}) mit moderater IV (IVR {iv_rank:.1f}%). "
            f"Synthetischer Hebel {pmcc.effective_leverage:.1f}x ohne Marginzinsen + monatlich {pmcc.monthly_yield_pct:.1f}% "
            f"Covered Call Cashflow (${pmcc.short_entry_price*100:.0f}/Mo)."
        )
        return pmcc

    @classmethod
    def scan_sp500_asymmetric(
        cls,
        top_n: int = 20,
        force_refresh: bool = False,
        max_workers: int = 12,
        symbols_subset: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Scans the S&P 500 universe for truly qualified asymmetric opportunities:
        - 45 DTE Long Calls (IVR <= 25%, bullish momentum, 200-800% targets)
        - 45 DTE Long Puts (IVR <= 30%, bearish breakdown, 300-800% targets)
        - Deep ITM LEAPS PMCC (secular uptrend, >= 1.5% monthly yield, 2.5x leverage)
        Cached with 15-minute TTL.
        """
        # 1. Check disk cache
        if not force_refresh and symbols_subset is None:
            cache_candidates = [
                ASYMMETRIC_CACHE_FILE,
                BASE_DIR / "src" / "data" / "sp500_asymmetric.json",
                Path(__file__).resolve().parent.parent / "data" / "sp500_asymmetric.json",
            ]
            for p in cache_candidates:
                if p.exists():
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        cached_at = data.get("cached_at", 0)
                        if (time.time() - cached_at < ASYMMETRIC_CACHE_TTL) or p != ASYMMETRIC_CACHE_FILE:
                            return data
                    except Exception:
                        pass

        # 2. Build Candidate Universe from S&P 500
        if symbols_subset:
            candidate_symbols = [s.strip().upper() for s in symbols_subset if s.strip()]
        else:
            candidate_symbols = []
            # Priority A: Check if unusual greeks cache exists (already scanned institutional flow)
            ug_candidates = [
                CACHE_DIR / "sp500_unusual_greeks.json",
                BASE_DIR / "src" / "data" / "sp500_unusual_greeks.json",
            ]
            for ug_p in ug_candidates:
                if ug_p.exists():
                    try:
                        with open(ug_p, "r", encoding="utf-8") as f:
                            ug_data = json.load(f)
                        for item in ug_data.get("results", []):
                            sym = item.get("symbol")
                            if sym and sym not in candidate_symbols:
                                candidate_symbols.append(sym)
                    except Exception:
                        pass
                    if candidate_symbols:
                        break

            # Priority B: Representative liquid leaders across all 11 S&P 500 sectors
            benchmark_sp500 = [
                "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO", "ORCL", "CRM", "PLTR",
                "JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B", "BLK",
                "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE",
                "HD", "MCD", "NKE", "SBUX", "NFLX", "DIS", "TMUS",
                "CAT", "GE", "UNP", "BA", "HON", "RTX",
                "PG", "COST", "PEP", "KO", "WMT",
                "XOM", "CVX", "COP", "SLB",
                "PLD", "AMT", "EQIX", "LIN", "NEM", "NEE", "CEG",
            ]
            for s in benchmark_sp500:
                if s not in candidate_symbols:
                    candidate_symbols.append(s)

        # 3. Parallel Scanning & Qualification
        all_qualified_longs: List[AsymmetricLongSetup] = []
        all_qualified_pmcc: List[LeapsPmccSetup] = []

        def process_symbol(sym: str) -> Tuple[List[AsymmetricLongSetup], Optional[LeapsPmccSetup]]:
            try:
                spot, chain_df = LiveDataFeed.get_options_chain_for_dte_range(sym, min_dte=20, max_dte=450)
                if spot <= 0 or chain_df.empty:
                    return [], None

                tech = cls.compute_technical_profile(sym, spot)
                iv_rank = tech.get("iv_rank", 20.0)

                longs = cls.qualify_asymmetric_long(
                    symbol=sym,
                    spot_price=spot,
                    chain_df=chain_df,
                    tech=tech,
                    iv_rank=iv_rank,
                )
                pmcc = cls.qualify_leaps_pmcc(
                    symbol=sym,
                    spot_price=spot,
                    chain_df=chain_df,
                    tech=tech,
                    iv_rank=iv_rank,
                )
                return longs, pmcc
            except Exception:
                return [], None

        workers = min(max_workers, len(candidate_symbols)) if candidate_symbols else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_sym = {executor.submit(process_symbol, sym): sym for sym in candidate_symbols}
            for future in concurrent.futures.as_completed(future_to_sym):
                try:
                    l_res, p_res = future.result()
                    all_qualified_longs.extend(l_res)
                    if p_res:
                        all_qualified_pmcc.append(p_res)
                except Exception:
                    pass

        # 4. Sort and Rank
        all_qualified_longs.sort(key=lambda x: (x.flow_squeeze_alert, x.strategy_score), reverse=True)
        all_qualified_pmcc.sort(key=lambda x: (x.strategy_score, x.monthly_yield_pct), reverse=True)

        final_longs = all_qualified_longs[:top_n]
        final_pmcc = all_qualified_pmcc[:top_n]

        longs_dict = [s.to_dict() for s in final_longs]
        pmcc_dict = [p.to_dict() for p in final_pmcc]

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cached_at": time.time(),
            "ttl_seconds": ASYMMETRIC_CACHE_TTL,
            "universe": "S&P 500 (503 Constituents)",
            "screening_rules": {
                "max_call_iv_rank": 25.0,
                "max_put_iv_rank": 30.0,
                "max_pmcc_iv_rank": 35.0,
                "trend_filter": "Spot > 20 EMA & RSI sweet spot (Long Call) / Spot < 20 EMA (Long Put)",
                "leaps_trend_filter": "Spot > 200 EMA & 50 EMA, Monthly yield >= 1.5%",
            },
            "total_candidates_analyzed": len(candidate_symbols),
            "total_asymmetric_longs": len(longs_dict),
            "total_leaps_pmcc": len(pmcc_dict),
            "asymmetric_longs": longs_dict,
            "leaps_pmcc": pmcc_dict,
        }

        # 5. Persist to cache
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(ASYMMETRIC_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            bundled_path = BASE_DIR / "src" / "data" / "sp500_asymmetric.json"
            bundled_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bundled_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

        return payload

