"""
Unusual Greeks Volume & Institutional Flow Engine
Detects:
1. Unusual Gamma Volume ($M): Heavy Gamma concentration creating dealer pinning / magnet strikes.
2. Unusual Vanna Volume ($M): Directional volatility flows that create dynamic support/resistance when IV changes.
3. Unusual Vega Volume ($k): Large volatility expansion/compression bets.
4. Unusual Delta Flow ($M): Directional whale sweeps with Vol/OI > 2.0.
5. Structural Levels:
   - Primary Gamma Magnet Strike (price gravitation zone)
   - Dynamic Call Resistance Wall
   - Dynamic Put Support Wall
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.engine.black_scholes import BlackScholesEngine
from src.data.live_feed import LiveDataFeed


@dataclass
class GreekAnomaly:
    symbol: str
    strike: float
    option_type: str
    expiration: str
    dte: int
    volume: int
    open_interest: int
    vol_oi_ratio: float
    iv_pct: float
    anomaly_type: str        # "UNUSUAL_GAMMA", "UNUSUAL_VANNA", "UNUSUAL_VEGA", "UNUSUAL_SWEEP"
    gamma_vol_m: float       # In Million Dollars
    vanna_vol_m: float       # In Million Dollars
    vega_vol_k: float        # In Thousand Dollars
    delta_vol_m: float       # In Million Dollars
    significance_score: float # 0 - 100
    description: str


@dataclass
class GreekStructureReport:
    symbol: str
    spot_price: float
    primary_magnet_strike: float
    magnet_distance_pct: float
    magnet_pull_force: str       # "STRONG_PINNING", "MODERATE", "NEUTRAL"
    call_resistance_strike: float
    put_support_strike: float
    total_gamma_volume_m: float
    total_vanna_volume_m: float
    anomalies: List[GreekAnomaly]


class UnusualGreeksEngine:
    @classmethod
    def analyze_ticker_anomalies(
        cls,
        symbol: str,
        min_vol_oi_ratio: float = 1.5,
        min_volume: int = 50,
        risk_free_rate: float = 0.045,
    ) -> Optional[GreekStructureReport]:
        """
        Pulls the live options chain, calculates 1st and 2nd order Greek volumes,
        and identifies anomalies and magnet strikes.
        """
        symbol = symbol.upper()
        try:
            spot_price, chain_df = LiveDataFeed.get_options_chain_for_dte_range(symbol, min_dte=5, max_dte=60)
            if chain_df.empty:
                return None
        except Exception:
            return None

        # Filter realistic strikes (+- 20% around spot)
        df = chain_df[chain_df['strike'].between(spot_price * 0.80, spot_price * 1.20)].copy()
        if df.empty:
            return None

        df['volume'] = df['volume'].fillna(0.0).astype(float)
        df['open_interest'] = df['open_interest'].fillna(0.0).astype(float)

        # Calculate Greeks per strike and Greek Volumes
        g_vol_m_list = []
        va_vol_m_list = []
        ve_vol_k_list = []
        de_vol_m_list = []
        vol_oi_list = []

        for _, row in df.iterrows():
            k = float(row['strike'])
            o_type = str(row['option_type']).lower()
            dte = max(1, int(row['dte']))
            T = dte / 365.0
            iv = max(0.05, float(row['implied_volatility']))
            vol = float(row['volume']) if pd.notna(row['volume']) else 0.0
            oi = float(row['open_interest']) if pd.notna(row['open_interest']) else 0.0

            greeks = BlackScholesEngine.calculate_all_greeks(o_type, spot_price, k, T, risk_free_rate, iv)

            # 1. Gamma Dollar Volume ($M): Vol * Gamma * Spot^2 * 0.01 / 1e6
            g_vol_m = vol * greeks.gamma * (spot_price ** 2) * 0.01 / 1e6

            # 2. Vanna Exposure Volume ($M): Vol * Vanna * Spot * 100 / 1e6
            va_vol_m = vol * abs(greeks.vanna) * spot_price * 100.0 / 1e6

            # 3. Vega Dollar Volume ($k): Vol * Vega * 100 / 1e3
            ve_vol_k = vol * greeks.vega * 100.0 / 1e3

            # 4. Delta Dollar Flow ($M): Vol * Delta * Spot * 100 / 1e6
            de_vol_m = vol * abs(greeks.delta) * spot_price * 100.0 / 1e6

            ratio = vol / max(1.0, oi)

            g_vol_m_list.append(round(g_vol_m, 3))
            va_vol_m_list.append(round(va_vol_m, 3))
            ve_vol_k_list.append(round(ve_vol_k, 2))
            de_vol_m_list.append(round(de_vol_m, 3))
            vol_oi_list.append(round(ratio, 2))

        df['gamma_vol_m'] = g_vol_m_list
        df['vanna_vol_m'] = va_vol_m_list
        df['vega_vol_k'] = ve_vol_k_list
        df['delta_vol_m'] = de_vol_m_list
        df['vol_oi_ratio'] = vol_oi_list

        # Identify Structural Levels (Aggregated across all strikes)
        strike_group = df.groupby('strike').agg({
            'gamma_vol_m': 'sum',
            'vanna_vol_m': 'sum',
            'delta_vol_m': 'sum',
            'volume': 'sum',
        }).reset_index()

        # 1. Primary Gamma Magnet: Strike with maximum cumulative Gamma Volume
        top_gamma_strike = strike_group.sort_values('gamma_vol_m', ascending=False).iloc[0]
        magnet_strike = float(top_gamma_strike['strike'])
        magnet_dist_pct = round(((magnet_strike - spot_price) / spot_price) * 100.0, 2)

        # Pull Force evaluation
        if abs(magnet_dist_pct) < 1.0:
            pull_force = "STRONG_PINNING"
        elif abs(magnet_dist_pct) < 3.0:
            pull_force = "MODERATE"
        else:
            pull_force = "NEUTRAL"

        # 2. Dynamic Call Resistance & Put Support
        calls_df = df[df['option_type'].isin(['c', 'call'])]
        puts_df = df[df['option_type'].isin(['p', 'put'])]

        if not calls_df.empty:
            call_res = float(calls_df.groupby('strike')['gamma_vol_m'].sum().idxmax())
        else:
            call_res = magnet_strike * 1.02

        if not puts_df.empty:
            put_sup = float(puts_df.groupby('strike')['gamma_vol_m'].sum().idxmax())
        else:
            put_sup = magnet_strike * 0.98

        # Find Anomaly Rows
        anomalies: List[GreekAnomaly] = []

        gamma_thresh = max(0.5, df['gamma_vol_m'].quantile(0.92))
        vanna_thresh = max(0.5, df['vanna_vol_m'].quantile(0.92))
        vega_thresh = max(10.0, df['vega_vol_k'].quantile(0.92))

        for _, r in df.iterrows():
            vol = int(float(r.get('volume', 0.0) or 0.0)) if pd.notna(r.get('volume')) else 0
            oi = int(float(r.get('open_interest', 0.0) or 0.0)) if pd.notna(r.get('open_interest')) else 0
            ratio = float(r.get('vol_oi_ratio', 0.0) or 0.0)
            k = float(r['strike'])
            o_type = str(r['option_type']).upper()
            exp = str(r['expiration'])
            dte = int(r['dte'])
            iv_pct = round(float(r['implied_volatility']) * 100.0, 1)

            if vol < min_volume:
                continue

            # Check for specific anomaly conditions
            anomaly_detected = None
            desc = ""
            score = 0.0

            if r['gamma_vol_m'] >= gamma_thresh and ratio >= min_vol_oi_ratio:
                anomaly_detected = "UNUSUAL_GAMMA"
                desc = f"Massives Gamma-Volumen (${r['gamma_vol_m']}M) bei Strike ${k}. Fungiert als Pinning-Magnet."
                score = min(99.0, 75.0 + (r['gamma_vol_m'] / gamma_thresh) * 10.0)

            elif r['vanna_vol_m'] >= vanna_thresh and ratio >= min_vol_oi_ratio:
                anomaly_detected = "UNUSUAL_VANNA"
                desc = f"Ungewöhnliches Vanna-Expositionsvolumen (${r['vanna_vol_m']}M). Starker Hebel bei IV-Änderungen."
                score = min(95.0, 70.0 + (r['vanna_vol_m'] / vanna_thresh) * 10.0)

            elif r['vega_vol_k'] >= vega_thresh and ratio >= 2.0:
                anomaly_detected = "UNUSUAL_VEGA"
                desc = f"Großer Volatilitäts-Block (${r['vega_vol_k']}k Vega). Institutionelle Vol-Positionierung."
                score = min(90.0, 68.0 + (r['vega_vol_k'] / vega_thresh) * 8.0)

            elif ratio >= 3.0 and vol >= 300:
                anomaly_detected = "UNUSUAL_SWEEP"
                desc = f"Aggressiver Sweep mit Vol/OI von {ratio}x ({vol} Kontrakte gegen {oi} OI)."
                score = min(95.0, 70.0 + ratio * 5.0)

            if anomaly_detected:
                anomalies.append(GreekAnomaly(
                    symbol=symbol,
                    strike=k,
                    option_type=o_type,
                    expiration=exp,
                    dte=dte,
                    volume=vol,
                    open_interest=oi,
                    vol_oi_ratio=ratio,
                    iv_pct=iv_pct,
                    anomaly_type=anomaly_detected,
                    gamma_vol_m=float(r['gamma_vol_m']),
                    vanna_vol_m=float(r['vanna_vol_m']),
                    vega_vol_k=float(r['vega_vol_k']),
                    delta_vol_m=float(r['delta_vol_m']),
                    significance_score=round(score, 1),
                    description=desc,
                ))

        # Sort anomalies by significance score
        anomalies.sort(key=lambda x: x.significance_score, reverse=True)

        return GreekStructureReport(
            symbol=symbol,
            spot_price=round(spot_price, 2),
            primary_magnet_strike=round(magnet_strike, 1),
            magnet_distance_pct=magnet_dist_pct,
            magnet_pull_force=pull_force,
            call_resistance_strike=round(call_res, 1),
            put_support_strike=round(put_sup, 1),
            total_gamma_volume_m=round(float(df['gamma_vol_m'].sum()), 2),
            total_vanna_volume_m=round(float(df['vanna_vol_m'].sum()), 2),
            anomalies=anomalies[:12],  # Top 12 anomalies
        )
