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
6. S&P 500 Constituent Parallel Scanner:
   - Institutional Significance Scoring (0-100)
   - Anomaly Classification: WHALE_CALL_SWEEP, WHALE_PUT_SWEEP, GAMMA_PINNING, VOL_SQUEEZE, VANNA_SURGE
   - 15-minute Disk TTL Caching (data_cache/sp500_unusual_greeks.json)
"""

import os
import time
import json
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from src.engine.black_scholes import BlackScholesEngine
from src.engine.dealer_greeks import DealerGreeksEngine
from src.data.live_feed import LiveDataFeed
from src.data.sp500_constituents import get_sp500_symbols, get_symbol_sector

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "data_cache"
CACHE_FILE = CACHE_DIR / "sp500_unusual_greeks.json"
CACHE_TTL = 15 * 60  # 15 minutes TTL in seconds


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

    @classmethod
    def scan_ticker_anomaly(
        cls,
        symbol: str,
        min_vol_oi: float = 1.2,
        risk_free_rate: float = 0.045,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyzes a single S&P 500 stock for options volume, Vol/OI ratio,
        Net GEX ($M), Gamma Regime, Vanna volume, Magnet strike, and anomaly classification.
        """
        symbol = symbol.strip().upper()
        try:
            spot_price, chain_df = LiveDataFeed.get_options_chain_for_dte_range(symbol, min_dte=5, max_dte=60)
            if chain_df.empty or spot_price <= 0:
                return None
        except Exception:
            return None

        # Focus on strikes within +-25% of spot
        df = chain_df[chain_df['strike'].between(spot_price * 0.75, spot_price * 1.25)].copy()
        if df.empty:
            return None

        df['volume'] = df['volume'].fillna(0.0).astype(float)
        df['open_interest'] = df['open_interest'].fillna(0.0).astype(float)

        calls_df = df[df['option_type'].isin(['c', 'call'])]
        puts_df = df[df['option_type'].isin(['p', 'put'])]
        call_vol = float(calls_df['volume'].sum())
        put_vol = float(puts_df['volume'].sum())
        total_vol = call_vol + put_vol

        if total_vol < 10.0:
            return None

        put_call_ratio = round(put_vol / max(1.0, call_vol), 2)

        # Greeks and Volume calculations per strike
        g_vol_m_list = []
        va_vol_m_list = []
        vol_oi_list = []

        for _, row in df.iterrows():
            k = float(row['strike'])
            o_type = str(row['option_type']).lower()
            dte = max(1, int(row.get('dte', 30)))
            T = dte / 365.0
            iv = max(0.05, float(row.get('implied_volatility', 0.20)))
            vol = float(row['volume'])
            oi = float(row['open_interest'])

            greeks = BlackScholesEngine.calculate_all_greeks(o_type, spot_price, k, T, risk_free_rate, iv)

            # Gamma Dollar Volume ($M)
            g_vol_m = vol * greeks.gamma * (spot_price ** 2) * 0.01 / 1e6
            # Vanna Volume ($M)
            va_vol_m = vol * abs(greeks.vanna) * spot_price * 100.0 / 1e6
            ratio = vol / max(1.0, oi)

            g_vol_m_list.append(g_vol_m)
            va_vol_m_list.append(va_vol_m)
            vol_oi_list.append(ratio)

        df['gamma_vol_m'] = g_vol_m_list
        df['vanna_vol_m'] = va_vol_m_list
        df['vol_oi_ratio'] = vol_oi_list

        # Maximum Vol/OI strike ratio and contract volume
        active_df = df[df['volume'] >= 20]
        if active_df.empty:
            active_df = df[df['volume'] >= 5]
        if active_df.empty:
            active_df = df

        max_row = active_df.sort_values('vol_oi_ratio', ascending=False).iloc[0]
        max_vol_oi_ratio = round(float(max_row['vol_oi_ratio']), 2)
        max_vol_oi_strike = round(float(max_row['strike']), 2)
        max_vol_oi_contract_vol = int(float(max_row['volume']))
        max_vol_oi_type = str(max_row['option_type']).upper()

        # Dealer Greeks analysis (Net GEX & Gamma Regime)
        dealer_profile = DealerGreeksEngine.analyze_options_chain(chain_df, spot_price, risk_free_rate=risk_free_rate)
        net_gex_m = round(dealer_profile.net_gex_dollar_1pct / 1e6, 2)
        gamma_regime = dealer_profile.gamma_regime

        # Total Vanna volume ($M)
        total_vanna_m = round(float(df['vanna_vol_m'].sum()), 2)

        # Primary Gamma Magnet Strike and distance %
        strike_gamma = df.groupby('strike')['gamma_vol_m'].sum()
        primary_magnet_strike = round(float(strike_gamma.idxmax()), 2)
        magnet_dist_pct = round(((primary_magnet_strike - spot_price) / spot_price) * 100.0, 2)

        # Anomaly Classification:
        # WHALE_CALL_SWEEP, WHALE_PUT_SWEEP, GAMMA_PINNING, VOL_SQUEEZE, VANNA_SURGE
        if gamma_regime == "NEGATIVE_GAMMA" and (max_vol_oi_type in ['C', 'CALL'] or call_vol > put_vol * 1.3) and max_vol_oi_ratio >= min_vol_oi:
            classification = "VOL_SQUEEZE"
        elif abs(magnet_dist_pct) <= 1.5 and gamma_regime == "POSITIVE_GAMMA" and net_gex_m > 0.5:
            classification = "GAMMA_PINNING"
        elif total_vanna_m >= 5.0 and (total_vanna_m > abs(net_gex_m) * 1.5 or total_vanna_m >= 15.0):
            classification = "VANNA_SURGE"
        elif max_vol_oi_type in ['P', 'PUT'] and (put_call_ratio >= 1.1 or max_vol_oi_ratio >= min_vol_oi * 1.3 or put_vol > call_vol):
            classification = "WHALE_PUT_SWEEP"
        elif max_vol_oi_type in ['C', 'CALL'] and max_vol_oi_ratio >= min_vol_oi:
            classification = "WHALE_CALL_SWEEP"
        else:
            classification = "WHALE_CALL_SWEEP" if call_vol >= put_vol else "WHALE_PUT_SWEEP"

        # Institutional Significance Score (0 - 100)
        voi_score = min(40.0, (max_vol_oi_ratio / 3.0) * 25.0)
        vol_score = min(25.0, float(np.log10(max(10.0, total_vol))) * 6.0)
        gex_score = min(20.0, (abs(net_gex_m) / 5.0) * 15.0)
        vanna_score = min(15.0, (total_vanna_m / 5.0) * 10.0)
        raw_score = voi_score + vol_score + gex_score + vanna_score

        if classification == "VOL_SQUEEZE":
            raw_score += 6.0
        elif classification == "GAMMA_PINNING" and abs(magnet_dist_pct) < 0.8:
            raw_score += 5.0
        elif classification in ["WHALE_CALL_SWEEP", "WHALE_PUT_SWEEP"] and max_vol_oi_ratio >= 3.0:
            raw_score += 5.0

        significance_score = round(min(99.0, max(15.0, raw_score)), 1)
        sector = get_symbol_sector(symbol) or "Uncategorized"

        return {
            "symbol": symbol,
            "sector": sector,
            "spot_price": round(spot_price, 2),
            "total_volume": int(total_vol),
            "call_volume": int(call_vol),
            "put_volume": int(put_vol),
            "put_call_ratio": put_call_ratio,
            "max_vol_oi_ratio": max_vol_oi_ratio,
            "max_vol_oi_strike": max_vol_oi_strike,
            "max_vol_oi_contract_volume": max_vol_oi_contract_vol,
            "max_vol_oi_option_type": max_vol_oi_type,
            "net_gex_m": net_gex_m,
            "gamma_regime": gamma_regime,
            "total_vanna_m": total_vanna_m,
            "primary_magnet_strike": primary_magnet_strike,
            "magnet_distance_pct": magnet_dist_pct,
            "anomaly_classification": classification,
            "significance_score": significance_score,
        }

    @classmethod
    def scan_sp500_anomalies(
        cls,
        top_n: int = 50,
        max_workers: int = 16,
        min_vol_oi: float = 1.2,
        symbols: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Scans S&P 500 stocks in parallel using concurrent.futures.ThreadPoolExecutor.
        Calculates Spot price, Total options volume, Put/Call ratio, Max Vol/OI strike ratio,
        Net GEX ($M), Gamma Regime, Total Vanna volume ($M), Primary Gamma Magnet Strike,
        Anomaly classification, and Institutional Significance Score (0-100).
        Results are cached to data_cache/sp500_unusual_greeks.json with a 15-minute TTL.
        """
        # 1. Check disk cache
        if not force_refresh and symbols is None:
            cache_candidates = [
                CACHE_FILE,
                BASE_DIR / "src" / "data" / "sp500_unusual_greeks.json",
                Path(__file__).resolve().parent.parent / "data" / "sp500_unusual_greeks.json",
            ]
            for p in cache_candidates:
                if p.exists():
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            cache_payload = json.load(f)
                        cached_at = cache_payload.get("cached_at", 0)
                        # On serverless cold start, allow older cache if no fresh cache exists
                        if (time.time() - cached_at < CACHE_TTL) or p != CACHE_FILE:
                            cached_results = cache_payload.get("results", [])
                            filtered = [r for r in cached_results if r.get("max_vol_oi_ratio", 0) >= min_vol_oi]
                            filtered.sort(key=lambda x: x.get("significance_score", 0), reverse=True)
                            return filtered[:top_n]
                    except Exception:
                        pass

        # 2. Determine symbols to scan
        if symbols is None:
            symbols_to_scan = get_sp500_symbols(normalize_for_yf=True)
        else:
            symbols_to_scan = [s.strip().upper().replace(".", "-") for s in symbols if s.strip()]

        # 3. Parallel scan across workers
        results: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(cls.scan_ticker_anomaly, sym, min_vol_oi): sym
                for sym in symbols_to_scan
            }
            for future in concurrent.futures.as_completed(future_to_symbol):
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception:
                    continue

        # Sort by significance score descending
        results.sort(key=lambda x: x.get("significance_score", 0), reverse=True)

        # 4. Save to disk cache
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cached_at": time.time(),
                "ttl_seconds": CACHE_TTL,
                "total_scanned": len(symbols_to_scan),
                "total_anomalies": len(results),
                "results": results,
            }
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f, indent=2)
        except Exception:
            pass

        # Filter by min_vol_oi and slice top_n
        filtered = [r for r in results if r.get("max_vol_oi_ratio", 0) >= min_vol_oi]
        return filtered[:top_n]


def scan_sp500_anomalies(
    top_n: int = 50,
    max_workers: int = 16,
    min_vol_oi: float = 1.2,
    symbols: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience module-level function for scanning S&P 500 options anomalies.
    """
    return UnusualGreeksEngine.scan_sp500_anomalies(
        top_n=top_n,
        max_workers=max_workers,
        min_vol_oi=min_vol_oi,
        symbols=symbols,
        force_refresh=force_refresh,
    )
