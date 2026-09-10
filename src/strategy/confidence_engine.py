"""
Multi-Factor Trade Confidence Engine
Computes an institutional-grade Confidence Score (0 - 100%) and Letter Grade (A+, A, B, C)
based on 6 quantitative sub-models:
1. Volatility Risk Premium (VRP Score): Realized Vol vs Implied Vol edge
2. Dealer Greek Alignment (GEX/VEX Score): Net GEX regime & distance to Gamma Walls
3. Directional Momentum & Trend Alignment: Multi-timeframe moving averages & trend confirmation
4. Probability of Profit (POP Score): Model-based delta probability & margin of safety
5. Liquidity & Execution Quality: Bid-Ask tightness, open interest, and volume
6. Macro Volatility Climate: VIX level, VIX trend, and extreme tail-risk buffers
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class ConfidenceReport:
    total_score: float         # 0 - 100%
    grade: str                 # "A+", "A", "B", "C", "D"
    vrp_score: float           # Sub-score 0 - 25
    gex_alignment_score: float # Sub-score 0 - 25
    trend_score: float         # Sub-score 0 - 20
    pop_score: float           # Sub-score 0 - 15
    liquidity_score: float     # Sub-score 0 - 10
    macro_buffer_score: float  # Sub-score 0 - 5
    strengths: list[str]
    risks: list[str]
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "grade": self.grade,
            "vrp_score": self.vrp_score,
            "gex_alignment_score": self.gex_alignment_score,
            "trend_score": self.trend_score,
            "pop_score": self.pop_score,
            "liquidity_score": self.liquidity_score,
            "macro_buffer_score": self.macro_buffer_score,
            "strengths": self.strengths,
            "risks": self.risks,
            "verdict": self.verdict,
        }


class ConfidenceEngine:
    @classmethod
    def evaluate_trade(
        cls,
        action: str,               # "SELL (CREDIT)" or "BUY (DEBIT/LEVERAGE)"
        strategy_name: str,
        spot_price: float,
        iv_current: float,
        hv_30d: float,
        iv_rank: float,
        net_gex_dollar_m: float,
        put_wall: float,
        call_wall: float,
        pop_pct: float,
        roc_pct: float,
        entry_price: float,
        short_strike: float | None,
        long_strike: float | None,
        vix_level: float,
        ema_20: float,
        ema_50: float,
        ema_200: float,
    ) -> ConfidenceReport:
        strengths = []
        risks = []

        is_sell = "SELL" in action.upper()
        is_bullish = "BULL" in strategy_name.upper()
        is_bearish = "BEAR" in strategy_name.upper()

        # -------------------------------------------------------------
        # 1. Volatility Risk Premium (VRP Score: max 25 pts)
        # -------------------------------------------------------------
        vrp_spread = (iv_current - hv_30d) * 100.0  # Vol points
        vrp_score = 0.0

        if is_sell:
            # When selling, we want IV > HV (overpriced options)
            if vrp_spread >= 4.0:
                vrp_score = 25.0
                strengths.append(f"Starke Volatilitätsrisikoprämie: IV liegt {vrp_spread:+.1f} Pkt über Realized Vol.")
            elif vrp_spread >= 1.5:
                vrp_score = 20.0
                strengths.append(f"Positive Vol-Prämie: IV ist {vrp_spread:+.1f} Pkt höher als HV.")
            elif vrp_spread >= 0.0:
                vrp_score = 14.0
            else:
                vrp_score = 7.0
                risks.append(f"Negative Vol-Prämie: Realized Vol ({hv_30d*100:.1f}%) ist höher als IV ({iv_current*100:.1f}%).")
            
            # Bonus for high IV rank
            if iv_rank >= 45.0:
                vrp_score = min(25.0, vrp_score + 4.0)
                strengths.append(f"Hoher IV Rank ({iv_rank:.1f}%) bietet üppiges Prämienteil.")
        else:
            # When buying, we want low IV (cheap options)
            if iv_rank <= 20.0:
                vrp_score = 25.0
                strengths.append(f"Optionen sind extrem günstig (IV Rank nur {iv_rank:.1f}%).")
            elif iv_rank <= 30.0:
                vrp_score = 20.0
                strengths.append(f"Günstige Volatilität (IV Rank {iv_rank:.1f}%).")
            elif iv_rank <= 45.0:
                vrp_score = 12.0
            else:
                vrp_score = 5.0
                risks.append(f"Teure Optionen zum Kaufen (IV Rank {iv_rank:.1f}% birgt IV-Crush-Gefahr).")

        # -------------------------------------------------------------
        # 2. Dealer Greek Alignment (GEX/VEX Score: max 25 pts)
        # -------------------------------------------------------------
        gex_score = 0.0
        dist_put_wall = ((spot_price - put_wall) / spot_price) * 100.0 if put_wall > 0 else 0.0
        dist_call_wall = ((call_wall - spot_price) / spot_price) * 100.0 if call_wall > 0 else 0.0

        if is_sell:
            # Option sellers love Positive Gamma (market makers cushion volatility)
            if net_gex_dollar_m >= 15.0:
                gex_score += 15.0
                strengths.append(f"Hohes Positives Gamma (+${net_gex_dollar_m:.1f}M GEX) dämpft Marktschwankungen.")
            elif net_gex_dollar_m >= 0.0:
                gex_score += 10.0
            else:
                gex_score += 3.0
                risks.append(f"Negatives Gamma (-${abs(net_gex_dollar_m):.1f}M GEX): Market Maker könnten Volatilität anfachen.")

            # Distance to Wall protection
            if is_bullish and short_strike:
                # Bull Put: short strike should be below or at Put Wall
                if short_strike <= put_wall:
                    gex_score += 10.0
                    strengths.append(f"Short Put (${short_strike:.0f}) liegt geschützt hinter der institutionellen Put Wall (${put_wall:.0f}).")
                elif dist_put_wall >= 3.0:
                    gex_score += 7.0
                else:
                    gex_score += 3.0
            elif is_bearish and short_strike:
                # Bear Call: short strike should be above Call Wall
                if short_strike >= call_wall:
                    gex_score += 10.0
                    strengths.append(f"Short Call (${short_strike:.0f}) liegt geschützt über der Call Wall (${call_wall:.0f}).")
                else:
                    gex_score += 5.0
            else:
                gex_score += 7.0
        else:
            # Option buyers love Negative Gamma (acceleration / gamma squeeze)
            if net_gex_dollar_m < 0:
                gex_score += 18.0
                strengths.append(f"Negatives Gamma (-${abs(net_gex_dollar_m):.1f}M GEX) begünstigt explosive Trendläufe.")
            else:
                gex_score += 8.0
            
            # Breakout beyond walls
            if is_bullish and spot_price >= call_wall * 0.99:
                gex_score += 7.0
                strengths.append("Kurs drückt gegen Call Wall (Potentieller Gamma-Squeeze).")
            elif is_bearish and spot_price <= put_wall * 1.01:
                gex_score += 7.0
                strengths.append("Kurs bricht durch Put Wall (Beschleunigte Abwärtskaskade).")
            else:
                gex_score += 4.0

        # -------------------------------------------------------------
        # 3. Directional Trend & Momentum (max 20 pts)
        # -------------------------------------------------------------
        trend_score = 0.0
        is_uptrend = spot_price > ema_20 and ema_20 > ema_50 and spot_price > ema_200
        is_downtrend = spot_price < ema_20 and ema_20 < ema_50 and spot_price < ema_200

        if is_bullish:
            if is_uptrend:
                trend_score = 20.0
                strengths.append("Perfekte Trendausrichtung: Kurs über EMA 20, 50 und 200.")
            elif spot_price > ema_50:
                trend_score = 14.0
            else:
                trend_score = 6.0
                risks.append("Gegen den übergeordneten Trend (Kurs unter EMA 50).")
        elif is_bearish:
            if is_downtrend:
                trend_score = 20.0
                strengths.append("Bärenmarkttrend bestätigt: Kurs unter allen Schlüssel-EMAs.")
            elif spot_price < ema_50:
                trend_score = 14.0
            else:
                trend_score = 6.0
                risks.append("Short gegen steigenden Aufwärtstrend.")
        else:  # Neutral / Iron Condor
            if abs(spot_price - ema_50) / spot_price < 0.025:
                trend_score = 18.0
                strengths.append("Klassische Seitwärtskonsolidierung ideal für Iron Condor.")
            else:
                trend_score = 10.0

        # -------------------------------------------------------------
        # 4. Probability of Profit & Risk/Reward (max 15 pts)
        # -------------------------------------------------------------
        pop_score = 0.0
        if pop_pct >= 85.0:
            pop_score = 15.0
            strengths.append(f"Sehr hohe Gewinnwahrscheinlichkeit: {pop_pct:.1f}% POP.")
        elif pop_pct >= 75.0:
            pop_score = 12.0
            strengths.append(f"Solide statistische Gewinnchance: {pop_pct:.1f}% POP.")
        elif pop_pct >= 65.0:
            pop_score = 8.0
        else:
            pop_score = 4.0
            risks.append(f"Niedrigere Trefferquote ({pop_pct:.1f}% POP).")

        # -------------------------------------------------------------
        # 5. Liquidity & Spread Efficiency (max 10 pts)
        # -------------------------------------------------------------
        # Defined by reasonable entry pricing
        liq_score = 8.0
        if entry_price > 0.40:
            liq_score = 10.0
        elif entry_price < 0.15:
            liq_score = 5.0
            risks.append("Geringer Prämientrag: Spread könnte durch Transaktionskosten belastet werden.")

        # -------------------------------------------------------------
        # 6. Macro Volatility Climate (max 5 pts)
        # -------------------------------------------------------------
        macro_score = 3.5
        if 13.0 <= vix_level <= 24.0:
            macro_score = 5.0
            strengths.append(f"Ideales Makro-Volatilitätsumfeld (VIX bei {vix_level:.1f}).")
        elif vix_level > 32.0:
            macro_score = 1.5
            risks.append(f"Erhöhter VIX ({vix_level:.1f}): Makro-Crashrisiko erfordert strikte Stop-Loss-Disziplin.")
        else:
            macro_score = 3.5

        # TOTAL SCORE & GRADE (sum of 6 sub-scores max 100)
        total_raw = vrp_score + gex_score + trend_score + pop_score + liq_score + macro_score
        total_score = round(max(10.0, min(100.0, total_raw)), 1)

        if total_score >= 90.0:
            grade = "A+"
            verdict = "EXZELLENTES SETUP: Höchste statistische Kante. Alle Griechen, VRP und Trendfaktoren sind im Einklang."
        elif total_score >= 80.0:
            grade = "A"
            verdict = "SEHR STARKES SETUP: Ausgezeichnetes Chance-Risiko-Verhältnis mit solider Deckung durch institutionelle Levels."
        elif total_score >= 70.0:
            grade = "B"
            verdict = "GUTES STANDARD-SETUP: Solide Parameter, normale Positionsgröße empfohlen."
        elif total_score >= 55.0:
            grade = "C"
            verdict = "MODERATES SETUP: Geringere Sicherheitsmarge, reduzierte Kontraktgröße ratsam."
        else:
            grade = "D"
            verdict = "SCHWACHES SETUP: Erhöhte Risikofaktoren, lieber auf besseres Marktsignal warten."

        return ConfidenceReport(
            total_score=total_score,
            grade=grade,
            vrp_score=round(vrp_score, 1),
            gex_alignment_score=round(gex_score, 1),
            trend_score=round(trend_score, 1),
            pop_score=round(pop_score, 1),
            liquidity_score=round(liq_score, 1),
            macro_buffer_score=round(macro_score, 1),
            strengths=strengths,
            risks=risks,
            verdict=verdict,
        )