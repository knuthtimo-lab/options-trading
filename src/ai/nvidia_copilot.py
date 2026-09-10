"""
NVIDIA NIM AI Quant Copilot
Integrates with NVIDIA NIM API (https://integrate.api.nvidia.com/v1) using the 120B/340B supermodel class
(e.g. nvidia/llama-3.1-nemotron-70b-instruct or nvidia/nemotron-4-340b-instruct).

Equipped with live tools:
1. Yahoo Finance real-time stock price & technical indicators (EMA 20/50/200, RSI 14, ATR).
2. Options Greeks & Dealer Positioning (Net GEX, VEX, Put/Call Walls, Gamma Regime).
3. Unusual Greek Volume Anomalies & Gamma Magnet Pinning Strikes.
4. Trade Setups & Multi-Factor Confidence Ratings.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error
import yfinance as yf
import numpy as np

from src.data.live_feed import LiveDataFeed
from src.strategy.signal_generator import SignalGenerator
from src.engine.unusual_greeks import UnusualGreeksEngine

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


class NvidiaQuantCopilot:
    _api_key_override: Optional[str] = None
    _model_override: Optional[str] = None

    @classmethod
    def set_api_key(cls, key: str):
        cls._api_key_override = key.strip() if key else None

    @classmethod
    def set_model(cls, model: str):
        cls._model_override = model.strip() if model else None

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        if cls._api_key_override:
            return cls._api_key_override
        return os.getenv("NVIDIA_API_KEY")

    @classmethod
    def get_model(cls) -> str:
        if cls._model_override:
            return cls._model_override
        return os.getenv("NVIDIA_MODEL", DEFAULT_MODEL)

    @classmethod
    def extract_symbols_from_prompt(cls, prompt: str) -> List[str]:
        """Extracts valid uppercase stock tickers from user message."""
        common_symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "META", "MSFT", "AMZN", "GOOGL", "IWM"]
        found = []
        prompt_upper = prompt.upper()
        for sym in common_symbols:
            # Match word boundary or standalone symbol
            if re.search(rf"\b{sym}\b", prompt_upper):
                found.append(sym)
        if not found:
            # Fallback to SPY and NVDA for general market questions
            found = ["SPY", "NVDA"]
        return list(dict.fromkeys(found))[:3]

    @classmethod
    def gather_live_market_context(cls, symbols: List[str]) -> Dict[str, Any]:
        """Fetches live market data, indicators, greeks, and unusual volume for the symbols."""
        context_data = {}
        for sym in symbols:
            try:
                # 1. Price & Technical Indicators via yfinance
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="6mo")
                if hist.empty:
                    continue

                spot = float(hist['Close'].iloc[-1])
                day_change_pct = round(float((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100.0), 2) if len(hist) >= 2 else 0.0
                vol_today = int(hist['Volume'].iloc[-1])

                # Moving Averages
                ema_20 = float(hist['Close'].ewm(span=20).mean().iloc[-1])
                ema_50 = float(hist['Close'].ewm(span=50).mean().iloc[-1])
                ema_200 = float(hist['Close'].ewm(span=200).mean().iloc[-1]) if len(hist) >= 150 else ema_50

                # RSI (14)
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-6)
                rsi_14 = round(float(100 - (100 / (1 + rs.iloc[-1]))), 1)

                # 2. Options Overview & Greeks
                overview = LiveDataFeed.get_ticker_overview(sym)

                # 3. Unusual Greeks & Gamma Magnet
                unusual = UnusualGreeksEngine.analyze_ticker_anomalies(sym)

                # 4. Strategy Setup
                signal = SignalGenerator.analyze_ticker(sym)

                context_data[sym] = {
                    "spot_price": round(spot, 2),
                    "day_change_pct": day_change_pct,
                    "volume": vol_today,
                    "ema_20": round(ema_20, 2),
                    "ema_50": round(ema_50, 2),
                    "ema_200": round(ema_200, 2),
                    "rsi_14": rsi_14,
                    "trend": "BULLISH" if spot > ema_50 > ema_200 else ("BEARISH" if spot < ema_50 < ema_200 else "NEUTRAL/CHOPPY"),
                    "iv_rank": round(overview.iv_rank_1y, 1),
                    "historical_vol_30d": round(overview.historical_vol_30d * 100.0, 1),
                    "current_iv": round(overview.current_iv_estimate * 100.0, 1),
                    "vix_level": round(overview.vix_level, 1),
                    "gamma_magnet": unusual.primary_magnet_strike if unusual else None,
                    "magnet_distance_pct": unusual.magnet_distance_pct if unusual else None,
                    "magnet_pull_force": unusual.magnet_pull_force if unusual else None,
                    "call_resistance": unusual.call_resistance_strike if unusual else None,
                    "put_support": unusual.put_support_strike if unusual else None,
                    "anomalies_detected": [a.description for a in unusual.anomalies[:3]] if unusual else [],
                    "recommended_strategy": signal.trade.strategy_name if signal and signal.trade else "NEUTRAL_WAIT",
                    "action": signal.trade.action if signal and signal.trade else "NONE",
                    "legs": signal.trade.legs_summary if signal and signal.trade else "NONE",
                }
            except Exception as e:
                continue

        return context_data

    @classmethod
    def chat(cls, user_message: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Processes a user question, retrieves real-time market data & indicators,
        and queries the NVIDIA NIM Supermodel.
        """
        api_key = cls.get_api_key()
        model_name = cls.get_model()

        # Extract symbols and gather real-time data
        symbols = cls.extract_symbols_from_prompt(user_message)
        market_context = cls.gather_live_market_context(symbols)

        # Build comprehensive quantitative system prompt
        context_json = json.dumps(market_context, indent=2)

        system_prompt = f"""Du bist die institutional quantitative KI-Copilot für Optionen-Trading und Derivate-Analyse, angetrieben von NVIDIAs 120B/340B Supermodel-Architektur.
Du hast direkten Zugriff auf Yahoo Finance Live-Kurse, technische Indikatoren (EMA 20/50/200, RSI 14), 1. & 2. Ordnung Griechen (Delta, Gamma, Vanna, Charm, Volga), Dealer Net GEX, ungewöhnliches Optionsvolumen und Gamma-Magnet-Pinning-Strikes.

AKTUELLE LIVE-MARKT-DATEN & QUANTITATIVE ANALYSE (DIREKT AUS YAHOO FINANCE & DEM GREEKS ENGINE):
{context_json}

RICHTLINIEN FÜR DEINE ANTWORTEN:
1. Sei präzise, quantitativ fundiert und professionell wie ein Senior Volatility Trader bei einem Hedgefonds.
2. Nutze immer die konkreten Zahlen aus dem Datenkontext (z.B. exakter Spot-Kurs, EMA-Level, RSI, Net GEX, Gamma Magnet Strike, DTE und Strikes).
3. Erkläre bei Nachfragen verständlich:
   - Warum ein Gamma-Magnet den Kurs anzieht (Dealer Delta-Hedging Pinning).
   - Wie sich Vanna und Charm auf den Trade auswirken.
   - Warum Defined-Risk Spreads (Credit vs. Debit) nackten Optionen vorzuziehen sind.
4. Gib konkrete Handlungsoptionen mit Einstieg, Stop-Loss und Take-Profit (45-50%).
5. Antworte auf Deutsch im professionellen Markdown-Format mit Hervorhebungen.
"""

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history[-4:])
        messages.append({"role": "user", "content": user_message})

        # If no API key is provided, return intelligent local quantitative analysis
        if not api_key:
            return {
                "status": "NO_API_KEY",
                "model": model_name,
                "symbols_analyzed": symbols,
                "market_context": market_context,
                "reply": cls._generate_offline_analysis(user_message, symbols, market_context),
                "notice": "Tipp: Hinterlege deinen NVIDIA API Key im Einstellungs-Menü, um die volle Rechenpower des NVIDIA 120B/340B Supermodels live zu nutzen."
            }

        # Query NVIDIA NIM API
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
            "stream": False,
        }

        try:
            req = urllib.request.Request(
                NVIDIA_API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                  "Content-Type": "application/json",
                  "Authorization": f"Bearer {api_key}",
                  "User-Agent": "OptionsTrading-NvidiaCopilot/2.0"
                }
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                reply_text = resp_json["choices"][0]["message"]["content"]
                return {
                    "status": "SUCCESS",
                    "model": model_name,
                    "symbols_analyzed": symbols,
                    "market_context": market_context,
                    "reply": reply_text,
                }
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            return {
                "status": "API_ERROR",
                "model": model_name,
                "error": f"NVIDIA API Fehler ({e.code}): {err_msg}",
                "reply": cls._generate_offline_analysis(user_message, symbols, market_context) + f"\n\n*(Hinweis: NVIDIA API meldete: HTTP {e.code} - Fallback auf interne Quant-Engine)*",
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "model": model_name,
                "error": str(e),
                "reply": cls._generate_offline_analysis(user_message, symbols, market_context),
            }

    @classmethod
    def _generate_offline_analysis(cls, prompt: str, symbols: List[str], ctx: Dict[str, Any]) -> str:
        """Fallback quantitative intelligence engine when API key is awaiting configuration."""
        if not symbols or not ctx:
            return "Bitte stelle eine Frage zu einem konkreten Ticker wie SPY, QQQ, NVDA, AAPL oder TSLA."

        sym = symbols[0]
        data = ctx.get(sym, {})
        spot = data.get("spot_price", "---")
        rsi = data.get("rsi_14", "---")
        trend = data.get("trend", "---")
        magnet = data.get("gamma_magnet", "---")
        pull = data.get("magnet_pull_force", "---")
        strategy = data.get("recommended_strategy", "---")
        legs = data.get("legs", "---")
        iv_rank = data.get("iv_rank", "---")

        return f"""### 🤖 NVIDIA Quant-Analyse für **{sym}**

**1. Live Markt- & Indikatoren-Profil (Yahoo Finance):**
- **Aktueller Kurs:** `${spot}` ({data.get('day_change_pct', 0.0):+}% heute)
- **Trend-Status:** `{trend}` (EMA 20: `${data.get('ema_20')}`, EMA 50: `${data.get('ema_50')}`, EMA 200: `${data.get('ema_200')}`)
- **RSI (14):** `{rsi}` ({'Überverkauft' if float(rsi or 50) < 35 else 'Überkauft' if float(rsi or 50) > 70 else 'Neutral'})
- **IV Rank:** `{iv_rank}%` (VIX: `{data.get('vix_level')}`)

**2. Dealer Greeks & Ungewöhnliches Volumen:**
- **Gamma Magnet Strike:** **`${magnet}`** (Anziehungskraft: `{pull}`)
- **Dynamischer Support (Put Wall):** `${data.get('put_support')}`
- **Dynamischer Widerstand (Call Wall):** `${data.get('call_resistance')}`
{f'- **Erkannte Anomalie:** *{data["anomalies_detected"][0]}*' if data.get('anomalies_detected') else ''}

**3. Quantitatives Handels-Setup:**
- **Empfehlung:** `{strategy}` ({data.get('action')})
- **Struktur:** `{legs}`
- **Begründung:** Der Kurs wird durch den Gamma-Magneten bei `${magnet}` stabilisiert. Bei einem IV-Rank von `{iv_rank}%` bietet ein definierter Credit Spread das höchste statistische Edge ohne ungedecktes Tail-Risiko.

> *Hinweis: Trage deinen persönlichen **NVIDIA API Key** in den Einstellungen ein, um Live-Streaming-Antworten des NVIDIA 120B/340B Nemotron-Modells zu aktivieren.*
"""
