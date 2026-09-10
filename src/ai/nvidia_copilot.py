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
    def set_api_key(cls, key: Optional[str]):
        cls._api_key_override = key.strip() if key is not None else None

    @classmethod
    def set_model(cls, model: str):
        cls._model_override = model.strip() if model else None

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        if cls._api_key_override is not None:
            return cls._api_key_override if cls._api_key_override else None
        key = os.getenv("NVIDIA_API_KEY")
        return key.strip() if key else None

    @classmethod
    def get_model(cls) -> str:
        if cls._model_override:
            return cls._model_override
        return os.getenv("NVIDIA_MODEL", DEFAULT_MODEL)

    STOPWORDS = {
        'DER', 'DIE', 'DAS', 'DEN', 'DEM', 'DES', 'EIN', 'EINE', 'EINER', 'EINEM', 'EINEN', 'EINES',
        'UND', 'ODER', 'ABER', 'DENN', 'DOCH', 'WEIL', 'WENN', 'DASS', 'ALS', 'WIE', 'OB',
        'FÜR', 'FUR', 'MIT', 'VON', 'BEI', 'NACH', 'ZU', 'ZUM', 'ZUR', 'AUF', 'AUS', 'IN', 'IM', 'AN', 'AM',
        'VOR', 'HINTER', 'ÜBER', 'UBER', 'UNTER', 'NEBEN', 'ZWISCHEN', 'DURCH', 'GEGEN', 'OHNE', 'UM',
        'IST', 'SIND', 'WAR', 'WAREN', 'WIRD', 'WERDEN', 'HAT', 'HATTE', 'HABEN', 'KANN', 'KÖNNEN', 'KONNTE',
        'MUSS', 'MÜSSEN', 'MUSSTE', 'SOLL', 'SOLLTE', 'SOLLTEN', 'WILL', 'WOLLEN', 'WOLLTE', 'MÖCHTE', 'MOCHTE',
        'GIBT', 'GEHT', 'STEHT', 'LIEGT', 'MACHT', 'TUN', 'SIEHT', 'SCHAUST', 'GUCKST',
        'GUCK', 'GUCKE', 'SCHAU', 'SCHAUE', 'ZEIG', 'ZEIGE', 'FINDE', 'FINDEN', 'PRÜFE', 'PRUFE',
        'ANALYSEN', 'ANALYSE', 'ANALYSIERE', 'ANALYSIS', 'CHECK', 'CHECKE', 'TEST', 'TESTE',
        'BITTE', 'MAL', 'DIR', 'MIR', 'UNS', 'EUCH', 'IHNEN', 'SIE', 'ER', 'ES', 'ICH', 'DU', 'WIR', 'IHR',
        'HIER', 'DORT', 'DA', 'JETZT', 'NUN', 'HEUTE', 'GESTERN', 'MORGEN', 'TAG', 'TAGE', 'WOCHE', 'MONAT',
        'KURS', 'PREIS', 'TREND', 'CALL', 'CALLS', 'PUT', 'PUTS', 'OPTION', 'OPTIONEN', 'GREEKS', 'GRIECHEN',
        'SPREAD', 'SPREADS', 'TRADE', 'TRADES', 'TRADING', 'BUY', 'SELL', 'LONG', 'SHORT', 'HOLD',
        'WALL', 'WALLS', 'NET', 'GEX', 'OI', 'IV', 'RSI', 'EMA', 'SMA', 'MACD', 'DTE', 'ATM', 'ITM', 'OTM',
        'AI', 'KI', 'API', 'KEY', 'MODEL', 'MODELL', 'NVIDIA', 'NIM', 'DEPO', 'PORTFOLIO', 'BACKTEST', 'CASH',
        'WAS', 'WER', 'WO', 'WANN', 'WARUM', 'WIESO', 'WOHIN', 'WOHER', 'WELCHE', 'WELCHER', 'WELCHES',
        'GUT', 'SCHLECHT', 'BESSER', 'MEHR', 'WENIGER', 'VIEL', 'SEHR', 'NICHT', 'KEIN', 'KEINE', 'JA', 'NEIN',
        'AUCH', 'NOCH', 'SCHON', 'NUR', 'WIEDER', 'IMMER', 'NIE', 'ALLES', 'ETWAS', 'NICHTS',
        'THE', 'A', 'AN', 'AND', 'OR', 'BUT', 'NOR', 'FOR', 'YET', 'SO', 'AT', 'BY', 'FROM', 'IN', 'INTO', 'OF',
        'ON', 'TO', 'WITH', 'ABOUT', 'AGAINST', 'BETWEEN', 'THROUGH', 'DURING', 'BEFORE', 'AFTER', 'ABOVE', 'BELOW',
        'IS', 'AM', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING', 'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID',
        'WILL', 'WOULD', 'SHALL', 'SHOULD', 'CAN', 'COULD', 'MAY', 'MIGHT', 'MUST',
        'HOW', 'WHAT', 'WHICH', 'WHO', 'WHOM', 'WHOSE', 'WHY', 'WHERE', 'WHEN',
        'THIS', 'THAT', 'THESE', 'THOSE', 'MY', 'YOUR', 'HIS', 'HER', 'ITS', 'OUR', 'THEIR',
        'I', 'YOU', 'HE', 'SHE', 'IT', 'WE', 'THEY', 'ME', 'HIM', 'HER', 'US', 'THEM',
        'LOOK', 'SEE', 'SHOW', 'TELL', 'GIVE', 'GET', 'KNOW', 'THINK', 'TAKE', 'MAKE',
        'STOCK', 'STOCKS', 'PRICE', 'PRICES', 'CHART', 'HIGH', 'LOW', 'OPEN', 'CLOSE'
    }

    @classmethod
    def extract_symbols_from_prompt(cls, prompt: str) -> List[str]:
        """Extracts valid stock tickers from user message, handling both uppercase, lowercase and contextual phrases."""
        found = []
        
        # 1. Look for explicit dollar sign notation ($GRAB, $SPY, $NVDA)
        for m in re.finditer(r'\$([A-Za-z]{1,5})\b', prompt):
            sym = m.group(1).upper()
            if sym not in cls.STOPWORDS:
                found.append(sym)

        # 2. Contextual indicators like 'guck dir mal GRAB an', 'analysiere grab', 'aktie grab', 'bei grab'
        context_patterns = [
            r'(?:guck\s+dir\s+(?:mal\s+)?|schau\s+dir\s+(?:mal\s+)?|analysiere\s+|wie\s+steht\s+|aktie\s+|ticker\s+|symbol\s+|für\s+|fuer\s+|bei\s+|zu\s+|über\s+|ueber\s+)([A-Za-z]{1,5})\b',
        ]
        for pat in context_patterns:
            for m in re.finditer(pat, prompt, re.IGNORECASE):
                sym = m.group(1).upper()
                if sym not in cls.STOPWORDS:
                    found.append(sym)

        # 3. Known major universe check
        known_universe = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "META", "MSFT", "AMZN", "GOOGL", "IWM", "PLTR", "ARM", "SMCI", "COIN", "GRAB", "INTC", "BABA", "DIS", "NFLX", "UBER"]
        for sym in known_universe:
            if re.search(rf"\b{sym}\b", prompt, re.IGNORECASE):
                found.append(sym)

        # 4. Standalone uppercase words of 2-5 chars (e.g. GRAB, SOFI, RIVN) not in stopwords
        tokens = re.findall(r'\b([A-Za-z]{2,5})\b', prompt)
        for tok in tokens:
            u = tok.upper()
            if tok.isupper() and u not in cls.STOPWORDS:
                found.append(u)

        dedup = list(dict.fromkeys(found))
        if not dedup:
            # Fallback only when no specific ticker was mentioned
            dedup = ["SPY"]
        return dedup[:3]

    @classmethod
    def gather_live_market_context(cls, symbols: List[str]) -> Dict[str, Any]:
        """Fetches live market data, indicators, greeks, and unusual volume for the symbols."""
        context_data = {}
        for sym in symbols:
            try:
                # 1. Price & Technical Indicators via yfinance (3mo for rapid fetching)
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="3mo")
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

                # 4. Strategy Setup & Dynamic Strikes
                signal = SignalGenerator.analyze_ticker(sym)
                regime = signal.regime if signal else None
                trade = signal.trade if signal else None
                
                strategy_name = trade.strategy_name if trade else (regime.strategy_type if regime else "NEUTRAL_WAIT")
                action = trade.action if trade else (regime.action if regime else "NONE")
                legs_summary = trade.legs_summary if trade else "NONE"
                
                # Dynamic strikes tailored to ticker spot price level if no fixed spread exists
                if legs_summary == "NONE" and spot > 0:
                    width = 5.0 if spot > 100 else (2.5 if spot > 30 else (1.0 if spot > 10 else 0.5))
                    put_wall = unusual.put_support_strike if unusual and unusual.put_support_strike else round(spot * 0.95, 2)
                    call_wall = unusual.call_resistance_strike if unusual and unusual.call_resistance_strike else round(spot * 1.05, 2)
                    if "BUY" in action:
                        legs_summary = f"Long Put: ${put_wall} | Long Call: ${call_wall}"
                    else:
                        short_p = round(put_wall, 2)
                        long_p = round(put_wall - width, 2)
                        short_c = round(call_wall, 2)
                        long_c = round(call_wall + width, 2)
                        legs_summary = f"Puts: {short_p}/{long_p} | Calls: {short_c}/{long_c}"

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
                    "recommended_strategy": strategy_name,
                    "action": action,
                    "legs": legs_summary,
                }
            except Exception as e:
                continue

        return context_data

    @classmethod
    def chat(cls, user_message: str, chat_history: List[Dict[str, str]] = None, api_key_override: Optional[str] = None, model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a user question, retrieves real-time market data & indicators,
        and queries the NVIDIA NIM Supermodel.
        """
        api_key = api_key_override.strip() if api_key_override else cls.get_api_key()
        model_name = model_override.strip() if model_override else cls.get_model()

        # Extract symbols and gather real-time data
        symbols = cls.extract_symbols_from_prompt(user_message)
        market_context = cls.gather_live_market_context(symbols)

        # Build comprehensive quantitative system prompt
        context_json = json.dumps(market_context, indent=2)

        system_prompt = f"""Du bist der institutionelle quantitative KI-Copilot für Optionen-Trading und Derivate-Analyse, angetrieben von NVIDIAs 120B/340B Supermodel-Architektur.
Du hast direkten Zugriff auf Yahoo Finance Live-Kurse, technische Indikatoren (EMA 20/50/200, RSI 14), 1. & 2. Ordnung Griechen (Delta, Gamma, Vanna, Charm, Volga), Dealer Net GEX, ungewöhnliches Optionsvolumen und Gamma-Magnet-Pinning-Strikes.

AKTUELLE LIVE-MARKT-DATEN & QUANTITATIVE ANALYSE (DIREKT AUS YAHOO FINANCE & DEM GREEKS ENGINE):
{context_json}

RICHTLINIEN FÜR DEINE ANTWORTEN:
1. Sei präzise, quantitativ fundiert und professionell wie ein Senior Volatility Trader bei einem Hedgefonds.
2. Beziehe dich IMMER exakt auf das vom Benutzer angefragte Symbol (z.B. {symbols[0] if symbols else 'den Markt'}).
3. Nutze immer die konkreten Zahlen aus dem Datenkontext (exakter Spot-Kurs, EMA-Level, RSI, Net GEX, Gamma Magnet Strike, DTE und Strikes).
4. Erkläre bei Nachfragen verständlich:
   - Warum ein Gamma-Magnet den Kurs anzieht (Dealer Delta-Hedging Pinning).
   - Wie sich Vanna und Charm auf den Trade auswirken.
   - Warum Defined-Risk Spreads (Credit vs. Debit) nackten Optionen vorzuziehen sind.
5. Gib konkrete Handlungsoptionen mit Einstieg, Stop-Loss und Take-Profit (45-50%).
6. Antworte auf Deutsch im professionellen Markdown-Format direkt mit deiner Analyse. Keine internen Denkprozesse voranstellen.
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
            "max_tokens": 768,
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
            with urllib.request.urlopen(req, timeout=25) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                raw_reply = resp_json["choices"][0]["message"]["content"]
                
                # Clean up reasoning tokens if model outputs thinking blocks
                reply_text = re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL).strip()
                if reply_text.startswith("Here's a thinking process:"):
                    parts = re.split(r"\n\n(?=[A-Z0-9#])", reply_text)
                    if len(parts) > 1:
                        reply_text = "\n\n".join(parts[1:]).strip()

                return {
                    "status": "SUCCESS",
                    "model": model_name,
                    "symbols_analyzed": symbols,
                    "market_context": market_context,
                    "reply": reply_text if reply_text else raw_reply,
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
        """Fallback quantitative intelligence engine when API key is awaiting configuration or timing out."""
        if not symbols or not ctx:
            return "Bitte stelle eine Frage zu einem konkreten Ticker wie GRAB, SPY, QQQ, NVDA, AAPL oder TSLA."

        sym = symbols[0]
        data = ctx.get(sym, {})
        spot = data.get("spot_price", "---")
        rsi = data.get("rsi_14", "---")
        trend = data.get("trend", "---")
        magnet = data.get("gamma_magnet", "---")
        pull = data.get("magnet_pull_force", "---")
        strategy = data.get("recommended_strategy", "---")
        action = data.get("action", "---")
        legs = data.get("legs", "---")
        iv_rank = data.get("iv_rank", "---")

        try:
            rsi_val = float(rsi)
        except (ValueError, TypeError):
            rsi_val = 50.0

        anom_line = f"- **Erkannte Anomalie:** *{data['anomalies_detected'][0]}*" if data.get('anomalies_detected') else "- **Volumen-Status:** Normales institutionelles Orderbuch ohne extreme Block-Sweeps."

        return f"""### 🤖 NVIDIA Quant-Analyse für **{sym}**

**1. Live Markt- & Indikatoren-Profil (Yahoo Finance):**
- **Aktueller Kurs:** `${spot}` ({data.get('day_change_pct', 0.0):+}% heute)
- **Trend-Status:** `{trend}` (EMA 20: `${data.get('ema_20')}`, EMA 50: `${data.get('ema_50')}`, EMA 200: `${data.get('ema_200')}`)
- **RSI (14):** `{rsi}` ({'Stark überverkauft' if rsi_val < 30 else 'Überverkauft' if rsi_val < 40 else 'Überkauft' if rsi_val > 70 else 'Neutral'})
- **IV Rank:** `{iv_rank}%` (VIX: `{data.get('vix_level')}`)

**2. Dealer Greeks & Ungewöhnliches Volumen:**
- **Gamma Magnet Strike:** **`${magnet}`** (Anziehungskraft: `{pull}`)
- **Dynamischer Support (Put Wall):** `${data.get('put_support')}`
- **Dynamischer Widerstand (Call Wall):** `${data.get('call_resistance')}`
{anom_line}

**3. Quantitatives Handels-Setup:**
- **Empfehlung:** `{strategy}` ({action})
- **Struktur:** `{legs}`
- **Begründung:** Der Kurs von {sym} (${spot}) orientiert sich stark am Gamma-Magneten bei `${magnet}` (Put Support `${data.get('put_support')}` / Call Resistance `${data.get('call_resistance')}`). Bei einem IV-Rank von `{iv_rank}%` und einem RSI(14) von `{rsi}` bietet ein risiko-definiertes Setup um die Support-Zone statistisch das höchste Risk/Reward-Verhältnis.
"""
