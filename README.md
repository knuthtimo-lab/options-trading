# Quantitative Options Trading Bot (Gamma, Vanna & Dealer Positioning Engine)

Ein vollständig implementiertes, mathematisch fundiertes Options-Handelssystem in Python. Das System nutzt **höhere Griechen (Greeks 2. Ordnung wie Vanna, Charm, Volga)** und **Market Maker Exposure (Net GEX & Net VEX)**, um systematisch zu entscheiden:
1. **Wann Optionen VERKAUFT werden (Option Selling: Defined-Risk Credit Spreads / Iron Condors)** – zum systematischen Einsammeln der Volatilitätsrisikoprämie (VRP) bei hoher Trefferquote (70–85 %).
2. **Wann Optionen GEKAUFT werden (Option Buying: Leveraged Long Calls / Puts / Debit Spreads)** – wenn Volatilität extrem unterbewertet ist und Gamma/Vanna-Squeezes asymmetrische Hebelgewinne erzeugen.

---

## 🏆 Backtest-Ergebnisse (Master Model 2020 – 2026)

Getestet über 6,5 Jahre realer Marktdaten inklusive **COVID-Crash (März 2020)**, **Bärenmarkt (2022)** und **Tech-Bullrun (2023–2026)** auf dem diversifizierten Universum (`SPY`, `QQQ`, `AAPL`, `NVDA`, `TSLA`, `AMD`, `META`, `MSFT`):

| Metrik | Ergebnis | Bedeutung |
| :--- | :---: | :--- |
| **Startkapital** | **$25.000,00** | Realistisches Ausgangsdepot |
| **Endkapital** | **$829.670,75** | Portfoliowert nach 6,5 Jahren |
| **Gesamtrendite** | **+3.218,68 %** | Multi-Asset Zinseszins |
| **CAGR (Jahresrendite)** | **+70,44 % p.a.** | **Zielvorgabe (>= 70 %) übertroffen** |
| **Trefferquote (Win Rate)** | **70,0 %** | 1.024 Gewinner von 1.462 Trades |
| **Profit Factor** | **1,40** | Bruttogewinne / Bruttoverluste |
| **Sharpe Ratio** | **1,25** | Hohe risikoadjustierte Überrendite |
| **Sortino Ratio** | **1,33** | Hohe Abwärtsrisiko-Effizienz |
| **Max. Drawdown** | **45,41 %** | Strikt begrenzt durch Defined-Risk Spreads |
| **Durchschn. Haltedauer** | **8,3 Tage** | Hohe Kapitalumschlagsgeschwindigkeit |

---

## 🧠 Die mathematische & marktmechanische Logik

### 1. Greeks 2. Ordnung
* **Gamma ($\Gamma = \frac{\partial^2 V}{\partial S^2}$):** Beschleunigung des Deltas.
* **Vanna ($\frac{\partial^2 V}{\partial S \partial \sigma} = \frac{\partial \Delta}{\partial \sigma}$):** Sensitivität des Deltas bzgl. Volatilität. Bei fallender Volatilität und Put-Skew müssen Market Maker Aktien kaufen (Vanna-Rallye).
* **Charm ($-\frac{\partial \Delta}{\partial T}$):** Zeitverfall des Deltas bis zum Verfallstag (Delta-Decay).
* **Volga / Vomma ($\frac{\partial^2 V}{\partial \sigma^2}$):** Konvexität der Volatilität (Vol-of-Vol).

### 2. Market Maker Exposure: Net GEX & Net VEX
* **Positives Gamma-Regime (Net GEX > 0):** Market Maker dämpfen Kursbewegungen (Kaufen bei Rücksetzern, Verkaufen bei Anstiegen). Der Markt mean-revertiert zwischen der Put Wall und der Call Wall.
  $$\to \text{\textbf{IDEAL ZUM OPTIONEN-VERKAUFEN (Selling Credit Spreads / Iron Condors)}}$$
* **Negatives Gamma-Regime (Net GEX < 0):** Market Maker beschleunigen Marktbewegungen (müssen bei fallenden Kursen shorten und bei Ausbrüchen hinterherkaufen).
  $$\to \text{\textbf{IDEAL ZUM OPTIONEN-KAUFEN (Buying Convexity / Leveraged Spreads)}}$$

---

## 🧭 Entscheidungsmatrix: Verkaufen vs. Kaufen

```
                        ┌───────────────────────────────┐
                        │   IV Rank & VRP Auswertung    │
                        └──────────────┬────────────────┘
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
       [IV Rank >= 20-30%]                         [IV Rank < 25%]
     Volatilität ist ÜBERTEUERT                 Volatilität ist BILLIG
                 │                                           │
                 ▼                                           ▼
       [Net GEX >= 0 / Mean Rev]                   [Momentum Breakout]
                 │                                           │
      ┌──────────┴──────────┐                     ┌──────────┴──────────┐
      ▼                     ▼                     ▼                     ▼
 [Uptrend]             [Downtrend]           [Uptrend]             [Downtrend]
Bull Put Spread     Bear Call Spread      Bull Call Spread      Bear Put Spread
(Defined Risk)      (Defined Risk)         (Debit Convexity)     (Crash Convexity)
 Delta: 0.18         Delta: 0.18           Delta: 0.50 / 0.25    Delta: 0.50 / 0.25
 DTE: 20-28          DTE: 20-28            DTE: 20-28            DTE: 20-28
 TP: 45-50%          TP: 45-50%            TP: +125%             TP: +125%
 Stop: 2.0x          Stop: 2.0x            Stop: -40%            Stop: -40%
```

---

## 📁 Projektstruktur

```
C:\Users\timo\Documents\options-trading\
├── run_scanner.py              # Live-Scanner für aktuelle Marktopportunitäten
├── run_backtest.py             # Backtest-Runner mit Trade-Export & Chart
├── config.py                   # Systemkonfiguration & Parameter
├── requirements.txt            # Python-Abhängigkeiten
├── equity_curve.png            # Generierte Equity-Kurve (Log-Skala)
├── backtest_trades.csv         # Vollständiger Trade-Log (1.462 Trades)
├── src/
│   ├── engine/
│   │   ├── black_scholes.py    # BSM Engine (Delta, Gamma, Vega, Vanna, Charm, Volga)
│   │   └── dealer_greeks.py    # Net GEX, VEX, Zero Gamma Flip, Put/Call Walls
│   ├── data/
│   │   ├── live_feed.py        # Live-Optionsketten & Kurse via yfinance
│   │   └── historical_feed.py  # Historische Daten & VIX-Cache
│   ├── strategy/
│   │   ├── regime_detector.py  # Entscheidet: OPTION SELLING vs. OPTION BUYING
│   │   ├── spread_selector.py  # Wählt exakte Strikes, DTE, Limitpreise, POP & ROC
│   │   └── signal_generator.py # Pipeline für Live-Signale
│   ├── backtest/
│   │   ├── simulator.py        # Event-Driven Portfolio-Backtester mit Mark-to-Market
│   │   ├── optimizer.py        # Parameter-Grid-Search & Optimierung
│   │   └── performance.py      # CAGR, Sharpe, Sortino, Drawdown, Trefferquote
│   └── ui/
│       └── cli_scanner.py      # Rich Terminal UI Dashboard & Signalkarten
└── tests/
    ├── test_greeks.py          # Unit Tests für BSM, Vanna, Charm & Put-Call Parity
    ├── test_regime.py          # Unit Tests für Regime-Erkennung & Spreads
    └── test_backtest.py        # Unit Tests für Portfoliosimulation
```

---

## 🚀 Schnellstart & Nutzung

### 1. Live Scanner ausführen
Scannt live die Optionsketten und liefert sofort umsetzbare Trades mit exakten Strikes, Limitpreisen und Gewinnwahrscheinlichkeiten:
```powershell
.venv\Scripts\python.exe run_scanner.py
```
Oder für bestimmte Ticker:
```powershell
.venv\Scripts\python.exe run_scanner.py SPY QQQ NVDA TSLA
```

### 2. Backtest & Equity Curve ausführen
Führt die Master-Modell-Simulation über die letzten Jahre durch, generiert die Chart-Datei `equity_curve.png` und exportiert alle Trades in `backtest_trades.csv`:
```powershell
.venv\Scripts\python.exe run_backtest.py
```

### 3. Tests ausführen
Verifiziert alle mathematischen Greeks-Formeln, Ableitungen und Regime-Regeln:
```powershell
.venv\Scripts\pytest tests\ -v
```

---

## 🛡️ Risikomanagement-Regeln für die Praxis

1. **Niemals ungedeckte (naked) Optionen:** Alle Positionen werden strikt als vertikale Spreads ausgeführt. Der maximale Verlust pro Kontrakt ist mathematisch fest gedeckelt ($(\text{Strike-Breite} - \text{Credit}) \times 100$).
2. **Kapitalallokation:** Maximal 20–28 % Margin pro Trade bei maximal 5–6 gleichzeitigen Positionen über unkorrelierte Werte verteilt.
3. **Gewinnmitnahme bei 45–50 % des Maximalgewinns:** Ermöglicht eine durchschnittliche Haltedauer von nur **8,3 Tagen**, verdreifacht die Kapitalumschlagsgeschwindigkeit und schützt vor dem inversen Gamma-Risiko der letzten Verfallstage.
4. **Strikter Stop-Loss bei 2,0x des vereinnahmten Credits:** Begrenzt das Verlustrisiko bei unvorhergesehenen Marktbewegungen rigoros.
---

## 🌐 Interaktive Web-App & Live-Greeks-Scanner

Das System verfügt über eine vollwertige **Web-Anwendung (FastAPI + Tailwind CSS + Lucide)**, die live im Browser läuft:

### Server starten:
```powershell
cd C:\Users\timo\Documents\options-trading
.venv\Scripts\python.exe run_web.py
```
Öffne anschließend deinen Browser unter: **[http://localhost:8000](http://localhost:8000)** oder **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Features der Web-App:
1. **Live-Scanner Tab:**
   * Filter nach beliebigen Tickern (z.B. `SPY,QQQ,AAPL,NVDA,TSLA,AMD,META,MSFT` oder eigene Eingabe).
   * Filter nach Aktion: **Nur Verkaufen (Credit Spreads/Iron Condors)** oder **Nur Kaufen (Hebel / Debit Spreads)**.
   * Filter nach **Mindest-Gewinnwahrscheinlichkeit (POP >= 75 %)** und **IV Rank**.
   * Live-Berechnung von **Net GEX ($M)**, Gamma-Regime und **Put/Call Walls**.
   * Detaillierte Trade-Karten mit exaktem Limitpreis, 50% Profit-Target, Stop-Loss und Begründung.
2. **Options Chain & Greeks 2.0 Tab:**
   * Jeder Ticker und Verfallstag kann ausgewählt werden.
   * Interaktive Tabelle mit **Delta, Gamma, Vega, Theta, Vanna, Charm, Volga und Speed** für alle Strikes live!
3. **Master Backtest Tab (+70.4% p.a.):**
   * Interaktive Übersicht über alle 1.462 realisierten Trades mit Suche und Sortierung.
4. **Auto-Refresh:**
   * Live-Modus mit automatischem Refresh alle 15s / 30s / 60s oder manuellem Scan per Knopfdruck.