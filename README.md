# QQQ Trend Prediction mit Top Tech-Aktien

### Problem Definition:
### Ziel

Vorhersage der Preisrichtung über die nächsten t=[5, 10, 15, 20, 30, 60, 120] Minuten für den Invesco QQQ ETF (QQQ) unter Verwendung der Top 5 Tech-Aktien als Einflussfaktoren.

Für jede Minute vom 2022-01-03 bis zum 2025-11-21 berechnen wir die erwartete Preisveränderung über das zukünftige Fenster t, während wir aktuelle technische Features von sowohl QQQ als auch Top-Tech-Aktien als Input-Predictors verwenden.

### Input Features

*QQQ Technische Features:*
- Normalisierte VWAP und Volumen
- Normalisierte exponentielle gleitende Durchschnitte (EMA) über [5, 10, 20] Minuten
- EMA Differenz (EMA5 - EMA20)
- Kurz- und mittelfristige Returns (5, 15, 30 Minuten)
- Realisierte Volatilität (10 Minuten)

*Top Tech-Aktien Features (NVDA, AAPL, MSFT, GOOGL, AMZN)*:
- Normalisierte VWAP und Volumen für jede Aktie
- EMA über [5, 10, 20] Minuten für jede Aktie
- Kurz- und mittelfristige Returns (5, 15, 30 Minuten) für jede Aktie
- EMA slopes für jede Aktie

*Multi-Asset Relationship Features*:

- Korrelation zwischen QQQ und Tech-Aktien (15 Minuten)
- Relative Stärke (QQQ Performance vs Tech-Durchschnitt)
- Tech Momentum Leader (führende Tech-Aktie identifizieren)

### Verfahrensübersicht:
- Sammelt Minuten-Bars für QQQ und Top 5 Tech-Aktien von 2022-01-03 → 2025-11-21
- Berechnet technische Features für QQQ und vereinfachte Cross-Asset Features
- Sagt Trendrichtung über nächste 30 Minuten vorher mittels Neural Network
- Nutzt Decision Tree zur Identifikation von Entry-Points
- Implementiert Trading-Strategie in Alpaca



**Wir erwarten Muster zu finden, bei denen Tech-Aktien Momentum und Korrelationsänderungen QQQ Trendbewegungen vorausgehen.**

---
## 01 Data Acquisition
Bezieht Rohmarktdaten für QQQ und Top Tech-Aktien von 2022-01-03 bis 2025-11-21, verwendet Alpaca Markets API als exklusive Datenquelle. Die Daten sind gefiltert auf reguläre Handelszeiten.

**Script**

[scripts/01_data_acquisition/01_data_acquisition.py](scripts/01_data_acquisition/01_data_acquisition.py)

Zieht **1-minute** adjustierte bars von **2022-01-03 → 2025-11-21** und schreibt `symbol.parquet` Dateien nach `../trading-board-project/data/raw/QQQ_1m`

**APIs Used**
- Alpaca Markets API v2

**Parameter**
- `symbol`: QQQ, NVDA, AAPL, MSFT, GOOGL, AMZN
- `timeframe`: 1Min (1-Minuten Bars)
- `feed`: iex (kostenloser IEX Daten-Feed mit 15-minütiger Verzögerung) 
- `adjustment`: all (automatische Anpassung für Splits und Dividenden)
- `limit`: 10000
- `start`: 2022-01-03 (Startdatum)  
- `end`: aktuelles Datum (Enddatum)
- `sort`: asc (Sortierreihenfolge, neueste zuerst)

**Datenspeicherung**
- Parquet-Dateien in `../trading-board-project/data/raw/QQQ_1m/`
- Gefiltert für reguläre Handelszeiten
- `columns`: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `trade_count`,`vwap`,   

QQQ Rohdaten Beispiel:

<img src="images/01_QQQ_bar_data.png" alt="drawing" width="800"/>

---

## 02 Data Understanding

**QQQ Close Price**

 Zeigt den Close-Preisverlauf von QQQ über die Zeit, inklusive markierter wichtiger Ereignisse wie dem ChatGPT-Launch, der SVB-Krise und starken NVIDIA-Gewinnen.

![QQQ Close](images/data_understanding/qqq_close.png)

**Durchschnittliches Intraday-Volumen** 

Zeigt das durchschnittliche Volumen pro Minute von QQQ.

![Avg Intraday Volume](images/data_understanding/qqq_avg_intraday_volume.png)

**1‑Minuten-Rendite Histogramm**

Zeigt die Renditeverteilung und Abweichungen von der Normalverteilung.

![Returns Histogram](images/data_understanding/qqq_returns_hist_improved.png)

**Korrelations-Heatmap**

Zeigt die Korrelationen der 1‑Minuten-Renditen zwischen QQQ und Top-Tech-Aktien.

![Correlation Heatmap](images/data_understanding/corr_heatmap.png)




