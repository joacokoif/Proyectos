# 📈 Stocks — Análisis Cuantitativo de Mercados

> Conjunto de scripts de análisis cuantitativo para mercados financieros: desde un screener de acciones de Mega Caps hasta una estrategia algorítmica de trading en BTC con MACD + ATR.

---

## 🗂️ Scripts incluidos

### 1. `stock_screener_mvp.py` — Escáner Cuantitativo de Acciones

Screener automático para **19 Mega Caps del S&P 500** usando datos históricos de 1 año.

**Lógica:**
- Descarga precios desde **Yahoo Finance** (`yfinance`)
- Calcula **SMA 50**, **SMA 200** y **RSI 14**
- Clasifica cada acción en una señal:

| Señal | Condición |
|---|---|
| `BUY (Pullback en Tendencia)` | Precio > SMA 200 y RSI < 45 |
| `BUY (Rebote de Riesgo)` | Precio < SMA 200 y RSI < 30 |
| `SELL (Sobrecomprada)` | RSI > 70 |
| `HOLD` | Resto de casos |

**Universo analizado:**
`SPY AAPL MSFT GOOGL AMZN NVDA META TSLA BRK-B JPM V WMT JNJ PG UNH HD MA CVX LLY`

```bash
python stock_screener_mvp.py
```

---

### 2. `predictor.py` — Predictor de Precios

Script de predicción de precios con modelos estadísticos.

---

## 🚀 Instalación

```bash
pip install requests pandas numpy yfinance
```

> **Nota:** `btc_macd_atr_strategy.py` usa la API pública de Binance (sin autenticación). No se ejecutan órdenes reales, solo análisis de señales.

---

## 🛠️ Stack técnico

| Librería | Uso |
|---|---|
| `yfinance` | Datos históricos de acciones (Yahoo Finance) |
| `requests` | API de Binance para datos OHLCV en tiempo real |
| `pandas` / `numpy` | Cálculo de indicadores técnicos |

---

## ⚠️ Disclaimer

Este proyecto es de **uso educativo y analítico**. No constituye asesoramiento financiero. Las señales generadas son para análisis personal y no garantizan resultados de inversión.
