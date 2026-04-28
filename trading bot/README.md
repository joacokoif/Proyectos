# 📈 BTC MACD-ATR Strategy — Algoritmo de Trading para Bitcoin

> Estrategia algorítmica de trading para **BTC/USDT** que combina **MACD**, **EMA 50** y **ATR** para detectar señales de entrada Long/Short con gestión de riesgo dinámica. Consume datos en tiempo real desde la **API pública de Binance**.

---

## 🧠 Lógica de la estrategia

```
Datos OHLCV (Binance API, 1h)
        ↓
Indicadores: EMA 50 + MACD(12,26,9) + ATR(14)
        ↓
Condición de entrada detectada?
        ├── LONG:  cruce alcista MACD + Precio > EMA 50
        └── SHORT: cruce bajista MACD + Precio < EMA 50
        ↓
Niveles de riesgo calculados con ATR (ratio 1:1.5)
```

---

## 📐 Indicadores

### EMA 50 — Filtro de tendencia
Determina la dirección del mercado. Solo se toman Longs cuando el precio está **por encima** de la EMA 50, y Shorts cuando está **por debajo**.

### MACD (12, 26, 9)
- `MACD Line = EMA(12) - EMA(26)`
- `Signal Line = EMA(9) de la MACD Line`
- Se detecta un **cruce alcista** (bullish crossover) o **bajista** (bearish crossunder)

### ATR 14 — Average True Range
Mide la volatilidad real del mercado. Se usa para calcular el Stop Loss y Take Profit de forma dinámica, adaptándose a las condiciones del mercado.

> ⚙️ El ATR usa la fórmula RMA (Running Moving Average) de TradingView: `alpha = 1/14`

---

## 🎯 Condiciones de entrada

| Señal | MACD | Precio vs EMA 50 |
|---|---|---|
| 🟢 **LONG** | Cruce alcista (crossover) | Precio **>** EMA 50 |
| 🔴 **SHORT** | Cruce bajista (crossunder) | Precio **<** EMA 50 |

---

## 💰 Gestión de riesgo (ratio 1:1.5)

| | LONG | SHORT |
|---|---|---|
| **Stop Loss** | `Precio - 1.0 × ATR` | `Precio + 1.0 × ATR` |
| **Take Profit** | `Precio + 1.5 × ATR` | `Precio - 1.5 × ATR` |

---

## 🖥️ Ejemplo de salida

```
[14:32:01] Iniciando estrategia para BTCUSDT (1h)...

==================================================
[ DATOS DEL MERCADO (2024-04-28 14:00:00) ]
==================================================
Precio actual (Close): 63,412.50
EMA 50:                61,820.30
MACD Line / Signal:    142.80 / 98.20
ATR (14):              890.45

--- SEÑALES ESTRATEGIA ---
[LONG] ALERTA LONG (COMPRA)! - Condiciones cumplidas
    -> Precio Entrada: 63,412.50
    -> Stop Loss    (-1.0 ATR): 62,522.05
    -> Take Profit  (+1.5 ATR): 64,748.18
==================================================
```

---

## 🚀 Uso

```bash
pip install requests pandas numpy
python btc_macd_atr_strategy.py
```

> No requiere API key — usa el endpoint público de Binance.

---

## ⚙️ Configuración

Editá estas variables al inicio del script para personalizar:

```python
symbol   = 'BTCUSDT'   # Par a analizar (ej. ETHUSDT, SOLUSDT)
interval = '1h'        # Temporalidad: '15m', '1h', '4h', '1d'
limit    = 1000        # Cantidad de velas históricas
```

---

## 🛠️ Stack técnico

| Librería | Uso |
|---|---|
| `requests` | Consumo de la API pública de Binance |
| `pandas` | Procesamiento de datos OHLCV |
| `numpy` | Cálculo de indicadores técnicos |

---

## ⚠️ Disclaimer

Este script es de **uso educativo y analítico**. No ejecuta órdenes reales ni conecta con ninguna wallet. Las señales generadas son para análisis personal y no constituyen asesoramiento financiero.
