# ⚽ FPL Predictor — ML + Optimización Matemática

> Pipeline completo de Machine Learning + Programación Lineal Entera para armar el equipo óptimo en Fantasy Premier League (FPL).

---

## 🧠 ¿Cómo funciona?

El pipeline se ejecuta en 6 pasos automáticos:

```
API FPL → Datos históricos → Feature Engineering → LightGBM → Predicción xP → MILP Knapsack
```

1. **Fetch** — descarga datos de todos los jugadores disponibles desde la API oficial de FPL.
2. **Historial** — obtiene el historial partido a partido de los 300 jugadores con mayor `selected_by_percent`.
3. **Features** — calcula rolling averages de 5 gameweeks: puntos, minutos e ICT index.
4. **LightGBM** — entrena un modelo de regresión (`gbdt`) para predecir `total_points` del próximo GW.
5. **Predicción xP** — aplica el modelo al próximo fixture de cada jugador.
6. **Optimización** — resuelve un problema **MILP** (PuLP / CBC) para seleccionar el equipo de 15 jugadores que maximiza los xP respetando las restricciones de FPL.

---

## 📐 Restricciones del optimizador

| Restricción | Valor |
|---|---|
| Presupuesto total | £100m |
| Jugadores totales | 15 |
| Porteros (GK) | 2 |
| Defensas (DEF) | 5 |
| Centrocampistas (MID) | 5 |
| Delanteros (FWD) | 3 |
| Máx. por club | 3 |

---

## 🗂️ Estructura del proyecto

```
fpl_predictor/
├── fpl_pipeline.py            # Pipeline principal (ML + optimización)
├── fpl_transfer_recommender.py # Recomendador de transferencias
├── debug_chiesa.py            # Debug de jugadores específicos
└── debug_fpl.py               # Debug de datos de API
```

---

## 🚀 Uso

```bash
pip install requests pandas numpy pulp lightgbm
python fpl_pipeline.py
```

La salida muestra el equipo óptimo agrupado por posición:

```
Position   Player Name               Team ID  Cost (£m)   xP
-----------------------------------------------------------------
GK         Flekken                   11       £4.5        4.21
DEF        Alexander-Arnold          12       £7.2        6.80
...
```

---

## 🛠️ Stack técnico

| Librería | Uso |
|---|---|
| `lightgbm` | Modelo de regresión para predecir xP |
| `pulp` + CBC | Solver MILP para optimización del equipo |
| `pandas` / `numpy` | Procesamiento de datos y feature engineering |
| `requests` | Consumo de la API oficial de FPL |

---

## 📌 Features del modelo

- `rolling_points_5` — suma de puntos en los últimos 5 GW
- `rolling_minutes_5` — promedio de minutos en los últimos 5 GW
- `rolling_ict_5` — promedio del ICT Index en los últimos 5 GW
- `fixture_difficulty` — dificultad del próximo partido (FDR)

---

## ⚠️ Notas

- El script limita el fetch de historial a **300 jugadores** (ordenados por ownership) para evitar timeouts.
- Los jugadores sin historial o sin fixture próximo reciben `xP = 0`.
- Se requiere conexión a internet para acceder a la API de FPL.
