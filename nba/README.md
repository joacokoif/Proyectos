# 🏀 NBA Back-to-Back Calculator — Modelo Log5

> Herramienta de análisis cuantitativo del rendimiento de franquicias NBA en situaciones de **Back-to-Back** (dos partidos en días consecutivos), con predicción probabilística via **Fórmula Log5** y generación de **reportes PDF profesionales**.

---

## 📊 ¿Qué hace?

1. **Descarga** el historial de partidos de la temporada elegida desde la NBA API oficial.
2. **Detecta** todos los back-to-backs jugados por cada equipo.
3. **Calcula métricas** empíricas reales: Win Rate en Game 1 y Game 2, distancia viajada, fatiga del rival.
4. **Predice** probabilidades del próximo B2B de un equipo usando la **Fórmula Log5** ajustada por:
   - Ventaja de localía (+/- 3.5%)
   - Fatiga propia en Game 2 (-4.5%)
   - Distancia de viaje (+penalidad si > 1000 km o > 2000 km)
   - Fatiga del rival (si también viene de B2B)
5. **Genera un PDF** completo con 6 gráficos analíticos.

---

## 📄 Reporte PDF

El reporte incluye:

| Sección | Descripción |
|---|---|
| Cabecera del equipo | Win% base (True Talent) |
| Último B2B disputado | Resultados, matchups, distancia, rivales cansados |
| Próximo B2B | Predicción Log5 Game 1 y Game 2 |
| Donut chart | Probabilidades W-W, W-L, L-W, L-L |
| Stacked bar | Histórico real vs. modelo Log5 |
| Radar chart | Perfil del equipo (talento, local, Game 2, sweep) |
| Fatigue Matrix | Heatmap 2x2: escenarios de fatiga propia y rival |
| Game 2 Context | Win Rate en distintos contextos de fatiga |
| Top 10 resilience | Ranking de las 10 franquicias más resilientes en B2B |
| Tabla global | Los 30 equipos con sus probabilidades B2B |

---

## 🗂️ Estructura del proyecto

```
nba/
├── calculadora_b2b.py   # Lógica principal: Log5, análisis, fetch NBA API
├── B2B_nba_pdf.py       # Generador de reporte PDF con 6 gráficos
├── nba_tv_hoy.py        # Partidos de hoy en TV
├── nba_test.py          # Tests y validaciones
└── requirements.txt     # Dependencias
```

---

## 🚀 Uso

### Calculadora interactiva (terminal)

```bash
pip install -r requirements.txt
python calculadora_b2b.py
```

```
> Temporada (ej. 25/26): 25/26
> Fecha de corte (opcional): 01/04/2026
> Equipo (ej. NYK, LAL): NYK
```

### Generador de reporte PDF

```bash
python B2B_nba_pdf.py
```

Genera un archivo como: `Reporte_B2B_NYK_2025-26_01-04-2026.pdf`

---

## 🧮 Fórmula Log5

```python
P(A gana B) = (pA × (1 - pB)) / (pA × (1 - pB) + pB × (1 - pA))
```

Con ajustes de ecosistema NBA:
- 🏠 **Localía**: ±3.5%
- 😴 **Fatiga B2B**: -4.5%
- ✈️ **Viaje > 1000 km**: -1.5% adicional
- ✈️ **Viaje > 2000 km**: -2.5% adicional
- ⚔️ **Rival también en B2B**: penalidad simétrica aplicada al oponente

---

## 🛠️ Stack técnico

| Librería | Uso |
|---|---|
| `nba_api` | Datos oficiales de partidos NBA |
| `fpdf` | Generación de reportes PDF |
| `matplotlib` / `seaborn` | Gráficos analíticos (donut, radar, heatmap, bar) |
| `pandas` / `numpy` | Procesamiento de datos |
| `requests` | Scraping de calendarios desde Basketball-Reference |

---

## 📌 Abreviaturas de equipos soportadas

`ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS`
