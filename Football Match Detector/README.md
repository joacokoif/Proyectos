# ⚔️ David vs Goliat — Football Match Detector

> Script de terminal que detecta automáticamente enfrentamientos entre equipos del **Top 5** y los **últimos 3** de la tabla en las principales ligas de fútbol europeo, usando la API de [football-data.org](https://www.football-data.org).

---

## 🧠 ¿Qué hace?

Para cada liga configurada, el script:

1. **Descarga** la tabla de posiciones actualizada.
2. **Obtiene** los próximos partidos (ventana configurable en días).
3. **Detecta** enfrentamientos **Goliat** (Top N) vs **David** (Últimos N).
4. **Muestra** un reporte visual en terminal con `rich` (colores, paneles, tablas).

---

## 🖥️ Ejemplo de salida

```
⚔️   DAVID vs GOLIAT
Top 5 vs Últimos 3  ·  próximos 7 días  ·  5 ligas activas

━━━━━━━━━━━━  Premier League  ━━━━━━━━━━━━

  #  Equipo               PTS  PJ
  1  Manchester City 🏰   72   32
  ...
 18  Luton Town 🗡️         28   32

  ⚔️  1 enfrentamiento(s) encontrado(s):

  ┌─ 15/04/2024 · Jornada 33 ─────────────────┐
  │ Manchester City vs Luton Town              │
  │ Local (Goliat) → Visitante (David)         │
  │ 🏰 Goliat: #1 Manchester City  (Top 5)    │
  │ 🗡️  David:  #18 Luton Town  (Últimos 3)   │
  └────────────────────────────────────────────┘
```

---

## 🗂️ Estructura del proyecto

```
David_vs_Goliat/
├── main.py          # Pipeline principal + visualización con rich
├── fetcher.py       # Descarga de tabla y fixtures via API
├── analyzer.py      # Lógica de detección David vs Goliat
├── config.py        # API Key, ligas activas, parámetros TOP_N / BOTTOM_N
└── requirements.txt # Dependencias
```

---

## ⚙️ Configuración rápida

### 1. Obtener API Key (gratis)

Registrate en [football-data.org/client/register](https://www.football-data.org/client/register) y confirmá tu email.

### 2. Configurar `config.py`

```python
FOOTBALL_DATA_API_KEY = "tu_api_key_aqui"

LIGAS_ACTIVAS = ["PL", "PD", "BL1", "SA", "FL1"]  # Premier, LaLiga, Bundesliga, Serie A, Ligue 1

TOP_N    = 5   # Cuántos equipos se consideran "Goliat"
BOTTOM_N = 3   # Cuántos equipos se consideran "David"
DIAS_ADELANTE = 7  # Ventana de búsqueda de partidos
```

### 3. Ejecutar

```bash
pip install -r requirements.txt
python main.py
```

---

## 🔧 Ligas disponibles

| Código | Liga |
|---|---|
| `PL` | Premier League (Inglaterra) |
| `PD` | La Liga (España) |
| `BL1` | Bundesliga (Alemania) |
| `SA` | Serie A (Italia) |
| `FL1` | Ligue 1 (Francia) |

---

## 🛠️ Stack técnico

| Librería | Uso |
|---|---|
| `rich` | Terminal UI: tablas, paneles, colores |
| `requests` | Consumo de la API football-data.org |
| Python stdlib | `datetime`, `sys`, `io` |

---

## 📌 Notas

- El plan gratuito de football-data.org tiene **límite de 10 requests/minuto**.
- Si agregás muchas ligas, puede llegar al rate limit — el script lo maneja con mensajes de error informativos.
- Compatible con Windows (UTF-8 forzado en `sys.stdout`).
