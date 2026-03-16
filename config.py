"""
config.py — Configuración central del bot de value betting
=========================================================
Editá este archivo con tus API keys y preferencias.
"""

# ──────────────────────────────────────────────
# API KEYS
# ──────────────────────────────────────────────
# Obtené tu key gratuita en: https://the-odds-api.com/
THE_ODDS_API_KEY = "1ea934ce8f0fb58ef580d25420c9e873"

# Gratuita en: https://www.football-data.org/
FOOTBALL_DATA_API_KEY = "TU_KEY_AQUI"

# ──────────────────────────────────────────────
# LIGAS A ANALIZAR (códigos de football-data.org)
# ──────────────────────────────────────────────
LEAGUES = {
    "PL":  "Premier League",
    "PD":  "La Liga",
    "SA":  "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "CL":  "Champions League",
}

# Sports key para The Odds API
ODDS_SPORT_KEY = "soccer"

# Regiones de bookmakers (eu, uk, us, au)
ODDS_REGIONS = "eu"

# Mercados: h2h = 1X2, spreads = Asian Handicap, totals = over/under
ODDS_MARKETS = "h2h"

# ──────────────────────────────────────────────
# PARÁMETROS DE VALUE BETTING
# ──────────────────────────────────────────────
# Edge mínimo para considerar una apuesta como value (5% = 0.05)
MIN_VALUE_EDGE = 0.05

# Probabilidad mínima del modelo para considerar la apuesta (evita outliers)
MIN_MODEL_PROB = 0.10

# Cuota mínima y máxima (línea shopping)
MIN_ODDS = 1.30
MAX_ODDS = 10.0

# ──────────────────────────────────────────────
# GESTIÓN DE BANKROLL (Kelly Criterion)
# ──────────────────────────────────────────────
# Bankroll total en tu moneda
BANKROLL = 1000.0

# Fracción del Kelly completo (0.25 = Quarter Kelly, más conservador)
KELLY_FRACTION = 0.25

# Stake máximo por apuesta como % del bankroll (cap de riesgo)
MAX_STAKE_PCT = 0.05   # 5% máximo

# Stake mínimo por apuesta
MIN_STAKE = 5.0        # en tu moneda

# ──────────────────────────────────────────────
# AJUSTE VENTAJA DE LA CASA (Home Advantage)
# ──────────────────────────────────────────────
# Factor multiplicador sobre goles esperados del local
HOME_ATTACK_BOOST = 1.10

# ──────────────────────────────────────────────
# OPERACIÓN DEL BOT
# ──────────────────────────────────────────────
# Intervalo de polling en segundos cuando está en modo live
POLLING_INTERVAL_SECONDS = 300   # 5 minutos

# Número de semanas de datos históricos para el modelo
HISTORY_WEEKS = 20

# Guardar señales en CSV para backtesting
SAVE_SIGNALS_CSV = True
SIGNALS_CSV_PATH = "signals_log.csv"
