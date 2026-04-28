"""
config.py — David vs Goliat
============================
Editá este archivo para personalizar el análisis.

Para obtener una API key gratuita:
  1. Registrate en https://www.football-data.org/client/register
  2. Confirmá el mail
  3. Pegá tu token en FOOTBALL_DATA_API_KEY
"""

# ─── API KEY ──────────────────────────────────────────────────────────────────
FOOTBALL_DATA_API_KEY = "TU_API_KEY_AQUI"  # → Registrate gratis en https://www.football-data.org/client/register

# ─── LIGAS ACTIVAS ────────────────────────────────────────────────────────────
# Agregá o quitá códigos según qué ligas querés analizar.
# Lista completa de ligas disponibles (plan gratuito):
#   PL  = Premier League (Inglaterra)
#   PD  = La Liga (España)
#   BL1 = Bundesliga (Alemania)
#   SA  = Serie A (Italia)
#   FL1 = Ligue 1 (Francia)
#   PPL = Primeira Liga (Portugal)
#   DED = Eredivisie (Países Bajos)
#   BSA = Brasileirão (Brasil)
#   WC  = Copa del Mundo
#   CL  = Champions League

LIGAS_ACTIVAS = [
    "PL",   # Premier League
    "PD",   # La Liga
    "BL1",  # Bundesliga
    "SA",   # Serie A
    "FL1",  # Ligue 1
    "PPL",  # Primeira Liga
    "DED",  # Eredivisie
    "ELC",  # Championship
]

# Nombres bonitos para mostrar en pantalla
LIGAS_NOMBRES = {
    "PL":  "🏴󠁧󠁢󠁥󠁮󠁧󠁿  Premier League",
    "PD":  "🇪🇸  La Liga",
    "BL1": "🇩🇪  Bundesliga",
    "SA":  "🇮🇹  Serie A",
    "FL1": "🇫🇷  Ligue 1",
    "PPL": "🇵🇹  Primeira Liga",
    "DED": "🇳🇱  Eredivisie",
    "BSA": "🇧🇷  Brasileirão",
    "WC":  "🌍  Copa del Mundo",
    "CL":  "🏆  Champions League",
}

# ─── PARÁMETROS DE ANÁLISIS ───────────────────────────────────────────────────
# Cuántos equipos del TOP se consideran "Goliat"
TOP_N = 5

# Cuántos equipos del FONDO se consideran "David"
BOTTOM_N = 3

# Cuántos días hacia adelante buscar partidos (desde hoy)
DIAS_ADELANTE = 14
