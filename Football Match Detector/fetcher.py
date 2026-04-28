"""
fetcher.py — David vs Goliat
==============================
Llama a la API de football-data.org para obtener:
  - Tabla de posiciones de una liga
  - Partidos próximos (fixtures) de una liga
"""

import time
from datetime import datetime, timedelta, timezone

import requests

from config import FOOTBALL_DATA_API_KEY, DIAS_ADELANTE

BASE_URL = "https://api.football-data.org/v4"

HEADERS = {
    "X-Auth-Token": FOOTBALL_DATA_API_KEY,
}

# Rate limit: plan gratuito permite ~10 req/min → esperamos un poco entre llamadas
_SLEEP_BETWEEN_REQUESTS = 6.5  # segundos


def _get(endpoint: str, params: dict = None) -> dict | None:
    """Hace un GET a la API y devuelve el JSON, o None si falla."""
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        status = resp.status_code
        if status in (400, 401, 403, 429):
            try:
                error_data = resp.json()
                msg = error_data.get("message", "Error desconocido en la API.")
            except Exception:
                msg = f"Error HTTP {status}"

            prefix = "⚠️  Rate limit" if status == 429 else "❌  Error de API"
            raise RuntimeError(f"{prefix}: {msg}") from e
        return None
    except requests.exceptions.RequestException:
        return None


def get_standings(liga_code: str) -> list[dict]:
    """
    Devuelve la lista de equipos ordenados por posición en la tabla.
    Cada elemento: {"position": int, "team": str, "team_id": int}
    """
    data = _get(f"/competitions/{liga_code}/standings")
    time.sleep(_SLEEP_BETWEEN_REQUESTS)

    if not data or "standings" not in data:
        return []

    # La API devuelve 3 tipos de standings (TOTAL, HOME, AWAY). Usamos TOTAL.
    tabla = next(
        (s for s in data["standings"] if s.get("type") == "TOTAL"),
        data["standings"][0] if data["standings"] else None,
    )
    if not tabla:
        return []

    return [
        {
            "position": row["position"],
            "team":     row["team"]["name"],
            "team_id":  row["team"]["id"],
            "pts":      row["points"],
            "played":   row["playedGames"],
            "won":      row["won"],
            "draw":     row["draw"],
            "lost":     row["lost"],
        }
        for row in tabla.get("table", [])
    ]


def get_upcoming_fixtures(liga_code: str) -> list[dict]:
    """
    Devuelve los partidos programados de los próximos DIAS_ADELANTE días.
    Cada elemento: {"match_id", "date", "home_id", "home", "away_id", "away", "competition"}
    """
    hoy = datetime.now(timezone.utc)
    hasta = hoy + timedelta(days=DIAS_ADELANTE)

    data = _get(
        f"/competitions/{liga_code}/matches",
        params={
            "status":    "SCHEDULED",
            "dateFrom":  hoy.strftime("%Y-%m-%d"),
            "dateTo":    hasta.strftime("%Y-%m-%d"),
        },
    )
    time.sleep(_SLEEP_BETWEEN_REQUESTS)

    if not data or "matches" not in data:
        return []

    fixtures = []
    for m in data["matches"]:
        utc_str = m.get("utcDate", "")
        try:
            dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
            fecha_local = dt.strftime("%a %d/%m  %H:%Mh")
        except ValueError:
            fecha_local = utc_str

        fixtures.append(
            {
                "match_id":    m["id"],
                "date":        fecha_local,
                "date_raw":    utc_str,
                "home_id":     m["homeTeam"]["id"],
                "home":        m["homeTeam"]["name"],
                "away_id":     m["awayTeam"]["id"],
                "away":        m["awayTeam"]["name"],
                "competition": m.get("competition", {}).get("name", liga_code),
                "matchday":    m.get("matchday"),
            }
        )
    return fixtures
