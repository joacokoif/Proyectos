"""
analyzer.py — David vs Goliat
================================
Detecta partidos donde un equipo del Top N se enfrenta
a uno de los últimos M de la tabla.
"""

from dataclasses import dataclass


@dataclass
class DavidVsGoliat:
    date:        str
    competition: str
    matchday:    int | None
    home:        str
    home_pos:    int
    away:        str
    away_pos:    int
    david:       str   # nombre del equipo "pequeño"
    goliat:      str   # nombre del equipo "grande"
    david_pos:   int
    goliat_pos:  int
    david_is_home: bool


def find_david_vs_goliat(
    standings: list[dict],
    fixtures:  list[dict],
    top_n:     int,
    bottom_n:  int,
) -> list[DavidVsGoliat]:
    """
    standings : lista ordenada por posición (resultado de fetcher.get_standings)
    fixtures  : lista de partidos (resultado de fetcher.get_upcoming_fixtures)
    top_n     : cuántos del tope son "Goliat"
    bottom_n  : cuántos del fondo son "David"

    Devuelve una lista de DavidVsGoliat para cada enfrentamiento detectado.
    """
    if not standings or not fixtures:
        return []

    total = len(standings)

    # IDs de equipos top (Goliat) y bottom (David)
    top_ids    = {row["team_id"]: row["position"] for row in standings[:top_n]}
    bottom_ids = {row["team_id"]: row["position"] for row in standings[total - bottom_n:]}

    # Mapeo id → nombre
    id_to_name = {row["team_id"]: row["team"] for row in standings}

    matches = []
    for fix in fixtures:
        h_id = fix["home_id"]
        a_id = fix["away_id"]

        # Caso 1: local es Goliat, visitante es David
        if h_id in top_ids and a_id in bottom_ids:
            matches.append(DavidVsGoliat(
                date        = fix["date"],
                competition = fix["competition"],
                matchday    = fix["matchday"],
                home        = fix["home"],
                home_pos    = top_ids[h_id],
                away        = fix["away"],
                away_pos    = bottom_ids[a_id],
                goliat      = fix["home"],
                goliat_pos  = top_ids[h_id],
                david       = fix["away"],
                david_pos   = bottom_ids[a_id],
                david_is_home = False,
            ))

        # Caso 2: local es David, visitante es Goliat
        elif h_id in bottom_ids and a_id in top_ids:
            matches.append(DavidVsGoliat(
                date        = fix["date"],
                competition = fix["competition"],
                matchday    = fix["matchday"],
                home        = fix["home"],
                home_pos    = bottom_ids[h_id],
                away        = fix["away"],
                away_pos    = top_ids[a_id],
                goliat      = fix["away"],
                goliat_pos  = top_ids[a_id],
                david       = fix["home"],
                david_pos   = bottom_ids[h_id],
                david_is_home = True,
            ))

    # Ordenar cronológicamente
    matches.sort(key=lambda m: m.date)
    return matches
