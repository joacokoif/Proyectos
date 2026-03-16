"""
data_fetcher.py — Descarga stats históricas de football-data.org
================================================================
Calcula attack_strength y defence_strength por equipo
para alimentar el modelo de Poisson bivariado.

API gratuita: https://www.football-data.org/
Rate limit: 10 requests/minuto (tier free)
"""

import requests
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from config import FOOTBALL_DATA_API_KEY, LEAGUES, HISTORY_WEEKS


@dataclass
class TeamStats:
    """Estadísticas calculadas de un equipo en la temporada actual."""
    name: str
    matches_home: int = 0
    matches_away: int = 0
    goals_scored_home: int = 0
    goals_scored_away: int = 0
    goals_conceded_home: int = 0
    goals_conceded_away: int = 0

    @property
    def avg_scored_home(self) -> float:
        return self.goals_scored_home / self.matches_home if self.matches_home else 0.0

    @property
    def avg_scored_away(self) -> float:
        return self.goals_scored_away / self.matches_away if self.matches_away else 0.0

    @property
    def avg_conceded_home(self) -> float:
        return self.goals_conceded_home / self.matches_home if self.matches_home else 0.0

    @property
    def avg_conceded_away(self) -> float:
        return self.goals_conceded_away / self.matches_away if self.matches_away else 0.0


@dataclass
class LeagueStats:
    """Estadísticas a nivel de liga para normalizar los strengths."""
    league_code: str
    team_stats: Dict[str, TeamStats] = field(default_factory=dict)
    avg_home_goals: float = 0.0   # media de goles del local en la liga
    avg_away_goals: float = 0.0   # media de goles del visitante en la liga

    def get_attack_strength(self, team: str, is_home: bool) -> float:
        """
        Attack Strength = avg goles del equipo / avg goles de la liga.
        Valores > 1.0 indican ataque por encima de la media.
        """
        if team not in self.team_stats:
            return 1.0
        ts = self.team_stats[team]
        if is_home:
            ref = self.avg_home_goals
            team_avg = ts.avg_scored_home
        else:
            ref = self.avg_away_goals
            team_avg = ts.avg_scored_away
        return (team_avg / ref) if ref else 1.0

    def get_defence_strength(self, team: str, is_home: bool) -> float:
        """
        Defence Strength = avg goles concedidos del equipo / avg goles de la liga.
        Valores < 1.0 indican mejor defensa que la media.
        """
        if team not in self.team_stats:
            return 1.0
        ts = self.team_stats[team]
        if is_home:
            ref = self.avg_home_goals
            team_avg = ts.avg_conceded_home
        else:
            ref = self.avg_away_goals
            team_avg = ts.avg_conceded_away
        return (team_avg / ref) if ref else 1.0


class DataFetcher:
    """
    Obtiene resultados históricos de football-data.org y
    calcula strengths atacante/defensivo por equipo.
    """

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: str = FOOTBALL_DATA_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": self.api_key})
        self._cache: Dict[str, LeagueStats] = {}

    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        try:
            resp = self.session.get(
                f"{self.BASE_URL}{endpoint}",
                params=params or {},
                timeout=15
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                print("[DataFetcher] Rate limit alcanzado, esperando 60s...")
                time.sleep(60)
                return self._get(endpoint, params)
            print(f"[DataFetcher] HTTP Error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[DataFetcher] Error: {e}")
            return None

    def fetch_league_stats(self, league_code: str) -> LeagueStats:
        """
        Descarga los partidos jugados de una liga y calcula:
        - attack_strength por equipo (local/visitante)
        - defence_strength por equipo (local/visitante)
        - media de goles de la liga
        """
        if league_code in self._cache:
            return self._cache[league_code]

        print(f"[DataFetcher] Descargando stats de liga: {league_code}...")
        data = self._get(f"/competitions/{league_code}/matches", {
            "status": "FINISHED",
        })

        if not data or "matches" not in data:
            print(f"[DataFetcher] Sin datos para {league_code}, usando defaults.")
            return LeagueStats(league_code=league_code)

        matches = data["matches"]
        stats = LeagueStats(league_code=league_code)
        team_data: Dict[str, TeamStats] = {}
        total_home_goals = 0
        total_away_goals = 0
        total_matches = 0

        for match in matches:
            score = match.get("score", {}).get("fullTime", {})
            hg = score.get("home")
            ag = score.get("away")
            if hg is None or ag is None:
                continue

            home_name = match["homeTeam"]["shortName"]
            away_name = match["awayTeam"]["shortName"]

            if home_name not in team_data:
                team_data[home_name] = TeamStats(name=home_name)
            if away_name not in team_data:
                team_data[away_name] = TeamStats(name=away_name)

            team_data[home_name].matches_home += 1
            team_data[home_name].goals_scored_home += hg
            team_data[home_name].goals_conceded_home += ag

            team_data[away_name].matches_away += 1
            team_data[away_name].goals_scored_away += ag
            team_data[away_name].goals_conceded_away += hg

            total_home_goals += hg
            total_away_goals += ag
            total_matches += 1

        stats.team_stats = team_data
        stats.avg_home_goals = total_home_goals / total_matches if total_matches else 1.35
        stats.avg_away_goals = total_away_goals / total_matches if total_matches else 1.10

        print(f"[DataFetcher] {league_code}: {total_matches} partidos | "
              f"Avg goles: Local={stats.avg_home_goals:.2f}, Visita={stats.avg_away_goals:.2f}")

        self._cache[league_code] = stats
        return stats

    def fetch_all_leagues(self) -> Dict[str, LeagueStats]:
        """Descarga stats de todas las ligas configuradas."""
        result = {}
        for code in LEAGUES:
            result[code] = self.fetch_league_stats(code)
            time.sleep(6)  # respetar rate limit (10 req/min)
        return result


# ──────────────────────────────────────────────────────
# DATOS DEMO para modo sin API key
# ──────────────────────────────────────────────────────
def get_demo_league_stats() -> Dict[str, LeagueStats]:
    """Retorna stats simuladas de equipos para modo demo."""
    prem = LeagueStats(league_code="PL", avg_home_goals=1.55, avg_away_goals=1.20)
    prem.team_stats = {
        "Man City":  TeamStats("Man City",  18, 17, 48, 30, 15, 20),
        "Arsenal":   TeamStats("Arsenal",   17, 18, 35, 25, 18, 25),
    }

    liga = LeagueStats(league_code="PD", avg_home_goals=1.60, avg_away_goals=1.15)
    liga.team_stats = {
        "Real Madrid": TeamStats("Real Madrid", 17, 18, 52, 38, 10, 18),
        "Barcelona":   TeamStats("Barcelona",   18, 17, 48, 35, 12, 20),
    }

    serie = LeagueStats(league_code="SA", avg_home_goals=1.45, avg_away_goals=1.05)
    serie.team_stats = {
        "Napoli":   TeamStats("Napoli",   16, 17, 38, 28, 18, 22),
        "Juventus": TeamStats("Juventus", 18, 16, 30, 20, 15, 20),
    }

    bundes = LeagueStats(league_code="BL1", avg_home_goals=1.80, avg_away_goals=1.35)
    bundes.team_stats = {
        "Bayern Munich":      TeamStats("Bayern Munich",      18, 17, 65, 45, 12, 18),
        "Borussia Dortmund":  TeamStats("Borussia Dortmund",  17, 18, 45, 35, 22, 30),
    }

    ligue = LeagueStats(league_code="FL1", avg_home_goals=1.50, avg_away_goals=1.10)
    ligue.team_stats = {
        "PSG":       TeamStats("PSG",       18, 17, 60, 42, 10, 15),
        "Marseille": TeamStats("Marseille", 17, 17, 35, 28, 25, 32),
    }

    return {"PL": prem, "PD": liga, "SA": serie, "BL1": bundes, "FL1": ligue}
