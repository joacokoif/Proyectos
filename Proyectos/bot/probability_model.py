"""
probability_model.py — Modelo Poisson Bivariado para fútbol
============================================================
Metodología estándar de la industria de apostadores profesionales.

Calcula la probabilidad real de cada resultado (1 / X / 2)
usando las fuerzas de ataque y defensa de cada equipo
y la distribución de Poisson para simular la distribución de goles.

Referencia: Dixon & Coles (1997) - "Modelling Association Football Scores"
"""

import math
from dataclasses import dataclass
from typing import Tuple, List
from data_fetcher import LeagueStats
from config import HOME_ATTACK_BOOST


@dataclass
class MatchProbabilities:
    """Probabilidades del modelo para un partido (deben sumar 1.0)."""
    home_team: str
    away_team: str

    prob_home: float   # P(victoria local)
    prob_draw: float   # P(empate)
    prob_away: float   # P(victoria visitante)

    lambda_home: float  # goles esperados del local
    lambda_away: float  # goles esperados del visitante

    @property
    def total(self) -> float:
        return self.prob_home + self.prob_draw + self.prob_away

    def as_dict(self) -> dict:
        return {
            "home": self.prob_home,
            "draw": self.prob_draw,
            "away": self.prob_away,
        }


def _poisson_pmf(k: int, lam: float) -> float:
    """
    Función de masa de probabilidad de Poisson: P(X = k | λ)
    P(k, λ) = (e^(-λ) × λ^k) / k!
    """
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


class PoissonModel:
    """
    Modelo Poisson Bivariado para predicción de partidos de fútbol.

    Funcionamiento:
    ─────────────────────────────────────────────────────────────
    1. Calcula los goles esperados para cada equipo:
       λ_home = attack_home × defence_away × avg_home_goals_liga × HOME_BOOST
       λ_away = attack_away × defence_home × avg_away_goals_liga

    2. Construye una matriz de probabilidades de marcadores
       usando distribuciones de Poisson independientes.

    3. Suma las probabilidades de todos los marcadores posibles
       para obtener: P(Home), P(Draw), P(Away)
    ─────────────────────────────────────────────────────────────
    Esta es la misma metodología que usan los mercados de apuestas
    para fijar sus cuotas iniciales ("líneas").
    """

    MAX_GOALS = 8   # máximo de goles a simular por equipo

    def predict(
        self,
        home_team: str,
        away_team: str,
        league_stats: LeagueStats,
    ) -> MatchProbabilities:
        """
        Calcula las probabilidades del partido usando el modelo Poisson.

        Args:
            home_team: Nombre del equipo local (debe coincidir con team_stats)
            away_team: Nombre del equipo visitante
            league_stats: Estadísticas calculadas de la liga

        Returns:
            MatchProbabilities con P(H), P(D), P(A) y lambdas
        """
        # Fuerzas de cada equipo (normalizadas respecto a la media de la liga)
        att_home = league_stats.get_attack_strength(home_team, is_home=True)
        def_away = league_stats.get_defence_strength(away_team, is_home=False)
        att_away = league_stats.get_attack_strength(away_team, is_home=False)
        def_home = league_stats.get_defence_strength(home_team, is_home=True)

        # Goles esperados: λ = attack_self × defence_opponent × avg_liga
        lambda_home = (
            att_home
            * def_away
            * league_stats.avg_home_goals
            * HOME_ATTACK_BOOST
        )
        lambda_away = (
            att_away
            * def_home
            * league_stats.avg_away_goals
        )

        # Clamp lambdas a rango razonable
        lambda_home = max(0.2, min(lambda_home, 6.0))
        lambda_away = max(0.2, min(lambda_away, 6.0))

        # Construir matriz de probabilidades de goles (home_goals × away_goals)
        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0

        for hg in range(self.MAX_GOALS + 1):
            p_hg = _poisson_pmf(hg, lambda_home)
            for ag in range(self.MAX_GOALS + 1):
                p_ag = _poisson_pmf(ag, lambda_away)
                p_score = p_hg * p_ag

                if hg > ag:
                    prob_home += p_score
                elif hg == ag:
                    prob_draw += p_score
                else:
                    prob_away += p_score

        # Normalizar (la suma debería ser ~1.0, pero hay truncamiento en MAX_GOALS)
        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total

        return MatchProbabilities(
            home_team=home_team,
            away_team=away_team,
            prob_home=round(prob_home, 4),
            prob_draw=round(prob_draw, 4),
            prob_away=round(prob_away, 4),
            lambda_home=round(lambda_home, 2),
            lambda_away=round(lambda_away, 2),
        )

    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        league_stats: LeagueStats,
        max_goals: int = 5,
    ) -> List[List[float]]:
        """
        Retorna la matriz de probabilidades de marcadores exactos.
        matrix[hg][ag] = P(local hg : visitante ag)

        Útil para analizar mercados de over/under y BTTS.
        """
        probs = self.predict(home_team, away_team, league_stats)
        matrix = []
        total = 0.0
        raw = []

        for hg in range(max_goals + 1):
            row = []
            raw_row = []
            for ag in range(max_goals + 1):
                p = _poisson_pmf(hg, probs.lambda_home) * _poisson_pmf(ag, probs.lambda_away)
                raw_row.append(p)
                total += p
            raw.append(raw_row)

        # Normalizar
        for row in raw:
            matrix.append([round(p / total, 4) for p in row])

        return matrix
