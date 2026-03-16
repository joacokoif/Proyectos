"""
value_finder.py — Detector de Value Bets
=========================================
Compara las probabilidades del modelo Poisson con las cuotas
del mercado para encontrar apuestas con Expected Value positivo.

Fórmula:
    EV = (prob_modelo × cuota_decimal) - 1
    Value% = EV × 100

Una apuesta tiene "value" cuando EV > MIN_VALUE_EDGE,
es decir, cuando el mercado está subpreciando la probabilidad real.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from odds_fetcher import OddsLine
from probability_model import MatchProbabilities
from config import MIN_VALUE_EDGE, MIN_MODEL_PROB, MIN_ODDS, MAX_ODDS


@dataclass
class ValueBet:
    """Una apuesta identificada con value positivo."""

    # Identificación
    match_id: str
    home_team: str
    away_team: str
    league: str
    commence_time: datetime

    # Selección de la apuesta
    selection: str       # "Home" | "Draw" | "Away"
    outcome_emoji: str   # 🏠 | ➕ | ✈️

    # Probabilidades
    model_prob: float    # P del modelo Poisson
    implied_prob: float  # P implícita en la cuota (1/cuota)

    # Cuota
    odds: float          # Cuota decimal (mejor precio disponible)
    bookmaker: str       # Bookmaker que ofrece esa cuota

    # Métricas de value
    expected_value: float  # EV = (prob × odds) - 1
    value_pct: float       # EV × 100

    # Contexto del modelo
    lambda_home: float   # Goles esperados local
    lambda_away: float   # Goles esperados visitante

    @property
    def rating(self) -> str:
        """Rating cualitativo del value encontrado."""
        if self.value_pct >= 15:
            return "🔥 ALTO"
        elif self.value_pct >= 8:
            return "⚡ MEDIO"
        else:
            return "✅ BAJO"

    @property
    def hours_to_match(self) -> float:
        """Horas que faltan hasta el partido."""
        now = datetime.now(tz=self.commence_time.tzinfo)
        delta = self.commence_time - now
        return max(0.0, delta.total_seconds() / 3600)

    def __str__(self) -> str:
        return (
            f"{self.outcome_emoji} {self.home_team} vs {self.away_team} | "
            f"{self.selection} @ {self.odds:.2f} | "
            f"Modelo: {self.model_prob:.1%} | "
            f"Value: +{self.value_pct:.1f}%"
        )


class ValueFinder:
    """
    Compara probabilidades del modelo con cuotas del mercado
    y retorna solo las apuestas con value positivo.

    Analogía con trading:
    - model_prob > implied_prob → el mercado subestima la probabilidad
    - Es como encontrar un activo subvalorado respecto a su valor intrínseco
    """

    OUTCOME_META = {
        "Home": ("🏠", lambda l: (l.best_odds_home, l.bk_home)),
        "Draw": ("➕", lambda l: (l.best_odds_draw, l.bk_draw)),
        "Away": ("✈️",  lambda l: (l.best_odds_away, l.bk_away)),
    }

    def find_value_bets(
        self,
        odds_line: OddsLine,
        model_probs: MatchProbabilities,
        min_edge: float = MIN_VALUE_EDGE,
    ) -> List[ValueBet]:
        """
        Analiza un partido y retorna las apuestas con value positivo.

        Args:
            odds_line: Cuotas del mercado para el partido
            model_probs: Probabilidades calculadas por el modelo Poisson
            min_edge: Umbral mínimo de EV (default: 5%)

        Returns:
            Lista de ValueBet (puede ser vacía si no hay value)
        """
        candidates = [
            ("Home", model_probs.prob_home),
            ("Draw", model_probs.prob_draw),
            ("Away", model_probs.prob_away),
        ]

        value_bets: List[ValueBet] = []

        for selection, model_prob in candidates:
            emoji, price_fn = self.OUTCOME_META[selection]
            odds, bookmaker = price_fn(odds_line)

            # Validaciones de integridad
            if not odds or odds < MIN_ODDS or odds > MAX_ODDS:
                continue
            if model_prob < MIN_MODEL_PROB:
                continue

            # Cálculo del Expected Value
            ev = (model_prob * odds) - 1.0
            value_pct = ev * 100.0

            # Solo retornar si tiene edge positivo suficiente
            if ev >= min_edge:
                implied_prob = 1.0 / odds
                vb = ValueBet(
                    match_id=odds_line.match_id,
                    home_team=odds_line.home_team,
                    away_team=odds_line.away_team,
                    league=odds_line.league,
                    commence_time=odds_line.commence_time,
                    selection=selection,
                    outcome_emoji=emoji,
                    model_prob=round(model_prob, 4),
                    implied_prob=round(implied_prob, 4),
                    odds=odds,
                    bookmaker=bookmaker,
                    expected_value=round(ev, 4),
                    value_pct=round(value_pct, 2),
                    lambda_home=model_probs.lambda_home,
                    lambda_away=model_probs.lambda_away,
                )
                value_bets.append(vb)

        return value_bets

    def scan(
        self,
        odds_lines: List[OddsLine],
        model_probs_map: dict,  # {match_id: MatchProbabilities}
        min_edge: float = MIN_VALUE_EDGE,
    ) -> List[ValueBet]:
        """
        Escanea todos los partidos disponibles y retorna value bets ordenados.
        Equivalente a un screener de acciones — filtra el mercado completo.

        Returns:
            Lista de ValueBet ordenada de mayor a menor EV
        """
        all_bets: List[ValueBet] = []

        for line in odds_lines:
            probs = model_probs_map.get(line.match_id)
            if not probs:
                continue
            bets = self.find_value_bets(line, probs, min_edge)
            all_bets.extend(bets)

        # Ordenar por expected value descendente (como un screener)
        all_bets.sort(key=lambda b: b.expected_value, reverse=True)
        return all_bets
