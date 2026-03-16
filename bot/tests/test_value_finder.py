"""
tests/test_value_finder.py — Tests del detector de value bets
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
import pytest
from odds_fetcher import OddsLine
from probability_model import MatchProbabilities
from value_finder import ValueFinder, ValueBet


def make_line(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> OddsLine:
    return OddsLine(
        match_id="test_001",
        home_team="Team A",
        away_team="Team B",
        sport="soccer_test",
        league="Test League",
        commence_time=datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc),
        best_odds_home=home_odds,
        best_odds_draw=draw_odds,
        best_odds_away=away_odds,
        bk_home="Pinnacle",
        bk_draw="Bet365",
        bk_away="Betfair",
        avg_odds_home=home_odds,
        avg_odds_draw=draw_odds,
        avg_odds_away=away_odds,
        bookmakers=["Pinnacle", "Bet365"],
    )


def make_probs(ph: float, pd: float, pa: float) -> MatchProbabilities:
    return MatchProbabilities(
        home_team="Team A",
        away_team="Team B",
        prob_home=ph,
        prob_draw=pd,
        prob_away=pa,
        lambda_home=1.5,
        lambda_away=1.1,
    )


class TestValueFinder:
    def setup_method(self):
        self.finder = ValueFinder()

    def test_clear_value_bet_detected(self):
        """Si el modelo estima 60% y la cuota da 45% implícita → value."""
        # Cuota 2.00 implica 50%, modelo dice 60% → EV = 0.60×2.00 - 1 = +0.20
        line  = make_line(2.00, 3.50, 3.80)
        probs = make_probs(0.60, 0.25, 0.15)
        bets  = self.finder.find_value_bets(line, probs, min_edge=0.05)
        assert len(bets) >= 1
        home_bets = [b for b in bets if b.selection == "Home"]
        assert home_bets, "Debería detectar value en Home"
        assert home_bets[0].expected_value == pytest.approx(0.20, abs=0.001)

    def test_no_value_when_model_underestimates(self):
        """Si la cuota ya refleja o supera la probabilidad del modelo → sin value."""
        # Cuota 1.50 implica 66.7%, modelo dice 60% → EV = -0.10 (negativo)
        line  = make_line(1.50, 3.80, 5.00)
        probs = make_probs(0.60, 0.25, 0.15)
        bets  = self.finder.find_value_bets(line, probs, min_edge=0.05)
        home_bets = [b for b in bets if b.selection == "Home"]
        assert not home_bets

    def test_value_pct_calculation(self):
        """value_pct debe ser (EV × 100)."""
        line  = make_line(3.00, 3.50, 2.50)
        probs = make_probs(0.40, 0.30, 0.30)
        bets  = self.finder.find_value_bets(line, probs, min_edge=0.0)
        for bet in bets:
            expected = (bet.model_prob * bet.odds - 1) * 100
            assert abs(bet.value_pct - expected) < 0.01

    def test_ev_formula(self):
        """EV = model_prob × odds - 1."""
        line  = make_line(2.50, 3.20, 2.80)
        probs = make_probs(0.45, 0.30, 0.25)
        bets  = self.finder.find_value_bets(line, probs, min_edge=0.0)
        for bet in bets:
            ev_manual = bet.model_prob * bet.odds - 1
            assert abs(bet.expected_value - ev_manual) < 0.001

    def test_scan_returns_sorted_by_ev(self):
        """El scan debe retornar bets ordenados de mayor a menor EV."""
        from odds_fetcher import get_demo_odds
        from data_fetcher import get_demo_league_stats
        from probability_model import PoissonModel
        from main import _find_league_code, _find_team_name

        odds   = get_demo_odds()
        leagues = get_demo_league_stats()
        model  = PoissonModel()

        probs_map = {}
        for line in odds:
            code  = _find_league_code(line.league)
            stats = leagues.get(code)
            if not stats:
                continue
            h = _find_team_name(line.home_team, stats)
            a = _find_team_name(line.away_team, stats)
            probs_map[line.match_id] = model.predict(h, a, stats)

        bets = self.finder.scan(odds, probs_map, min_edge=0.0)
        evs  = [b.expected_value for b in bets]
        assert evs == sorted(evs, reverse=True), "Los bets deben estar ordenados por EV desc"

    def test_min_prob_filter(self):
        """No se deben incluir apuestas con prob_modelo < MIN_MODEL_PROB."""
        line  = make_line(20.0, 3.50, 1.40)
        # prob_away extremadamente baja → filtrar aunque la cuota sea alta
        probs = make_probs(0.05, 0.30, 0.65)
        bets  = self.finder.find_value_bets(line, probs, min_edge=0.0)
        home_bets = [b for b in bets if b.selection == "Home"]
        # prob_home = 0.05 < MIN_MODEL_PROB=0.10 → no debe aparecer
        assert not home_bets
