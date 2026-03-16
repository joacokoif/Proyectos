"""
tests/test_poisson_model.py — Tests del modelo Poisson
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from data_fetcher import LeagueStats, TeamStats, get_demo_league_stats
from probability_model import PoissonModel, _poisson_pmf


class TestPoissonPMF:
    def test_pmf_sums_to_one(self):
        """La suma de todos los valores Poisson de 0 a infinito debe ser 1."""
        lam = 1.5
        total = sum(_poisson_pmf(k, lam) for k in range(50))
        assert abs(total - 1.0) < 0.001

    def test_pmf_lambda_zero(self):
        assert _poisson_pmf(0, 0.0) == 1.0
        assert _poisson_pmf(1, 0.0) == 0.0

    def test_pmf_peak(self):
        """Para λ=2, el valor más probable es k=1 o k=2."""
        probs = [_poisson_pmf(k, 2.0) for k in range(10)]
        most_likely = probs.index(max(probs))
        assert most_likely in (1, 2)


class TestPoissonModel:
    def setup_method(self):
        self.model = PoissonModel()
        self.leagues = get_demo_league_stats()

    def test_probabilities_sum_to_one(self):
        """P(H) + P(D) + P(A) debe ser exactamente 1.0."""
        stats = self.leagues["PL"]
        probs = self.model.predict("Man City", "Arsenal", stats)
        total = probs.prob_home + probs.prob_draw + probs.prob_away
        assert abs(total - 1.0) < 0.001, f"Total = {total:.6f}"

    def test_probabilities_in_range(self):
        """Cada probabilidad debe estar entre 0 y 1."""
        for code, stats in self.leagues.items():
            teams = list(stats.team_stats.keys())
            if len(teams) >= 2:
                probs = self.model.predict(teams[0], teams[1], stats)
                assert 0 <= probs.prob_home <= 1
                assert 0 <= probs.prob_draw <= 1
                assert 0 <= probs.prob_away <= 1

    def test_strong_home_favorite(self):
        """Un equipo local fuerte debe tener prob_home > prob_away."""
        stats = self.leagues["BL1"]
        probs = self.model.predict("Bayern Munich", "Borussia Dortmund", stats)
        assert probs.prob_home > probs.prob_away, (
            f"Bayern en casa debería tener prob_home > prob_away, "
            f"got {probs.prob_home:.3f} vs {probs.prob_away:.3f}"
        )

    def test_lambdas_positive(self):
        """Los goles esperados deben ser positivos."""
        stats = self.leagues["PD"]
        probs = self.model.predict("Real Madrid", "Barcelona", stats)
        assert probs.lambda_home > 0
        assert probs.lambda_away > 0

    def test_unknown_team_uses_defaults(self):
        """Un equipo desconocido no debe romper el modelo."""
        stats = self.leagues["PL"]
        probs = self.model.predict("Equipo Desconocido A", "Equipo Desconocido B", stats)
        total = probs.prob_home + probs.prob_draw + probs.prob_away
        assert abs(total - 1.0) < 0.001

    def test_score_matrix_sums(self):
        """La matriz de marcadores debe sumar ~1.0."""
        stats = self.leagues["PL"]
        matrix = self.model.predict_score_matrix("Man City", "Arsenal", stats)
        total = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[0])))
        assert abs(total - 1.0) < 0.01
