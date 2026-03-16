"""
tests/test_kelly.py — Tests del Kelly Criterion
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from kelly import KellyCriterion


class TestKellyCriterion:
    def setup_method(self):
        self.kelly = KellyCriterion(
            bankroll=1000.0,
            fraction=0.25,
            max_stake_pct=0.05,
            min_stake=5.0,
        )

    def test_full_kelly_formula(self):
        """
        Para prob=0.6, odds=2.0:
        b = 1.0, p=0.6, q=0.4
        f* = (1×0.6 - 0.4) / 1 = 0.20
        """
        rec = self.kelly.calculate(prob=0.6, odds=2.0)
        assert rec.kelly_full == pytest.approx(0.20, abs=0.001)

    def test_quarter_kelly_applied(self):
        """kelly_adjusted debe ser 25% del kelly_full."""
        rec = self.kelly.calculate(prob=0.6, odds=2.0)
        assert rec.kelly_adjusted == pytest.approx(rec.kelly_full * 0.25, abs=0.001)

    def test_negative_ev_returns_no_bet(self):
        """Cuando EV < 0, Kelly retorna 0 (no apostar)."""
        # prob=0.4, odds=2.0 → EV = 0.4×2 - 1 = -0.20
        rec = self.kelly.calculate(prob=0.4, odds=2.0)
        assert rec.kelly_full == 0.0
        assert not rec.is_value

    def test_stake_capped_at_max(self):
        """El stake no debe superar el límite de MAX_STAKE_PCT."""
        # Kelly muy alto: prob=0.9, odds=5.0
        rec = self.kelly.calculate(prob=0.9, odds=5.0)
        assert rec.stake_pct <= 0.05

    def test_stake_amount_matches_pct(self):
        """stake_amount ≈ bankroll × stake_pct (tolerancia por redondeo)."""
        rec = self.kelly.calculate(prob=0.55, odds=2.10)
        if rec.is_value:
            expected = self.kelly.bankroll * rec.stake_pct
            assert abs(rec.stake_amount - expected) < 0.10   # tolerancia de redondeo

    def test_break_even_edge(self):
        """Cuando prob × odds = 1, EV = 0 y no debería haber apuesta."""
        # prob = 0.5, odds = 2.0 → EV = 0
        rec = self.kelly.calculate(prob=0.5, odds=2.0)
        assert rec.kelly_full == pytest.approx(0.0, abs=0.001)

    def test_is_value_flag(self):
        """is_value debe ser True solo cuando kelly_full > 0."""
        rec_pos = self.kelly.calculate(prob=0.6, odds=2.0)
        rec_neg = self.kelly.calculate(prob=0.4, odds=2.0)
        assert rec_pos.is_value is True
        assert rec_neg.is_value is False
