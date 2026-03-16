"""
kelly.py — Kelly Criterion para sizing de stakes
=================================================
Determina el tamaño óptimo de cada apuesta en función del edge
y la probabilidad estimada, maximizando el crecimiento logarítmico
del bankroll a largo plazo.

Fórmula Kelly completa:
    f* = (b×p - q) / b
    donde:
        b = cuota_decimal - 1  (ganancia neta por unidad)
        p = probabilidad del modelo
        q = 1 - p              (probabilidad de pérdida)

Fracción de Kelly:
    Usamos Quarter Kelly (f* × 0.25) para reducir volatilidad.
    Apostadores profesionales usan entre 10%-33% del Kelly completo.
"""

from dataclasses import dataclass
from typing import Optional
from config import BANKROLL, KELLY_FRACTION, MAX_STAKE_PCT, MIN_STAKE


@dataclass
class StakeRecommendation:
    """Resultado del cálculo de stake para una apuesta."""
    kelly_full: float        # Kelly fraction completa (%)
    kelly_adjusted: float    # Fracción de Kelly aplicada (%)
    stake_pct: float         # % del bankroll recomendado
    stake_amount: float      # Monto en moneda
    bankroll: float          # Bankroll de referencia
    is_value: bool           # True si kelly_full > 0
    capped: bool             # True si el stake fue limitado por MAX_STAKE_PCT

    @property
    def stake_units(self) -> float:
        """Stake expresado en 'unidades' (1 unidad = 1% del bankroll)."""
        return self.stake_pct * 100

    def __str__(self) -> str:
        cap_str = " (🔒 capped)" if self.capped else ""
        return (
            f"Kelly: {self.kelly_full:.1%} | "
            f"Ajustado: {self.kelly_adjusted:.1%} | "
            f"Stake: ${self.stake_amount:.2f}{cap_str}"
        )


class KellyCriterion:
    """
    Calcula stakes óptimos usando el Criterio de Kelly.

    Por qué Kelly funciona:
    ─────────────────────────────────────────────────────────────
    - Apuesta demasiado poco → subóptimo, crecimiento lento
    - Apuesta demasiado → riesgo de ruina
    - Kelly → punto matemáticamente óptimo para max crecimiento

    Por qué usar Fracción de Kelly:
    - El Kelly completo asume que el modelo es 100% preciso (no lo es)
    - Quarter Kelly reduce la varianza enormemente
    - Protege el bankroll de errores del modelo o sorpresas del mercado
    ─────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        bankroll: float = BANKROLL,
        fraction: float = KELLY_FRACTION,
        max_stake_pct: float = MAX_STAKE_PCT,
        min_stake: float = MIN_STAKE,
    ):
        self.bankroll = bankroll
        self.fraction = fraction
        self.max_stake_pct = max_stake_pct
        self.min_stake = min_stake

    def calculate(self, prob: float, odds: float) -> StakeRecommendation:
        """
        Calcula el stake óptimo para una apuesta con value.

        Args:
            prob: Probabilidad del modelo (0-1)
            odds: Cuota decimal (ej: 2.10)

        Returns:
            StakeRecommendation con el stake en monto y porcentaje
        """
        b = odds - 1.0   # ganancia neta por unidad apostada
        q = 1.0 - prob   # probabilidad complementaria

        # Kelly completo
        if b <= 0:
            kelly_full = 0.0
        else:
            kelly_full = (b * prob - q) / b

        # Sin value → no apostar
        if kelly_full <= 0:
            return StakeRecommendation(
                kelly_full=0.0,
                kelly_adjusted=0.0,
                stake_pct=0.0,
                stake_amount=0.0,
                bankroll=self.bankroll,
                is_value=False,
                capped=False,
            )

        # Fracción de Kelly (reducción de volatilidad)
        kelly_adjusted = kelly_full * self.fraction

        # Cap de riesgo máximo por apuesta
        capped = kelly_adjusted > self.max_stake_pct
        stake_pct = min(kelly_adjusted, self.max_stake_pct)

        # Monto en moneda
        stake_amount = self.bankroll * stake_pct

        # Stake mínimo (no vale la pena apostar menos)
        if stake_amount < self.min_stake:
            return StakeRecommendation(
                kelly_full=kelly_full,
                kelly_adjusted=kelly_adjusted,
                stake_pct=stake_pct,
                stake_amount=stake_amount,
                bankroll=self.bankroll,
                is_value=True,
                capped=capped,
            )

        return StakeRecommendation(
            kelly_full=round(kelly_full, 4),
            kelly_adjusted=round(kelly_adjusted, 4),
            stake_pct=round(stake_pct, 4),
            stake_amount=round(stake_amount, 2),
            bankroll=self.bankroll,
            is_value=True,
            capped=capped,
        )

    def ruin_probability(self, edge: float, kelly_frac: float, n_bets: int = 100) -> float:
        """
        Estimación simplificada de la probabilidad de ruina en N apuestas.
        Mayor edge y menor fracción → menor probabilidad de ruina.
        """
        # Aproximación: P(ruina) ≈ exp(-2 × edge × n_bets × kelly_frac)
        import math
        return round(math.exp(-2 * edge * n_bets * kelly_frac), 4)
