"""
odds_fetcher.py — Obtiene cuotas en vivo desde The Odds API
===========================================================
Implementa line shopping: toma la mejor cuota disponible
entre todos los bookmakers retornados por la API.

Documentación API: https://the-odds-api.com/liveapi/guides/v4/
"""

import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timezone

from config import (
    THE_ODDS_API_KEY, ODDS_SPORT_KEY, ODDS_REGIONS,
    ODDS_MARKETS, MIN_ODDS, MAX_ODDS
)


@dataclass
class OddsLine:
    """Representa las cuotas de un partido provenientes del mercado."""
    match_id: str
    home_team: str
    away_team: str
    sport: str
    league: str
    commence_time: datetime

    # Mejores cuotas disponibles (line shopping)
    best_odds_home: float = 0.0
    best_odds_draw: float = 0.0
    best_odds_away: float = 0.0

    # Bookmaker que ofrece cada mejor cuota
    bk_home: str = ""
    bk_draw: str = ""
    bk_away: str = ""

    # Cuotas promedio de mercado
    avg_odds_home: float = 0.0
    avg_odds_draw: float = 0.0
    avg_odds_away: float = 0.0

    # Bookmakers incluidos
    bookmakers: List[str] = field(default_factory=list)

    @property
    def overround(self) -> float:
        """Margen de la casa (vig). Valores < 1.10 indican mercado eficiente."""
        if self.avg_odds_home and self.avg_odds_draw and self.avg_odds_away:
            return (1/self.avg_odds_home + 1/self.avg_odds_draw + 1/self.avg_odds_away)
        return 0.0

    @property
    def implied_prob_home(self) -> float:
        return (1 / self.best_odds_home) if self.best_odds_home else 0.0

    @property
    def implied_prob_draw(self) -> float:
        return (1 / self.best_odds_draw) if self.best_odds_draw else 0.0

    @property
    def implied_prob_away(self) -> float:
        return (1 / self.best_odds_away) if self.best_odds_away else 0.0


class OddsFetcher:
    """
    Conecta con The Odds API y retorna cuotas del mercado.
    Usa line shopping para obtener el mejor precio disponible.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str = THE_ODDS_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.requests_remaining = None
        self.requests_used = None

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        params["apiKey"] = self.api_key
        try:
            resp = self.session.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=15)
            self.requests_remaining = resp.headers.get("x-requests-remaining", "?")
            self.requests_used = resp.headers.get("x-requests-used", "?")
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[OddsFetcher] Error: {e}")
            return None

    def get_sports(self) -> List[dict]:
        """Retorna todos los deportes disponibles en la API."""
        return self._get("/sports", {"all": "true"}) or []

    def get_events(self, sport_key: str = ODDS_SPORT_KEY) -> List[OddsLine]:
        """
        Obtiene los próximos eventos con cuotas en vivo.
        Retorna lista de OddsLine con las mejores cuotas encontradas.
        """
        data = self._get(
            f"/sports/{sport_key}/odds",
            {
                "regions": ODDS_REGIONS,
                "markets": ODDS_MARKETS,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            }
        )
        if not data:
            return []

        lines: List[OddsLine] = []
        for event in data:
            line = self._parse_event(event, sport_key)
            if line:
                lines.append(line)

        print(f"[OddsFetcher] {len(lines)} partidos | "
              f"Requests usados: {self.requests_used} | Restantes: {self.requests_remaining}")
        return lines

    def _parse_event(self, event: dict, sport_key: str) -> Optional[OddsLine]:
        """Parsea un evento de la API y hace line shopping."""
        try:
            commence_time = datetime.fromisoformat(
                event["commence_time"].replace("Z", "+00:00")
            )

            line = OddsLine(
                match_id=event["id"],
                home_team=event["home_team"],
                away_team=event["away_team"],
                sport=sport_key,
                league=event.get("sport_title", "Unknown League"),
                commence_time=commence_time,
            )

            # Arrays para calcular promedios
            home_prices, draw_prices, away_prices = [], [], []
            bk_names = []

            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker["key"]
                bk_names.append(bk_name)

                for market in bookmaker.get("markets", []):
                    if market["key"] != "h2h":
                        continue

                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    h = outcomes.get(event["home_team"], 0.0)
                    d = outcomes.get("Draw", 0.0)
                    a = outcomes.get(event["away_team"], 0.0)

                    if not all([h, d, a]):
                        continue

                    home_prices.append((h, bk_name))
                    draw_prices.append((d, bk_name))
                    away_prices.append((a, bk_name))

            if not home_prices:
                return None

            # Line shopping: mejor cuota
            best_h = max(home_prices, key=lambda x: x[0])
            best_d = max(draw_prices, key=lambda x: x[0])
            best_a = max(away_prices, key=lambda x: x[0])

            line.best_odds_home = best_h[0]
            line.best_odds_draw = best_d[0]
            line.best_odds_away = best_a[0]
            line.bk_home = best_h[1]
            line.bk_draw = best_d[1]
            line.bk_away = best_a[1]

            # Promedios
            line.avg_odds_home = sum(p for p, _ in home_prices) / len(home_prices)
            line.avg_odds_draw = sum(p for p, _ in draw_prices) / len(draw_prices)
            line.avg_odds_away = sum(p for p, _ in away_prices) / len(away_prices)
            line.bookmakers = bk_names

            # Filtro de cuotas inválidas
            if not (MIN_ODDS <= line.best_odds_home <= MAX_ODDS):
                return None

            return line

        except (KeyError, ValueError) as e:
            return None


# ──────────────────────────────────────────────────────
# MODO DEMO: datos simulados sin API key
# ──────────────────────────────────────────────────────
DEMO_ODDS: List[OddsLine] = [
    OddsLine(
        match_id="demo_001",
        home_team="Manchester City",
        away_team="Arsenal",
        sport="soccer_epl",
        league="Premier League",
        commence_time=datetime(2026, 3, 15, 15, 0, tzinfo=timezone.utc),
        best_odds_home=1.85, best_odds_draw=3.60, best_odds_away=4.20,
        avg_odds_home=1.82, avg_odds_draw=3.55, avg_odds_away=4.10,
        bk_home="Pinnacle", bk_draw="Bet365", bk_away="Betfair",
        bookmakers=["Pinnacle", "Bet365", "Betfair", "William Hill"],
    ),
    OddsLine(
        match_id="demo_002",
        home_team="Real Madrid",
        away_team="Barcelona",
        sport="soccer_spain_la_liga",
        league="La Liga",
        commence_time=datetime(2026, 3, 15, 20, 0, tzinfo=timezone.utc),
        best_odds_home=2.10, best_odds_draw=3.30, best_odds_away=3.40,
        avg_odds_home=2.05, avg_odds_draw=3.25, avg_odds_away=3.30,
        bk_home="Pinnacle", bk_draw="Unibet", bk_away="Bet365",
        bookmakers=["Pinnacle", "Bet365", "Unibet", "Betano"],
    ),
    OddsLine(
        match_id="demo_003",
        home_team="Napoli",
        away_team="Juventus",
        sport="soccer_italy_serie_a",
        league="Serie A",
        commence_time=datetime(2026, 3, 16, 18, 0, tzinfo=timezone.utc),
        best_odds_home=2.40, best_odds_draw=3.20, best_odds_away=2.95,
        avg_odds_home=2.35, avg_odds_draw=3.15, avg_odds_away=2.90,
        bk_home="Pinnacle", bk_draw="Bet365", bk_away="Pinnacle",
        bookmakers=["Pinnacle", "Bet365", "Betfair"],
    ),
    OddsLine(
        match_id="demo_004",
        home_team="Bayern Munich",
        away_team="Borussia Dortmund",
        sport="soccer_germany_bundesliga",
        league="Bundesliga",
        commence_time=datetime(2026, 3, 16, 15, 30, tzinfo=timezone.utc),
        best_odds_home=1.65, best_odds_draw=4.00, best_odds_away=5.10,
        avg_odds_home=1.62, avg_odds_draw=3.95, avg_odds_away=5.00,
        bk_home="Pinnacle", bk_draw="Bet365", bk_away="Betfair",
        bookmakers=["Pinnacle", "Bet365", "Betfair", "Tipico"],
    ),
    OddsLine(
        match_id="demo_005",
        home_team="PSG",
        away_team="Marseille",
        sport="soccer_france_ligue_one",
        league="Ligue 1",
        commence_time=datetime(2026, 3, 17, 21, 0, tzinfo=timezone.utc),
        best_odds_home=1.55, best_odds_draw=4.10, best_odds_away=5.80,
        avg_odds_home=1.50, avg_odds_draw=4.00, avg_odds_away=5.60,
        bk_home="Pinnacle", bk_draw="Unibet", bk_away="Bet365",
        bookmakers=["Pinnacle", "Bet365", "Unibet"],
    ),
]


def get_demo_odds() -> List[OddsLine]:
    """Retorna datos de cuotas simulados para modo demo."""
    return DEMO_ODDS
