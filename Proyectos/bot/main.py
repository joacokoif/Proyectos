"""
main.py — Orchestrador principal del Value Bet Scanner
=======================================================
Modos de operación:
  python main.py --demo      →  usa datos simulados (sin API key)
  python main.py             →  usa APIs reales
  python main.py --watch     →  polling continuo cada N minutos

Flujo:
  1. Obtener cuotas del mercado (odds_fetcher)
  2. Obtener stats históricas de equipos (data_fetcher)
  3. Calcular probabilidades Poisson (probability_model)
  4. Detectar value bets con EV > umbral (value_finder)
  5. Calcular stakes (Kelly criterion)
  6. Renderizar dashboard + guardar señales CSV
"""

import argparse
import sys
import io
import os
import time
from datetime import datetime
from typing import List

# ── Forzar UTF-8 en Windows (evita UnicodeEncodeError con emojis en cp1252) ──
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from rich.console import Console

from config import POLLING_INTERVAL_SECONDS, LEAGUES
from odds_fetcher import OddsFetcher, OddsLine, get_demo_odds
from data_fetcher import DataFetcher, LeagueStats, get_demo_league_stats
from probability_model import PoissonModel
from value_finder import ValueFinder, ValueBet
from kelly import KellyCriterion
from dashboard import render_dashboard, log_signals_csv

console = Console()


# ══════════════════════════════════════════════════════
# Mapeador de league names a códigos football-data
# ══════════════════════════════════════════════════════

LEAGUE_NAME_TO_CODE = {
    "Premier League":     "PL",
    "La Liga":            "PD",
    "Serie A":            "SA",
    "Bundesliga":         "BL1",
    "Ligue 1":            "FL1",
    "Champions League":   "CL",
    # Fallback para nombres parciales de The Odds API
    "EPL":                "PL",
    "Spain - La Liga":    "PD",
    "Italy - Serie A":    "SA",
    "Germany - Bundesliga": "BL1",
    "France - Ligue 1":   "FL1",
}


def _find_league_code(league_name: str) -> str:
    """Busca el código de liga dado el nombre de The Odds API."""
    for key, code in LEAGUE_NAME_TO_CODE.items():
        if key.lower() in league_name.lower():
            return code
    return "PL"   # Fallback a Premier League


def _find_team_name(team: str, league_stats: LeagueStats) -> str:
    """
    Intenta encontrar el nombre del equipo en las stats (fuzzy match básico).
    The Odds API usa nombres largos; football-data usa nombres cortos.
    """
    # Coincidencia exacta
    if team in league_stats.team_stats:
        return team
    # Coincidencia parcial (ej: "Manchester City" → "Man City")
    for known_team in league_stats.team_stats:
        if known_team.lower() in team.lower() or team.lower() in known_team.lower():
            return known_team
    return team   # Devuelve el original si no hay match (el modelo usará defaults)


# ══════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════

class ValueBetScanner:
    """Orquesta todo el pipeline de detección de value bets."""

    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.model = PoissonModel()
        self.value_finder = ValueFinder()
        self.kelly = KellyCriterion()

        if not demo_mode:
            self.odds_fetcher = OddsFetcher()
            self.data_fetcher = DataFetcher()

    def run(self) -> List[ValueBet]:
        """
        Ejecuta un ciclo completo del scanner.
        Returns: Lista de value bets detectadas.
        """
        if self.demo_mode:
            return self._run_demo()
        else:
            return self._run_live()

    def _run_demo(self) -> List[ValueBet]:
        """Pipeline con datos simulados (sin API keys)."""
        console.print("[yellow]🎮 Modo DEMO — usando datos simulados[/]")

        odds_lines = get_demo_odds()
        league_stats_map = get_demo_league_stats()

        return self._process(odds_lines, league_stats_map, mode="DEMO")

    def _run_live(self) -> List[ValueBet]:
        """Pipeline con datos reales de las APIs."""
        console.print("[green]🌐 Modo LIVE — conectando con APIs...[/]")

        # 1. Obtener cuotas del mercado
        console.print("[dim]  → Obteniendo cuotas (The Odds API)...[/]")
        odds_lines = self.odds_fetcher.get_events()
        if not odds_lines:
            console.print("[red]Sin datos de cuotas. ¿API key configurada?[/]")
            return []

        # 2. Obtener stats históricas de ligas involucradas
        console.print("[dim]  → Descargando stats históricas (football-data.org)...[/]")
        leagues_needed = set(_find_league_code(l.league) for l in odds_lines)
        league_stats_map = {}
        for code in leagues_needed:
            league_stats_map[code] = self.data_fetcher.fetch_league_stats(code)
            time.sleep(6)

        return self._process(odds_lines, league_stats_map, mode="LIVE")

    def _process(
        self,
        odds_lines: list,
        league_stats_map: dict,
        mode: str,
    ) -> List[ValueBet]:
        """Corre el modelo y el value finder sobre los datos disponibles."""

        console.print(f"[dim]  → Calculando probabilidades Poisson ({len(odds_lines)} partidos)...[/]")

        model_probs_map = {}
        for line in odds_lines:
            league_code = _find_league_code(line.league)
            stats = league_stats_map.get(league_code)
            if not stats:
                continue

            home = _find_team_name(line.home_team, stats)
            away = _find_team_name(line.away_team, stats)

            probs = self.model.predict(home, away, stats)
            model_probs_map[line.match_id] = probs

        # Detectar value bets
        console.print("[dim]  → Escaneando value bets...[/]")
        value_bets = self.value_finder.scan(odds_lines, model_probs_map)

        # Dashboard
        render_dashboard(value_bets, len(odds_lines), self.kelly, mode=mode)

        # Logging CSV
        log_signals_csv(value_bets, self.kelly)

        return value_bets


# ══════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="⚽ Value Bet Scanner — detecta apuestas con edge positivo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py --demo          Correr con datos simulados (sin API key)
  python main.py                 Correr una vez con APIs reales
  python main.py --watch         Polling continuo cada 5 minutos
  python main.py --watch --demo  Polling en modo demo (para testear)
        """
    )
    parser.add_argument("--demo",  action="store_true", help="Usar datos simulados sin API keys")
    parser.add_argument("--watch", action="store_true", help="Polling continuo (modo live)")
    args = parser.parse_args()

    scanner = ValueBetScanner(demo_mode=args.demo)

    if args.watch:
        console.rule("[bold bright_blue]⚽ VALUE BET SCANNER — Modo Watch[/]")
        console.print(f"[dim]Polling cada {POLLING_INTERVAL_SECONDS}s. Ctrl+C para detener.[/]")
        while True:
            try:
                bets = scanner.run()
                console.print(f"[dim]⏳ Próximo scan en {POLLING_INTERVAL_SECONDS}s...[/]")
                time.sleep(POLLING_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                console.print("\n[yellow]Scanner detenido.[/]")
                sys.exit(0)
    else:
        scanner.run()


if __name__ == "__main__":
    main()
