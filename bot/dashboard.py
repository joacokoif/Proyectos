"""
dashboard.py — Terminal Dashboard con Rich
==========================================
Muestra las value bets en tiempo real con una tabla premium
en el terminal, con colores y métricas clave por apuesta.
"""

import csv
import os
from datetime import datetime, timezone
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.style import Style
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich import box
from rich.rule import Rule

from value_finder import ValueBet
from kelly import KellyCriterion, StakeRecommendation
from config import BANKROLL, MIN_VALUE_EDGE, SAVE_SIGNALS_CSV, SIGNALS_CSV_PATH

console = Console()


# ──────────────────────────────────────────────────────
# Helpers de formato
# ──────────────────────────────────────────────────────

def _ev_color(value_pct: float) -> str:
    if value_pct >= 15:
        return "bold bright_green"
    elif value_pct >= 8:
        return "bold yellow"
    else:
        return "green"


def _prob_bar(prob: float, width: int = 10) -> Text:
    """Mini barra horizontal de probabilidad."""
    filled = round(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "bright_cyan" if prob > 0.5 else "bright_magenta" if prob > 0.35 else "white"
    t = Text()
    t.append(bar, style=color)
    t.append(f" {prob:.0%}", style="dim")
    return t


def _format_datetime(dt: datetime) -> str:
    local_tz = datetime.now().astimezone().tzinfo
    local_dt = dt.astimezone(local_tz)
    return local_dt.strftime("%a %d/%m  %H:%M")


def _hours_label(hours: float) -> str:
    if hours < 1:
        return f"[red bold]{int(hours*60)}m[/]"
    elif hours < 6:
        return f"[yellow]{hours:.1f}h[/]"
    else:
        return f"[bright_white]{hours:.0f}h[/]"


# ──────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────

def render_header(n_scanned: int, n_value: int, mode: str = "DEMO") -> Panel:
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    mode_color = "yellow" if mode == "DEMO" else "bright_green"

    title = Text()
    title.append("⚽  VALUE BET SCANNER  ", style="bold white")
    title.append(f"[{mode}]", style=f"bold {mode_color}")

    subtitle = Text(justify="center")
    subtitle.append(f"🕐 {now}   ", style="dim")
    subtitle.append(f"📊 Partidos analizados: {n_scanned}   ", style="bright_white")
    subtitle.append(f"💡 Value bets encontradas: {n_value}   ", style="bold bright_green" if n_value > 0 else "dim")
    subtitle.append(f"💰 Bankroll: ${BANKROLL:,.0f}   ", style="cyan")
    subtitle.append(f"🎯 Edge mínimo: {MIN_VALUE_EDGE:.0%}", style="magenta")

    body = Align.center(
        Text.assemble(("\n", ""), subtitle)
    )

    return Panel(
        body,
        title=title,
        border_style="bright_blue",
        padding=(0, 2),
    )


# ──────────────────────────────────────────────────────
# Tabla principal de value bets
# ──────────────────────────────────────────────────────

def render_value_table(
    bets: List[ValueBet],
    kelly: KellyCriterion,
) -> Table:

    table = Table(
        title=f"[bold bright_yellow]🔍 VALUE BETS DETECTADAS  ({len(bets)} señales)[/]",
        box=box.SIMPLE_HEAVY,
        border_style="bright_blue",
        header_style="bold bright_cyan",
        show_lines=True,
        expand=True,
    )

    # Columnas
    table.add_column("#",        style="dim",            width=3,  justify="right")
    table.add_column("Liga",     style="bright_white",   width=14)
    table.add_column("Partido",  style="white",          min_width=22)
    table.add_column("Apuesta",  style="bold",           width=7,  justify="center")
    table.add_column("Cuota",    style="bold yellow",    width=7,  justify="center")
    table.add_column("Libro",    style="dim cyan",       width=12)
    table.add_column("Modelo",   min_width=14)
    table.add_column("Mercado",  min_width=14)
    table.add_column("Value %",  style="bold",           width=9,  justify="right")
    table.add_column("EV",       width=8,                justify="right")
    table.add_column("Stake $",  style="bold white",     width=9,  justify="right")
    table.add_column("⏱ Faltan", width=8,                justify="center")

    for i, bet in enumerate(bets, 1):
        stake_rec = kelly.calculate(bet.model_prob, bet.odds)
        ev_style = _ev_color(bet.value_pct)
        stake_str = f"${stake_rec.stake_amount:.0f}" if stake_rec.is_value and stake_rec.stake_amount >= 1 else "[dim]–[/]"

        # Modelo vs mercado (barras visuales)
        model_bar  = _prob_bar(bet.model_prob)
        market_bar = _prob_bar(bet.implied_prob)

        table.add_row(
            str(i),
            bet.league[:14],
            f"{bet.home_team} vs {bet.away_team}",
            f"{bet.outcome_emoji} {bet.selection}",
            f"[bold yellow]{bet.odds:.2f}[/]",
            bet.bookmaker[:12],
            model_bar,
            market_bar,
            Text(f"+{bet.value_pct:.1f}%", style=ev_style),
            Text(f"{bet.expected_value:+.3f}", style=ev_style),
            stake_str,
            _hours_label(bet.hours_to_match),
        )

    return table


# ──────────────────────────────────────────────────────
# Panel de métricas de bankroll
# ──────────────────────────────────────────────────────

def render_bankroll_panel(bets: List[ValueBet], kelly: KellyCriterion) -> Panel:
    if not bets:
        return Panel("[dim]Sin señales activas[/]", title="💰 Bankroll", border_style="dim")

    total_stake = sum(
        kelly.calculate(b.model_prob, b.odds).stake_amount for b in bets
    )
    avg_ev = sum(b.value_pct for b in bets) / len(bets)
    best = bets[0]

    text = Text()
    text.append(f"  Stakes totales sugeridos: ", style="white")
    text.append(f"${total_stake:.2f} / ${BANKROLL:.0f}\n", style="bold cyan")
    text.append(f"  Exposición: ", style="white")
    text.append(f"{total_stake/BANKROLL:.1%} del bankroll\n", style="yellow")
    text.append(f"  EV promedio de señales: ", style="white")
    text.append(f"+{avg_ev:.1f}%\n", style="bold bright_green")
    text.append(f"  🏆 Mejor apuesta: ", style="white")
    text.append(f"{best.home_team} vs {best.away_team} {best.outcome_emoji} @ {best.odds:.2f}", style="bold bright_yellow")

    return Panel(text, title="[bold]💰 Resumen de posiciones[/]", border_style="cyan")


# ──────────────────────────────────────────────────────
# Panel de ayuda / leyenda
# ──────────────────────────────────────────────────────

def render_legend() -> Panel:
    text = Text()
    text.append("  EV = (Prob.Modelo × Cuota) − 1  ", style="dim")
    text.append("│  ", style="bright_black")
    text.append("🔥 ALTO ≥15%  ", style="bright_green")
    text.append("⚡ MEDIO ≥8%  ", style="yellow")
    text.append("✅ BAJO ≥5%  ", style="green")
    text.append("│  ", style="bright_black")
    text.append("Stake = Quarter-Kelly × Bankroll", style="dim")
    return Panel(text, border_style="dim", padding=(0, 1))


# ──────────────────────────────────────────────────────
# Renderizado completo
# ──────────────────────────────────────────────────────

def render_dashboard(
    bets: List[ValueBet],
    n_scanned: int,
    kelly: KellyCriterion,
    mode: str = "DEMO",
) -> None:
    """Renderiza el dashboard completo en la consola."""
    console.clear()
    console.print(render_header(n_scanned, len(bets), mode))
    console.print()

    if not bets:
        console.print(
            Panel(
                Align.center(
                    Text("\n🔎  No se encontraron value bets con el umbral actual.\n"
                         "    Probá bajando MIN_VALUE_EDGE en config.py o esperá más partidos.\n",
                         style="dim")
                ),
                border_style="dim",
            )
        )
    else:
        console.print(render_value_table(bets, kelly))
        console.print()
        console.print(render_bankroll_panel(bets, kelly))

    console.print()
    console.print(render_legend())


# ──────────────────────────────────────────────────────
# CSV Logger
# ──────────────────────────────────────────────────────

def log_signals_csv(bets: List[ValueBet], kelly: KellyCriterion) -> None:
    """Guarda las señales en CSV para backtesting posterior."""
    if not SAVE_SIGNALS_CSV or not bets:
        return

    file_exists = os.path.isfile(SIGNALS_CSV_PATH)
    with open(SIGNALS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "match_id", "league",
                "home_team", "away_team", "selection",
                "odds", "bookmaker", "model_prob", "implied_prob",
                "expected_value", "value_pct",
                "lambda_home", "lambda_away",
                "stake_amount", "stake_pct",
                "commence_time",
            ])
        now = datetime.now().isoformat()
        for bet in bets:
            stake = kelly.calculate(bet.model_prob, bet.odds)
            writer.writerow([
                now, bet.match_id, bet.league,
                bet.home_team, bet.away_team, bet.selection,
                bet.odds, bet.bookmaker,
                bet.model_prob, bet.implied_prob,
                bet.expected_value, bet.value_pct,
                bet.lambda_home, bet.lambda_away,
                stake.stake_amount, stake.stake_pct,
                bet.commence_time.isoformat(),
            ])
    console.print(f"[dim]💾 Señales guardadas en {SIGNALS_CSV_PATH}[/]")
