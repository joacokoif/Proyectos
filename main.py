"""
main.py — David vs Goliat
==========================
Dale Play a este archivo y el script hace todo:
  1. Descarga la tabla de posiciones de cada liga configurada
  2. Busca los próximos partidos de esas ligas
  3. Detecta enfrentamientos Top 5 vs Últimos 3
  4. Muestra el reporte en pantalla

Configuración rápida:
  → Registrate gratis en https://www.football-data.org/client/register
  → Pegá tu API key en config.py (variable FOOTBALL_DATA_API_KEY)
  → Para agregar ligas, editá la lista LIGAS_ACTIVAS en config.py
"""

import sys
import io

# Forzar UTF-8 en Windows para evitar errores con emojis
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text
from rich         import box

from config   import (
    FOOTBALL_DATA_API_KEY, LIGAS_ACTIVAS, LIGAS_NOMBRES,
    TOP_N, BOTTOM_N, DIAS_ADELANTE
)
from fetcher  import get_standings, get_upcoming_fixtures
from analyzer import find_david_vs_goliat

console = Console(highlight=False, force_terminal=True)


# ═══════════════════════════════════════════════════════
# HELPERS VISUALES
# ═══════════════════════════════════════════════════════

def nombre_liga(code: str) -> str:
    return LIGAS_NOMBRES.get(code, code)


def render_match_card(match, top_n: int, bottom_n: int):
    """Imprime una tarjeta por partido detectado."""
    david_home = match.david_is_home

    home_label = f"#{match.home_pos} {match.home}"
    away_label = f"#{match.away_pos} {match.away}"

    # Colores: Goliat = amarillo, David = magenta
    if david_home:
        home_rich  = f"[bold magenta]{home_label}[/]  🗡️  David"
        away_rich  = f"[bold yellow]{away_label}[/]  🏰 Goliat"
    else:
        home_rich  = f"[bold yellow]{home_label}[/]  🏰 Goliat"
        away_rich  = f"[bold magenta]{away_label}[/]  🗡️  David"

    jornada = f"  · Jornada {match.matchday}" if match.matchday else ""
    title   = f"[bold white]{match.date}[/][dim]{jornada}[/]"

    content = Text.assemble(
        (f"{match.home}", "bold yellow" if not david_home else "bold magenta"),
        (" vs ", "dim white"),
        (f"{match.away}", "bold yellow" if david_home else "bold magenta"),
        ("\n", ""),
        (f"{'Local (David)' if david_home else 'Local (Goliat)'}  →  ", "dim"),
        (f"{'Visitante (Goliat)' if david_home else 'Visitante (David)'}", "dim"),
        ("\n\n", ""),
        (f"🏰 Goliat: ", "bold yellow"),
        (f"#{match.goliat_pos} {match.goliat}  ", "white"),
        (f"(Top {top_n})\n", "dim"),
        (f"🗡️  David:  ", "bold magenta"),
        (f"#{match.david_pos} {match.david}  ", "white"),
        (f"(Últimos {bottom_n})", "dim"),
    )

    console.print(
        Panel(
            content,
            title=title,
            border_style="cyan",
            padding=(0, 2),
        )
    )


def render_standings_mini(standings: list[dict], top_n: int, bottom_n: int, liga_code: str):
    """Tabla chica de posiciones con top y bottom resaltados."""
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        title=f"[bold]{nombre_liga(liga_code)}[/]",
        title_style="bold cyan",
        min_width=40,
    )
    table.add_column("#",    style="dim", width=4, justify="right")
    table.add_column("Equipo",            width=26)
    table.add_column("PTS",  justify="right")
    table.add_column("PJ",   justify="right")

    total = len(standings)
    for row in standings:
        pos  = row["position"]
        name = row["team"]
        pts  = str(row["pts"])
        pj   = str(row["played"])

        if pos <= top_n:
            style = "bold yellow"
            tag   = " 🏰"
        elif pos > total - bottom_n:
            style = "bold magenta"
            tag   = " 🗡️"
        else:
            style = "dim"
            tag   = ""

        table.add_row(str(pos), name + tag, pts, pj, style=style)

    console.print(table)


# ═══════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════

def run():
    # ── Verificar API key ──────────────────────────────
    # Un token válido de football-data tiene 32 caracteres (hex)
    es_placeholder = any(x in FOOTBALL_DATA_API_KEY.upper() for x in ("AQUI", "YOUR_API_KEY", "TU_TOKEN"))
    es_vacio      = not FOOTBALL_DATA_API_KEY or len(FOOTBALL_DATA_API_KEY) < 20
    
    if es_placeholder or es_vacio:
        console.print(Panel(
            Text.assemble(
                ("Sin API key configurada — no se puede conectar a la API.\n\n", "yellow"),
                ("Para usar el script:\n", "white"),
                ("  1. Registrate gratis en ", "dim"),
                ("https://www.football-data.org/client/register\n", "bright_cyan"),
                ("  2. Confirmá el mail y activá tu token\n", "dim"),
                ("  3. Abrí config.py\n", "dim"),
                (f'  4. Reemplazá "{FOOTBALL_DATA_API_KEY}" por tu key real\n', "dim"),
                ("  5. Dale Play de nuevo", "green"),
            ),
            title="[bold yellow]⚠️  CONFIGURACIÓN REQUERIDA[/]",
            border_style="yellow",
        ))
        sys.exit(1)

    # ── Banner ─────────────────────────────────────────
    console.print()
    console.print(Panel(
        Text.assemble(
            ("Top ", "dim white"),
            (str(TOP_N), "bold yellow"),
            (" vs Últimos ", "dim white"),
            (str(BOTTOM_N), "bold magenta"),
            (f"  ·  próximos {DIAS_ADELANTE} días  ·  ", "dim"),
            (f"{len(LIGAS_ACTIVAS)} ligas activas", "dim"),
        ),
        title="[bold white]⚔️   DAVID vs GOLIAT[/]",
        border_style="bright_blue",
        padding=(0, 4),
    ))
    console.print()

    total_encontrados = 0

    try:
        for liga_code in LIGAS_ACTIVAS:
            console.rule(f"[bold cyan]{nombre_liga(liga_code)}[/]")
            console.print(f"[dim]  Descargando tabla de posiciones...[/]")

            standings = get_standings(liga_code)
            if not standings:
                console.print(f"  [red]No se pudo obtener la tabla de {liga_code}.[/]\n")
                continue

            render_standings_mini(standings, TOP_N, BOTTOM_N, liga_code)

            console.print(f"[dim]  Buscando partidos próximos...[/]")
            fixtures = get_upcoming_fixtures(liga_code)

            if not fixtures:
                console.print(f"  [dim]Sin partidos programados en los próximos {DIAS_ADELANTE} días.[/]\n")
                continue

            matches = find_david_vs_goliat(standings, fixtures, TOP_N, BOTTOM_N)

            if not matches:
                console.print(
                    f"  [dim green]✔  Ningún enfrentamiento David vs Goliat en los próximos {DIAS_ADELANTE} días.[/]\n"
                )
            else:
                total_encontrados += len(matches)
                console.print(
                    f"  [bold green]⚔️   {len(matches)} enfrentamiento(s) encontrado(s):[/]\n"
                )
                for m in matches:
                    render_match_card(m, TOP_N, BOTTOM_N)

            console.print()

    except RuntimeError as e:
        console.print(f"\n[bold red]{e}[/]")
        sys.exit(1)

    # ── Resumen final ──────────────────────────────────
    if total_encontrados == 0:
        console.print(Panel(
            f"[dim]Sin enfrentamientos David vs Goliat en los próximos {DIAS_ADELANTE} días.[/]",
            border_style="dim",
        ))
    else:
        console.print(Panel(
            Text.assemble(
                ("Total: ", "dim"),
                (str(total_encontrados), "bold white"),
                (" partido(s) donde 🗡️  David enfrenta a 🏰 Goliat", "white"),
            ),
            border_style="bright_blue",
        ))


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    run()
