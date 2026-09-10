"""
Interactive Quantitative Options Scanner & Dashboard (Rich UI)
Scans live options chains, computes Dealer Greeks (Net GEX, VEX, Gamma Walls),
and outputs actionable BUY vs. SELL trade setups with exact strikes and pricing.
"""

from typing import List, Optional
import os
import sys
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from src.strategy.signal_generator import SignalGenerator, MarketSignal

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
console = Console(highlight=False, legacy_windows=False)


class OptionsDashboard:
    @staticmethod
    def render_market_overview(signals: List[MarketSignal]):
        """Renders summary overview table of all scanned tickers."""
        table = Table(
            title=">>> QUANTITATIVE OPTIONS MARKET SCANNER (GAMMA & VANNA ENGINE) <<<",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="bright_blue",
        )

        table.add_column("Ticker", style="bold yellow", justify="center")
        table.add_column("Spot", justify="right")
        table.add_column("HV (30d)", justify="right")
        table.add_column("IV Rank", justify="right")
        table.add_column("Net GEX ($M)", justify="right")
        table.add_column("Gamma Regime", justify="center")
        table.add_column("Key Walls (Put / Call)", justify="center")
        table.add_column("Recommended Action", style="bold", justify="center")
        table.add_column("Strategy", justify="left")

        for sig in signals:
            ov = sig.overview
            gp = sig.gamma_profile
            rg = sig.regime
            tr = sig.trade

            gex_m = f"${gp.net_gex_dollar_1pct / 1e6:+.1f}M"
            gex_color = "green" if gp.gamma_regime == "POSITIVE_GAMMA" else "red"
            regime_text = f"[{gex_color}]{gp.gamma_regime}[/{gex_color}]"

            walls = f"${gp.put_wall_strike:.0f} / ${gp.call_wall_strike:.0f}"

            action_color = "green" if rg.action == "SELL_OPTIONS" else ("cyan" if rg.action == "BUY_OPTIONS" else "yellow")
            action_text = f"[{action_color}]{rg.action}[/{action_color}]"

            strategy_name = tr.strategy_name if tr else rg.strategy_type

            table.add_row(
                sig.symbol,
                f"${ov.spot_price:.2f}",
                f"{ov.historical_vol_30d * 100:.1f}%",
                f"{ov.iv_rank_1y:.1f}%",
                gex_m,
                regime_text,
                walls,
                action_text,
                strategy_name,
            )

        console.print(table)

    @staticmethod
    def render_detailed_trade(sig: MarketSignal):
        """Renders a detailed trade card for a single actionable signal."""
        tr = sig.trade
        if not tr:
            return

        is_sell = "SELL" in tr.action
        card_color = "green" if is_sell else "cyan"
        title_text = f"[+] ACTIONABLE TRADE SETUP: {sig.symbol} - {tr.strategy_name.upper()}"

        content = []
        content.append(f"[bold white]Underlying Spot:[/bold white] ${tr.spot_price:.2f} | [bold white]Action:[/bold white] [{card_color}]{tr.action}[/{card_color}]")
        content.append(f"[bold white]Legs:[/bold white] [yellow]{tr.legs_summary}[/yellow] (Expiration: [bold]{tr.expiration}[/bold] | [bold]{tr.dte} DTE[/bold])")
        content.append("-" * 65)
        content.append(f"[bold white]Entry Limit Price:[/bold white] [bold green]${tr.entry_limit_price:.2f}[/bold green] (Credit/Debit per share)")
        content.append(f"[bold white]Target Profit Exit:[/bold white] [bold cyan]${tr.target_exit_price:.2f}[/bold cyan] (Take Profit at 45-50% / +100%)")
        content.append(f"[bold white]Stop Loss Exit:[/bold white]     [bold red]${tr.stop_loss_price:.2f}[/bold red] (Strict Defined-Risk Cut)")
        content.append("-" * 65)
        content.append(f"[bold white]Max Profit (1 Ctr):[/bold white] [green]${tr.max_profit_dollar:.0f}[/green] | [bold white]Max Loss (1 Ctr):[/bold white] [red]${tr.max_loss_dollar:.0f}[/red]")
        content.append(f"[bold white]Win Probability (POP):[/bold white] [bold]{tr.probability_of_profit_pct:.1f}%[/bold] | [bold white]Return on Capital (ROC):[/bold white] [bold]{tr.return_on_capital_pct:.1f}%[/bold]")
        content.append(f"[bold white]Theta Decay:[/bold white] [green]+${tr.net_theta_daily_dollar:.2f}/day[/green] | [bold white]Net Vanna:[/bold white] {tr.net_vanna:.2f} | [bold white]Net Delta:[/bold white] {tr.net_delta:.2f}")
        content.append("-" * 65)
        content.append(f"[italic yellow]Rationale:[/italic yellow] {tr.reasoning}")

        panel = Panel(
            "\n".join(content),
            title=title_text,
            border_style=card_color,
            box=box.ROUNDED,
            expand=False,
        )
        console.print(panel)


def run_cli_scan(symbols: Optional[List[str]] = None):
    """Executes live market scan and renders the terminal dashboard."""
    if not symbols:
        symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]

    console.print("\n[bold cyan]Scanning live options chains across universe...[/bold cyan]")
    signals = SignalGenerator.scan_universe(symbols)

    if not signals:
        console.print("[bold red]No active signals found.[/bold red]")
        return

    console.print("\n")
    OptionsDashboard.render_market_overview(signals)
    console.print("\n[bold yellow]DETAILED ACTIONABLE TRADE CARDS:[/bold yellow]\n")

    for sig in signals:
        if sig.trade:
            OptionsDashboard.render_detailed_trade(sig)