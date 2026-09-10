"""
Run Scanner Entrypoint
Executes real-time options scanning for top tickers and outputs trade recommendations.
Usage:
    python run_scanner.py
    python run_scanner.py SPY QQQ NVDA
"""

import sys
from src.ui.cli_scanner import run_cli_scan

if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]
    run_cli_scan(symbols)