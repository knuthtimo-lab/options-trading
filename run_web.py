"""
Run Web Dashboard Server
Usage:
    python run_web.py
    Open browser at: http://localhost:8000
"""

import os
import sys
from src.web.app import app
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print("=" * 65)
    print("QUANT OPTIONS LIVE DASHBOARD & SCANNER")
    print(f"Server laeuft auf: http://{host}:{port}")
    print("Druecke CTRL+C zum Beenden.")
    print("=" * 65)
    uvicorn.run(app, host=host, port=port, log_level="info")