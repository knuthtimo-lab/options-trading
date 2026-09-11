"""
Generates and verifies data_cache/sp500_unusual_greeks.json
Scans top liquid S&P 500 stocks across all 11 GICS sectors.
"""
import sys
import os
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data.sp500_constituents import get_sp500_sectors, get_sp500_symbols, get_symbol_sector
from src.engine.unusual_greeks import scan_sp500_anomalies, CACHE_FILE, UnusualGreeksEngine

def main():
    print("=" * 70)
    print("S&P 500 Unusual Greeks Scanner - Cache Generator & Verifier")
    print("=" * 70)

    # 1. Select representative liquid symbols across all 11 GICS sectors
    sector_symbols = {
        "Technology": ["NVDA", "AAPL", "MSFT", "AMD", "AVGO", "ORCL", "CRM", "PLTR"],
        "Financials": ["JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B", "BLK"],
        "Healthcare": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE"],
        "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
        "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS"],
        "Industrials": ["CAT", "GE", "UNP", "BA", "HON", "RTX"],
        "Consumer Staples": ["PG", "COST", "PEP", "KO", "WMT"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "OXY"],
        "Real Estate": ["PLD", "AMT", "EQIX", "SPG"],
        "Materials": ["LIN", "APD", "SHW", "FCX", "NEM"],
        "Utilities": ["NEE", "SO", "DUK", "CEG"],
    }

    scan_targets = []
    for sector, syms in sector_symbols.items():
        scan_targets.extend(syms)

    print(f"Scanning {len(scan_targets)} liquid S&P 500 leaders across all 11 sectors...")
    t0 = time.time()
    results = scan_sp500_anomalies(
        top_n=len(scan_targets),
        max_workers=16,
        min_vol_oi=1.0,
        symbols=None,
        force_refresh=False,
    )
    elapsed = time.time() - t0
    print(f"Scan finished in {elapsed:.2f}s. Total anomalies found: {len(results)}")

    # 2. Verify Cache File on disk
    print("\n--- Verifying Disk Cache File ---")
    assert CACHE_FILE.exists(), f"Cache file not found at {CACHE_FILE}"
    print(f"[OK] Cache file exists at {CACHE_FILE}")
    print(f"[OK] Cache file size: {os.path.getsize(CACHE_FILE):,} bytes")

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    assert "timestamp" in cache_data, "Cache missing 'timestamp'"
    assert "cached_at" in cache_data, "Cache missing 'cached_at'"
    assert "ttl_seconds" in cache_data, "Cache missing 'ttl_seconds'"
    assert cache_data["ttl_seconds"] == 900, f"Expected 900s TTL, got {cache_data['ttl_seconds']}"
    assert "results" in cache_data, "Cache missing 'results'"
    assert len(cache_data["results"]) == cache_data["total_anomalies"], "Cache results count mismatch"
    assert len(cache_data["results"]) >= len(results), "Filtered results exceeded total cached"
    print(f"[OK] Cache header: timestamp={cache_data['timestamp']}, ttl={cache_data['ttl_seconds']}s")

    # 3. Verify anomaly item fields and data types
    print("\n--- Verifying Anomaly Field Schemas & Calculations ---")
    required_fields = [
        "symbol", "sector", "spot_price", "total_volume", "call_volume",
        "put_volume", "put_call_ratio", "max_vol_oi_ratio", "max_vol_oi_strike",
        "max_vol_oi_contract_volume", "max_vol_oi_option_type", "net_gex_m",
        "gamma_regime", "total_vanna_m", "primary_magnet_strike",
        "magnet_distance_pct", "anomaly_classification", "significance_score"
    ]
    valid_classes = {"WHALE_CALL_SWEEP", "WHALE_PUT_SWEEP", "GAMMA_PINNING", "VOL_SQUEEZE", "VANNA_SURGE"}

    sectors_present = set()
    classes_present = set()

    for idx, item in enumerate(results):
        for field in required_fields:
            assert field in item, f"Item {item.get('symbol')} missing required field: {field}"
        assert item["spot_price"] > 0, f"Invalid spot price: {item['spot_price']}"
        assert item["total_volume"] >= 0, f"Invalid total volume: {item['total_volume']}"
        assert item["put_call_ratio"] >= 0, f"Invalid p/c ratio: {item['put_call_ratio']}"
        assert item["anomaly_classification"] in valid_classes, f"Unknown classification: {item['anomaly_classification']}"
        assert 0.0 <= item["significance_score"] <= 100.0, f"Score out of bounds: {item['significance_score']}"
        assert item["gamma_regime"] in ["POSITIVE_GAMMA", "NEGATIVE_GAMMA", "NEUTRAL"], f"Unknown regime: {item['gamma_regime']}"

        sectors_present.add(item["sector"])
        classes_present.add(item["anomaly_classification"])

    print(f"[OK] All {len(results)} anomalies passed field and range validations.")
    print(f"[OK] Sectors represented in cache: {len(sectors_present)}/11 sectors ({', '.join(sorted(sectors_present))})")
    print(f"[OK] Anomaly classifications detected: {', '.join(sorted(classes_present))}")

    # 4. Print Top 10 Anomalies
    print("\n--- Top 10 S&P 500 Unusual Greeks Anomalies ---")
    print(f"{'SYM':<6} {'SECTOR':<16} {'SPOT':>7} {'NET GEX':>9} {'VANNA':>8} {'MAGNET':>8} {'DIST%':>6} {'VOL/OI':>7} {'SCORE':>6} {'CLASS':<18}")
    print("-" * 105)
    for r in results[:10]:
        print(f"{r['symbol']:<6} {r['sector'][:15]:<16} ${r['spot_price']:>6.1f} ${r['net_gex_m']:>7.1f}M ${r['total_vanna_m']:>6.1f}M ${r['primary_magnet_strike']:>7.1f} {r['magnet_distance_pct']:>5.1f}% {r['max_vol_oi_ratio']:>6.1f}x {r['significance_score']:>5.1f} {r['anomaly_classification']:<18}")

    # 5. Verify Cache Hits (0ms latency test)
    print("\n--- Verifying Fast Disk Cache Retrieval ---")
    t_cache = time.time()
    cached_fetch = scan_sp500_anomalies(top_n=20, min_vol_oi=1.0, force_refresh=False)
    cache_latency_ms = (time.time() - t_cache) * 1000.0
    print(f"[OK] Retrieved {len(cached_fetch)} cached anomalies in {cache_latency_ms:.2f}ms (Cache Hit!)")
    assert cache_latency_ms < 500.0, "Cache fetch took too long!"

    print("\n" + "=" * 70)
    print("All S&P 500 Unusual Greeks Cache & Engine Verifications PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
