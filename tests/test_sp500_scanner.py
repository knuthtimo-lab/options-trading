import pytest
from fastapi.testclient import TestClient
from src.data.sp500_constituents import get_sp500_symbols, get_sp500_sectors, get_symbol_sector
from src.engine.unusual_greeks import UnusualGreeksEngine, scan_sp500_anomalies, CACHE_FILE
from src.web.app import app


def test_sp500_constituents_structure():
    """Verify official 11 GICS sectors and symbol extraction."""
    sectors = get_sp500_sectors()
    assert len(sectors) == 11
    
    expected_sectors = [
        "Technology", "Financials", "Healthcare", "Consumer Discretionary",
        "Communication Services", "Industrials", "Consumer Staples", "Energy",
        "Real Estate", "Materials", "Utilities"
    ]
    for s in expected_sectors:
        assert s in sectors, f"Sector {s} not found in constituents"
        assert len(sectors[s]) > 0

    symbols = get_sp500_symbols()
    assert len(symbols) >= 500
    assert "NVDA" in symbols
    assert "AAPL" in symbols
    assert "JPM" in symbols
    assert "XOM" in symbols

    assert get_symbol_sector("NVDA") == "Technology"
    assert get_symbol_sector("JPM") == "Financials"
    assert get_symbol_sector("LLY") == "Healthcare"
    assert get_symbol_sector("AMZN") == "Consumer Discretionary"
    assert get_symbol_sector("XOM") == "Energy"
    assert get_symbol_sector("BRK.B") == "Financials"
    assert get_symbol_sector("BRK-B") == "Financials"


def test_sp500_scanner_cache_and_anomalies():
    """Verify that scan_sp500_anomalies loads cached anomalies and satisfies field contracts."""
    anomalies = scan_sp500_anomalies(top_n=20, min_vol_oi=1.0, force_refresh=False)
    assert len(anomalies) > 0

    required_fields = [
        "symbol", "sector", "spot_price", "total_volume", "call_volume",
        "put_volume", "put_call_ratio", "max_vol_oi_ratio", "max_vol_oi_strike",
        "max_vol_oi_contract_volume", "max_vol_oi_option_type", "net_gex_m",
        "gamma_regime", "total_vanna_m", "primary_magnet_strike",
        "magnet_distance_pct", "anomaly_classification", "significance_score"
    ]
    valid_classes = {"WHALE_CALL_SWEEP", "WHALE_PUT_SWEEP", "GAMMA_PINNING", "VOL_SQUEEZE", "VANNA_SURGE"}

    for a in anomalies:
        for f in required_fields:
            assert f in a, f"Anomaly {a.get('symbol')} missing field: {f}"
        assert a["spot_price"] > 0
        assert a["anomaly_classification"] in valid_classes
        assert 0.0 <= a["significance_score"] <= 100.0


def test_api_sp500_unusual_endpoint():
    """Verify /api/sp500/unusual endpoint filtering and sorting."""
    client = TestClient(app)

    # 1. Base query
    res = client.get("/api/sp500/unusual?sort_by=significance&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "anomalies" in data
    assert "sectors" in data
    assert len(data["sectors"]) == 11
    assert len(data["anomalies"]) <= 10

    # 2. Sector filtering
    res_tech = client.get("/api/sp500/unusual?sector=Technology&limit=10")
    assert res_tech.status_code == 200
    data_tech = res_tech.json()
    for item in data_tech["anomalies"]:
        assert item["sector"] == "Technology"

    # 3. Sorting verification
    for sort_param in ["vol_oi", "net_gex", "vanna", "volume", "significance"]:
        r = client.get(f"/api/sp500/unusual?sort_by={sort_param}&limit=5")
        assert r.status_code == 200


def test_api_sp500_top_symbols_endpoint():
    """Verify /api/sp500/top_symbols returns comma-separated top tickers."""
    client = TestClient(app)

    # Count = 8
    res8 = client.get("/api/sp500/top_symbols?count=8")
    assert res8.status_code == 200
    data8 = res8.json()
    assert data8["count"] == 8
    assert len(data8["symbol_list"]) == 8
    assert len(data8["symbols"].split(",")) == 8

    # Count = 16
    res16 = client.get("/api/sp500/top_symbols?count=16")
    assert res16.status_code == 200
    data16 = res16.json()
    assert data16["count"] == 16
    assert len(data16["symbol_list"]) == 16
    assert len(data16["symbols"].split(",")) == 16
