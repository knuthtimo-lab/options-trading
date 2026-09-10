import pytest
from fastapi.testclient import TestClient
from src.web.app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_backtest_21_iterations_endpoint(client):
    r = client.get('/api/backtest/iterations')
    assert r.status_code == 200
    data = r.json()
    assert data['total_iterations'] == 21
    assert len(data['iterations']) == 21
    champ = [it for it in data['iterations'] if it['iteration'] == 15][0]
    assert champ['cagr_pct'] >= 65.0
    assert champ['profit_factor'] >= 1.40

def test_options_chain_safe_nan_handling(client):
    r = client.get('/api/chain/GRAB')
    assert r.status_code == 200
    data = r.json()
    assert data['symbol'] == 'GRAB'
    assert len(data['contracts']) > 0
    for c in data['contracts']:
        assert 'type' in c
        assert c['type'] in ('CALL', 'PUT')
        assert isinstance(c['open_interest'], int)
        assert isinstance(c['volume'], int)

def test_curated_flow_setups(client):
    r = client.get('/api/setups/curated?symbols=SPY')
    assert r.status_code == 200
    data = r.json()
    assert data['count'] >= 1
    setup = data['setups'][0]
    assert 'when_to_sell_rule' in setup
    assert 'when_to_stop_rule' in setup
    assert 'quant_rationale' in setup
    assert 'short_strike' in setup
    assert 'long_strike' in setup
