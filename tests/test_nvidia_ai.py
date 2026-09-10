import pytest
from src.ai.nvidia_copilot import NvidiaQuantCopilot


def test_symbol_extraction():
    symbols = NvidiaQuantCopilot.extract_symbols_from_prompt("Was denkst du über NVDA und SPY bezüglich Gamma-Pinning?")
    assert "NVDA" in symbols
    assert "SPY" in symbols


def test_live_market_context_gathering():
    ctx = NvidiaQuantCopilot.gather_live_market_context(["SPY"])
    assert "SPY" in ctx
    spy_data = ctx["SPY"]
    assert "spot_price" in spy_data
    assert spy_data["spot_price"] > 0
    assert "ema_20" in spy_data
    assert "rsi_14" in spy_data
    assert "iv_rank" in spy_data


def test_ai_copilot_offline_chat(monkeypatch):
    monkeypatch.setattr(
        NvidiaQuantCopilot,
        "gather_live_market_context",
        lambda syms: {
            s: {
                "spot_price": 500.0,
                "rsi_14": 50.0,
                "trend": "BULLISH",
                "gamma_magnet": 500.0,
                "magnet_pull_force": "HIGH",
                "recommended_strategy": "BULL_PUT_SPREAD",
                "action": "SELL",
                "legs": "500/495",
                "iv_rank": 40.0,
            }
            for s in syms
        },
    )
    NvidiaQuantCopilot.set_api_key("")
    try:
        res = NvidiaQuantCopilot.chat("Analysiere bitte NVDA und den Gamma-Magneten.")
        assert res["status"] == "NO_API_KEY"
        assert "model" in res
        assert "reply" in res
        assert len(res["reply"]) > 50
        assert "NVDA" in res["reply"]
    finally:
        NvidiaQuantCopilot.set_api_key(None)


def test_nemotron_model_configuration(monkeypatch):
    models_to_test = [
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-super-120b-a12b",
    ]
    monkeypatch.setattr(
        NvidiaQuantCopilot,
        "gather_live_market_context",
        lambda syms: {s: {"spot_price": 500.0} for s in syms},
    )
    NvidiaQuantCopilot.set_api_key("")
    try:
        for model in models_to_test:
            NvidiaQuantCopilot.set_model(model)
            assert NvidiaQuantCopilot.get_model() == model
            res = NvidiaQuantCopilot.chat("Kurze Analyse SPY")
            assert res["model"] == model
            assert "SPY" in res["symbols_analyzed"]
    finally:
        NvidiaQuantCopilot.set_api_key(None)

