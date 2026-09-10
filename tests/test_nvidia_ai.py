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


def test_ai_copilot_offline_chat():
    res = NvidiaQuantCopilot.chat("Analysiere bitte NVDA und den Gamma-Magneten.")
    assert "status" in res
    assert "model" in res
    assert "reply" in res
    assert len(res["reply"]) > 50
    assert "NVDA" in res["reply"]
