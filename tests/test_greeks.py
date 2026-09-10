"""
Unit tests for BlackScholesEngine
"""

import pytest
import numpy as np
from src.engine.black_scholes import BlackScholesEngine


def test_put_call_parity():
    S = 100.0
    K = 105.0
    T = 0.5
    r = 0.05
    sigma = 0.25
    q = 0.02

    call_price = BlackScholesEngine.price("call", S, K, T, r, sigma, q)
    put_price = BlackScholesEngine.price("put", S, K, T, r, sigma, q)

    parity_lhs = call_price - put_price
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(parity_lhs - parity_rhs) < 1e-6


def test_analytical_greeks_vs_finite_difference():
    S = 500.0
    K = 500.0
    T = 45 / 365.0
    r = 0.045
    sigma = 0.20
    q = 0.015

    base = BlackScholesEngine.calculate_all_greeks("call", S, K, T, r, sigma, q)

    # 1. Delta via dS
    eps_s = 0.01
    p_up = BlackScholesEngine.price("call", S + eps_s, K, T, r, sigma, q)
    p_dn = BlackScholesEngine.price("call", S - eps_s, K, T, r, sigma, q)
    fd_delta = (p_up - p_dn) / (2 * eps_s)
    assert abs(base.delta - fd_delta) < 1e-4

    # 2. Gamma via dS^2
    fd_gamma = (p_up - 2 * base.price + p_dn) / (eps_s ** 2)
    assert abs(base.gamma - fd_gamma) < 1e-4

    # 3. Vega via dSigma (vega in profile is per 1% change, raw vega is * 100)
    eps_vol = 0.0001
    p_vol_up = BlackScholesEngine.price("call", S, K, T, r, sigma + eps_vol, q)
    p_vol_dn = BlackScholesEngine.price("call", S, K, T, r, sigma - eps_vol, q)
    fd_vega_raw = (p_vol_up - p_vol_dn) / (2 * eps_vol)
    assert abs(base.vega * 100.0 - fd_vega_raw) < 1e-3

    # 4. Vanna: d(Delta) / d(sigma)
    delta_vol_up = BlackScholesEngine.calculate_all_greeks("call", S, K, T, r, sigma + eps_vol, q).delta
    delta_vol_dn = BlackScholesEngine.calculate_all_greeks("call", S, K, T, r, sigma - eps_vol, q).delta
    fd_vanna = (delta_vol_up - delta_vol_dn) / (2 * eps_vol)
    assert abs(base.vanna - fd_vanna) < 1e-3


def test_implied_volatility_inversion():
    S = 450.0
    K = 460.0
    T = 30 / 365.0
    r = 0.04
    true_sigma = 0.28
    
    call_price = BlackScholesEngine.price("call", S, K, T, r, true_sigma)
    calc_sigma = BlackScholesEngine.implied_volatility(call_price, "call", S, K, T, r)
    assert calc_sigma is not None
    assert abs(calc_sigma - true_sigma) < 1e-4
