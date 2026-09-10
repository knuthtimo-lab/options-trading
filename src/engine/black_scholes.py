"""
Black-Scholes-Merton Analytical Engine
Calculates:
- Option Prices (Call, Put)
- 1st Order Greeks: Delta, Vega, Theta, Rho
- 2nd Order Greeks: Gamma, Vanna, Charm, Volga (Vomma)
- 3rd Order Greeks: Speed
- Implied Volatility Inversion (Brentq)
"""

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class GreekProfile:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    charm: float
    volga: float
    speed: float


class BlackScholesEngine:
    @staticmethod
    def _d1_d2(
        S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> tuple[float, float]:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0, 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @classmethod
    def price(
        cls,
        option_type: str,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """Calculate theoretical option price under BSM."""
        if T <= 0:
            if option_type.lower() in ("c", "call"):
                return max(0.0, S - K)
            return max(0.0, K - S)

        if sigma <= 0.0001:
            if option_type.lower() in ("c", "call"):
                return max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
            return max(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))

        d1, d2 = cls._d1_d2(S, K, T, r, sigma, q)
        if option_type.lower() in ("c", "call"):
            return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))

    @classmethod
    def calculate_all_greeks(
        cls,
        option_type: str,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> GreekProfile:
        """Calculate all primary and higher-order Greeks."""
        is_call = option_type.lower() in ("c", "call")
        
        # Handle zero or negative time
        if T <= 0 or sigma <= 0.0001:
            intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
            delta = 1.0 if (is_call and S > K) else (-1.0 if (not is_call and S < K) else 0.0)
            return GreekProfile(
                price=intrinsic,
                delta=delta,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
                vanna=0.0,
                charm=0.0,
                volga=0.0,
                speed=0.0,
            )

        d1, d2 = cls._d1_d2(S, K, T, r, sigma, q)
        sqrt_T = np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)

        # Price
        if is_call:
            price = S * disc_q * cdf_d1 - K * disc_r * cdf_d2
            delta = disc_q * cdf_d1
        else:
            price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
            delta = -disc_q * norm.cdf(-d1)

        # Gamma (identical for Call & Put)
        gamma = (disc_q * pdf_d1) / (S * sigma * sqrt_T)

        # Vega (per 1% change in sigma, hence / 100)
        raw_vega = S * disc_q * pdf_d1 * sqrt_T
        vega_1pct = raw_vega / 100.0

        # Theta (per calendar day decay, hence / 365)
        common_theta = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
        if is_call:
            theta_annual = common_theta - r * K * disc_r * cdf_d2 + q * S * disc_q * cdf_d1
        else:
            theta_annual = common_theta + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        theta_daily = theta_annual / 365.0

        # Rho (per 1% change in interest rate)
        if is_call:
            rho = (K * T * disc_r * cdf_d2) / 100.0
        else:
            rho = (-K * T * disc_r * norm.cdf(-d2)) / 100.0

        # --- HIGHER ORDER GREEKS ---
        # 1. Vanna: d(Delta)/d(sigma) = d(Vega)/d(S)
        # Vanna measures the sensitivity of delta to volatility, and is critical for skew dynamics.
        vanna = -disc_q * pdf_d1 * (d2 / sigma)

        # 2. Charm: d(Delta)/d(t) = -d(Delta)/d(T)
        # Charm is delta decay over time, crucial for weekend & expiration decay positioning.
        term_charm = (2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (2.0 * T * sigma * sqrt_T)
        if is_call:
            charm = q * disc_q * cdf_d1 - disc_q * pdf_d1 * term_charm
        else:
            charm = -q * disc_q * norm.cdf(-d1) - disc_q * pdf_d1 * term_charm

        # 3. Volga (Vomma): d(Vega)/d(sigma)
        # Measures convexity of volatility (vol-of-vol exposure)
        volga = raw_vega * (d1 * d2 / sigma) / 100.0

        # 4. Speed: d(Gamma)/d(S)
        speed = -(gamma / S) * ((d1 / (sigma * sqrt_T)) + 1.0)

        return GreekProfile(
            price=float(max(0.0, price)),
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega_1pct),
            theta=float(theta_daily),
            rho=float(rho),
            vanna=float(vanna),
            charm=float(charm),
            volga=float(volga),
            speed=float(speed),
        )

    @classmethod
    def implied_volatility(
        cls,
        target_price: float,
        option_type: str,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
    ) -> Optional[float]:
        """Calculates Implied Volatility from option market price using Brent's method."""
        if target_price <= 0 or T <= 0:
            return None

        # Intrinsic value bounds
        is_call = option_type.lower() in ("c", "call")
        intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
        if target_price < intrinsic:
            return None

        def objective(sigma):
            return cls.price(option_type, S, K, T, r, sigma, q) - target_price

        try:
            low_val = objective(0.001)
            high_val = objective(5.0)
            if low_val * high_val > 0:
                high_val = objective(15.0)
                if low_val * high_val > 0:
                    return None
                return float(brentq(objective, 0.001, 15.0, maxiter=100))
            return float(brentq(objective, 0.001, 5.0, maxiter=100))
        except Exception:
            return None
