from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BSParams:
    """Black-Scholes parameters for a European option.

    S0: spot price
    K: strike
    r: continuously compounded risk-free rate
    q: continuously compounded dividend yield (use 0.0 if none)
    sigma: volatility (annualized)
    T: time to maturity in years
    """

    S0: float
    K: float
    r: float
    q: float
    sigma: float
    T: float


def _validate(p: BSParams) -> None:
    if p.S0 <= 0.0:
        raise ValueError("S0 must be > 0")
    if p.K <= 0.0:
        raise ValueError("K must be > 0")
    if p.sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if p.T < 0.0:
        raise ValueError("T must be >= 0")


def _d1_d2(p: BSParams) -> tuple[float, float]:
    """Compute d1 and d2. Assumes validated p and sigma>0, T>0."""
    vol_sqrt_t = p.sigma * math.sqrt(p.T)
    d1 = (
        math.log(p.S0 / p.K) + (p.r - p.q + 0.5 * p.sigma * p.sigma) * p.T
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(p: BSParams, option: OptionType) -> float:
    """Closed-form Black-Scholes price for a European call/put."""
    _validate(p)

    # Handle T=0: option is worth its intrinsic value.
    if p.T == 0.0:
        if option == "call":
            return max(p.S0 - p.K, 0.0)
        return max(p.K - p.S0, 0.0)

    # Handle sigma=0: deterministic forward under r-q
    if p.sigma == 0.0:
        fwd = p.S0 * math.exp((p.r - p.q) * p.T)
        disc = math.exp(-p.r * p.T)
        if option == "call":
            return disc * max(fwd - p.K, 0.0)
        return disc * max(p.K - fwd, 0.0)

    d1, d2 = _d1_d2(p)
    df_r = math.exp(-p.r * p.T)
    df_q = math.exp(-p.q * p.T)

    if option == "call":
        return p.S0 * df_q * norm.cdf(d1) - p.K * df_r * norm.cdf(d2)
    else:
        return p.K * df_r * norm.cdf(-d2) - p.S0 * df_q * norm.cdf(-d1)


def bs_delta(p: BSParams, option: OptionType) -> float:
    """Black-Scholes delta (dV/dS0)."""
    _validate(p)

    if p.T == 0.0:
        # At expiry delta is not well-defined at-the-money; use a convention.
        if option == "call":
            return 1.0 if p.S0 > p.K else 0.0
        return -1.0 if p.S0 < p.K else 0.0

    if p.sigma == 0.0:
        # Deterministic forward: delta is a step depending on whether fwd > K.
        fwd = p.S0 * math.exp((p.r - p.q) * p.T)
        df_q = math.exp(-p.q * p.T)
        if option == "call":
            return df_q * (1.0 if fwd > p.K else 0.0)
        return df_q * (-1.0 if fwd < p.K else 0.0)

    d1, _ = _d1_d2(p)
    df_q = math.exp(-p.q * p.T)

    if option == "call":
        return df_q * norm.cdf(d1)
    else:
        return df_q * (norm.cdf(d1) - 1.0)


def bs_gamma(p: BSParams) -> float:
    """Black-Scholes gamma (d^2V/dS0^2). Same for call and put."""
    _validate(p)

    if p.T == 0.0 or p.sigma == 0.0:
        return 0.0

    d1, _ = _d1_d2(p)
    df_q = math.exp(-p.q * p.T)
    return df_q * norm.pdf(d1) / (p.S0 * p.sigma * math.sqrt(p.T))


def bs_vega(p: BSParams) -> float:
    """Black-Scholes vega (dV/dsigma)."""
    _validate(p)

    if p.T == 0.0 or p.sigma == 0.0:
        return 0.0

    d1, _ = _d1_d2(p)
    df_q = math.exp(-p.q * p.T)
    return p.S0 * df_q * norm.pdf(d1) * math.sqrt(p.T)


def put_call_parity(p: BSParams) -> float:
    """Return C - P (should equal S0*e^{-qT} - K*e^{-rT})."""
    c = bs_price(p, "call")
    put = bs_price(p, "put")
    return c - put
