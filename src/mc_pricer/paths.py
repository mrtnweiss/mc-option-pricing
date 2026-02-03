from __future__ import annotations

import math
from typing import Optional

import numpy as np


def simulate_gbm_terminal(
    *,
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: Optional[int] = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate terminal values S_T under GBM (Black–Scholes) under risk-neutral measure.

    S_T = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z), Z~N(0,1)

    Args:
        S0, r, q, sigma, T: model parameters
        n_paths: number of Monte Carlo samples
        seed: RNG seed for reproducibility
        antithetic: if True, uses antithetic variates (Z and -Z) to reduce variance

    Returns:
        np.ndarray of shape (n_paths,) with simulated terminal prices.
    """
    if S0 <= 0.0:
        raise ValueError("S0 must be > 0")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if T < 0.0:
        raise ValueError("T must be >= 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    # Deterministic cases
    if T == 0.0:
        return np.full(shape=(n_paths,), fill_value=S0, dtype=float)

    if sigma == 0.0:
        # Deterministic forward under r-q
        ST_det = S0 * math.exp((r - q) * T)
        return np.full(shape=(n_paths,), fill_value=ST_det, dtype=float)

    rng = np.random.default_rng(seed)
    vol_sqrt_t = sigma * math.sqrt(T)
    drift = (r - q - 0.5 * sigma * sigma) * T

    if not antithetic:
        z = rng.standard_normal(n_paths)
    else:
        # Generate ceil(n/2) normals, mirror them, then slice to n_paths
        m = (n_paths + 1) // 2
        z_half = rng.standard_normal(m)
        z = np.concatenate([z_half, -z_half])[:n_paths]

    ST = S0 * np.exp(drift + vol_sqrt_t * z)
    return ST
