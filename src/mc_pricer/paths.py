from __future__ import annotations

import math

import numpy as np


def simulate_gbm_terminal(
    *,
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate terminal values S_T under GBM (Black–Scholes)
    under the risk-neutral measure.

    S_T = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z), Z~N(0,1)
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

def simulate_gbm_paths(
    *,
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate GBM price paths on an equidistant grid.

    Returns array of shape (n_paths, n_steps + 1) including S0 at t=0.
    Uses exact discretization in log-space.
    """
    if S0 <= 0.0:
        raise ValueError("S0 must be > 0")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if T < 0.0:
        raise ValueError("T must be >= 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")

    dt = T / n_steps

    # Deterministic cases
    if T == 0.0:
        out = np.full((n_paths, n_steps + 1), S0, dtype=float)
        return out

    if sigma == 0.0:
        t_grid = np.linspace(0.0, T, n_steps + 1)
        path = S0 * np.exp((r - q) * t_grid)
        return np.tile(path[None, :], (n_paths, 1))

    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    if not antithetic:
        z = rng.standard_normal((n_paths, n_steps))
    else:
        m = (n_paths + 1) // 2
        z_half = rng.standard_normal((m, n_steps))
        z_full = np.concatenate([z_half, -z_half], axis=0)[:n_paths]
        z = z_full

    log_inc = drift + vol * z
    log_S = np.cumsum(log_inc, axis=1)

    out = np.empty((n_paths, n_steps + 1), dtype=float)
    out[:, 0] = S0
    out[:, 1:] = S0 * np.exp(log_S)
    return out