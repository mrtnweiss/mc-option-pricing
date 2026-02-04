from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mc_pricer.bs_closed_form import BSParams
from mc_pricer.products import payoff_call, payoff_put

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class GreekResult:
    value: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    seed: int | None
    antithetic: bool

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)


def _z_for_paths(n_paths: int, seed: int | None, antithetic: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if not antithetic:
        return rng.standard_normal(n_paths)

    m = (n_paths + 1) // 2
    z_half = rng.standard_normal(m)
    return np.concatenate([z_half, -z_half])[:n_paths]


def _terminal_from_z(
    *, S0: float, r: float, q: float, sigma: float, T: float, z: np.ndarray
) -> np.ndarray:
    if T == 0.0:
        return np.full_like(z, fill_value=S0, dtype=float)
    if sigma == 0.0:
        st_det = S0 * math.exp((r - q) * T)
        return np.full_like(z, fill_value=st_det, dtype=float)

    drift = (r - q - 0.5 * sigma * sigma) * T
    vol_sqrt_t = sigma * math.sqrt(T)
    return S0 * np.exp(drift + vol_sqrt_t * z)


def _payoff(option: OptionType, ST: np.ndarray, K: float) -> np.ndarray:
    if option == "call":
        return payoff_call(ST, K)
    return payoff_put(ST, K)


def _mean_stderr(x: np.ndarray) -> tuple[float, float]:
    n = x.size
    if n <= 1:
        raise ValueError("Need at least 2 paths.")
    mean = float(np.mean(x))
    stdev = float(np.std(x, ddof=1))
    return mean, stdev / math.sqrt(n)


def _ci(mean: float, stderr: float, level: float = 0.95) -> tuple[float, float]:
    if abs(level - 0.95) < 1e-12:
        z = 1.959963984540054
    else:
        from scipy.stats import norm  # local import

        z = float(norm.ppf(0.5 + level / 2.0))
    return mean - z * stderr, mean + z * stderr


def mc_delta_pathwise(
    p: BSParams,
    option: OptionType,
    *,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
    ci_level: float = 0.95,
) -> GreekResult:
    """Pathwise delta estimator for European call/put under GBM."""
    if p.S0 <= 0.0:
        raise ValueError("S0 must be > 0")
    if p.K <= 0.0:
        raise ValueError("K must be > 0")
    if p.sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if p.T < 0.0:
        raise ValueError("T must be >= 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    z = _z_for_paths(n_paths, seed, antithetic)
    ST = _terminal_from_z(S0=p.S0, r=p.r, q=p.q, sigma=p.sigma, T=p.T, z=z)

    disc = math.exp(-p.r * p.T)

    if p.T == 0.0 or p.sigma == 0.0:
        # Deterministic / expiry conventions: use step-like behavior
        if option == "call":
            delta = 1.0 if ST[0] > p.K else 0.0
        else:
            delta = -1.0 if ST[0] < p.K else 0.0
        return GreekResult(delta, 0.0, delta, delta, n_paths, seed, antithetic)

    dST_dS0 = ST / p.S0

    if option == "call":
        per_path = disc * (ST > p.K) * dST_dS0
    else:
        per_path = disc * (-(ST < p.K) * dST_dS0)

    mean, stderr = _mean_stderr(per_path.astype(float))
    lo, hi = _ci(mean, stderr, ci_level)
    return GreekResult(mean, stderr, lo, hi, n_paths, seed, antithetic)


def mc_delta_fd_crn(
    p: BSParams,
    option: OptionType,
    *,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
    eps_rel: float = 1e-4,
    ci_level: float = 0.95,
) -> GreekResult:
    """Finite-difference delta using Common Random Numbers (CRN)."""
    if eps_rel <= 0.0:
        raise ValueError("eps_rel must be > 0")
    eps = p.S0 * eps_rel
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    z = _z_for_paths(n_paths, seed, antithetic)
    disc = math.exp(-p.r * p.T)

    ST_plus = _terminal_from_z(S0=p.S0 + eps, r=p.r, q=p.q, sigma=p.sigma, T=p.T, z=z)
    ST_minus = _terminal_from_z(S0=p.S0 - eps, r=p.r, q=p.q, sigma=p.sigma, T=p.T, z=z)

    payoff_plus = _payoff(option, ST_plus, p.K)
    payoff_minus = _payoff(option, ST_minus, p.K)

    per_path = disc * (payoff_plus - payoff_minus) / (2.0 * eps)

    mean, stderr = _mean_stderr(per_path)
    lo, hi = _ci(mean, stderr, ci_level)
    return GreekResult(mean, stderr, lo, hi, n_paths, seed, antithetic)


def mc_vega_fd_crn(
    p: BSParams,
    option: OptionType,
    *,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
    eps_abs: float = 1e-4,
    ci_level: float = 0.95,
) -> GreekResult:
    """Finite-difference vega using CRN (bump sigma)."""
    if eps_abs <= 0.0:
        raise ValueError("eps_abs must be > 0")
    if p.sigma < 0.0:
        raise ValueError("sigma must be >= 0")

    z = _z_for_paths(n_paths, seed, antithetic)
    disc = math.exp(-p.r * p.T)

    sig_plus = p.sigma + eps_abs
    sig_minus = max(p.sigma - eps_abs, 0.0)

    ST_plus = _terminal_from_z(S0=p.S0, r=p.r, q=p.q, sigma=sig_plus, T=p.T, z=z)
    ST_minus = _terminal_from_z(S0=p.S0, r=p.r, q=p.q, sigma=sig_minus, T=p.T, z=z)

    payoff_plus = _payoff(option, ST_plus, p.K)
    payoff_minus = _payoff(option, ST_minus, p.K)

    denom = sig_plus - sig_minus
    per_path = disc * (payoff_plus - payoff_minus) / denom

    mean, stderr = _mean_stderr(per_path)
    lo, hi = _ci(mean, stderr, ci_level)
    return GreekResult(mean, stderr, lo, hi, n_paths, seed, antithetic)
