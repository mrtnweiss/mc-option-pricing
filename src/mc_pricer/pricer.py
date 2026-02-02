from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from mc_pricer.bs_closed_form import BSParams
from mc_pricer.paths import simulate_gbm_terminal
from mc_pricer.products import payoff_call, payoff_put

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    seed: Optional[int]
    antithetic: bool


def _mc_mean_and_stderr(discounted_payoff: np.ndarray) -> tuple[float, float]:
    """Return (mean, stderr) from discounted payoffs."""
    n = discounted_payoff.size
    if n <= 1:
        raise ValueError("Need at least 2 paths for a meaningful stderr.")
    mean = float(np.mean(discounted_payoff))
    # sample std with ddof=1
    stdev = float(np.std(discounted_payoff, ddof=1))
    stderr = stdev / math.sqrt(n)
    return mean, stderr


def mc_price_european_vanilla(
    p: BSParams,
    option: OptionType,
    *,
    n_paths: int,
    seed: Optional[int] = None,
    antithetic: bool = False,
    ci_level: float = 0.95,
) -> MCResult:
    """Monte Carlo price for a European call/put under Black–Scholes GBM.

    Returns price + standard error and a normal-approx CI.
    """
    if ci_level <= 0.0 or ci_level >= 1.0:
        raise ValueError("ci_level must be in (0,1)")

    ST = simulate_gbm_terminal(
        S0=p.S0,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        T=p.T,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )

    if option == "call":
        payoff = payoff_call(ST, p.K)
    else:
        payoff = payoff_put(ST, p.K)

    disc = math.exp(-p.r * p.T)
    discounted_payoff = disc * payoff

    price, stderr = _mc_mean_and_stderr(discounted_payoff)

    # Normal approx CI. For interview purposes, z=1.96 for 95% is fine.
    # If you later want exact z for other levels, you can use scipy.stats.norm.ppf.
    if abs(ci_level - 0.95) < 1e-12:
        z = 1.959963984540054  # ~1.96
    else:
        # minimal fallback: use scipy if available
        from scipy.stats import norm  # local import

        z = float(norm.ppf(0.5 + ci_level / 2.0))

    ci_low = price - z * stderr
    ci_high = price + z * stderr

    return MCResult(
        price=price,
        stderr=stderr,
        ci_low=ci_low,
        ci_high=ci_high,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
