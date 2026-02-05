from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mc_pricer.bs_closed_form import BSParams
from mc_pricer.paths import simulate_gbm_paths, simulate_gbm_terminal
from mc_pricer.products import (
    payoff_asian_arithmetic_call,
    payoff_asian_arithmetic_put,
    payoff_call,
    payoff_put,
)

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    seed: int | None
    antithetic: bool

    # Optional metadata (harmless for existing code)
    control: str = "none"
    beta: float | None = None

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)


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


def _z_for_ci(ci_level: float) -> float:
    """Two-sided normal critical value for CI level (e.g. 0.95 -> ~1.96)."""
    if abs(ci_level - 0.95) < 1e-12:
        return 1.959963984540054  # ~1.96
    from scipy.stats import norm  # local import

    return float(norm.ppf(0.5 + ci_level / 2.0))


def _apply_control_variate(
    y: np.ndarray, x: np.ndarray, ex: float
) -> tuple[np.ndarray, float]:
    """
    Control variate estimator:
        y_cv = y - beta * (x - E[x]),
    with beta = Cov(y,x)/Var(x).

    Returns:
        (y_cv, beta)
    """
    x_centered = x - float(ex)
    var_x = float(np.var(x_centered, ddof=1))
    if var_x == 0.0:
        return y, 0.0

    cov_yx = float(np.cov(y, x_centered, ddof=1)[0, 1])
    beta = cov_yx / var_x
    y_cv = y - beta * x_centered
    return y_cv, float(beta)


def mc_price_european_vanilla(
    p: BSParams,
    option: OptionType,
    *,
    n_paths: int,
    seed: int | None = None,
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

    z = _z_for_ci(ci_level)
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
        control="none",
        beta=None,
    )


def mc_price_european_vanilla_cv(
    p: BSParams,
    option: OptionType,
    *,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
    ci_level: float = 0.95,
) -> MCResult:
    """
    Monte Carlo price for a European call/put with Control Variate.

    Control variate choice:
        X = e^{-rT} * S_T
        E[X] = S0 * e^{-qT}

    This typically reduces variance materially for vanilla options under BS.
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

    payoff = payoff_call(ST, p.K) if option == "call" else payoff_put(ST, p.K)

    disc = math.exp(-p.r * p.T)

    # Target variable: discounted payoff
    y = disc * payoff

    # Control: discounted stock (known expectation under risk-neutral measure)
    x = disc * ST
    ex = p.S0 * math.exp(-p.q * p.T)

    y_cv, beta = _apply_control_variate(y=y, x=x, ex=ex)

    price, stderr = _mc_mean_and_stderr(y_cv)

    z = _z_for_ci(ci_level)
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
        control="disc_stock",
        beta=beta,
    )


def mc_price_asian_arithmetic(
    p: BSParams,
    option: OptionType,
    *,
    n_paths: int,
    n_steps: int = 50,
    seed: int | None = None,
    antithetic: bool = False,
    ci_level: float = 0.95,
) -> MCResult:
    """Monte Carlo price for arithmetic-average Asian option (discrete monitoring)."""
    if ci_level <= 0.0 or ci_level >= 1.0:
        raise ValueError("ci_level must be in (0,1)")
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")

    paths = simulate_gbm_paths(
        S0=p.S0,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        T=p.T,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
    )

    disc = math.exp(-p.r * p.T)

    payoff = (
        payoff_asian_arithmetic_call(paths, p.K)
        if option == "call"
        else payoff_asian_arithmetic_put(paths, p.K)
    )

    discounted_payoff = disc * payoff
    price, stderr = _mc_mean_and_stderr(discounted_payoff)

    z = _z_for_ci(ci_level)
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
        control="none",
        beta=None,
    )
