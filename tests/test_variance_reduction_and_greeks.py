import math

import pytest

from mc_pricer.bs_closed_form import BSParams, bs_delta, bs_price, bs_vega
from mc_pricer.greeks import mc_delta_fd_crn, mc_delta_pathwise, mc_vega_fd_crn
from mc_pricer.pricer import mc_price_european_vanilla, mc_price_european_vanilla_cv


def _ci_to_stderr(ci95: tuple[float, float]) -> float:
    """Approx stderr from a 95% CI assuming normal approx: CI = mean +/- 1.96*stderr."""
    lo, hi = ci95
    return (hi - lo) / (2.0 * 1.959963984540054)


@pytest.mark.slow
def test_control_variate_reduces_standard_error():
    """
    Control variate should materially reduce the MC standard error for vanilla options.
    We use enough paths + fixed seed + antithetic to make this stable in CI.
    """
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)
    n_paths = 120_000
    seed = 123
    antithetic = True

    plain = mc_price_european_vanilla(
        p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic
    )
    cv = mc_price_european_vanilla_cv(
        p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic
    )

    # sanity
    assert plain.stderr > 0.0
    assert cv.stderr > 0.0

    # CV should reduce stderr noticeably (threshold can be tuned if needed)
    assert cv.stderr < 0.85 * plain.stderr

    # Both should be statistically consistent with BS
    bs = bs_price(p, "call")
    assert abs(plain.price - bs) <= 4.0 * plain.stderr
    assert abs(cv.price - bs) <= 4.0 * cv.stderr

    # If your CV result exposes beta, it should exist and be finite
    if hasattr(cv, "beta"):
        assert cv.beta is not None
        assert math.isfinite(float(cv.beta))


@pytest.mark.slow
def test_delta_pathwise_close_to_bs():
    """
    Pathwise delta should be close to analytical BS delta.
    Use CI if available; otherwise use stderr inferred from CI.
    """
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)
    n_paths = 200_000
    seed = 42
    antithetic = True

    res = mc_delta_pathwise(p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic)
    target = bs_delta(p, "call")

    assert hasattr(res, "value")
    assert hasattr(res, "ci95")

    lo, hi = res.ci95
    # strongest, most robust check: BS delta should lie in the MC CI
    assert lo <= target <= hi

    # optional: also check reasonable proximity in terms of stderr
    stderr = getattr(res, "stderr", None)
    if stderr is None:
        stderr = _ci_to_stderr(res.ci95)
    assert abs(res.value - target) <= 4.0 * float(stderr)


@pytest.mark.slow
def test_delta_fd_crn_close_to_bs():
    """
    Finite-difference delta with common random numbers (CRN) should be close to BS delta.
    """
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)
    n_paths = 200_000
    seed = 42
    antithetic = True

    res = mc_delta_fd_crn(p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic)
    target = bs_delta(p, "call")

    assert hasattr(res, "value")
    assert hasattr(res, "ci95")

    lo, hi = res.ci95
    assert lo <= target <= hi

    stderr = getattr(res, "stderr", None)
    if stderr is None:
        stderr = _ci_to_stderr(res.ci95)
    assert abs(res.value - target) <= 4.0 * float(stderr)


@pytest.mark.slow
def test_vega_fd_crn_close_to_bs():
    """
    FD vega with CRN should be close to analytical BS vega.
    """
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)
    n_paths = 200_000
    seed = 42
    antithetic = True

    res = mc_vega_fd_crn(p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic)
    target = bs_vega(p)

    assert hasattr(res, "value")
    assert hasattr(res, "ci95")

    lo, hi = res.ci95
    assert lo <= target <= hi

    stderr = getattr(res, "stderr", None)
    if stderr is None:
        stderr = _ci_to_stderr(res.ci95)
    assert abs(res.value - target) <= 4.0 * float(stderr)
