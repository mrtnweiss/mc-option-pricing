import math

from mc_pricer.bs_closed_form import BSParams, bs_price
from mc_pricer.pricer import mc_price_european_vanilla


def test_mc_price_matches_bs_within_confidence():
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)

    res = mc_price_european_vanilla(
        p,
        "call",
        n_paths=120_000,
        seed=123,
        antithetic=True,
    )
    bs = bs_price(p, "call")

    # Basic sanity
    assert res.stderr > 0.0
    assert res.ci_low < res.price < res.ci_high

    # Statistical consistency: error should be within a few standard errors
    err = abs(res.price - bs)
    assert err <= 4.0 * res.stderr


def test_mc_put_call_parity_approx_holds():
    p = BSParams(S0=100.0, K=110.0, r=0.01, q=0.0, sigma=0.25, T=0.5)

    call = mc_price_european_vanilla(
        p, "call", n_paths=150_000, seed=7, antithetic=True
    )
    put = mc_price_european_vanilla(p, "put", n_paths=150_000, seed=7, antithetic=True)

    lhs = call.price - put.price
    rhs = p.S0 * math.exp(-p.q * p.T) - p.K * math.exp(-p.r * p.T)

    # allow statistical tolerance (combine standard errors conservatively)
    tol = 4.0 * (call.stderr + put.stderr)
    assert abs(lhs - rhs) <= tol
