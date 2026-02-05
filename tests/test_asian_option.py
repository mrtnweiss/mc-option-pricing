import math

from mc_pricer.bs_closed_form import BSParams
from mc_pricer.pricer import mc_price_asian_arithmetic, mc_price_european_vanilla


def test_asian_call_reasonable_bounds_vs_european():
    # Asian call should typically be <= European call (less variance due to averaging).
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)
    n_paths = 120_000
    seed = 7

    euro = mc_price_european_vanilla(p, "call", n_paths=n_paths, seed=seed, antithetic=True)
    asian = mc_price_asian_arithmetic(p, "call", n_paths=n_paths, n_steps=50, seed=seed, antithetic=True)

    # Soft statistical check with combined tolerance
    tol = 4.0 * (euro.stderr + asian.stderr)
    assert asian.price <= euro.price + tol


def test_asian_price_increases_with_spot():
    p1 = BSParams(S0=90.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)
    p2 = BSParams(S0=110.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)

    n_paths = 80_000
    seed = 42

    a1 = mc_price_asian_arithmetic(p1, "call", n_paths=n_paths, n_steps=50, seed=seed, antithetic=True)
    a2 = mc_price_asian_arithmetic(p2, "call", n_paths=n_paths, n_steps=50, seed=seed, antithetic=True)

    tol = 4.0 * (a1.stderr + a2.stderr)
    assert a2.price >= a1.price - tol
