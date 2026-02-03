from mc_pricer.bs_closed_form import BSParams, bs_delta, bs_vega
from mc_pricer.greeks import mc_delta_fd_crn, mc_delta_pathwise, mc_vega_fd_crn


def test_delta_pathwise_matches_bs_within_ci():
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)

    res = mc_delta_pathwise(p, "call", n_paths=120_000, seed=123, antithetic=True)
    target = bs_delta(p, "call")

    err = abs(res.value - target)
    assert err <= 4.0 * res.stderr


def test_delta_fd_crn_matches_bs_within_ci():
    p = BSParams(S0=100.0, K=110.0, r=0.01, q=0.0, sigma=0.25, T=0.5)

    res = mc_delta_fd_crn(
        p, "call", n_paths=140_000, seed=7, antithetic=True, eps_rel=1e-4
    )
    target = bs_delta(p, "call")

    err = abs(res.value - target)
    assert err <= 4.0 * res.stderr


def test_vega_fd_crn_matches_bs_within_ci():
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)

    res = mc_vega_fd_crn(
        p, "call", n_paths=160_000, seed=42, antithetic=True, eps_abs=1e-4
    )
    target = bs_vega(p)

    err = abs(res.value - target)
    assert err <= 4.0 * res.stderr
