import math

import pytest

from mc_pricer.bs_closed_form import BSParams, bs_delta, bs_gamma, bs_price, bs_vega, put_call_parity

def test_dummy():
    assert True


def test_put_call_parity_holds():
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)
    lhs = put_call_parity(p)
    rhs = p.S0 * math.exp(-p.q * p.T) - p.K * math.exp(-p.r * p.T)
    assert abs(lhs - rhs) < 1e-10


def test_basic_greeks_sane_ranges():
    p = BSParams(S0=100.0, K=110.0, r=0.01, q=0.0, sigma=0.25, T=0.5)

    c = bs_price(p, "call")
    put = bs_price(p, "put")
    assert c >= 0.0
    assert put >= 0.0

    dc = bs_delta(p, "call")
    dp = bs_delta(p, "put")
    assert 0.0 <= dc <= 1.0
    assert -1.0 <= dp <= 0.0

    g = bs_gamma(p)
    v = bs_vega(p)
    assert g >= 0.0
    assert v >= 0.0


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        bs_price(BSParams(S0=-1.0, K=100.0, r=0.0, q=0.0, sigma=0.2, T=1.0), "call")
    with pytest.raises(ValueError):
        bs_price(BSParams(S0=100.0, K=0.0, r=0.0, q=0.0, sigma=0.2, T=1.0), "call")
    with pytest.raises(ValueError):
        bs_price(BSParams(S0=100.0, K=100.0, r=0.0, q=0.0, sigma=-0.1, T=1.0), "call")
    with pytest.raises(ValueError):
        bs_price(BSParams(S0=100.0, K=100.0, r=0.0, q=0.0, sigma=0.2, T=-1.0), "call")
