from __future__ import annotations

import sys

from mc_pricer.bs_closed_form import BSParams, bs_delta, bs_price, bs_vega
from mc_pricer.greeks import mc_delta_fd_crn, mc_delta_pathwise, mc_vega_fd_crn
from mc_pricer.pricer import mc_price_european_vanilla, mc_price_european_vanilla_cv


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    # Keep it "smoke": fast enough for CI, but stable via seeds + antithetic.
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)
    n_paths = 80_000  # CI-friendly
    seed = 42
    antithetic = True

    print("E2E smoke parameters:")
    print(
        f"  S0={p.S0}, K={p.K}, r={p.r}, q={p.q}, sigma={p.sigma}, T={p.T}\n"
        f"  n_paths={n_paths}, seed={seed}, antithetic={antithetic}\n"
    )

    # ---------- Pricing smoke (MC vs BS) ----------
    for opt in ("call", "put"):
        mc = mc_price_european_vanilla(
            p, option=opt, n_paths=n_paths, seed=seed, antithetic=antithetic
        )
        bs = bs_price(p, opt)

        err = abs(mc.price - bs)
        k = 5.0  # allow a few standard errors

        print(f"{opt.upper():>4} price:")
        print(
            f"  MC={mc.price:.6f}  stderr={mc.stderr:.6f}  CI95={mc.ci95}  "
            f"BS={bs:.6f}  |err|={err:.6f}"
        )
        _assert(
            err <= k * mc.stderr,
            f"{opt} MC price too far from BS: |err|={err} > {k}*stderr={k * mc.stderr}",
        )

        # ---------- Control Variate smoke ----------
        mc_cv = mc_price_european_vanilla_cv(
            p, option=opt, n_paths=n_paths, seed=seed, antithetic=antithetic
        )
        # CV should reduce variance typically; use a soft check.
        # We accept "no worse than" in rare edge cases.
        print(
            f"  CV={mc_cv.price:.6f}  stderr={mc_cv.stderr:.6f}  CI95={mc_cv.ci95}  "
            f"beta={mc_cv.beta:.4f}"
        )

        _assert(
            mc_cv.stderr <= mc.stderr * 1.05,
            f"{opt} CV stderr not improved: cv={mc_cv.stderr} vs plain={mc.stderr}",
        )

    # ---------- Greeks smoke (CALL) ----------
    print("\nGreeks (CALL):")
    bs_d = bs_delta(p, "call")
    bs_v = bs_vega(p)

    delta_pw = mc_delta_pathwise(
        p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic
    )
    delta_fd = mc_delta_fd_crn(
        p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic
    )
    vega_fd = mc_vega_fd_crn(
        p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic
    )

    print(
        f"  Delta (pathwise): {delta_pw.value:.6f}  CI95={delta_pw.ci95}  BS={bs_d:.6f}"
    )
    print(
        f"  Delta (FD+CRN):   {delta_fd.value:.6f}  CI95={delta_fd.ci95}  BS={bs_d:.6f}"
    )
    print(
        f"  Vega  (FD+CRN):   {vega_fd.value:.6f}  CI95={vega_fd.ci95}  BS={bs_v:.6f}"
    )

    # Use stderr-based tolerances (robust across platforms).
    _assert(
        abs(delta_pw.value - bs_d) <= 5.0 * delta_pw.stderr,
        (
            "Pathwise delta too far from BS: "
            f"|err|={abs(delta_pw.value - bs_d)} > 5*stderr={5 * delta_pw.stderr}"
        ),
    )

    _assert(
        abs(delta_fd.value - bs_d) <= 5.0 * delta_fd.stderr,
        (
            "FD+CRN delta too far from BS: "
            f"|err|={abs(delta_fd.value - bs_d)} > 5*stderr={5 * delta_fd.stderr}"
        ),
    )

    _assert(
        abs(vega_fd.value - bs_v) <= 6.0 * vega_fd.stderr,
        (
            "FD+CRN vega too far from BS: "
            f"|err|={abs(vega_fd.value - bs_v)} > 6*stderr={6 * vega_fd.stderr}"
        ),
    )

    print("\nE2E smoke passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\nE2E smoke FAILED: {e}", file=sys.stderr)
        raise
