from __future__ import annotations

from mc_pricer.bs_closed_form import BSParams, bs_delta, bs_price, bs_vega
from mc_pricer.greeks import mc_delta_fd_crn, mc_delta_pathwise, mc_vega_fd_crn
from mc_pricer.pricer import mc_price_european_vanilla, mc_price_european_vanilla_cv


def main() -> None:
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)
    n_paths = 200_000
    seed = 42
    antithetic = True

    print("Parameters:")
    print(
        f"  S0={p.S0}, K={p.K}, r={p.r}, q={p.q}, sigma={p.sigma}, T={p.T}\n"
        f"  n_paths={n_paths}, seed={seed}, antithetic={antithetic}\n"
    )

    for opt in ("call", "put"):
        mc = mc_price_european_vanilla(
            p, option=opt, n_paths=n_paths, seed=seed, antithetic=antithetic
        )
        mc_cv = mc_price_european_vanilla_cv(
            p, option=opt, n_paths=n_paths, seed=seed, antithetic=antithetic
        )
        bs = bs_price(p, opt)

        lo, hi = mc.ci95
        print(
            f"{opt.upper():>5} | MC= {mc.price:9.6f}  stderr={mc.stderr:8.6f}  "
            f"CI95=[{lo:9.6f}, {hi:9.6f}]  BS={bs:9.6f}"
        )

        lo2, hi2 = mc_cv.ci95
        beta = mc_cv.beta if mc_cv.beta is not None else float("nan")
        print(
            f"{'':>5} | CV= {mc_cv.price:9.6f}  stderr={mc_cv.stderr:8.6f}  "
            f"CI95=[{lo2:9.6f}, {hi2:9.6f}]  beta={beta:.4f}"
        )

    delta_pw = mc_delta_pathwise(
        p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic
    )
    delta_fd = mc_delta_fd_crn(
        p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic
    )
    vega_fd = mc_vega_fd_crn(
        p, "call", n_paths=n_paths, seed=seed, antithetic=antithetic
    )

    print("\nGreeks (CALL):")
    print(
        f"  Delta (pathwise): {delta_pw.value:.6f}  CI95={delta_pw.ci95}  "
        f"BS={bs_delta(p, 'call'):.6f}"
    )

    print(
        f"  Delta (FD+CRN):   {delta_fd.value:.6f}  CI95={delta_fd.ci95}  "
        f"BS={bs_delta(p, 'call'):.6f}"
    )

    print(
        f"  Vega  (FD+CRN):   {vega_fd.value:.6f}  CI95={vega_fd.ci95}  "
        f"BS={bs_vega(p):.6f}"
    )


if __name__ == "__main__":
    main()
