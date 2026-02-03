from __future__ import annotations


from mc_pricer.bs_closed_form import BSParams, bs_price
from mc_pricer.pricer import mc_price_european_vanilla


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
        bs = bs_price(p, opt)

        lo, hi = mc.ci95
        print(
            f"{opt.upper():>5} | MC={mc.price:10.6f}  stderr={mc.stderr:8.6f}  "
            f"CI95=[{lo:10.6f}, {hi:10.6f}]  BS={bs:10.6f}"
        )


if __name__ == "__main__":
    main()
