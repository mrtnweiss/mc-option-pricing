from __future__ import annotations

from mc_pricer.bs_closed_form import BSParams, bs_price
from mc_pricer.pricer import mc_price_european_vanilla


def main() -> None:
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)

    n_paths = 200_000
    seed = 42

    mc_call = mc_price_european_vanilla(p, "call", n_paths=n_paths, seed=seed, antithetic=True)
    bs_call = bs_price(p, "call")

    mc_put = mc_price_european_vanilla(p, "put", n_paths=n_paths, seed=seed, antithetic=True)
    bs_put = bs_price(p, "put")

    print("Parameters:")
    print(f"  S0={p.S0}, K={p.K}, r={p.r}, q={p.q}, sigma={p.sigma}, T={p.T}")
    print(f"  n_paths={n_paths}, seed={seed}, antithetic=True")
    print()

    def line(label: str, mc, bs):
        print(
            f"{label:>5} | MC={mc.price:10.6f}  stderr={mc.stderr:9.6f}  "
            f"CI95=[{mc.ci_low:10.6f}, {mc.ci_high:10.6f}]  BS={bs:10.6f}"
        )

    line("CALL", mc_call, bs_call)
    line("PUT", mc_put, bs_put)


if __name__ == "__main__":
    main()
