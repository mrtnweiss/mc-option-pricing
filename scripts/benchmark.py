from __future__ import annotations

import time

from mc_pricer.bs_closed_form import BSParams, bs_price
from mc_pricer.pricer import mc_price_european_vanilla, mc_price_european_vanilla_cv


def _run(label: str, fn) -> tuple[str, float, float, float]:
    t0 = time.perf_counter()
    res = fn()
    dt = time.perf_counter() - t0
    return (label, res.price, res.stderr, dt)


def main() -> None:
    p = BSParams(S0=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, T=1.0)
    option = "call"
    n_paths = 200_000
    seed = 42

    bs = bs_price(p, option)

    rows: list[tuple[str, float, float, float]] = []
    rows.append(
        _run(
            "plain",
            lambda: mc_price_european_vanilla(
                p, option=option, n_paths=n_paths, seed=seed, antithetic=False
            ),
        )
    )
    rows.append(
        _run(
            "antithetic",
            lambda: mc_price_european_vanilla(
                p, option=option, n_paths=n_paths, seed=seed, antithetic=True
            ),
        )
    )
    rows.append(
        _run(
            "control-variate (antithetic)",
            lambda: mc_price_european_vanilla_cv(
                p, option=option, n_paths=n_paths, seed=seed, antithetic=True
            ),
        )
    )

    print(f"Benchmark: European {option.upper()}  n_paths={n_paths}  seed={seed}")
    print(f"BS price: {bs:.6f}\n")
    print(f"{'method':<26} {'price':>12} {'stderr':>12} {'|err|':>12} {'time(s)':>10}")
    print("-" * 76)
    for label, price, stderr, dt in rows:
        err = abs(price - bs)
        print(f"{label:<26} {price:12.6f} {stderr:12.6f} {err:12.6f} {dt:10.3f}")


if __name__ == "__main__":
    main()
