from __future__ import annotations

import argparse

from mc_pricer.bs_closed_form import BSParams, bs_delta, bs_price, bs_vega
from mc_pricer.greeks import mc_delta_fd_crn, mc_delta_pathwise, mc_vega_fd_crn
from mc_pricer.pricer import mc_price_european_vanilla, mc_price_european_vanilla_cv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mc-pricer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser(
        "demo", help="Price European options and" + "compare against Black–Scholes."
    )
    demo.add_argument("--s0", type=float, default=100.0)
    demo.add_argument("--k", type=float, default=100.0)
    demo.add_argument("--r", type=float, default=0.02)
    demo.add_argument("--q", type=float, default=0.01)
    demo.add_argument("--sigma", type=float, default=0.2)
    demo.add_argument("--t", type=float, default=1.0)

    demo.add_argument("--option", choices=["call", "put", "both"], default="both")
    demo.add_argument("--n-paths", type=int, default=200_000)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--antithetic", action="store_true")

    demo.add_argument(
        "--cv", action="store_true", help="Also run control variate pricing."
    )
    demo.add_argument(
        "--greeks", action="store_true", help="Also compute MC Greeks for CALL."
    )
    demo.add_argument(
        "--bump-s0", type=float, default=1e-4, help="Relative bump for S0 in FD."
    )
    demo.add_argument(
        "--bump-sigma", type=float, default=1e-4, help="Absolute bump for sigma in FD."
    )

    return parser


def cmd_demo(args: argparse.Namespace) -> None:
    p = BSParams(S0=args.s0, K=args.k, r=args.r, q=args.q, sigma=args.sigma, T=args.t)

    print("Parameters:")
    print(
        f"  S0={p.S0}, K={p.K}, r={p.r}, q={p.q}, sigma={p.sigma}, T={p.T}\n"
        f"  n_paths={args.n_paths}, seed={args.seed}, antithetic={args.antithetic}\n"
    )

    opts = ("call", "put") if args.option == "both" else (args.option,)

    for opt in opts:
        mc = mc_price_european_vanilla(
            p,
            option=opt,
            n_paths=args.n_paths,
            seed=args.seed,
            antithetic=args.antithetic,
        )
        bs = bs_price(p, opt)
        lo, hi = mc.ci95

        print(
            f"{opt.upper():>5} | MC= {mc.price:9.6f}  stderr={mc.stderr:8.6f}  "
            f"CI95=[{lo:9.6f}, {hi:9.6f}]  BS={bs:9.6f}"
        )

        if args.cv:
            mc_cv = mc_price_european_vanilla_cv(
                p,
                option=opt,
                n_paths=args.n_paths,
                seed=args.seed,
                antithetic=args.antithetic,
            )
            lo2, hi2 = mc_cv.ci95
            beta = mc_cv.beta if mc_cv.beta is not None else float("nan")
            print(
                f"{'':>5} | CV= {mc_cv.price:9.6f}  stderr={mc_cv.stderr:8.6f}  "
                f"CI95=[{lo2:9.6f}, {hi2:9.6f}]  beta={beta:.4f}"
            )

    if args.greeks:
        print("\nGreeks (CALL):")

        delta_pw = mc_delta_pathwise(
            p, "call", n_paths=args.n_paths, seed=args.seed, antithetic=args.antithetic
        )
        delta_fd = mc_delta_fd_crn(
            p,
            "call",
            n_paths=args.n_paths,
            seed=args.seed,
            antithetic=args.antithetic,
            eps_rel=args.bump_s0,
        )
        vega_fd = mc_vega_fd_crn(
            p,
            "call",
            n_paths=args.n_paths,
            seed=args.seed,
            antithetic=args.antithetic,
            eps_abs=args.bump_sigma,
        )

        bs_d = bs_delta(p, "call")
        bs_v = bs_vega(p)

        print(
            f"  Delta (pathwise): {delta_pw.value:.6f}  CI95={delta_pw.ci95}  "
            f"BS={bs_d:.6f}"
        )
        print(
            f"  Delta (FD+CRN):   {delta_fd.value:.6f}  CI95={delta_fd.ci95}  "
            f"BS={bs_d:.6f}"
        )
        print(
            f"  Vega  (FD+CRN):   {vega_fd.value:.6f}  CI95={vega_fd.ci95}  "
            f"BS={bs_v:.6f}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()
