from __future__ import annotations

import argparse

from mc_pricer.demo import main as demo_main


def main() -> None:
    parser = argparse.ArgumentParser(prog="mc-pricer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="Run demo pricing + greeks output")

    args = parser.parse_args()
    if args.cmd == "demo":
        demo_main()
