# Changelog

## [0.1.0] - Initial release
- Monte Carlo pricer for European options under Black–Scholes
- Standard error + CI95
- Variance reduction: antithetic variates, control variate, CRN for FD Greeks
- Greeks: delta (pathwise), delta/vega (FD + CRN)
- CLI entry point (`mc-pricer`) and module entry point (`python -m mc_pricer`)
- CI: ruff/black/pytest + e2e smoke