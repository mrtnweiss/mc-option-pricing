# MC Option Pricing

Monte Carlo option pricing under **Black–Scholes (GBM)** with confidence intervals, Greeks, and variance reduction (antithetic variates, control variate, CRN finite differences).

## Features

- European call/put pricing via Monte Carlo (terminal simulation under GBM)
- Standard error and **95% confidence interval**
- Variance reduction
  - **Antithetic variates**
  - **Control variate** (uses a correlated control with known expectation)
  - **Common Random Numbers (CRN)** for finite-difference Greeks
- Greeks
  - **Delta (pathwise)**
  - **Delta (finite difference + CRN)**
  - **Vega (finite difference + CRN)**
- Clean entry points
  - CLI: `mc-pricer`
  - Module: `python -m mc_pricer`

---

## Installation

Create and activate a virtual environment, then install in editable mode:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e ".[dev]"
```

---

## Quickstart

### CLI demo

```bash
mc-pricer demo --cv --greeks --antithetic --n-paths 80000
```

Useful knobs:

- `--option call|put|both`
- `--n-paths N`
- `--seed SEED`
- `--antithetic`
- `--cv`
- `--greeks`
- `--bump-s0` (relative bump for delta FD)
- `--bump-sigma` (absolute bump for vega FD)

### Module entry point

```bash
python -m mc_pricer demo --cv --greeks --antithetic --n-paths 80000
```

### E2E smoke test

```bash
python scripts/e2e_smoke.py
```

---

## Method overview

### Model (risk-neutral GBM)

Under Black–Scholes, the terminal price under the risk-neutral measure is:

Sₜ = S₀ · exp((r − q − ½σ²)T + σ√T · Z),  where Z ~ N(0, 1)

A European option value is the discounted expectation of its payoff:

V = e^(−rT) · E[ payoff(Sₜ) ]

Monte Carlo approximates the expectation with a sample mean over n_paths. Uncertainty is quantified by the standard error:

stderr = s / √n

and a normal-approximation 95% confidence interval:

V̂ ± 1.96 · stderr

### Variance reduction

- **Antithetic variates:** simulate paired normals \(Z\) and \(-Z\) to reduce variance.
- **Control variate:** use a correlated quantity with known expectation to reduce estimator variance (beta is estimated from sample covariance).
- **CRN (common random numbers):** reuse identical random numbers for “bumped” simulations to reduce noise in finite-difference Greeks.

### Greeks

- **Delta (pathwise):** differentiates the payoff along simulated paths when applicable.
- **Delta/Vega (FD + CRN):** central differences using the same random draws across bumps.

---

## Development

### Lint / format / tests

```bash
ruff check .
black --check .
pytest -q
```

Run slow/stability tests (if marked):

```bash
pytest -q -m slow
```

---

## Benchmark

Run a small benchmark comparing variance reduction methods (plain vs antithetic vs control variate):

```bash
python scripts/benchmark.py

Benchmark: European CALL  n_paths=200000  seed=42
BS price: 8.349406

method                            price       stderr        |err|    time(s)
----------------------------------------------------------------------------
plain                          8.366485     0.030046     0.017079      0.007
antithetic                     8.373232     0.030022     0.023826      0.008
control-variate (antithetic)   8.364304     0.013245     0.014898      0.013
```

## Project layout

- `src/mc_pricer/` — library code
  - `paths.py` — GBM simulation
  - `pricer.py` — MC pricing + control variate
  - `greeks.py` — pathwise + FD/CRN Greeks
  - `bs_closed_form.py` — Black–Scholes closed form + Greeks
  - `cli.py` — CLI entry point
- `scripts/` — E2E smoke / utility scripts
- `tests/` — unit tests
- `.github/workflows/ci.yaml` — CI pipeline (lint/format/tests + optional build/e2e)

---
