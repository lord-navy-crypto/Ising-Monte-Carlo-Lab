# Contributing

Contributions are welcome when they preserve the project's central rule: a visual result is not accepted as scientific validation by itself.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

On Windows, activate with `.venv\Scripts\activate`.

## Pull-request expectations

1. Keep the reusable numerical engine in `ising_lab/` separate from Streamlit presentation logic in `app.py`.
2. State whether a change affects physics, sampling semantics, uncertainty estimates, or only presentation.
3. Add or update tests for changed behavior.
4. When possible, validate numerical work against an independent exact result, limiting case, or reproducible benchmark.
5. Preserve the distinction between finite-size Monte Carlo and thermodynamic-limit references.
6. Do not silently change estimator definitions or units.

## Reporting numerical changes

Include the model dimension, `N`, `J`, `h`, temperature, update method, burn-in, production length, thinning, seed, and reference used. If a change affects correlated Monte Carlo estimates, include autocorrelation/ESS information where relevant.

## User interface changes

The default presentation should remain trend-first: plots before tables, key values before complete raw data, and full data generated/displayed only when requested.
