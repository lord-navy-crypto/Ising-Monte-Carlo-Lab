# Ising Monte Carlo Lab

**Ising Monte Carlo Lab** is an interactive local computational-physics workbench for studying the 1D and 2D Ising models, Monte Carlo sampling, equilibration, finite-size effects, critical behavior, uncertainty, and comparison with independent analytic references.

The project grew from the original `notebooks/jason_1_original.ipynb`, but the current repository is not a notebook wrapper. Numerical physics, exact references, diagnostics, Streamlit presentation, launchers, tests, presets, and repository infrastructure are separated so the calculations can be checked and reused independently.

## Highlights

- 1D and 2D Ising systems with periodic boundaries
- Random-site Metropolis, checkerboard Metropolis, heat-bath/Glauber, and Wolff cluster updates
- Finite-`N` periodic 1D transfer-matrix reference
- Independent 2D Onsager/Yang thermodynamic-limit reference at supported zero-field settings
- Energy, magnetization, specific heat, standard susceptibility, and additional diagnostics
- Integrated autocorrelation time, effective sample size, blocking diagnostics, and multi-chain convergence checks
- Temperature scans, finite-size scans, equilibration studies, method comparison, snapshots, and external-data comparison
- Trend-first interface: figures first, key results second, complete/raw data only when explicitly requested
- CSV/JSON-oriented reproducibility and exports
- One-click macOS and Windows launchers with automatic free-port selection
- Automated GitHub CI across Python 3.10, 3.12, 3.13, and 3.14

## Scientific model

The Hamiltonian is

\[
\mathcal H=-J\sum_{\langle i,j\rangle}s_i s_j-h\sum_i s_i,
\qquad s_i\in\{-1,+1\}.
\]

The application keeps finite-size Monte Carlo results separate from independent analytic references. Smooth curves are not treated as proof of correct sampling.

### Important estimator conventions

The ordered-phase magnetization display uses

\[
\frac{\langle |M|\rangle}{N},
\]

while the standard fluctuation susceptibility uses

\[
\chi=\frac{\langle M^2\rangle-\langle M\rangle^2}{NT}.
\]

The `<|M|>`-connected fluctuation is retained under a different diagnostic name rather than being silently labelled as the standard susceptibility.

## Why the samplers are treated differently

The repository includes four update methods, but it does not force them into a misleading common definition of “one sweep.” In particular, a Wolff cluster transition is not equivalent to one local-lattice sweep. Method comparison therefore tracks spin touches and normalized work units.

Odd periodic lattices are also handled explicitly: checkerboard updates require a compatible even lattice, while heat-bath can use a correct sequential update path on odd sizes.

## Exact and analytic references

### 1D

The platform uses a finite-periodic transfer-matrix calculation for energy, magnetization, specific heat, and susceptibility. During finite-size scans, each lattice size receives its own finite-`N` reference.

### 2D

At supported zero-field settings, the platform provides an independent Onsager/Yang thermodynamic-limit reference. A finite Monte Carlo lattice is not described as the same ensemble as this infinite-volume result.

## Monte Carlo reliability diagnostics

The platform includes more than a visual “curve looks flat” check:

- integrated autocorrelation time `tau_int`
- effective sample size
- autocorrelation-aware uncertainty estimates
- block-based secondary error diagnostics
- random / all-plus / all-minus chain comparison
- R-hat-style agreement diagnostics
- equilibration drift and plateau information

These diagnostics are especially important around criticality, where local algorithms can exhibit critical slowing down.

## Interface philosophy: trend first, complete data on demand

Every major experiment follows the same presentation order:

1. **Plot / heatmap / trend first**
2. **Small set of high-value numerical results**
3. **Explicit More data / complete data action**
4. **Complete numerical tables, trajectories, arrays, spin coordinates, and downloads**

This keeps the default page readable without hiding the full numerical output from users who need it.

## Quick start — macOS

1. Clone or download this repository.
2. Double-click `RUN_ISING_LAB.command`.
3. The launcher creates `.venv`, installs/checks dependencies, finds a free localhost port from `8501` through `8520`, and starts Streamlit.

If macOS blocks the first launch:

```bash
chmod +x RUN_ISING_LAB.command
./RUN_ISING_LAB.command
```

## Quick start — Windows

Double-click:

```text
RUN_ISING_LAB.bat
```

The Windows launcher also creates a local `.venv` and chooses a free port automatically.

## Manual launch

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows activation:

```bat
.venv\Scripts\activate
```

## Development and tests

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m compileall -q app.py ising_lab tools tests
python -m pytest -q
```

The GitHub Actions workflow runs the complete test suite after installing all development dependencies.

## External data

The External Data page accepts measurement CSV files containing at least one of the following structures:

```text
temperature,energy
```

or

```text
temperature,magnetization
```

The platform sorts and validates the supplied measurements and aligns compatible data with the selected independent reference. Temperature uses `J/kB`-compatible units, energy is expressed per site in units of `J`, and magnetization per site is dimensionless.

## Notebook corrections preserved in the platform

The rebuilt platform intentionally corrects several weaknesses in the original notebook while preserving the notebook as a source artifact:

- `abs(mean(M))` could hide sign-flipping ordered behavior; the main display uses `<|M|>/N`.
- Extensive-energy variance was not a size-stable equilibration criterion; diagnostics use intensive quantities and correlation-aware statistics.
- The original calculated equilibration time was not consistently tied to production sampling.
- 1D plots should not inherit the 2D Onsager critical temperature; the 1D zero-field model has no finite-temperature phase transition.
- Finite-size Monte Carlo is not an exact reference and is not labelled as one.
- Temperature scans use temperature as the independent horizontal variable.

## Repository layout

```text
.
├── app.py
├── ising_lab/
│   ├── __init__.py
│   └── core.py
├── tools/
│   └── find_free_port.py
├── tests/
├── presets/
│   └── notebook_repaired.json
├── notebooks/
│   └── jason_1_original.ipynb
├── docs/
├── .github/
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/
│   ├── dependabot.yml
│   └── pull_request_template.md
├── RUN_ISING_LAB.command
├── RUN_ISING_LAB.bat
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── CHANGELOG.md
├── OPTIMIZATION_AUDIT.md
├── VALIDATION_RESULTS.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
└── RELEASE_CHECKLIST.md
```

## Validation record

See:

- `VALIDATION_RESULTS.md` — numerical and test results
- `OPTIMIZATION_AUDIT.md` — algorithmic/statistical audit history
- `DATA_DISPLAY_NOTES.md` — progressive data-display design
- `docs/PHYSICS_AND_NUMERICS.md` — conventions and estimator semantics
- `docs/ARCHITECTURE.md` — repository/code architecture

## Contributing

See `CONTRIBUTING.md`. Numerical changes should include a reproducible validation route whenever possible, preferably against an independent exact solution, limiting case, or benchmark.

## Security and local scope

The bundled launcher binds Streamlit to `localhost`. See `SECURITY.md` for repository guidance.

## Version

Current GitHub-ready package: **1.1.3**.

Version 1.1.3 is repository/launcher hardening. The Ising Hamiltonian, Monte Carlo estimators, analytic references, and v1.1.2 trend-first presentation semantics are unchanged.

## License status

This package does **not** select an open-source license on behalf of the repository owner. Before inviting public reuse or redistribution, add the license you want to use.
