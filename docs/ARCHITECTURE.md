# Architecture

## Layers

1. **Numerical core (`ising_lab/core.py`)** — Hamiltonian, update methods, exact references, diagnostics, scans, and reusable numerical functions.
2. **Interactive application (`app.py`)** — Streamlit controls, trend-first plots, key-data summaries, opt-in complete-data sections, downloads, and validation UI.
3. **Launch layer (`RUN_ISING_LAB.command`, `RUN_ISING_LAB.bat`, `tools/find_free_port.py`)** — local virtual environment creation and automatic free-port selection.
4. **Verification (`tests/`)** — numerical correctness, application structure, display-flow behavior, and launcher helper tests.

## Scientific separation

The Monte Carlo samplers and the independent references are deliberately separate. A sampler must not validate itself. For 1D, finite-periodic transfer-matrix quantities provide the reference. For 2D at zero field, the Onsager/Yang thermodynamic-limit formulas provide an independent comparison where applicable.

## UI data flow

The application follows a progressive-disclosure rule:

`controls → visual trend → key values → explicit complete-data action → raw/complete tables and exports`

This keeps default pages readable without sacrificing reproducibility or access to numerical results.
