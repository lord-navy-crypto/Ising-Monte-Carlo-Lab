# Changelog

## 1.1.3 — GitHub-ready release hardening

- Prepared the project directory to serve directly as a GitHub repository root.
- Added CI, Dependabot, issue forms, pull-request template, contribution/security/conduct documents, architecture/numerical documentation, and a release checklist.
- Added `.gitattributes` and `.editorconfig` for consistent repository behavior.
- macOS and Windows launchers now select the first free localhost port from 8501–8520 instead of failing when 8501 is occupied.
- Added a tested cross-platform `tools/find_free_port.py` helper.
- No Ising Hamiltonian, Monte Carlo estimator, exact-reference formula, or trend-first data-display semantics were changed in this release.

## 1.1.2 — trend-first progressive disclosure

- Reordered result pages so plots/trends/heatmaps appear before data tables.
- Replaced always-expanded complete tables with explicit `More data / complete ...` buttons.
- Added small key-data tables/summaries for overview, method comparison, equilibration, 1D/2D scans, finite-size scans, snapshots, external data, and validation.
- Preserved full numeric strings in visible key tables; no scientific precision is intentionally hidden.
- Kept complete tables, raw arrays, trajectories, spin coordinates, and CSV exports available on demand.
- Physics and Monte Carlo algorithms were not changed.

## 1.1.1 — clear-data display

This release intentionally changes only presentation and inspection of results; the 1.1.0 physics and Monte Carlo algorithms are unchanged.

### Data visibility

- Replaced compressed horizontal result cards with vertical tables: one physical/statistical quantity per row.
- Uses full numeric string representations instead of short `:.4g` / `:.6g` output for result inspection.
- Added complete long-form tables for 1D/2D temperature scans, finite-size scans, equilibration trajectories, method comparison, multi-chain results, external-data comparisons, and compliance checks.
- Added complete sampled-array displays for method comparison, equilibration, and snapshot runs.
- Snapshot spin states are listed by coordinate and spin value rather than relying only on a heatmap or a wide matrix.
- Full-width plots are stacked vertically in the major result pages so axes, legends, and hover values are not squeezed.
- Increased result-table height substantially; vertical space is preferred over hidden or truncated data.

## 1.1.0 — numerical-statistics hardening

This release keeps the original Ising project scope and UI structure, but hardens the Monte Carlo semantics and diagnostics.

### Correctness fixes

- Fixed odd-periodic-lattice heat-bath updates. Odd rings/tori are not bipartite, so heat-bath now switches to a sequential single-site update rather than updating adjacent same-color sites in parallel.
- Checkerboard Metropolis now rejects odd periodic `N`; interactive runs transparently fall back to random-site Metropolis and finite-size exports record `method_used`.
- Fixed a Wolff sampling bias. The previous build repeatedly generated clusters until cumulative flipped spins exceeded the lattice size; that state-dependent stopping rule changed the sampled chain. A Wolff update cycle is now exactly one cluster transition, and actual work is tracked separately as spin touches.
- 1D finite-size scans now recompute the finite-`N` transfer-matrix reference for every scanned `N` instead of copying the largest-`N` reference across the scan.

### Statistical semantics

- `susceptibility_standard = Var(M)/(N T)` is now the primary thermodynamic susceptibility.
- The quantity using `<|M|>` is retained as `susceptibility_abs_connected` and is clearly labelled as a diagnostic rather than silently called susceptibility.
- Added FFT autocorrelation, integrated autocorrelation time `tau_int`, effective sample size, and correlation-aware standard errors.
- Retained fixed-block errors as secondary legacy diagnostics (`*_block_error`).
- Added multi-chain convergence checks using random, all-plus, and all-minus initial conditions with energy and `|M|/N` R-hat summaries.

### Work and UI semantics

- Local Metropolis/checkerboard/heat-bath update cycles touch a full lattice; one Wolff update cycle is one cluster.
- `trajectory_work_units = cumulative_spin_touches / N_sites` provides a common computational-work axis for method comparisons.
- The UI now exposes `measure_every` and reports autocorrelation / effective sample size.
- Susceptibility diagnostics are visible in both 1D and 2D temperature scans.
- Monte Carlo budget labels now say “update cycles” instead of pretending every algorithm has the same sweep definition.

### Engineering

- GitHub Actions no longer assumes the repository contains an extra nested `Ising_Monte_Carlo_Lab/` directory.
- Package version updated to `1.1.0`.
- Expanded core regression suite to 22 passing tests in the build environment.
