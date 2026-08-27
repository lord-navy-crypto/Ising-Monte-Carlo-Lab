# Optimization Audit — Ising Monte Carlo Lab 1.1.0

## Basis

The optimized release was produced from the uploaded `Ising_Monte_Carlo_Lab.zip`. The original notebook remains in `notebooks/jason_1_original.ipynb`. The changes below are audit-driven corrections/extensions; they are not claims about what the original notebook itself already implemented.

## 1. Odd periodic checkerboard / heat-bath issue

A two-color parallel update is valid only when the periodic lattice is bipartite. For a periodic 1D ring or square 2D torus with odd linear size, the periodic wrap connects sites of the same parity. Therefore same-color sites are not independent.

Release 1.1.0 behavior:

- checkerboard Metropolis: requires even `N`;
- heat-bath: uses the fast two-color implementation for even `N`, and an exact sequential single-site heat-bath fallback for odd `N`;
- finite-size scans record any checkerboard-to-random-Metropolis fallback in `method_used`.

A direct 1D `N=15`, `T=1.5` validation after repair gave heat-bath energy close to the finite-N transfer-matrix result.

## 2. Wolff stopping-time bias

The previous core defined one Wolff “sweep” by repeatedly flipping clusters until the cumulative cluster sizes exceeded the number of lattice sites. The number of Markov transitions therefore depended on the current state and on realized cluster sizes. Sampling only at that state-dependent stopping time produced a measurable bias.

Audit reproduction on a 1D periodic ring (`N=15`, `T=1.5`) found the old Wolff result near `E/N ≈ -0.63`, while the transfer-matrix value is about `-0.58313`.

Release 1.1.0 performs exactly one Wolff cluster transition per update cycle. The same validation then produced values within a few `1e-3` of the finite-N exact energy in long runs.

Computational work is no longer conflated with transition count. The core records:

- `total_spin_touches`;
- `work_units = total_spin_touches / N_sites`;
- `trajectory_work_units`.

## 3. Susceptibility naming

The core previously used the `<|M|>`-connected fluctuation as the field named `susceptibility` and stored the standard signed fluctuation under `susceptibility_signed`.

Release 1.1.0 uses:

- `susceptibility_standard = Var(M)/(N T)` — primary thermodynamic fluctuation susceptibility;
- `susceptibility_abs_connected = (<M^2> - <|M|>^2)/(N T)` — symmetry-aware finite-size diagnostic;
- `susceptibility` remains as a compatibility alias for `susceptibility_standard`;
- `susceptibility_signed` remains as a compatibility alias for the same standard quantity.

## 4. Correlated Monte Carlo uncertainty

Fixed 8-block errors alone do not diagnose critical slowing down. Release 1.1.0 adds:

- FFT autocorrelation;
- initial-positive-pair integrated autocorrelation time;
- effective sample size;
- correlation-aware standard error for energy and `|M|/N`;
- fixed-block error retained separately for comparison.

The UI exposes these statistics in method comparison and equilibration workflows.

## 5. Equilibration interpretation

The original repaired build already improved the notebook by using energy per site. Release 1.1.0 keeps the plateau heuristic but explicitly labels it heuristic and supplements it with:

- half-to-half mean drift in units of overall standard deviation;
- integrated autocorrelation time;
- effective sample size;
- three-chain convergence comparison from random, all-plus, and all-minus starts.

No single one of these diagnostics is presented as proof of equilibrium.

## 6. Finite-size exact reference semantics

The 1D transfer-matrix reference is finite-N, so each lattice size has its own exact reference. Release 1.1.0 recomputes it for every size.

The 2D Onsager/Yang reference remains an infinite-volume comparison, and the UI continues to label it as not like-for-like with finite-N Monte Carlo.

## Validation performed in this environment

- Python syntax / AST parsing: pass.
- macOS launcher shell syntax: pass; executable mode remains `755`.
- Core pytest suite: **22 passed**.
- Full Streamlit AppTest suite could not be collected locally because this isolated runtime does not have `streamlit` installed. The repository CI installs `requirements-dev.txt` before running the full test suite.
