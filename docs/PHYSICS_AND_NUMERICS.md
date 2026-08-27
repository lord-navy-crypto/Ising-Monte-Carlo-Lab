# Physics and numerical conventions

The implemented Ising Hamiltonian is

\[
\mathcal H=-J\sum_{\langle i,j\rangle}s_i s_j-h\sum_i s_i,
\qquad s_i\in\{-1,+1\}.
\]

Temperature is expressed in `J/kB`-compatible units unless the user explicitly rescales the model.

## Monte Carlo methods

- Random-site Metropolis
- Checkerboard Metropolis on compatible even periodic lattices
- Heat-bath / Glauber, with a sequential path on odd periodic lattices
- Wolff single-cluster transitions

Wolff work is not labelled as an ordinary local sweep. Method comparisons track spin touches and normalize them by the number of sites.

## Magnetization and susceptibility

The ordered-phase visualization uses `<|M|>/N`. The standard zero-field fluctuation susceptibility is

\[
\chi=\frac{\langle M^2\rangle-\langle M\rangle^2}{NT}.
\]

The fluctuation formed around `<|M|>` is kept as a separately named diagnostic rather than silently replacing the standard susceptibility.

## Correlated samples

Monte Carlo measurements are serially correlated. The platform estimates integrated autocorrelation time and effective sample size, with blocking retained as an additional diagnostic.

## Reference semantics

- 1D: finite-`N`, periodic transfer matrix.
- 2D, `h=0`: Onsager/Yang thermodynamic-limit reference where implemented.
- Finite-size Monte Carlo and infinite-volume analytic results are not described as identical ensembles.
