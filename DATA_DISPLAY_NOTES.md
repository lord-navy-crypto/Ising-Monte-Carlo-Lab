# Progressive Data Display — v1.1.2

This patch changes presentation only. The numerical engine from v1.1.1 is preserved.

## Default page order

1. Plot / trend / heatmap first whenever a visual result exists.
2. A short key-data section follows the visual result.
3. Complete numerical output is not constructed until the user clicks `More data / complete ...`.
4. Complete CSV exports live with the complete-data section.

## Key-data policy

- Temperature scans: first point, representative point (or nearest-to-Tc point in 2D), and last point.
- Method comparison: method, energy/site, `<|M|>/N`, specific heat, and standard susceptibility.
- Equilibration: estimated equilibration cycle, drift, autocorrelation time, and effective samples.
- Finite-size: size, magnetization, reference magnetization, energy, and actual method used.
- Snapshot: method, temperature, energy, magnetization, heat capacity, susceptibility, and core sampling diagnostics.
- External data: trend first, then RMSE and a few aligned anchor rows.

## Complete-data policy

The complete-data controls retain the long-form tables introduced in v1.1.1, so every scalar, trajectory value, sampled array value, and spin coordinate can still be inspected without numeric truncation.
