# API guide

## Convenience interface

`pymeb.solve(centers, radii=None, ...)` is the recommended entry point. It
accepts points when `radii` is omitted and balls when a radius vector is
provided. The returned `MEBResult` exposes:

- `center`: the computed center, shape `[d]`;
- `radius`: the enclosing radius;
- `max_violation`: the largest full-set feasibility violation;
- `kkt_residual`: the solver's maximum KKT-style residual;
- `converged`: whether the selected solver met its stopping tests; and
- `solver_result`: complete state, history, and method-specific diagnostics.

The output tensors remain on the solver device. Convert a scalar with
`float(result.radius)` or move tensors with `result.center.cpu()` as needed.
`MEBResult.contains(...)` uses an absolute tolerance of `1e-8` by default,
matching the default final solver tolerance. Pass `atol=0` for a strict
floating-point comparison.

## Direct ripALM

Use the lower-level interface to reuse a state or inspect every outer
iteration:

```python
import torch
from pymeb import MEBProblem, RipALMOptions, ripalm

problem = MEBProblem(
    centers=torch.randn(10_000, 50, dtype=torch.float64, device="cuda"),
    radii=torch.zeros(10_000, dtype=torch.float64, device="cuda"),
)
options = RipALMOptions(
    outer_tol=1e-8,
    max_outer=100,
    max_inner=50,
    print_history=False,
)
result = ripalm(problem, options=options)
```

`result.state.r` and `result.state.c` are the enclosing radius and center.
`result.history` records iteration counts, penalty values, line-search work,
and residual components.

## Mixed-precision ripALM

```python
from pymeb import MixedPrecisionOptions, mixed_precision_ripalm

options = MixedPrecisionOptions(
    device="cuda",
    screening_ratio=0.98,
    feasibility_abs_tol=1e-8,
    max_refinements=2,
)
result = mixed_precision_ripalm(problem, options=options)
```

Important diagnostics include `kept_indices`, `reduced_problem`,
`max_violation`, `refinements`, `accepted`, and `refinement_history`. Set
`record_phase_timings=True` to collect synchronized phase timings. Timing adds
synchronization overhead, so use a separate uninstrumented run for total
performance measurements.

The master input should normally remain FP64. mp-ripALM creates its own FP32
copy for screening and uses the FP64 master data for refinement and the final
full-set check.

## Input conventions

- `centers` has shape `[m, d]`.
- `radii` has shape `[m]` and contains nonnegative values.
- Inputs must be floating-point tensors for `MEBProblem`; `solve` converts
  array-like inputs to its requested floating-point `dtype`.
- All tensors belonging to one problem/state pair must share a device and
  dtype.
- This implementation forms and solves a dense `(d + 1) x (d + 1)` Newton
  system, so very high-dimensional problems can be memory intensive.
