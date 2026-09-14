# PyMEB

PyMEB computes a minimum enclosing ball for a collection of points or balls.
It provides the high-accuracy ripALM solver and the paper's two-stage
mixed-precision variant, mp-ripALM. Both run on CPU or a CUDA-capable GPU
through PyTorch.

The mixed-precision solver uses an FP32 solve to identify likely boundary
balls, refines the reduced problem in FP64, and always checks the final ball
against every original constraint. Any discarded ball that violates the
requested tolerance is reinserted and the reduced problem is solved again.

## Install

PyMEB requires Python 3.10 or newer and PyTorch 2.2 or newer.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the additional packages needed for tests or paper experiments with:

```bash
python -m pip install -e ".[test,experiments]"
```

For GPU execution, install the PyTorch build appropriate for your CUDA setup
before installing PyMEB. See the official PyTorch installation selector.

## Quick start

The simplest interface accepts an `m x d` tensor of point coordinates:

```python
import torch
from pymeb import solve

points = torch.tensor(
    [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
    dtype=torch.float64,
)
result = solve(points)

print("center:", result.center)
print("radius:", float(result.radius))
print("KKT residual:", float(result.kkt_residual))
print("converged:", result.converged)
```

To enclose balls, provide one nonnegative radius per center:

```python
radii = torch.tensor([0.10, 0.25, 0.05], dtype=torch.float64)
result = solve(points, radii)
```

For a large problem on a GPU, select mp-ripALM explicitly:

```python
from pymeb import MixedPrecisionOptions, solve

options = MixedPrecisionOptions(device="cuda", screening_ratio=0.98)
result = solve(
    points,
    radii,
    method="mp-ripalm",
    device="cuda",
    options=options,
)
print("retained constraints:", result.solver_result.reduced_problem.m)
```

Solve and plot the paper's ten-ball toy example with:

```bash
python -m pip install -e ".[plot]"
python examples/quickstart.py
```

The example displays the input balls and their minimum enclosing ball.

The lower-level `MEBProblem`, `ripalm`, and `mixed_precision_ripalm` interfaces remain
available when you need full control over the primal-dual state and solver
history; see [the API guide](docs/api.md).

## Tests

```bash
python -m pytest
```

The unit tests cover the cone primitives, semismooth Newton step, ripALM
driver, mixed-precision workflow, deterministic synthetic generator, and
real-data preprocessing helpers. They do not download external datasets.

## Reproduce the paper experiments

The `experiments` directory contains the scripts and exact parameter settings
used for the ripALM and mp-ripALM results in the paper. Archived raw records
from the paper machine are in `experiments/results/paper`; new runs should be
written to `experiments/results/local`.

Start with the smoke-sized CPU command:

```bash
python -m experiments.benchmark_synthetic \
  --device cpu --d-values 20 --m-values 1000 --repeats 1 --warmups 0 \
  --output-stem experiments/results/local/synthetic-smoke
```

See [the reproduction guide](experiments/README.md) for the full commands,
hardware details, dataset downloads, and the mapping from scripts to paper
tables.

The comparison baselines from the paper (CGAL, SCS, and CLARABEL) and their
implementations are intentionally not included in this repository. The
included benchmark scripts run only ripALM and mp-ripALM.

## Project layout

```text
src/pymeb/                 solver package
examples/quickstart.py     solve and plot the paper's toy instance
tests/                     fast unit and integration tests
experiments/               paper-reproduction scripts and archived records
docs/api.md                public API notes
```

## Citation

If this code supports your work, please cite:

> Ling Liang and Lei Yang. *Mixed-Precision GPU Acceleration for Large-Scale
> Minimum Enclosing Ball Problems*. 2026.

A machine-readable record is provided in `CITATION.cff`.

## License

No open-source license has been selected yet. The authors should add the
intended license before publishing this repository; without one, normal
copyright restrictions apply.
