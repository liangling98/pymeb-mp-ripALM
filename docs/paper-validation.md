# Paper reproduction check — 2026-09-14

The checked code runs successfully and reproduces the archived solution
radii and screening decisions exactly for all cases rerun below. Runtime
values vary, even on the original hardware. The manuscript's KKT definition
also differs from the implemented stopping residual, although independent
checks of the manuscript formula passed on the 16 solutions tested.

Reference manuscript: `../R1/MEB_R1_v4.tex`. Reference solver records:
`experiments/results/paper`. No solver code or manuscript was changed by
this check.

## Environment and scope

Windows, Intel Core i7-13700K, NVIDIA GeForce RTX 4070, Python 3.12.9,
PyTorch 2.6.0+cu126. These CPU/GPU and Python/PyTorch versions match the
paper environment. Standard CPU and GPU runs used 16 PyTorch threads,
one warm-up, and three measured repeats. Benchmark jobs ran sequentially.

The local `.venv` uses the existing system PyTorch installation and an
editable installation of this package. All 39 tests passed. The earlier
package check also built a source archive and wheel and ran all 39 tests
and the quickstart using the installed wheel from an extracted source archive.

| Experiment | Cases and measured solves | Result |
|---|---|---|
| Standard synthetic, CPU | All 20 `(d,m)` pairs, both methods, 120 solves | All converged; exact archived median radii and retained counts |
| Standard synthetic, CUDA | All 20 pairs, both methods, 120 solves | All converged; exact archived median radii and retained counts |
| Real data, CUDA | Fashion-MNIST 60,000 points; HIGGS 1,000,000 points; FEMNIST 76,866 balls; 18 solves | All converged; exact archived median radii and retained counts |
| Boundary-dense, CUDA | 100,000 antipodal balls in dimension 50; 6 solves | All converged; exact archived median radii; all constraints retained |
| Higher-dimensional synthetic, CUDA | `d=100,500,1000`, `m=1000,5000`; both methods; 12 solves | All converged; exact archived radii and retained counts |
| Screening sensitivity, CUDA | All seven ratios, three original uniform seeds, full HIGGS and boundary-dense instances; 120 recorded solves | Exact archived run-wise radii, initial retention and refinement counts |

There were 396 recorded benchmark solves, excluding warm-ups, the additional
105 instrumented screening solves, and the 16 independent KKT-check solves.
All 396 recorded solves converged and their reported KKT residuals and
full-set violations were at most `1e-8`.

The real-data reruns read the existing caches under
`../pymeb/tests/.dataset_cache/real` through the current package's data
loaders. This checks the cached full-sized instances and solver execution;
it does not revalidate fresh downloads or rebuild the HIGGS/FEMNIST caches
from upstream raw data.

## Numerical comparisons

The maximum relative difference from archived radii was **zero** in every
comparison group, including the higher-dimensional subset and all screening
runs. This means equality of the recorded floating-point radius values;
it is not a claim of exact mathematical arithmetic or identical centers.

| Experiment | Largest code KKT residual | Largest full-set violation |
|---|---:|---:|
| Standard CPU | 9.899e-9 | 4.407e-9 |
| Standard CUDA | 9.418e-9 | 7.030e-9 |
| Real data | 8.975e-9 | 8.975e-9 |
| Boundary-dense | 9.993e-9 | 3.003e-13 |
| Higher-dimensional subset | 6.266e-9 | 4.127e-11 |
| Screening study, mixed method only | 2.939e-9 | 2.939e-9 |

The fresh standard-grid radii were also compared with the original CGAL
reference radii in `../pymeb/tests/meb_benchmark_records.json`. The resulting
16 dimension/method/device maximum relative errors reproduce the paper's
reported values at its displayed precision. The largest is `4.5264e-12`.
CGAL itself was not rerun.

Real-data mixed solves retained exactly 171 Fashion-MNIST points, 61 HIGGS
points, and 4 FEMNIST balls, with no refinements. PCA with rank 256 reproduced
the paper's 95% variance dimensions: 184, 23, and 69, respectively.

For the boundary-dense instance, an independent check found radius errors
below `3.01e-13` relative to the known optimum `1.1`, and center norms below
`3.18e-13` relative to the known center zero.

## Timing results

Representative standard GPU medians, in seconds, at `m=100000`:

| Dimension | Direct: archive → rerun | Mixed: archive → rerun |
|---|---:|---:|
| 20 | 0.292 → 0.385 | 0.296 → 0.184 |
| 30 | 0.508 → 0.578 | 0.446 → 0.293 |
| 40 | 0.543 → 0.593 | 0.338 → 0.251 |
| 50 | 0.932 → 1.038 | 0.498 → 0.392 |

The paper's statement that all standard GPU cases finish in under one second
does not hold literally for this rerun. It remains a description of the
original measurements, not a reproducibility guarantee.

Fresh real-data direct/mixed medians were 1.919/0.638 seconds for
Fashion-MNIST, 1.229/0.470 for HIGGS, and 1.868/0.700 for FEMNIST.
The real-data script instruments mixed-phase timings. The paper's HIGGS
table instead uses its separate uninstrumented screening-study runs;
those are present in `validation_screening.json` for a matching comparison.

Boundary-dense showed no useful screening benefit: all constraints remained.
Its standalone direct/mixed medians were 0.0720/0.0862 seconds. In the
separate uninstrumented sensitivity study, the ratio-0.98 speedup was about
1.015, effectively parity, versus the paper's 0.72. Thus the lack of reduction
is reproduced, but the exact slowdown is not stable across runs.

## Differences to resolve before claiming literal paper/code equivalence

1. **KKT formula.** `src/pymeb/ripalm.py:205` computes the maximum of full-set
   geometric violation, a combined Euclidean stationarity norm, and the sum
   of absolute complementarity terms. The manuscript instead takes the
   maximum of separate stationarity terms, primal and dual cone distances,
   and the maximum absolute complementarity term. Dual cone distance is not
   explicitly included in the code's residual. The code's other aggregations
   are conservative relative to the corresponding manuscript terms, but the
   quantities should not be described as identical.

   An independent closed-form cone-distance calculation evaluated all five
   manuscript terms on 16 fresh CUDA solutions: both methods on four standard
   synthetic instances, all three full-sized real instances, and boundary-dense.
   Every manuscript KKT value was below `1e-8`; the largest was `9.993e-9`.
   For example, mixed FEMNIST had code residual `8.975e-9` but manuscript KKT
   `6.346e-9`. Align the manuscript definition with the historical implementation,
   or explicitly document the implementation's stopping surrogate. Changing the
   stopping calculation would require new timing/iteration validation.

2. **Three rounding discrepancies.** Of 80 standard synthetic timing cells,
   77 agree with rounding the archived median directly to the shown precision.
   The remaining CPU entries are:

   | Case | Paper | Archived median | Direct rounding to 3 decimals |
   |---|---:|---:|---:|
   | Direct, `d=30,m=5000` | 0.198 | 0.197494600 | 0.197 |
   | Mixed, `d=30,m=10000` | 0.187 | 0.186465500 | 0.186 |
   | Mixed, `d=40,m=1000` | 0.152 | 0.151477100 | 0.151 |

   These are small presentation discrepancies, not solver failures.

3. **PCA command, corrected.** The real-data script defaults to PCA rank 128,
   which cannot report Fashion-MNIST's required 184 components. The reproduction
   guide now includes the tested `--pca-rank 256` option. This changes only the
   diagnostic, not the optimization problem.

4. **Library defaults versus paper settings.** Bare `RipALMOptions()` uses
   `tau=1e-4`, `rho=0.5`, and `max_inner=20`; the paper's direct solver uses
   `tau=1e-3`, `rho=0.9`, and `max_inner=50`. The experiment scripts explicitly
   set the paper options and were used for this validation. A bare `solve(...)`
   call is not a request to reproduce the paper's iteration counts or timing.

## Evidence and rerunning

Fresh JSON/CSV records, logs, and audit scripts are under
`experiments/results/local/validation_*`; this directory is intentionally
Git-ignored. `validation_comparison.csv` provides 100 case/method comparisons,
and `validation_comparison.json` adds aggregate checks, screening checks,
CGAL-reference errors, and manuscript rounding discrepancies.
`validation_kkt.json` records every independent KKT component.

From the repository root, using the local environment:

```powershell
.\.venv\Scripts\python.exe -m experiments.benchmark_synthetic --device cpu --threads 16 --output-stem experiments/results/local/validation_synthetic_cpu
.\.venv\Scripts\python.exe -m experiments.benchmark_synthetic --device cuda --threads 16 --output-stem experiments/results/local/validation_synthetic_cuda
.\.venv\Scripts\python.exe experiments/results/local/validation_remaining.py
.\.venv\Scripts\python.exe experiments/results/local/validation_kkt.py
.\.venv\Scripts\python.exe experiments/results/local/validation_compare.py
```

The local runner explicitly redirects real-data loading to the existing sibling
cache. For a standalone checkout, use the reproduction guide's download command
and its default cache location instead.

Not rerun: the `d=5000` and `d=10000` cases, CGAL/SCS/CLARABEL executables,
the separate low-tolerance/penalty sensitivity study, dedicated GPU profiler
and Nsight measurements, or a fresh-platform dependency/CI matrix. The results
above support numerical consistency for the tested scope, not a complete
independent reproduction of every experiment and baseline in the manuscript.
