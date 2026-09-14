# Paper experiment guide

These scripts reproduce the ripALM and mp-ripALM measurements reported in
*Mixed-Precision GPU Acceleration for Large-Scale Minimum Enclosing Ball
Problems*. They contain the paper's deterministic generators, solver
parameters, warm-up policy, repeat counts, timing synchronization, diagnostic
collection, and real-data preprocessing.

The repository intentionally contains no implementation or runner for any
comparison solver. Consequently, the commands below reproduce the paper rows,
figures, and diagnostics for ripALM and mp-ripALM, but do not regenerate the
comparison-solver columns or relative errors that depend on a comparison
solver's reference radius.

Run every command from the repository root after installing the experiment
dependencies:

```bash
python -m pip install -e ".[experiments]"
```

For the closest software match to the paper machine, first install the
appropriate CUDA 12.6 build of PyTorch 2.6.0/torchvision 0.21.0, then run:

```bash
python -m pip install -r experiments/requirements-paper.txt
python -m pip install -e . --no-deps
```

New records default to `experiments/results/local`, which is ignored by Git.
Use `--output-stem` to choose a different location where that option is
available.

## Reference environment

The archived measurements used Python 3.12.9, PyTorch 2.6.0 with CUDA 12.6,
an Intel Core i7-13700K (16 PyTorch CPU threads), 64 GB RAM, Windows 11, and an
NVIDIA GeForce RTX 4070 with 12 GB memory. Timing is hardware- and
software-dependent; radii, feasibility, retention, and KKT residuals are the
portable reproducibility targets.

Each standard timing result is the median of three measured runs after one
warm-up. The archived high-dimensional record contains one measured run per
case. CUDA timings are synchronized before and after each measurement.

## Synthetic CPU/GPU results

These commands reproduce the ripALM/mp-ripALM entries and CPU/GPU speedups in
the standard synthetic benchmark:

```bash
python -m experiments.benchmark_synthetic \
  --device cpu --threads 16 \
  --output-stem experiments/results/local/synthetic_cpu

python -m experiments.benchmark_synthetic \
  --device cuda \
  --output-stem experiments/results/local/synthetic_cuda
```

The deterministic generator implements the recurrence and value ordering
described in the paper. Generated tensors are cached below
`experiments/.dataset_cache`.

For the high-dimensional panel, run only the corresponding grid on CUDA:

```bash
python -m experiments.benchmark_synthetic \
  --device cuda --d-values 100 500 1000 5000 10000 \
  --m-values 1000 5000 --repeats 1
```

The largest cases materialize a dense `(d + 1) x (d + 1)` Newton system and
need substantial GPU memory and time.

## GPU phase and parameter diagnostics

```bash
python -m experiments.benchmark_diagnostics --device cuda --mode phases
python -m experiments.benchmark_diagnostics --device cuda --mode sensitivity
python -m experiments.benchmark_diagnostics --device cuda --mode profile
```

`phases` records synchronized low-solve, screening, high-solve, and full-check
times. `sensitivity` runs the one-factor low-tolerance, screening-ratio, and
initial-penalty experiments. `profile` uses the PyTorch profiler and therefore
requires CUDA. Profiling is deliberately separate from ordinary wall-clock
timing.

For optional Nsight Compute collection, invoke the narrow profiling target
under `ncu`; for example:

```bash
ncu python -m experiments.profile_ncu_target --target evaluate
ncu python -m experiments.profile_ncu_target --target hessian
```

## Screening-ratio study

The expanded study uses three independently seeded synthetic instances, one
million HIGGS rows, and the boundary-dense stress instance:

```bash
python -m experiments.benchmark_screening_sensitivity --device cuda
```

The HIGGS cache must already exist. Create it with the real-data command below
before running this study.

## Real-data and boundary-dense results

The full paper-sized command downloads Fashion-MNIST, streams and caches the
first one million HIGGS rows, and downloads FEMNIST through Hugging Face:

```bash
python -m experiments.benchmark_real_data \
  --device cuda --download \
  --fashion-limit 60000 \
  --higgs-limit 1000000 \
  --femnist-rows 814277 \
  --femnist-grouping writer_character \
  --femnist-min-group-size 2 \
  --pca-rank 256 \
  --output-stem experiments/results/local/real_data_full
```

Downloaded and preprocessed data are cached under
`experiments/.dataset_cache`. Check the source URLs and metadata stored in the
result JSON before publishing or redistributing any upstream data. The source
datasets themselves are not included here.

The `--pca-rank 256` setting is needed to reproduce the paper's Fashion-MNIST
95% variance dimension of 184. The script's default rank of 128 cannot reach
that threshold; this diagnostic setting does not change the MEB instance or
solver options.

Run the constructed worst-case screening instance separately with:

```bash
python -m experiments.benchmark_boundary_dense \
  --device cuda --dimension 50 --balls 100000 \
  --input-radius 0.1 --seed 20260825
```

## Two-phase illustration

```bash
python -m experiments.plot_two_phase_example
```

This writes PDF and PNG versions of the toy two-phase workflow to
`experiments/results/local`.

## Archived outputs

`results/paper` contains the non-baseline raw records and summaries used for
the paper:

- `meb_cpu_framework_records*` and `meb_cuda_framework_records*`: standard
  synthetic CPU/GPU runs;
- `meb_benchmark_records_dim*`: high-dimensional synthetic runs;
- `meb_r3_cuda_phase_breakdown*`, `meb_r3_cuda_parameter_sensitivity*`, and
  `meb_r3_gpu_profile*`: phase, parameter, and profiler diagnostics;
- `meb_r1_screening_path*`: expanded screening-ratio study;
- `meb_real_data_r3_2_full*`: full Fashion-MNIST, HIGGS, and FEMNIST study;
- `meb_boundary_dense_r1_5*`: boundary-dense stress test.

JSON files contain environment metadata, exact arguments and solver options,
raw records, and summaries. CSV files provide table-friendly raw and summary
views.
