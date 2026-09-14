"""Expanded screening-ratio study for the paper.

The study varies only the screening ratio. It covers independently generated
synthetic instances, a large real dataset, and an adversarial boundary-dense
instance. CGAL, SCS, and CLARABEL are intentionally not rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

RESULT_DIR = Path(__file__).resolve().parent / "results" / "local"

from .benchmark_boundary_dense import boundary_dense_problem
from .benchmark_synthetic import MIXED_PRECISION_OPTIONS, RIPALM_OPTIONS
from .benchmark_diagnostics import (
    _measure_direct,
    _measure_mixed,
    _synchronize,
    _warm_up_direct,
    _warm_up_mixed,
)
from pymeb import MEBProblem, MixedPrecisionOptions, mixed_precision_ripalm
from .real_datasets import higgs_problem


DEFAULT_RATIOS = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0)
DEFAULT_SEEDS = (20260825, 20260826, 20260827)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--synthetic-seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--synthetic-d", type=int, default=50)
    parser.add_argument("--synthetic-m", type=int, default=100_000)
    parser.add_argument("--higgs-limit", type=int, default=1_000_000)
    parser.add_argument("--boundary-d", type=int, default=50)
    parser.add_argument("--boundary-m", type=int, default=100_000)
    parser.add_argument("--boundary-radius", type=float, default=0.1)
    parser.add_argument("--boundary-seed", type=int, default=20260825)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=RESULT_DIR / "screening_sensitivity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    if any(not 0.0 < ratio <= 1.0 for ratio in args.ratios):
        raise ValueError("all screening ratios must lie in (0, 1]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    records: list[dict[str, Any]] = []
    for geometry, instance, master in _instances(args):
        problem = master.to(device=device)
        print(f"\n{instance}: m={problem.m}, d={problem.d}", flush=True)

        _warm_up_direct(problem, device, args.warmups, RIPALM_OPTIONS)
        for repeat in range(1, args.repeats + 1):
            record = _measure_direct(problem, device, RIPALM_OPTIONS)
            record.update(
                geometry=geometry,
                instance=instance,
                m=problem.m,
                d=problem.d,
                screening_ratio=None,
                repeat=repeat,
            )
            records.append(record)

        for ratio in args.ratios:
            options = replace(
                MIXED_PRECISION_OPTIONS,
                screening_ratio=ratio,
                device=device,
                record_phase_timings=False,
            )
            _warm_up_mixed(problem, device, args.warmups, options)
            print(f"  gamma={ratio:g}", flush=True)
            for repeat in range(1, args.repeats + 1):
                record = _measure_mixed_pair(problem, device, options)
                record.update(
                    geometry=geometry,
                    instance=instance,
                    m=problem.m,
                    d=problem.d,
                    screening_ratio=ratio,
                    repeat=repeat,
                )
                records.append(record)

        del problem, master
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = _summarize(records, args.ratios)
    _save(args, device, records, summary)
    _print_summary(summary)


def _instances(args: argparse.Namespace) -> Iterator[tuple[str, str, MEBProblem]]:
    for seed in args.synthetic_seeds:
        yield (
            "synthetic",
            f"uniform_seed_{seed}",
            _uniform_problem(args.synthetic_m, args.synthetic_d, seed),
        )
    real = higgs_problem(limit=args.higgs_limit, download=False)
    yield "real", real.name, real.problem
    yield (
        "boundary_dense",
        "antipodal_boundary_dense_balls",
        boundary_dense_problem(
            dimension=args.boundary_d,
            balls=args.boundary_m,
            input_radius=args.boundary_radius,
            seed=args.boundary_seed,
        ),
    )


def _uniform_problem(m: int, d: int, seed: int) -> MEBProblem:
    """Use the main benchmark's [0, 100) ranges with an independent seed."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = 100.0 * torch.rand((m, d + 1), generator=generator, dtype=torch.float64)
    return MEBProblem(values[:, 1:], values[:, 0], copy=False)


def _measure_mixed_pair(
    problem: MEBProblem,
    device: torch.device,
    options: MixedPrecisionOptions,
) -> dict[str, Any]:
    """Measure ordinary wall time, then collect synchronized phase diagnostics."""

    _synchronize(device)
    started_at = time.perf_counter()
    result = mixed_precision_ripalm(problem, options=options)
    _synchronize(device)
    elapsed_sec = time.perf_counter() - started_at

    instrumented = _measure_mixed(
        problem,
        device,
        replace(options, record_phase_timings=True),
    )
    instrumented["instrumented_elapsed_sec"] = instrumented["elapsed_sec"]
    instrumented["elapsed_sec"] = elapsed_sec
    initial_kept = result.refinement_history[0].kept_constraints
    instrumented.update(
        radius=float(result.state.r),
        max_violation=float(result.max_violation),
        kkt_residual=float(result.residual.value),
        converged=bool(result.converged),
        initial_kept=initial_kept,
        initial_kept_fraction=initial_kept / problem.m,
        kept=result.reduced_problem.m,
        kept_fraction=result.reduced_problem.m / problem.m,
        reinserted_constraints=result.reduced_problem.m - initial_kept,
        refinements=result.refinements,
        accepted=bool(result.accepted),
    )
    return instrumented


def _summarize(
    records: list[dict[str, Any]], ratios: list[float] | tuple[float, ...]
) -> list[dict[str, Any]]:
    direct = [record for record in records if record["solver"] == "ripALM"]
    mixed = [record for record in records if record["solver"] == "mp-ripALM"]
    direct_by_instance: dict[str, list[dict[str, Any]]] = {}
    for record in direct:
        direct_by_instance.setdefault(record["instance"], []).append(record)

    summary: list[dict[str, Any]] = []
    for geometry in ("synthetic", "real", "boundary_dense"):
        geometry_instances = sorted(
            {record["instance"] for record in mixed if record["geometry"] == geometry}
        )
        for ratio in ratios:
            group = [
                record
                for record in mixed
                if record["geometry"] == geometry
                and record["screening_ratio"] == ratio
            ]
            if not group:
                continue
            instance_speedups = []
            instance_relative_radius_errors = []
            for instance in geometry_instances:
                direct_group = direct_by_instance[instance]
                mixed_group = [record for record in group if record["instance"] == instance]
                direct_time = statistics.median(record["elapsed_sec"] for record in direct_group)
                mixed_time = statistics.median(record["elapsed_sec"] for record in mixed_group)
                direct_radius = statistics.median(record["radius"] for record in direct_group)
                mixed_radius = statistics.median(record["radius"] for record in mixed_group)
                instance_speedups.append(direct_time / mixed_time)
                instance_relative_radius_errors.append(
                    abs(mixed_radius - direct_radius) / max(abs(direct_radius), 1.0)
                )
            summary.append(
                {
                    "geometry": geometry,
                    "screening_ratio": ratio,
                    "instances": len(geometry_instances),
                    "timed_runs": len(group),
                    "median_initial_kept_fraction": statistics.median(
                        record["initial_kept_fraction"] for record in group
                    ),
                    "min_initial_kept_fraction": min(
                        record["initial_kept_fraction"] for record in group
                    ),
                    "max_initial_kept_fraction": max(
                        record["initial_kept_fraction"] for record in group
                    ),
                    "median_final_kept_fraction": statistics.median(
                        record["kept_fraction"] for record in group
                    ),
                    "max_reinserted_constraints": max(
                        record["reinserted_constraints"] for record in group
                    ),
                    "median_elapsed_sec": statistics.median(
                        record["elapsed_sec"] for record in group
                    ),
                    "median_low_solve_sec": statistics.median(
                        record["low_solve_sec"] for record in group
                    ),
                    "median_high_solve_sec": statistics.median(
                        record["high_solve_sec"] for record in group
                    ),
                    "median_speedup_over_ripalm": statistics.median(instance_speedups),
                    "min_speedup_over_ripalm": min(instance_speedups),
                    "max_speedup_over_ripalm": max(instance_speedups),
                    "max_refinements": max(record["refinements"] for record in group),
                    "accepted_runs": sum(bool(record["accepted"]) for record in group),
                    "all_accepted": all(bool(record["accepted"]) for record in group),
                    "all_converged": all(bool(record["converged"]) for record in group),
                    "max_full_set_violation": max(record["max_violation"] for record in group),
                    "max_kkt_residual": max(record["kkt_residual"] for record in group),
                    "max_relative_radius_difference": max(instance_relative_radius_errors),
                }
            )
    return summary


def _save(
    args: argparse.Namespace,
    device: torch.device,
    records: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> None:
    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": _jsonable(vars(args)),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "master_dtype": "torch.float64",
        "low_dtype": "torch.float32",
        "high_dtype": "torch.float64",
        "direct_options": _jsonable(asdict(RIPALM_OPTIONS)),
        "mixed_options_except_ratio": _jsonable(asdict(MIXED_PRECISION_OPTIONS)),
        "baselines_rerun": False,
        "timing_policy": (
            "reported elapsed time uses an uninstrumented run with synchronization only at "
            "the run boundaries; phase times use a separate run with synchronization at each "
            "phase boundary"
        ),
    }
    json_path = stem.with_suffix(".json")
    raw_path = stem.with_name(f"{stem.name}_raw.csv")
    summary_path = stem.with_name(f"{stem.name}_summary.csv")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "records": records, "summary": summary}, handle, indent=2)
    _write_csv(raw_path, records)
    _write_csv(summary_path, summary)
    print(f"Saved {json_path}\nSaved {raw_path}\nSaved {summary_path}", flush=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _print_summary(summary: list[dict[str, Any]]) -> None:
    print("\ngeometry       gamma   kept%    total(s)  high(s)  speedup  ref  accepted", flush=True)
    for row in summary:
        print(
            f"{row['geometry']:14s} {row['screening_ratio']:5.3f} "
            f"{100.0 * row['median_initial_kept_fraction']:7.3f} "
            f"{row['median_elapsed_sec']:9.4f} {row['median_high_solve_sec']:8.4f} "
            f"{row['median_speedup_over_ripalm']:8.3f} "
            f"{row['max_refinements']:3.0f} {str(row['all_accepted']):>9s}",
            flush=True,
        )


if __name__ == "__main__":
    main()
