"""Benchmark ripALM/mp-ripALM on reproducible real-data MEB instances.

The measurements cover boundary density, screening retention, refinement,
accuracy, phase costs, and the separate large-m/high-d/nonzero-radius regimes.
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
from typing import Any

import torch

RESULT_DIR = Path(__file__).resolve().parent / "results" / "local"

from .benchmark_synthetic import MIXED_PRECISION_OPTIONS, RIPALM_OPTIONS
from pymeb import MEBProblem, MixedPrecisionOptions, mixed_precision_ripalm, ripalm
from .real_datasets import (
    RealMEBInstance,
    fashion_mnist_problem,
    femnist_ball_problem,
    higgs_problem,
)


BOUNDARY_THRESHOLDS = (1e-8, 1e-6, 1e-4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("fashion", "higgs", "femnist"),
        default=("fashion", "higgs", "femnist"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--screening-ratio", type=float, default=0.98)
    parser.add_argument("--fashion-limit", type=int, default=60_000)
    parser.add_argument("--fashion-seed", type=int, default=20260825)
    parser.add_argument("--higgs-limit", type=int, default=100_000)
    parser.add_argument("--femnist-rows", type=int, default=100_000)
    parser.add_argument(
        "--femnist-grouping",
        choices=("writer", "writer_character"),
        default="writer_character",
    )
    parser.add_argument("--femnist-min-group-size", type=int, default=2)
    parser.add_argument("--femnist-max-groups", type=int)
    parser.add_argument("--pca-sample", type=int, default=10_000)
    parser.add_argument("--pca-rank", type=int, default=128)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=RESULT_DIR / "real_data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    if not (0.0 < args.screening_ratio <= 1.0):
        raise ValueError("screening-ratio must be in (0, 1]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    instances = _load_instances(args)
    mixed_options = replace(
        MIXED_PRECISION_OPTIONS,
        screening_ratio=args.screening_ratio,
        device=device,
        record_phase_timings=True,
    )
    records: list[dict[str, Any]] = []
    dataset_metadata: dict[str, Any] = {}

    for instance in instances:
        master = instance.problem
        if master.dtype != torch.float64:
            raise RuntimeError("real-data master inputs must remain float64")
        dataset_metadata[instance.name] = instance.metadata
        diagnostics = _dataset_diagnostics(
            master,
            sample_size=args.pca_sample,
            pca_rank=args.pca_rank,
            device=device,
        )
        print(
            f"\n{instance.name}: m={master.m}, d={master.d}, "
            f"nonzero radii={diagnostics['nonzero_radius_fraction']:.3f}",
            flush=True,
        )
        problem = master.to(device=device)
        try:
            _warm_up(problem, mixed_options, args.warmups, device)
            for repeat in range(1, args.repeats + 1):
                direct = _measure_direct(problem, device)
                direct.update(
                    dataset=instance.name,
                    m=master.m,
                    d=master.d,
                    repeat=repeat,
                    **diagnostics,
                )
                records.append(direct)
                _print_record(direct)

                mixed = _measure_mixed(problem, mixed_options, device)
                mixed.update(
                    dataset=instance.name,
                    m=master.m,
                    d=master.d,
                    repeat=repeat,
                    **diagnostics,
                )
                records.append(mixed)
                _print_record(mixed)
        except torch.OutOfMemoryError as error:
            records.append(
                {
                    "dataset": instance.name,
                    "m": master.m,
                    "d": master.d,
                    "solver": "failed",
                    "repeat": 0,
                    "error": f"CUDA out of memory: {error}",
                    **diagnostics,
                }
            )
            print(f"  CUDA out of memory: {error}", flush=True)
        finally:
            del problem
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = _summarize(records)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": vars(args),
        "device": device.type,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "master_dtype": "torch.float64",
        "low_dtype": "torch.float32",
        "high_dtype": "torch.float64",
        "baselines_rerun": False,
        "boundary_thresholds": BOUNDARY_THRESHOLDS,
        "direct_options": _jsonable(asdict(RIPALM_OPTIONS)),
        "mixed_options": _jsonable(asdict(mixed_options)),
        "datasets": dataset_metadata,
    }
    _save(args.output_stem, metadata, records, summary)


def _load_instances(args: argparse.Namespace) -> list[RealMEBInstance]:
    instances: list[RealMEBInstance] = []
    for name in args.datasets:
        if name == "fashion":
            instances.append(
                fashion_mnist_problem(
                    limit=args.fashion_limit,
                    seed=args.fashion_seed,
                    download=args.download,
                )
            )
        elif name == "higgs":
            instances.append(
                higgs_problem(limit=args.higgs_limit, download=args.download)
            )
        else:
            instances.append(
                femnist_ball_problem(
                    max_rows=args.femnist_rows,
                    grouping=args.femnist_grouping,
                    min_group_size=args.femnist_min_group_size,
                    max_groups=args.femnist_max_groups,
                    download=args.download,
                )
            )
    return instances


def _warm_up(
    problem: MEBProblem,
    mixed_options: MixedPrecisionOptions,
    warmups: int,
    device: torch.device,
) -> None:
    for _ in range(warmups):
        ripalm(problem, options=RIPALM_OPTIONS)
        mixed_precision_ripalm(problem, options=mixed_options)
    _synchronize(device)


def _measure_direct(problem: MEBProblem, device: torch.device) -> dict[str, Any]:
    _synchronize(device)
    started_at = time.perf_counter()
    result = ripalm(problem, options=RIPALM_OPTIONS)
    _synchronize(device)
    elapsed_sec = time.perf_counter() - started_at
    record: dict[str, Any] = {
        "solver": "ripALM",
        "elapsed_sec": elapsed_sec,
        "radius": float(result.state.r),
        "max_violation": float(problem.violation(result.state).clamp_min(0).max()),
        "kkt_residual": float(result.residual.value),
        "outer_iterations": result.outer_iterations,
        "inner_iterations": result.inner_iterations,
        "line_search_evaluations": result.line_search_evaluations,
        "converged": bool(result.converged),
        "accepted": None,
        "kept": None,
        "kept_fraction": None,
        "refinements": None,
    }
    record.update(_boundary_statistics(problem, result.state))
    return record


def _measure_mixed(
    problem: MEBProblem,
    options: MixedPrecisionOptions,
    device: torch.device,
) -> dict[str, Any]:
    _synchronize(device)
    started_at = time.perf_counter()
    result = mixed_precision_ripalm(problem, options=options)
    _synchronize(device)
    elapsed_sec = time.perf_counter() - started_at
    if result.timings is None:
        raise RuntimeError("phase timings were not recorded")
    record: dict[str, Any] = {
        "solver": "mp-ripALM",
        "elapsed_sec": elapsed_sec,
        "instrumented_total_sec": result.timings.total_sec,
        "low_solve_sec": result.timings.low_solve_sec,
        "screening_sec": result.timings.screening_sec,
        "high_solve_sec": result.timings.high_solve_sec,
        "full_check_sec": result.timings.full_feasibility_check_sec,
        "radius": float(result.state.r),
        "max_violation": float(result.max_violation),
        "kkt_residual": float(result.residual.value),
        "outer_iterations": None,
        "inner_iterations": None,
        "low_outer_iterations": result.low_result.outer_iterations,
        "low_inner_iterations": result.low_result.inner_iterations,
        "high_outer_iterations": result.high_result.outer_iterations,
        "high_inner_iterations": result.high_result.inner_iterations,
        "line_search_evaluations": (
            result.low_result.line_search_evaluations
            + result.high_result.line_search_evaluations
        ),
        "converged": bool(result.converged),
        "accepted": bool(result.accepted),
        "kept": result.reduced_problem.m,
        "kept_fraction": result.reduced_problem.m / problem.m,
        "refinements": result.refinements,
    }
    record.update(_boundary_statistics(problem, result.state))
    return record


def _boundary_statistics(problem: MEBProblem, state: Any) -> dict[str, float | int]:
    slack = problem.slack(state)
    scale = torch.maximum(
        state.r.abs(),
        torch.ones((), dtype=state.r.dtype, device=state.r.device),
    )
    relative_slack = slack / scale
    stats: dict[str, float | int] = {
        "minimum_relative_slack": float(relative_slack.min()),
    }
    for threshold in BOUNDARY_THRESHOLDS:
        key = f"near_boundary_fraction_{threshold:.0e}"
        stats[key] = float((relative_slack <= threshold).to(torch.float64).mean())
    return stats


def _dataset_diagnostics(
    problem: MEBProblem,
    *,
    sample_size: int,
    pca_rank: int,
    device: torch.device,
) -> dict[str, Any]:
    radii = problem.radii
    diagnostics: dict[str, Any] = {
        "nonzero_radius_fraction": float((radii > 0).to(torch.float64).mean()),
        "radius_min": float(radii.min()),
        "radius_median": float(radii.median()),
        "radius_max": float(radii.max()),
    }
    if sample_size <= 1 or pca_rank <= 0 or problem.m <= 1:
        return diagnostics

    count = min(sample_size, problem.m)
    generator = torch.Generator(device="cpu").manual_seed(20260825)
    indices = torch.randperm(problem.m, generator=generator)[:count]
    sample = problem.centers.index_select(0, indices).to(device=device, dtype=torch.float32)
    sample = sample - sample.mean(dim=0, keepdim=True)
    total_variance = sample.square().sum()
    q = min(pca_rank, sample.shape[0] - 1, sample.shape[1])
    if q <= 0 or float(total_variance) == 0.0:
        diagnostics.update(
            pca_sample=count,
            pca_computed_rank=0,
            pca_rank_95=0,
            pca_variance_captured=1.0,
        )
        return diagnostics
    torch.manual_seed(20260825)
    _, singular_values, _ = torch.pca_lowrank(sample, q=q, center=False, niter=2)
    cumulative = singular_values.square().cumsum(dim=0) / total_variance
    reached = torch.nonzero(cumulative >= 0.95, as_tuple=False).flatten()
    diagnostics.update(
        pca_sample=count,
        pca_computed_rank=q,
        pca_rank_95=(int(reached[0]) + 1 if reached.numel() else None),
        pca_variance_captured=float(cumulative[-1]),
    )
    del sample
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return diagnostics


def _summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    datasets = sorted({record["dataset"] for record in records})
    for dataset in datasets:
        group = [record for record in records if record["dataset"] == dataset]
        failed = [record for record in group if record["solver"] == "failed"]
        if failed:
            summaries.append(
                {
                    "dataset": dataset,
                    "m": failed[0]["m"],
                    "d": failed[0]["d"],
                    "status": "failed",
                    "error": failed[0]["error"],
                }
            )
            continue
        direct = [record for record in group if record["solver"] == "ripALM"]
        mixed = [record for record in group if record["solver"] == "mp-ripALM"]
        direct_time = statistics.median(record["elapsed_sec"] for record in direct)
        mixed_time = statistics.median(record["elapsed_sec"] for record in mixed)
        representative = mixed[0]
        summaries.append(
            {
                "dataset": dataset,
                "m": representative["m"],
                "d": representative["d"],
                "status": "ok",
                "repeats": len(mixed),
                "median_ripalm_sec": direct_time,
                "median_mp_ripalm_sec": mixed_time,
                "speedup_ripalm_over_mp": direct_time / mixed_time,
                "median_kept": statistics.median(record["kept"] for record in mixed),
                "median_kept_fraction": statistics.median(
                    record["kept_fraction"] for record in mixed
                ),
                "max_refinements": max(record["refinements"] for record in mixed),
                "all_accepted": all(record["accepted"] for record in mixed),
                "all_converged": all(record["converged"] for record in group),
                "max_violation": max(record["max_violation"] for record in group),
                "max_kkt_residual": max(record["kkt_residual"] for record in group),
                "near_boundary_fraction_1e-08": representative[
                    "near_boundary_fraction_1e-08"
                ],
                "near_boundary_fraction_1e-06": representative[
                    "near_boundary_fraction_1e-06"
                ],
                "near_boundary_fraction_1e-04": representative[
                    "near_boundary_fraction_1e-04"
                ],
                "pca_sample": representative.get("pca_sample"),
                "pca_computed_rank": representative.get("pca_computed_rank"),
                "pca_rank_95": representative.get("pca_rank_95"),
                "pca_variance_captured": representative.get("pca_variance_captured"),
                "nonzero_radius_fraction": representative["nonzero_radius_fraction"],
                "radius_min": representative["radius_min"],
                "radius_median": representative["radius_median"],
                "radius_max": representative["radius_max"],
            }
        )
    return summaries


def _save(
    stem: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(".json")
    raw_path = stem.with_name(f"{stem.name}_raw.csv")
    summary_path = stem.with_name(f"{stem.name}_summary.csv")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": _jsonable(metadata),
                "records": _jsonable(records),
                "summary": _jsonable(summary),
            },
            handle,
            indent=2,
        )
    _write_csv(raw_path, records)
    _write_csv(summary_path, summary)
    print(f"\nSaved {json_path}\nSaved {raw_path}\nSaved {summary_path}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _print_record(record: dict[str, Any]) -> None:
    suffix = ""
    if record["kept"] is not None:
        suffix = (
            f" kept={record['kept']} ({100.0 * record['kept_fraction']:.2f}%)"
            f" refinements={record['refinements']}"
        )
    print(
        f"  {record['solver']:<10} repeat={record['repeat']} "
        f"time={record['elapsed_sec']:.4f}s KKT={record['kkt_residual']:.3e}"
        f" boundary(1e-4)={100.0 * record['near_boundary_fraction_1e-04']:.2f}%"
        f"{suffix}",
        flush=True,
    )


if __name__ == "__main__":
    main()
