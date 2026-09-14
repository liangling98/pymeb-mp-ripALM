"""Record CPU or GPU performance of ripALM and mixed-precision ripALM.

The deterministic test instance is generated once in float64.  Direct ripALM
uses that float64 instance, while mixed-precision ripALM performs its float32
screening conversion internally and returns to the float64 master data for
high-precision refinement.

CPU is the default. This benchmark intentionally does not rerun CGAL, SCS, or
CLARABEL.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .datasets import paper_meb_problem
from pymeb import MixedPrecisionOptions, RipALMOptions, mixed_precision_ripalm, ripalm


DEFAULT_D_VALUES = [20, 30, 40, 50]
DEFAULT_M_VALUES = [10**3, 5 * 10**3, 10**4, 5 * 10**4, 10**5]
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "local"

RIPALM_OPTIONS = RipALMOptions(
    sigma=1.0,
    tau=1e-3,
    rho=0.9,
    outer_tol=1e-8,
    max_outer=100,
    max_inner=50,
    print_history=False,
    sigma_growth_factor=2.0,
)

MIXED_PRECISION_OPTIONS = MixedPrecisionOptions(
    screening_ratio=0.98,
    low_dtype=torch.float32,
    high_dtype=torch.float64,
    device="cpu",
    low_options=RipALMOptions(
        sigma=1.0,
        tau=1e-3,
        rho=0.9,
        outer_tol=1e-2,
        max_outer=100,
        max_inner=10,
        print_history=False,
        sigma_growth_factor=1.0,
    ),
    high_options=RipALMOptions(
        sigma=1.0,
        tau=1e-3,
        rho=0.9,
        outer_tol=1e-8,
        max_outer=100,
        max_inner=20,
        print_history=False,
        sigma_growth_factor=2.0,
    ),
    feasibility_abs_tol=1e-8,
    max_refinements=2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-values", nargs="+", type=int, default=DEFAULT_D_VALUES)
    parser.add_argument("--m-values", nargs="+", type=int, default=DEFAULT_M_VALUES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups must be nonnegative")
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    mixed_options = MixedPrecisionOptions(
        screening_ratio=MIXED_PRECISION_OPTIONS.screening_ratio,
        min_keep=MIXED_PRECISION_OPTIONS.min_keep,
        low_dtype=MIXED_PRECISION_OPTIONS.low_dtype,
        high_dtype=MIXED_PRECISION_OPTIONS.high_dtype,
        device=device,
        low_options=MIXED_PRECISION_OPTIONS.low_options,
        high_options=MIXED_PRECISION_OPTIONS.high_options,
        feasibility_abs_tol=MIXED_PRECISION_OPTIONS.feasibility_abs_tol,
        max_refinements=MIXED_PRECISION_OPTIONS.max_refinements,
    )

    d_values = sorted(set(args.d_values))
    m_values = sorted(set(args.m_values))
    _warm_up(d_values[0], m_values[0], args.warmups, device, mixed_options)

    raw_records: list[dict[str, Any]] = []
    for d in d_values:
        for m in m_values:
            master_problem = paper_meb_problem(
                m,
                d,
                dtype=torch.float64,
                device="cpu",
            )
            if master_problem.dtype != torch.float64:
                raise RuntimeError("master benchmark data must remain float64")
            problem = master_problem.to(device=device)

            print(f"\n{device.type.upper()} MEB instance: m={m}, d={d}", flush=True)
            for repeat in range(1, args.repeats + 1):
                direct = _measure_ripalm(problem, device)
                direct.update(d=d, m=m, repeat=repeat)
                raw_records.append(direct)
                _print_record(direct)

                mixed = _measure_mixed(problem, device, mixed_options)
                mixed.update(d=d, m=m, repeat=repeat)
                raw_records.append(mixed)
                _print_record(mixed)

    summary = _summarize(raw_records)
    metadata = _metadata(args, d_values, m_values, device)
    output_stem = args.output_stem or OUTPUT_DIR / f"meb_{device.type}_framework_records"
    _save(output_stem, metadata, raw_records, summary)


def _warm_up(
    d: int,
    m: int,
    warmups: int,
    device: torch.device,
    mixed_options: MixedPrecisionOptions,
) -> None:
    if warmups == 0:
        return
    problem = paper_meb_problem(
        min(m, 1000), min(d, 20), dtype=torch.float64, device=device
    )
    print(f"Warming up {device.type.upper()} solvers ({warmups} run(s) each)...", flush=True)
    for _ in range(warmups):
        ripalm(problem, options=RIPALM_OPTIONS)
        mixed_precision_ripalm(problem, options=mixed_options)
    _synchronize(device)


def _measure_ripalm(problem: Any, device: torch.device) -> dict[str, Any]:
    _synchronize(device)
    start = time.perf_counter()
    result = ripalm(problem, options=RIPALM_OPTIONS)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "solver": f"{device.type.upper()} ripALM",
        "elapsed_sec": elapsed,
        "radius": float(result.state.r),
        "max_violation": float(problem.violation(result.state).clamp_min(0).max()),
        "kkt_residual": float(result.residual.value),
        "outer_iterations": result.outer_iterations,
        "inner_iterations": result.inner_iterations,
        "low_outer_iterations": None,
        "low_inner_iterations": None,
        "high_outer_iterations": None,
        "high_inner_iterations": None,
        "kept": None,
        "refinements": None,
        "accepted": None,
        "converged": bool(result.converged),
    }


def _measure_mixed(
    problem: Any,
    device: torch.device,
    options: MixedPrecisionOptions,
) -> dict[str, Any]:
    _synchronize(device)
    start = time.perf_counter()
    result = mixed_precision_ripalm(problem, options=options)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "solver": f"{device.type.upper()} mp-ripALM",
        "elapsed_sec": elapsed,
        "radius": float(result.state.r),
        "max_violation": float(result.max_violation),
        "kkt_residual": float(result.residual.value),
        "outer_iterations": None,
        "inner_iterations": None,
        "low_outer_iterations": result.low_result.outer_iterations,
        "low_inner_iterations": result.low_result.inner_iterations,
        "high_outer_iterations": result.high_result.outer_iterations,
        "high_inner_iterations": result.high_result.inner_iterations,
        "kept": result.reduced_problem.m,
        "refinements": result.refinements,
        "accepted": bool(result.accepted),
        "converged": bool(result.converged),
    }


def _summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    keys = sorted({(record["d"], record["m"], record["solver"]) for record in records})
    for d, m, solver in keys:
        group = [
            record
            for record in records
            if record["d"] == d and record["m"] == m and record["solver"] == solver
        ]
        times = [float(record["elapsed_sec"]) for record in group]
        summary.append(
            {
                "d": d,
                "m": m,
                "solver": solver,
                "repeats": len(group),
                "median_sec": statistics.median(times),
                "min_sec": min(times),
                "max_sec": max(times),
                "median_radius": statistics.median(float(record["radius"]) for record in group),
                "max_violation": max(float(record["max_violation"]) for record in group),
                "max_kkt_residual": max(float(record["kkt_residual"]) for record in group),
                "kept": group[0]["kept"],
                "refinements": group[0]["refinements"],
                "all_accepted": all(record["accepted"] is not False for record in group),
                "all_converged": all(bool(record["converged"]) for record in group),
            }
        )
    return summary


def _metadata(
    args: argparse.Namespace,
    d_values: list[int],
    m_values: list[int],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": device.type,
        "master_dtype": "torch.float64",
        "low_dtype": "torch.float32",
        "high_dtype": "torch.float64",
        "screening_ratio": MIXED_PRECISION_OPTIONS.screening_ratio,
        "d_values": d_values,
        "m_values": m_values,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "baselines_rerun": False,
    }


def _save(
    output_stem: Path,
    metadata: dict[str, Any],
    raw_records: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    raw_csv_path = output_stem.with_name(f"{output_stem.name}_raw.csv")
    summary_csv_path = output_stem.with_name(f"{output_stem.name}_summary.csv")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "records": raw_records, "summary": summary}, handle, indent=2)
    _write_csv(raw_csv_path, raw_records)
    _write_csv(summary_csv_path, summary)

    print(f"\nSaved {json_path}")
    print(f"Saved {raw_csv_path}")
    print(f"Saved {summary_csv_path}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _print_record(record: dict[str, Any]) -> None:
    suffix = ""
    if record["kept"] is not None:
        suffix = f" kept={record['kept']} refinements={record['refinements']}"
    print(
        f"  {record['solver']:<14} repeat={record['repeat']} "
        f"time={record['elapsed_sec']:.4f}s residual={record['kkt_residual']:.3e}"
        f" violation={record['max_violation']:.3e}{suffix}",
        flush=True,
    )


if __name__ == "__main__":
    main()
