"""GPU implementation and solver-parameter diagnostics.

This script reruns only ripALM and mp-ripALM. It records synchronized phase
times, nonlinear/line-search work, one-factor parameter sensitivity, and
representative PyTorch CPU/CUDA profiler events.
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
from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, profile

RESULT_DIR = Path(__file__).resolve().parent / "results" / "local"

from .benchmark_synthetic import MIXED_PRECISION_OPTIONS, RIPALM_OPTIONS
from .datasets import paper_meb_problem
from pymeb import MixedPrecisionOptions, RipALMOptions, mixed_precision_ripalm, ripalm


PHASE_D_VALUES = [20, 30, 40, 50]
PHASE_M = 100_000
SENSITIVITY_D = 50
SENSITIVITY_M = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "phases", "sensitivity", "profile"),
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--phase-d-values", nargs="+", type=int, default=PHASE_D_VALUES)
    parser.add_argument("--phase-m", type=int, default=PHASE_M)
    parser.add_argument("--sensitivity-d", type=int, default=SENSITIVITY_D)
    parser.add_argument("--sensitivity-m", type=int, default=SENSITIVITY_M)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if args.mode in ("all", "phases"):
        run_phase_breakdown(args, device)
    if args.mode in ("all", "sensitivity"):
        run_sensitivity(args, device)
    if args.mode in ("all", "profile"):
        if device.type != "cuda":
            raise ValueError("GPU profiling requires --device cuda")
        run_profiles(args, device)


def run_phase_breakdown(args: argparse.Namespace, device: torch.device) -> None:
    records: list[dict[str, Any]] = []
    mixed_options = _mixed_options(device, record_phase_timings=True)
    for d in sorted(set(args.phase_d_values)):
        problem = _problem(args.phase_m, d, device)
        _warm_up(problem, device, args.warmups, RIPALM_OPTIONS, mixed_options)
        print(f"\nPhase breakdown: d={d}, m={args.phase_m}", flush=True)
        for repeat in range(1, args.repeats + 1):
            direct = _measure_direct(problem, device, RIPALM_OPTIONS)
            direct.update(experiment="phase_breakdown", d=d, m=args.phase_m, repeat=repeat)
            records.append(direct)
            _print_record(direct)

            mixed = _measure_mixed(problem, device, mixed_options)
            mixed.update(experiment="phase_breakdown", d=d, m=args.phase_m, repeat=repeat)
            records.append(mixed)
            _print_record(mixed)

    summary = _summarize(records, ("experiment", "d", "m", "solver"))
    _save_records(
        RESULT_DIR / f"{device.type}_phase_breakdown",
        _metadata(args, device, "phase_breakdown"),
        records,
        summary,
    )


def run_sensitivity(args: argparse.Namespace, device: torch.device) -> None:
    problem = _problem(args.sensitivity_m, args.sensitivity_d, device)
    records: list[dict[str, Any]] = []

    experiments: list[tuple[str, float, str, RipALMOptions | MixedPrecisionOptions]] = []
    for value in (0.95, 0.98, 0.995):
        experiments.append(
            (
                "screening_ratio",
                value,
                "mp-ripALM",
                _mixed_options(
                    device,
                    record_phase_timings=True,
                    screening_ratio=value,
                ),
            )
        )
    for value in (1e-1, 1e-2, 1e-3):
        low = replace(MIXED_PRECISION_OPTIONS.low_options, outer_tol=value)
        experiments.append(
            (
                "low_kkt_tolerance",
                value,
                "mp-ripALM",
                _mixed_options(
                    device,
                    record_phase_timings=True,
                    low_options=low,
                ),
            )
        )
    for value in (0.5, 1.0, 2.0):
        experiments.append(
            (
                "initial_penalty",
                value,
                "ripALM",
                replace(RIPALM_OPTIONS, sigma=value),
            )
        )

    for parameter, value, solver, options in experiments:
        if solver == "ripALM":
            assert isinstance(options, RipALMOptions)
            _warm_up_direct(problem, device, args.warmups, options)
        else:
            assert isinstance(options, MixedPrecisionOptions)
            _warm_up_mixed(problem, device, args.warmups, options)
        print(f"\nSensitivity: {parameter}={value:g} ({solver})", flush=True)
        for repeat in range(1, args.repeats + 1):
            if solver == "ripALM":
                record = _measure_direct(problem, device, options)
            else:
                record = _measure_mixed(problem, device, options)
            record.update(
                experiment="parameter_sensitivity",
                parameter=parameter,
                parameter_value=value,
                d=args.sensitivity_d,
                m=args.sensitivity_m,
                repeat=repeat,
            )
            records.append(record)
            _print_record(record)

    summary = _summarize(
        records,
        ("experiment", "parameter", "parameter_value", "d", "m", "solver"),
    )
    _save_records(
        RESULT_DIR / f"{device.type}_parameter_sensitivity",
        _metadata(args, device, "parameter_sensitivity"),
        records,
        summary,
    )


def run_profiles(args: argparse.Namespace, device: torch.device) -> None:
    cases: list[tuple[str, int, int, str]] = [
        ("constraint_dominated", 20, 100_000, "ripALM"),
        ("paper_largest_dimension", 50, 100_000, "mp-ripALM"),
        ("hessian_heavier", 500, 5_000, "ripALM"),
    ]
    all_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    tables: list[str] = []

    for case, d, m, solver in cases:
        problem = _problem(m, d, device)
        mixed_options = _mixed_options(device, record_phase_timings=False)
        if solver == "ripALM":
            _warm_up_direct(problem, device, args.warmups, RIPALM_OPTIONS)
            run: Callable[[], Any] = lambda: ripalm(problem, options=RIPALM_OPTIONS)
        else:
            _warm_up_mixed(problem, device, args.warmups, mixed_options)
            run = lambda: mixed_precision_ripalm(problem, options=mixed_options)

        print(f"\nProfiling {case}: {solver}, d={d}, m={m}", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        started_at = time.perf_counter()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            result = run()
            _synchronize(device)
        elapsed_sec = time.perf_counter() - started_at

        events = [_event_record(item) for item in prof.key_averages()]
        events.sort(key=lambda item: item["self_device_ms"], reverse=True)
        for event in events:
            event.update(case=case, solver=solver, d=d, m=m)
        all_events.extend(events)

        top_events = [event for event in events if event["self_device_ms"] > 0.0][:15]
        device_total_ms = sum(event["self_device_ms"] for event in events)
        summary = {
            "case": case,
            "solver": solver,
            "d": d,
            "m": m,
            "elapsed_sec_with_profiler": elapsed_sec,
            "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "converged": bool(result.converged),
            "kkt_residual": float(result.residual.value),
            "device_self_time_ms_sum": device_total_ms,
            "top_device_events": top_events,
        }
        if solver == "mp-ripALM":
            summary.update(
                kept=result.reduced_problem.m,
                refinements=result.refinements,
                accepted=bool(result.accepted),
            )
        summaries.append(summary)
        tables.append(_format_profile_table(summary, top_events))

    metadata = _metadata(args, device, "gpu_profile")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_DIR / "gpu_profile.json"
    events_path = RESULT_DIR / "gpu_profile_events.csv"
    table_path = RESULT_DIR / "gpu_profile_tables.txt"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "profiles": summaries}, handle, indent=2)
    _write_csv(events_path, all_events)
    table_path.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"Saved {json_path}\nSaved {events_path}\nSaved {table_path}")


def _mixed_options(
    device: torch.device,
    *,
    record_phase_timings: bool,
    **changes: Any,
) -> MixedPrecisionOptions:
    return replace(
        MIXED_PRECISION_OPTIONS,
        device=device,
        record_phase_timings=record_phase_timings,
        **changes,
    )


def _problem(m: int, d: int, device: torch.device) -> Any:
    master = paper_meb_problem(m, d, dtype=torch.float64, device="cpu")
    if master.dtype != torch.float64:
        raise RuntimeError("master data must be float64")
    return master.to(device=device)


def _warm_up(
    problem: Any,
    device: torch.device,
    warmups: int,
    direct_options: RipALMOptions,
    mixed_options: MixedPrecisionOptions,
) -> None:
    _warm_up_direct(problem, device, warmups, direct_options)
    _warm_up_mixed(problem, device, warmups, mixed_options)


def _warm_up_direct(
    problem: Any,
    device: torch.device,
    warmups: int,
    options: RipALMOptions,
) -> None:
    for _ in range(warmups):
        ripalm(problem, options=options)
    _synchronize(device)


def _warm_up_mixed(
    problem: Any,
    device: torch.device,
    warmups: int,
    options: MixedPrecisionOptions,
) -> None:
    for _ in range(warmups):
        mixed_precision_ripalm(problem, options=options)
    _synchronize(device)


def _measure_direct(
    problem: Any,
    device: torch.device,
    options: RipALMOptions,
) -> dict[str, Any]:
    _synchronize(device)
    started_at = time.perf_counter()
    result = ripalm(problem, options=options)
    _synchronize(device)
    elapsed_sec = time.perf_counter() - started_at
    return {
        "solver": "ripALM",
        "elapsed_sec": elapsed_sec,
        "outer_iterations": result.outer_iterations,
        "inner_iterations": result.inner_iterations,
        "line_search_evaluations": result.line_search_evaluations,
        "accepted_full_steps": result.accepted_full_steps,
        "ms_per_inner_iteration": 1000.0 * elapsed_sec / max(result.inner_iterations, 1),
        "line_search_evaluations_per_inner": (
            result.line_search_evaluations / max(result.inner_iterations, 1)
        ),
        "radius": float(result.state.r),
        "max_violation": float(problem.violation(result.state).clamp_min(0).max()),
        "kkt_residual": float(result.residual.value),
        "converged": bool(result.converged),
        "kept": None,
        "refinements": None,
        "accepted": None,
    }


def _measure_mixed(
    problem: Any,
    device: torch.device,
    options: MixedPrecisionOptions,
) -> dict[str, Any]:
    _synchronize(device)
    started_at = time.perf_counter()
    result = mixed_precision_ripalm(problem, options=options)
    _synchronize(device)
    elapsed_sec = time.perf_counter() - started_at
    if result.timings is None:
        raise RuntimeError("phase timing was not enabled")
    timings = asdict(result.timings)
    initial_kept = result.refinement_history[0].kept_constraints
    component_sum = sum(
        timings[key]
        for key in (
            "low_conversion_sec",
            "low_solve_sec",
            "screening_sec",
            "high_setup_sec",
            "high_solve_sec",
            "full_feasibility_check_sec",
            "refinement_update_sec",
            "final_kkt_sec",
        )
    )
    return {
        "solver": "mp-ripALM",
        "elapsed_sec": elapsed_sec,
        "instrumented_total_sec": timings["total_sec"],
        "phase_sum_sec": component_sum,
        "unattributed_sec": timings["total_sec"] - component_sum,
        "low_solve_fraction": timings["low_solve_sec"] / timings["total_sec"],
        "high_solve_fraction": timings["high_solve_sec"] / timings["total_sec"],
        "other_fraction": 1.0
        - (timings["low_solve_sec"] + timings["high_solve_sec"])
        / timings["total_sec"],
        "low_conversion_sec": timings["low_conversion_sec"],
        "low_solve_sec": timings["low_solve_sec"],
        "screening_sec": timings["screening_sec"],
        "high_setup_sec": timings["high_setup_sec"],
        "high_solve_sec": timings["high_solve_sec"],
        "full_feasibility_check_sec": timings["full_feasibility_check_sec"],
        "refinement_update_sec": timings["refinement_update_sec"],
        "final_kkt_sec": timings["final_kkt_sec"],
        "low_outer_iterations": result.low_result.outer_iterations,
        "low_inner_iterations": result.low_result.inner_iterations,
        "low_line_search_evaluations": result.low_result.line_search_evaluations,
        "low_accepted_full_steps": result.low_result.accepted_full_steps,
        "high_outer_iterations": result.high_result.outer_iterations,
        "high_inner_iterations": result.high_result.inner_iterations,
        "high_line_search_evaluations": result.high_result.line_search_evaluations,
        "high_accepted_full_steps": result.high_result.accepted_full_steps,
        "low_ms_per_inner_iteration": (
            1000.0 * timings["low_solve_sec"] / max(result.low_result.inner_iterations, 1)
        ),
        "high_ms_per_inner_iteration": (
            1000.0 * timings["high_solve_sec"] / max(result.high_result.inner_iterations, 1)
        ),
        "radius": float(result.state.r),
        "max_violation": float(result.max_violation),
        "kkt_residual": float(result.residual.value),
        "converged": bool(result.converged),
        "initial_kept": initial_kept,
        "initial_kept_fraction": initial_kept / problem.m,
        "kept": result.reduced_problem.m,
        "kept_fraction": result.reduced_problem.m / problem.m,
        "reinserted_constraints": result.reduced_problem.m - initial_kept,
        "refinements": result.refinements,
        "accepted": bool(result.accepted),
    }


def _summarize(
    records: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    keys = sorted({tuple(record[field] for field in group_fields) for record in records})
    summary: list[dict[str, Any]] = []
    for key in keys:
        group = [
            record
            for record in records
            if tuple(record[field] for field in group_fields) == key
        ]
        row: dict[str, Any] = dict(zip(group_fields, key))
        row["repeats"] = len(group)
        for field in group[0]:
            if field in group_fields or field in ("repeat", "solver"):
                continue
            values = [record.get(field) for record in group]
            numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if len(numeric) == len(values):
                row[f"median_{field}"] = statistics.median(numeric)
                row[f"min_{field}"] = min(numeric)
                row[f"max_{field}"] = max(numeric)
            elif all(isinstance(value, bool) for value in values):
                row[f"all_{field}"] = all(values)
            elif all(value is None for value in values):
                row[field] = None
        summary.append(row)
    return summary


def _event_record(item: Any) -> dict[str, Any]:
    self_device_us = getattr(item, "self_device_time_total", 0.0)
    device_us = getattr(item, "device_time_total", 0.0)
    if self_device_us == 0.0:
        self_device_us = getattr(item, "self_cuda_time_total", 0.0)
    if device_us == 0.0:
        device_us = getattr(item, "cuda_time_total", 0.0)
    return {
        "event": item.key,
        "count": item.count,
        "self_cpu_ms": item.self_cpu_time_total / 1000.0,
        "cpu_total_ms": item.cpu_time_total / 1000.0,
        "self_device_ms": self_device_us / 1000.0,
        "device_total_ms": device_us / 1000.0,
        "cpu_memory_bytes": item.cpu_memory_usage,
        "device_memory_bytes": getattr(item, "device_memory_usage", 0),
        "flops": item.flops,
    }


def _format_profile_table(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
) -> str:
    lines = [
        (
            f"{summary['case']}: {summary['solver']} d={summary['d']} m={summary['m']} "
            f"peak={summary['peak_cuda_memory_mb']:.1f} MiB"
        ),
        "event                                           self CUDA ms  total CUDA ms  count",
    ]
    for event in events:
        lines.append(
            f"{event['event'][:46]:46s} {event['self_device_ms']:12.3f} "
            f"{event['device_total_ms']:14.3f} {event['count']:6d}"
        )
    return "\n".join(lines)


def _metadata(
    args: argparse.Namespace,
    device: torch.device,
    experiment: str,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "command_arguments": vars(args),
        "device": device.type,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "master_dtype": "torch.float64",
        "low_dtype": "torch.float32",
        "high_dtype": "torch.float64",
        "repeats": args.repeats,
        "warmups": args.warmups,
        "baselines_rerun": False,
        "phase_timing_policy": "wall clock with CUDA synchronization at every phase boundary",
        "direct_options": _jsonable(asdict(RIPALM_OPTIONS)),
        "mixed_options": _jsonable(asdict(MIXED_PRECISION_OPTIONS)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_records(
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
            {"metadata": metadata, "records": _jsonable(records), "summary": _jsonable(summary)},
            handle,
            indent=2,
        )
    _write_csv(raw_path, records)
    _write_csv(summary_path, summary)
    print(f"Saved {json_path}\nSaved {raw_path}\nSaved {summary_path}")


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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _print_record(record: dict[str, Any]) -> None:
    suffix = ""
    if record.get("kept") is not None:
        suffix = f" kept={record['kept']} refinements={record['refinements']}"
    print(
        f"  {record['solver']:<10} repeat={record.get('repeat', 0)} "
        f"time={record['elapsed_sec']:.4f}s KKT={record['kkt_residual']:.3e}{suffix}",
        flush=True,
    )


if __name__ == "__main__":
    main()
