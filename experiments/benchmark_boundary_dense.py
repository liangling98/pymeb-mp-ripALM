"""Worst-case screening benchmark with every input ball on the MEB boundary."""

from __future__ import annotations

import argparse
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from pymeb import MEBProblem

from .benchmark_real_data import (
    MIXED_PRECISION_OPTIONS,
    RIPALM_OPTIONS,
    _measure_direct,
    _measure_mixed,
    _print_record,
    _save,
    _summarize,
    _warm_up,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=50)
    parser.add_argument("--balls", type=int, default=100_000)
    parser.add_argument("--input-radius", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--screening-ratio", type=float, default=0.98)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "local" / "boundary_dense",
    )
    return parser.parse_args()


def boundary_dense_problem(
    *, dimension: int, balls: int, input_radius: float, seed: int
) -> MEBProblem:
    """Create antipodal unit-norm centers, making every ball exactly active."""

    if dimension <= 0 or balls <= 0 or balls % 2:
        raise ValueError("dimension must be positive and balls must be positive and even")
    if input_radius < 0:
        raise ValueError("input-radius must be nonnegative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(
        balls // 2, dimension, generator=generator, dtype=torch.float64
    )
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    centers = torch.cat((directions, -directions), dim=0)
    radii = torch.full((balls,), input_radius, dtype=torch.float64)
    return MEBProblem(centers, radii, copy=False)


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    master = boundary_dense_problem(
        dimension=args.dimension,
        balls=args.balls,
        input_radius=args.input_radius,
        seed=args.seed,
    )
    problem = master.to(device=device)
    mixed_options = replace(
        MIXED_PRECISION_OPTIONS,
        screening_ratio=args.screening_ratio,
        device=device,
        record_phase_timings=True,
    )
    _warm_up(problem, mixed_options, args.warmups, device)

    diagnostics = {
        "nonzero_radius_fraction": float(args.input_radius > 0),
        "radius_min": args.input_radius,
        "radius_median": args.input_radius,
        "radius_max": args.input_radius,
    }
    records = []
    dataset = "antipodal_boundary_dense_balls"
    print(f"\n{dataset}: m={master.m}, d={master.d}", flush=True)
    for repeat in range(1, args.repeats + 1):
        for measure in (
            lambda: _measure_direct(problem, device),
            lambda: _measure_mixed(problem, mixed_options, device),
        ):
            record = measure()
            record.update(
                dataset=dataset,
                m=master.m,
                d=master.d,
                repeat=repeat,
                **diagnostics,
            )
            records.append(record)
            _print_record(record)

    summary = _summarize(records)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": vars(args),
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "construction": (
            "paired antipodal unit-norm centers with a common input radius; "
            "the exact MEB center is zero and every input ball is active"
        ),
        "exact_radius": 1.0 + args.input_radius,
        "direct_options": asdict(RIPALM_OPTIONS),
        "mixed_options": asdict(mixed_options),
        "baselines_rerun": False,
    }
    _save(args.output_stem, metadata, records, summary)


if __name__ == "__main__":
    main()
