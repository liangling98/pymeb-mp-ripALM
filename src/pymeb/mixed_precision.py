"""Two-phase mixed-precision ripALM driver."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import torch
from torch.profiler import record_function

from .problem import MEBProblem, MEBState
from .ripalm import RipALMOptions, RipALMResidual, RipALMResult, ripalm, ripalm_residual


@dataclass(frozen=True)
class MixedPrecisionOptions:
    """Options for low-precision screening followed by high-precision refinement."""

    screening_ratio: float = 0.98
    min_keep: int = 1
    low_dtype: torch.dtype = torch.float32
    high_dtype: torch.dtype = torch.float64
    device: Optional[Union[str, torch.device]] = None
    low_options: Optional[RipALMOptions] = None
    high_options: Optional[RipALMOptions] = None
    feasibility_abs_tol: float = 1e-8
    max_refinements: int = 2
    record_phase_timings: bool = False


@dataclass(frozen=True)
class MixedPrecisionPhaseTimings:
    """Synchronized wall-time breakdown for an instrumented mixed solve."""

    total_sec: float
    low_conversion_sec: float
    low_solve_sec: float
    screening_sec: float
    high_setup_sec: float
    high_solve_sec: float
    full_feasibility_check_sec: float
    refinement_update_sec: float
    final_kkt_sec: float
    high_solve_rounds_sec: tuple[float, ...]
    full_check_rounds_sec: tuple[float, ...]


@dataclass(frozen=True)
class MixedPrecisionRefinementInfo:
    """One high-precision reduced solve and full feasibility check."""

    round: int
    kept_constraints: int
    max_violation: torch.Tensor
    feasibility_tol: torch.Tensor
    accepted: bool


@dataclass(frozen=True)
class MixedPrecisionResult:
    """Result of the two-phase mixed-precision solver."""

    state: MEBState
    residual: RipALMResidual
    low_result: RipALMResult
    high_result: RipALMResult
    full_problem: MEBProblem
    reduced_problem: MEBProblem
    kept_indices: torch.Tensor
    screening_threshold: torch.Tensor
    max_violation: torch.Tensor
    feasibility_tol: torch.Tensor
    refinements: int
    accepted: bool
    converged: bool
    refinement_history: tuple[MixedPrecisionRefinementInfo, ...]
    timings: Optional[MixedPrecisionPhaseTimings]


@torch.no_grad()
def mixed_precision_ripalm(
    problem: MEBProblem,
    *,
    options: Optional[MixedPrecisionOptions] = None,
) -> MixedPrecisionResult:
    """Run float32 screening, then solve the reduced problem in higher precision."""

    opts = options or MixedPrecisionOptions()
    _validate_options(problem, opts)

    low_options = opts.low_options or _default_low_options()
    high_options = opts.high_options or _default_high_options()
    device = torch.device(opts.device) if opts.device is not None else problem.device
    timer = _PhaseTimer(device, enabled=opts.record_phase_timings)
    timer.start()

    with timer.phase("mp.low_conversion"):
        low_problem = problem.to(dtype=opts.low_dtype, device=device)
    with timer.phase("mp.low_solve"):
        low_result = ripalm(low_problem, options=low_options)
    with timer.phase("mp.screening"):
        kept_low, threshold = screen_by_boundary_ratio(
            low_problem,
            low_result.state,
            ratio=opts.screening_ratio,
            min_keep=opts.min_keep,
        )

    with timer.phase("mp.high_setup"):
        full_problem = problem.to(dtype=opts.high_dtype, device=device)
        kept = kept_low.to(device=full_problem.device)
        full_state = low_result.state.to(dtype=opts.high_dtype, device=device)

        high_result: Optional[RipALMResult] = None
        reduced_problem = full_problem.subset(kept)
        reduced_state = full_state.subset(kept)
        max_violation = torch.zeros((), dtype=full_problem.dtype, device=full_problem.device)
        feasibility_tol = torch.zeros_like(max_violation)
    accepted = False
    history: list[MixedPrecisionRefinementInfo] = []

    for refinement in range(opts.max_refinements + 1):
        with timer.phase("mp.high_solve"):
            high_result = ripalm(reduced_problem, state=reduced_state, options=high_options)
        with timer.phase("mp.full_feasibility_check"):
            full_state = high_result.state.align_constraints(reduced_problem, full_problem)
            violation = full_problem.violation(full_state)
            max_violation = violation.max().clamp_min(0.0)
            feasibility_tol = torch.as_tensor(
                opts.feasibility_abs_tol,
                dtype=full_problem.dtype,
                device=full_problem.device,
            )
            accepted = bool(max_violation <= feasibility_tol)
        history.append(
            MixedPrecisionRefinementInfo(
                round=refinement,
                kept_constraints=reduced_problem.m,
                max_violation=max_violation,
                feasibility_tol=feasibility_tol,
                accepted=accepted,
            )
        )
        if accepted or refinement == opts.max_refinements:
            break

        with timer.phase("mp.refinement_update"):
            violated = torch.nonzero(violation > feasibility_tol, as_tuple=False).flatten()
            kept = torch.sort(torch.unique(torch.cat((kept, violated)))).values
            reduced_problem = full_problem.subset(kept)
            reduced_state = full_state.subset(kept)

    if high_result is None:
        raise RuntimeError("high-precision phase did not run.")

    with timer.phase("mp.final_kkt"):
        residual = ripalm_residual(full_problem, full_state)
        screening_threshold = threshold.to(dtype=opts.high_dtype, device=device)
    timings = timer.finish()
    return MixedPrecisionResult(
        state=full_state,
        residual=residual,
        low_result=low_result,
        high_result=high_result,
        full_problem=full_problem,
        reduced_problem=reduced_problem,
        kept_indices=kept,
        screening_threshold=screening_threshold,
        max_violation=max_violation,
        feasibility_tol=feasibility_tol,
        refinements=len(history) - 1,
        accepted=accepted,
        converged=accepted and high_result.converged,
        refinement_history=tuple(history),
        timings=timings,
    )


class _PhaseTimer:
    def __init__(self, device: torch.device, *, enabled: bool) -> None:
        self.device = device
        self.enabled = enabled
        self._started_at: Optional[float] = None
        self._values: dict[str, list[float]] = {}

    def start(self) -> None:
        if self.enabled:
            self._synchronize()
            self._started_at = time.perf_counter()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        with record_function(name):
            started_at: Optional[float] = None
            if self.enabled:
                self._synchronize()
                started_at = time.perf_counter()
            yield
            if self.enabled:
                self._synchronize()
                assert started_at is not None
                self._values.setdefault(name, []).append(time.perf_counter() - started_at)

    def finish(self) -> Optional[MixedPrecisionPhaseTimings]:
        if not self.enabled:
            return None
        self._synchronize()
        assert self._started_at is not None
        total_sec = time.perf_counter() - self._started_at
        high_rounds = tuple(self._values.get("mp.high_solve", ()))
        check_rounds = tuple(self._values.get("mp.full_feasibility_check", ()))
        return MixedPrecisionPhaseTimings(
            total_sec=total_sec,
            low_conversion_sec=self._sum("mp.low_conversion"),
            low_solve_sec=self._sum("mp.low_solve"),
            screening_sec=self._sum("mp.screening"),
            high_setup_sec=self._sum("mp.high_setup"),
            high_solve_sec=sum(high_rounds),
            full_feasibility_check_sec=sum(check_rounds),
            refinement_update_sec=self._sum("mp.refinement_update"),
            final_kkt_sec=self._sum("mp.final_kkt"),
            high_solve_rounds_sec=high_rounds,
            full_check_rounds_sec=check_rounds,
        )

    def _sum(self, name: str) -> float:
        return sum(self._values.get(name, ()))

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


@torch.no_grad()
def screen_by_boundary_ratio(
    problem: MEBProblem,
    state: MEBState,
    *,
    ratio: float = 0.95,
    min_keep: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep balls whose reach is at least ``ratio`` times the current radius."""

    if not (0.0 < ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1].")
    if min_keep <= 0:
        raise ValueError("min_keep must be positive.")

    state.validate_for(problem)
    reach = problem.radii + problem.distances(state.c)
    threshold = torch.as_tensor(ratio, dtype=problem.dtype, device=problem.device) * state.r
    kept = torch.nonzero(reach >= threshold, as_tuple=False).flatten()

    keep_at_least = min(min_keep, problem.m)
    if kept.numel() < keep_at_least:
        kept = torch.cat((kept, torch.topk(reach, k=keep_at_least).indices))
    return torch.sort(torch.unique(kept)).values, threshold


def _default_low_options() -> RipALMOptions:
    return RipALMOptions(
        outer_tol=1e-4,
        gradient_tol=1e-4,
        max_outer=20,
        max_inner=10,
        fixed_outer_iterations=True,
        fixed_inner_iterations=True,
        print_history=False,
    )


def _default_high_options() -> RipALMOptions:
    return RipALMOptions(
        outer_tol=1e-8,
        gradient_tol=1e-10,
        max_outer=100,
        max_inner=20,
        fixed_inner_iterations=True,
        print_history=False,
    )


def _validate_options(problem: MEBProblem, opts: MixedPrecisionOptions) -> None:
    if problem.m == 0:
        raise ValueError("problem must contain at least one ball.")
    if not (0.0 < opts.screening_ratio <= 1.0):
        raise ValueError("screening_ratio must be in (0, 1].")
    if opts.min_keep <= 0:
        raise ValueError("min_keep must be positive.")
    if opts.feasibility_abs_tol < 0.0:
        raise ValueError("feasibility_abs_tol must be nonnegative.")
    if opts.max_refinements < 0:
        raise ValueError("max_refinements must be nonnegative.")
