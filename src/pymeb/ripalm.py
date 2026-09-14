"""Main ripALM driver built from the PyTorch core primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.profiler import record_function

from ._types import TensorLike
from .newton import line_search_newton_step, semismooth_newton_direction
from .problem import MEBProblem, MEBState
from .subproblem import (
    RelativeInexactCheck,
    RipALMSubproblemCandidate,
    check_relative_inexact_candidate,
    evaluate_subproblem_candidate,
)


@dataclass(frozen=True)
class RipALMOptions:
    """Numerical controls for :func:`ripalm`.

    ``sigma`` is the initial penalty, ``tau`` controls the proximal term, and
    ``rho`` controls the relative inner stopping rule. Set ``print_history``
    to display one diagnostic row per outer iteration.
    """

    sigma: float = 1e0
    sigma_growth_factor: float = 2.0
    tau: float = 1e-4
    rho: float = 5e-1
    outer_tol: float = 1e-8
    gradient_tol: Optional[float] = None
    max_outer: int = 100
    max_inner: int = 20
    fixed_outer_iterations: bool = False
    fixed_inner_iterations: bool = False
    line_search_max_iter: int = 10
    print_history: bool = False


@dataclass(frozen=True)
class RipALMResidual:
    value: torch.Tensor
    feasibility: torch.Tensor
    stationarity: torch.Tensor
    complementarity: torch.Tensor


@dataclass(frozen=True)
class RipALMIterationInfo:
    iteration: int
    inner_iterations: int
    line_search_evaluations: int
    accepted_full_steps: int
    sigma: torch.Tensor
    radius: torch.Tensor
    objective: torch.Tensor
    gradient_norm: torch.Tensor
    residual: RipALMResidual
    relative_lhs: torch.Tensor
    relative_rhs: torch.Tensor
    relative_satisfied: torch.Tensor
    accepted: torch.Tensor


@dataclass(frozen=True)
class RipALMResult:
    state: MEBState
    residual: RipALMResidual
    outer_iterations: int
    inner_iterations: int
    line_search_evaluations: int
    accepted_full_steps: int
    converged: bool
    history: tuple[RipALMIterationInfo, ...]


@torch.no_grad()
def ripalm(
    problem: MEBProblem,
    *,
    state: Optional[MEBState] = None,
    options: Optional[RipALMOptions] = None,
) -> RipALMResult:
    """Solve the MEB problem with ripALM and semismooth Newton subproblem solves."""

    opts = options or RipALMOptions()
    _validate_options(opts)

    current = MEBState.initial_feasible(problem) if state is None else state
    current.validate_for(problem)

    sigma = torch.as_tensor(opts.sigma, dtype=problem.dtype, device=problem.device)
    sigma_growth = torch.as_tensor(
        opts.sigma_growth_factor, dtype=problem.dtype, device=problem.device
    )
    tau = torch.as_tensor(opts.tau, dtype=problem.dtype, device=problem.device)
    rho = torch.as_tensor(opts.rho, dtype=problem.dtype, device=problem.device)
    gradient_tol = _gradient_tol(problem.dtype, opts.gradient_tol)

    total_inner = 0
    total_line_search_evaluations = 0
    total_accepted_full_steps = 0
    outer_iterations = 0
    converged = False
    history: list[RipALMIterationInfo] = []
    residual = ripalm_residual(problem, current)

    for outer in range(1, opts.max_outer + 1):
        check, inner_iters, line_search_evaluations, accepted_full_steps = _solve_subproblem(
            problem,
            current,
            sigma=sigma,
            tau=tau,
            rho=rho,
            gradient_tol=gradient_tol,
            max_inner=opts.max_inner,
            fixed_inner_iterations=opts.fixed_inner_iterations,
            line_search_max_iter=opts.line_search_max_iter,
        )
        total_inner += inner_iters
        total_line_search_evaluations += line_search_evaluations
        total_accepted_full_steps += accepted_full_steps
        outer_iterations = outer

        current = _outer_update(current, check.candidate, sigma)
        residual = ripalm_residual(problem, current)
        history.append(
            RipALMIterationInfo(
                iteration=outer,
                inner_iterations=inner_iters,
                line_search_evaluations=line_search_evaluations,
                accepted_full_steps=accepted_full_steps,
                sigma=sigma,
                radius=current.r,
                objective=check.candidate.objective,
                gradient_norm=check.gradient_norm,
                residual=residual,
                relative_lhs=check.lhs,
                relative_rhs=check.rhs,
                relative_satisfied=check.relative_satisfied,
                accepted=check.satisfied,
            )
        )

        if not opts.fixed_outer_iterations and bool(residual.value <= opts.outer_tol):
            converged = True
            break
        sigma = sigma * sigma_growth

    if opts.fixed_outer_iterations:
        converged = bool(residual.value <= opts.outer_tol)

    result = RipALMResult(
        state=current,
        residual=residual,
        outer_iterations=outer_iterations,
        inner_iterations=total_inner,
        line_search_evaluations=total_line_search_evaluations,
        accepted_full_steps=total_accepted_full_steps,
        converged=converged,
        history=tuple(history),
    )
    if opts.print_history:
        print_ripalm_history(result.history)
    return result


def print_ripalm_history(history: tuple[RipALMIterationInfo, ...]) -> None:
    """Print one compact line per ripALM outer iteration."""

    if not history:
        print("ripALM history: no iterations")
        return

    print(
        " iter inner        sigma        radius      residual          feas"
        "          stat          comp       ||g||      rel_lhs      rel_rhs"
    )
    for item in history:
        print(
            f"{item.iteration:5d}"
            f"{item.inner_iterations:6d}"
            f"{_scalar(item.sigma):13.3e}"
            f"{_scalar(item.radius):14.6e}"
            f"{_scalar(item.residual.value):14.6e}"
            f"{_scalar(item.residual.feasibility):14.6e}"
            f"{_scalar(item.residual.stationarity):14.6e}"
            f"{_scalar(item.residual.complementarity):14.6e}"
            f"{_scalar(item.gradient_norm):12.3e}"
            f"{_scalar(item.relative_lhs):13.3e}"
            f"{_scalar(item.relative_rhs):13.3e}"
        )


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


@torch.no_grad()
def ripalm_residual(problem: MEBProblem, state: MEBState) -> RipALMResidual:
    """Return a compact KKT-style residual for the MEB SOCP."""

    state.validate_for(problem)
    feasibility = problem.violation(state).max().clamp_min(0.0)
    stationarity = torch.linalg.vector_norm(
        torch.cat(((1.0 - state.u.sum()).reshape(1), -state.v.sum(dim=0)))
    )
    cone_t = state.r - problem.radii
    cone_x = state.c.unsqueeze(0) - problem.centers
    complementarity_rows = state.u * cone_t + (state.v * cone_x).sum(dim=1)
    complementarity = complementarity_rows.abs().sum()
    value = torch.stack((feasibility, stationarity, complementarity)).max()
    return RipALMResidual(
        value=value,
        feasibility=feasibility,
        stationarity=stationarity,
        complementarity=complementarity,
    )


def _solve_subproblem(
    problem: MEBProblem,
    state: MEBState,
    *,
    sigma: TensorLike,
    tau: TensorLike,
    rho: TensorLike,
    gradient_tol: float,
    max_inner: int,
    fixed_inner_iterations: bool,
    line_search_max_iter: int,
) -> tuple[RelativeInexactCheck, int, int, int]:
    line_search_evaluations = 0
    accepted_full_steps = 0
    with record_function("ripalm.evaluate_subproblem"):
        candidate = evaluate_subproblem_candidate(
            problem, state, state.r, state.c, sigma=sigma, tau=tau
        )

    if fixed_inner_iterations:
        for _ in range(max_inner):
            with record_function("ripalm.newton_direction"):
                newton = semismooth_newton_direction(
                    candidate,
                    sigma=sigma,
                    tau=tau,
                )
            with record_function("ripalm.line_search"):
                search = line_search_newton_step(
                    problem,
                    state,
                    candidate,
                    newton,
                    sigma=sigma,
                    tau=tau,
                    max_iter=line_search_max_iter,
                )
            candidate = search.candidate
            line_search_evaluations += search.iterations
            accepted_full_steps += _accepted_full_step(search, line_search_max_iter)
        with record_function("ripalm.inexact_check"):
            check = check_relative_inexact_candidate(
                problem,
                state,
                candidate,
                sigma=sigma,
                tau=tau,
                rho=rho,
                gradient_tol=gradient_tol,
            )
        return check, max_inner, line_search_evaluations, accepted_full_steps

    with record_function("ripalm.inexact_check"):
        check = check_relative_inexact_candidate(
            problem,
            state,
            candidate,
            sigma=sigma,
            tau=tau,
            rho=rho,
            gradient_tol=gradient_tol,
        )

    for inner in range(max_inner):
        if bool(check.satisfied):
            return check, inner, line_search_evaluations, accepted_full_steps

        with record_function("ripalm.newton_direction"):
            newton = semismooth_newton_direction(
                candidate,
                sigma=sigma,
                tau=tau,
            )
        with record_function("ripalm.line_search"):
            search = line_search_newton_step(
                problem,
                state,
                candidate,
                newton,
                sigma=sigma,
                tau=tau,
                max_iter=line_search_max_iter,
            )
        candidate = search.candidate
        line_search_evaluations += search.iterations
        accepted_full_steps += _accepted_full_step(search, line_search_max_iter)
        with record_function("ripalm.inexact_check"):
            check = check_relative_inexact_candidate(
                problem,
                state,
                candidate,
                sigma=sigma,
                tau=tau,
                rho=rho,
                gradient_tol=gradient_tol,
            )
        if bool(check.satisfied):
            return check, inner + 1, line_search_evaluations, accepted_full_steps

    return check, max_inner, line_search_evaluations, accepted_full_steps


def _accepted_full_step(search: object, line_search_max_iter: int) -> int:
    """Count a full step without adding a CUDA scalar synchronization."""

    iterations = getattr(search, "iterations")
    if iterations != 1:
        return 0
    if line_search_max_iter > 1:
        return 1
    return int(bool(getattr(search, "accepted")))


def _gradient_tol(dtype: torch.dtype, override: Optional[float]) -> float:
    if override is not None:
        return override
    return 1e-8 if dtype == torch.float64 else 1e-3


def _validate_options(options: RipALMOptions) -> None:
    if options.sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if options.sigma_growth_factor <= 0.0:
        raise ValueError("sigma_growth_factor must be positive.")
    if options.tau <= 0.0:
        raise ValueError("tau must be positive.")
    if options.rho < 0.0:
        raise ValueError("rho must be nonnegative.")
    if options.outer_tol < 0.0:
        raise ValueError("outer_tol must be nonnegative.")
    if options.gradient_tol is not None and options.gradient_tol < 0.0:
        raise ValueError("gradient_tol must be nonnegative.")
    if options.max_outer <= 0 or options.max_inner < 0:
        raise ValueError("max_outer must be positive and max_inner nonnegative.")
    if options.line_search_max_iter <= 0:
        raise ValueError("line_search_max_iter must be positive.")


def _outer_update(
    state: MEBState,
    candidate: RipALMSubproblemCandidate,
    sigma: torch.Tensor,
) -> MEBState:
    gradient = candidate.gradient
    return MEBState(
        candidate.r,
        candidate.c,
        candidate.u,
        candidate.v,
        alpha=state.alpha - sigma * gradient[0],
        beta=state.beta - sigma * gradient[1:],
    )
