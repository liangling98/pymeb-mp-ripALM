"""Semismooth Newton direction and line-search utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._types import TensorLike
from .problem import MEBProblem, MEBState
from .subproblem import (
    RipALMSubproblemCandidate,
    evaluate_subproblem_candidate,
    subproblem_hessian_matrix,
)


@dataclass(frozen=True)
class NewtonDirection:
    """Linear solve result for a semismooth Newton search direction."""

    direction: torch.Tensor
    residual: torch.Tensor
    residual_norm: torch.Tensor
    iterations: int
    solve_ok: torch.Tensor

    @property
    def converged(self) -> torch.Tensor:
        return self.solve_ok == 1.0

    @property
    def dr(self) -> torch.Tensor:
        return self.direction[0]

    @property
    def dc(self) -> torch.Tensor:
        return self.direction[1:]


@dataclass(frozen=True)
class LineSearchResult:
    """Serial Armijo backtracking result for a semismooth Newton step."""

    candidate: RipALMSubproblemCandidate
    step_size: torch.Tensor
    accepted: torch.Tensor
    objective: torch.Tensor
    iterations: int


@torch.no_grad()
def semismooth_newton_direction(
    candidate: RipALMSubproblemCandidate,
    *,
    sigma: TensorLike,
    tau: TensorLike,
) -> NewtonDirection:
    """Find a semismooth Newton direction with an explicit dense solve."""

    b = -candidate.gradient
    hessian = subproblem_hessian_matrix(candidate, sigma=sigma, tau=tau)

    cholesky, info = torch.linalg.cholesky_ex(hessian)
    if bool(info == 0):
        x = torch.cholesky_solve(b.unsqueeze(1), cholesky).squeeze(1)
    else:
        x = torch.linalg.solve(hessian, b)

    residual = b - hessian.matmul(x)
    residual_norm = torch.linalg.vector_norm(residual)
    return NewtonDirection(
        direction=x,
        residual=residual,
        residual_norm=residual_norm,
        iterations=1,
        solve_ok=torch.ones((), dtype=b.dtype, device=b.device),
    )


@torch.no_grad()
def line_search_newton_step(
    problem: MEBProblem,
    state: MEBState,
    candidate: RipALMSubproblemCandidate,
    newton: NewtonDirection,
    *,
    sigma: TensorLike,
    tau: TensorLike,
    max_iter: int = 20,
    reduction: float = 0.5,
    sufficient_decrease: float = 1e-4,
) -> LineSearchResult:
    """Armijo backtracking for a semismooth Newton direction."""

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not (0.0 < reduction < 1.0):
        raise ValueError("reduction must be in (0, 1).")

    dtype = candidate.gradient.dtype
    device = candidate.gradient.device
    direction = torch.as_tensor(newton.direction, dtype=dtype, device=device).flatten()
    if direction.shape != candidate.gradient.shape:
        raise ValueError("newton.direction must have shape [d + 1].")

    sigma_t = torch.as_tensor(sigma, dtype=dtype, device=device).reshape(())
    tau_t = torch.as_tensor(tau, dtype=dtype, device=device).reshape(())
    directional_derivative = (candidate.gradient * direction).sum()

    step = torch.ones((), dtype=dtype, device=device)
    accepted = torch.zeros((), dtype=torch.bool, device=device)
    trial = candidate
    trial_step = step
    iterations = 0

    for iteration in range(1, max_iter + 1):
        trial_step = step
        trial_r = candidate.r + step * direction[0]
        trial_c = candidate.c + step * direction[1:]
        trial = evaluate_subproblem_candidate(
            problem, state, trial_r, trial_c, sigma=sigma_t, tau=tau_t
        )
        accepted = (
            trial.objective
            <= candidate.objective
            + sufficient_decrease * step * directional_derivative
        )
        iterations = iteration
        if bool(accepted):
            break
        step = step * reduction

    return LineSearchResult(
        candidate=trial,
        step_size=trial_step,
        accepted=accepted,
        objective=trial.objective,
        iterations=iterations,
    )
