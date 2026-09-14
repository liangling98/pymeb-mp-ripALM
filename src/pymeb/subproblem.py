"""ripALM subproblem evaluation and stopping checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._types import TensorLike
from .cone import SOCProjection, project_soc
from .problem import MEBProblem, MEBState


@dataclass(frozen=True)
class RipALMSubproblemCandidate:
    """Evaluation data for a candidate ``(r, c)`` in a ripALM subproblem."""

    r: torch.Tensor
    c: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    gradient: torch.Tensor
    objective: torch.Tensor
    cone_projection: SOCProjection

    def primal_vector(self) -> torch.Tensor:
        return torch.cat((self.r.reshape(1), self.c))


@dataclass(frozen=True)
class RelativeInexactCheck:
    """Result of the ripALM relative-type subproblem accuracy check."""

    candidate: RipALMSubproblemCandidate
    satisfied: torch.Tensor
    relative_satisfied: torch.Tensor
    gradient_norm: torch.Tensor
    gradient_satisfied: torch.Tensor
    lhs: torch.Tensor
    rhs: torch.Tensor
    dual_step_norm_sq: torch.Tensor
    primal_step_norm_sq: torch.Tensor

    @property
    def gradient(self) -> torch.Tensor:
        return self.candidate.gradient

    @property
    def u_next(self) -> torch.Tensor:
        return self.candidate.u

    @property
    def v_next(self) -> torch.Tensor:
        return self.candidate.v


@torch.no_grad()
def evaluate_subproblem_candidate(
    problem: MEBProblem,
    state: MEBState,
    r: TensorLike,
    c: TensorLike,
    *,
    sigma: TensorLike,
    tau: TensorLike,
) -> RipALMSubproblemCandidate:
    """Evaluate ``Psi_k``, its gradient, and the next conic multipliers."""

    state.validate_for(problem)
    r_candidate = torch.as_tensor(
        r, dtype=problem.dtype, device=problem.device
    ).reshape(())
    c_candidate = torch.as_tensor(
        c, dtype=problem.dtype, device=problem.device
    ).flatten()
    sigma_t = torch.as_tensor(
        sigma, dtype=problem.dtype, device=problem.device
    ).reshape(())
    tau_t = torch.as_tensor(tau, dtype=problem.dtype, device=problem.device).reshape(())

    if c_candidate.shape != state.c.shape:
        raise ValueError("candidate c must have shape [d].")

    cone_points = torch.cat(
        (
            (problem.radii - r_candidate + state.u / sigma_t).unsqueeze(-1),
            problem.centers - c_candidate.unsqueeze(0) + state.v / sigma_t,
        ),
        dim=1,
    )
    cone_projection = project_soc(cone_points, return_info=True)
    projected = cone_projection.value
    dual_next = sigma_t * projected
    u_next = dual_next[:, 0]
    v_next = dual_next[:, 1:]

    primal_r_step = r_candidate - state.r
    primal_c_step = c_candidate - state.c
    grad_r = 1.0 - u_next.sum() + (tau_t / sigma_t) * primal_r_step
    grad_c = -v_next.sum(dim=0) + (tau_t / sigma_t) * primal_c_step
    gradient = torch.cat((grad_r.reshape(1), grad_c))
    projected_norm_sq = (projected * projected).sum()
    dual_norm_sq = (state.u * state.u).sum() + (state.v * state.v).sum()
    primal_step_norm_sq = primal_r_step.square() + (primal_c_step * primal_c_step).sum()
    objective = (
        r_candidate
        + 0.5 * sigma_t * projected_norm_sq
        - 0.5 * dual_norm_sq / sigma_t
        + 0.5 * tau_t * primal_step_norm_sq / sigma_t
    )

    return RipALMSubproblemCandidate(
        r=r_candidate,
        c=c_candidate,
        u=u_next,
        v=v_next,
        gradient=gradient,
        objective=objective,
        cone_projection=cone_projection,
    )


@torch.no_grad()
def subproblem_hessian_matrix(
    candidate: RipALMSubproblemCandidate,
    *,
    sigma: TensorLike,
    tau: TensorLike,
) -> torch.Tensor:
    """Materialize a generalized Hessian of ``Psi_k`` as a dense matrix.

    This forms the summed SOC Jacobian directly on the candidate device without
    allocating one dense Jacobian per ball.
    """

    dtype = candidate.gradient.dtype
    device = candidate.gradient.device
    sigma_t = torch.as_tensor(sigma, dtype=dtype, device=device).reshape(())
    tau_t = torch.as_tensor(tau, dtype=dtype, device=device).reshape(())

    info = candidate.cone_projection
    d = candidate.c.numel()
    n = d + 1
    jacobian_sum = torch.zeros(n, n, dtype=dtype, device=device)

    inside_count = info.inside.sum().to(dtype=dtype)
    diagonal = torch.full((n,), inside_count, dtype=dtype, device=device)

    middle = info.middle
    q_mid = info.unit_x[middle]
    gamma_mid = info.point_t[middle] / info.norm_x[middle].clamp_min(
        torch.finfo(dtype).tiny
    )
    middle_count = middle.sum().to(dtype=dtype)

    q_sum = q_mid.sum(dim=0)
    diagonal[0] = diagonal[0] + 0.5 * middle_count
    diagonal[1:] = diagonal[1:] + 0.5 * (middle_count + gamma_mid.sum())

    jacobian_sum[0, 1:] = 0.5 * q_sum
    jacobian_sum[1:, 0] = 0.5 * q_sum
    weighted_q = gamma_mid.unsqueeze(1) * q_mid
    jacobian_sum[1:, 1:] = -0.5 * q_mid.T.matmul(weighted_q)

    jacobian_sum.diagonal().add_(diagonal)
    hessian = sigma_t * jacobian_sum
    hessian.diagonal().add_(tau_t / sigma_t)
    return hessian


@torch.no_grad()
def check_relative_inexact_condition(
    problem: MEBProblem,
    state: MEBState,
    r: TensorLike,
    c: TensorLike,
    *,
    sigma: TensorLike,
    tau: TensorLike,
    rho: TensorLike,
    gradient_tol: TensorLike,
) -> RelativeInexactCheck:
    """Check the relative condition or the gradient-norm fallback."""

    candidate = evaluate_subproblem_candidate(
        problem, state, r, c, sigma=sigma, tau=tau
    )
    return check_relative_inexact_candidate(
        problem,
        state,
        candidate,
        sigma=sigma,
        tau=tau,
        rho=rho,
        gradient_tol=gradient_tol,
    )


@torch.no_grad()
def check_relative_inexact_candidate(
    problem: MEBProblem,
    state: MEBState,
    candidate: RipALMSubproblemCandidate,
    *,
    sigma: TensorLike,
    tau: TensorLike,
    rho: TensorLike,
    gradient_tol: TensorLike,
) -> RelativeInexactCheck:
    """Check the relative condition using an already evaluated candidate."""

    sigma_t = torch.as_tensor(
        sigma, dtype=problem.dtype, device=problem.device
    ).reshape(())
    tau_t = torch.as_tensor(tau, dtype=problem.dtype, device=problem.device).reshape(())
    rho_t = torch.as_tensor(rho, dtype=problem.dtype, device=problem.device).reshape(())

    alpha_beta_gap = torch.cat(
        ((state.alpha - candidate.r).reshape(1), state.beta - candidate.c)
    )
    gradient_norm_sq = (candidate.gradient * candidate.gradient).sum()
    gradient_norm = torch.sqrt(gradient_norm_sq)
    lhs = 2.0 * sigma_t * (alpha_beta_gap * candidate.gradient).sum()
    lhs = lhs.abs() + sigma_t.square() * gradient_norm_sq

    dual_u_step = candidate.u - state.u
    dual_v_step = candidate.v - state.v
    dual_step_norm_sq = (dual_u_step * dual_u_step).sum()
    dual_step_norm_sq = dual_step_norm_sq + (dual_v_step * dual_v_step).sum()
    primal_r_step = candidate.r - state.r
    primal_c_step = candidate.c - state.c
    primal_step_norm_sq = primal_r_step.square() + (primal_c_step * primal_c_step).sum()
    rhs = rho_t * (dual_step_norm_sq + tau_t * primal_step_norm_sq)
    relative_satisfied = lhs <= rhs
    gradient_tol_t = torch.as_tensor(
        gradient_tol, dtype=problem.dtype, device=problem.device
    ).reshape(())
    gradient_satisfied = gradient_norm <= gradient_tol_t

    return RelativeInexactCheck(
        candidate=candidate,
        satisfied=relative_satisfied | gradient_satisfied,
        relative_satisfied=relative_satisfied,
        gradient_norm=gradient_norm,
        gradient_satisfied=gradient_satisfied,
        lhs=lhs,
        rhs=rhs,
        dual_step_norm_sq=dual_step_norm_sq,
        primal_step_norm_sq=primal_step_norm_sq,
    )
