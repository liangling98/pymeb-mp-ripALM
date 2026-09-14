"""Small, user-facing convenience API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch

from ._types import TensorLike
from .mixed_precision import (
    MixedPrecisionOptions,
    MixedPrecisionResult,
    mixed_precision_ripalm,
)
from .problem import MEBProblem
from .ripalm import RipALMOptions, RipALMResult, ripalm

Method = Literal["ripalm", "mp-ripalm"]
SolverOptions = Union[RipALMOptions, MixedPrecisionOptions]
SolverResult = Union[RipALMResult, MixedPrecisionResult]


@dataclass(frozen=True)
class MEBResult:
    """A compact minimum-enclosing-ball result.

    Tensors stay on the device used by the solver. Use ``.cpu()`` or
    ``float(...)`` when a host value is required.
    """

    center: torch.Tensor
    radius: torch.Tensor
    max_violation: torch.Tensor
    kkt_residual: torch.Tensor
    converged: bool
    method: Method
    solver_result: SolverResult

    @torch.no_grad()
    def contains(
        self,
        centers: TensorLike,
        radii: Optional[TensorLike] = None,
        *,
        atol: float = 1e-8,
    ) -> torch.Tensor:
        """Return one Boolean enclosure check per supplied point or ball.

        The default absolute tolerance matches the default final solver
        tolerance. Set ``atol=0`` when an exact arithmetic comparison is
        specifically desired.
        """

        if atol < 0.0:
            raise ValueError("atol must be nonnegative.")
        centers_t = torch.as_tensor(
            centers,
            dtype=self.center.dtype,
            device=self.center.device,
        )
        if centers_t.ndim != 2 or centers_t.shape[1] != self.center.numel():
            raise ValueError("centers must have shape [m, d].")
        radii_t = (
            torch.zeros(centers_t.shape[0], dtype=centers_t.dtype, device=centers_t.device)
            if radii is None
            else torch.as_tensor(
                radii,
                dtype=centers_t.dtype,
                device=centers_t.device,
            ).reshape(-1)
        )
        if radii_t.numel() != centers_t.shape[0]:
            raise ValueError("radii must have length m.")
        reach = radii_t + torch.linalg.vector_norm(centers_t - self.center, dim=1)
        return reach <= self.radius + atol


@torch.no_grad()
def solve(
    centers: TensorLike,
    radii: Optional[TensorLike] = None,
    *,
    method: Method = "ripalm",
    dtype: torch.dtype = torch.float64,
    device: Optional[Union[str, torch.device]] = None,
    options: Optional[SolverOptions] = None,
) -> MEBResult:
    """Compute the minimum enclosing ball of points or balls.

    Args:
        centers: Center coordinates with shape ``[m, d]``.
        radii: Optional nonnegative input radii with shape ``[m]``. Omitting
            this argument treats every row of ``centers`` as a point.
        method: ``"ripalm"`` for a direct solve or ``"mp-ripalm"`` for
            low-precision screening followed by high-precision refinement.
        dtype: Master-data dtype. Float64 is recommended for final accuracy.
        device: PyTorch device used by the solver, such as ``"cpu"`` or
            ``"cuda"``.
        options: Options matching the selected method.

    Returns:
        A compact :class:`MEBResult`. Detailed iteration data remain available
        through its ``solver_result`` field.
    """

    centers_t = torch.as_tensor(centers, dtype=dtype, device=device)
    if centers_t.ndim != 2:
        raise ValueError("centers must have shape [m, d].")
    radii_t = (
        torch.zeros(centers_t.shape[0], dtype=dtype, device=centers_t.device)
        if radii is None
        else torch.as_tensor(radii, dtype=dtype, device=centers_t.device)
    )
    problem = MEBProblem(centers_t, radii_t, copy=False)

    if method == "ripalm":
        if options is not None and not isinstance(options, RipALMOptions):
            raise TypeError("method='ripalm' requires RipALMOptions.")
        direct_options = options or RipALMOptions(print_history=False)
        direct_result = ripalm(problem, options=direct_options)
        max_violation = problem.violation(direct_result.state).max().clamp_min(0.0)
        return MEBResult(
            center=direct_result.state.c,
            radius=direct_result.state.r,
            max_violation=max_violation,
            kkt_residual=direct_result.residual.value,
            converged=direct_result.converged,
            method=method,
            solver_result=direct_result,
        )

    if method == "mp-ripalm":
        if options is not None and not isinstance(options, MixedPrecisionOptions):
            raise TypeError("method='mp-ripalm' requires MixedPrecisionOptions.")
        mixed_options = options or MixedPrecisionOptions(device=problem.device)
        mixed_result = mixed_precision_ripalm(problem, options=mixed_options)
        return MEBResult(
            center=mixed_result.state.c,
            radius=mixed_result.state.r,
            max_violation=mixed_result.max_violation,
            kkt_residual=mixed_result.residual.value,
            converged=mixed_result.converged,
            method=method,
            solver_result=mixed_result,
        )

    raise ValueError("method must be 'ripalm' or 'mp-ripalm'.")
