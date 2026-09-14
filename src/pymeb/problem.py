"""Problem data and primal-dual state containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from ._types import TensorLike, _idx, _set


@dataclass(frozen=True)
class MEBProblem:
    """Ball data in structure-of-arrays form: ``radii [m]``, ``centers [m, d]``."""

    centers: torch.Tensor
    radii: torch.Tensor
    original_indices: torch.Tensor

    def __init__(
        self,
        centers: TensorLike,
        radii: TensorLike,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        original_indices: Optional[TensorLike] = None,
        copy: bool = False,
    ) -> None:
        C = torch.as_tensor(centers, dtype=dtype, device=device).detach()
        if not torch.is_floating_point(C):
            raise TypeError(
                "centers must be floating point; pass dtype=torch.float64 "
                "or use pymeb.solve for automatic conversion."
            )
        R = torch.as_tensor(radii, dtype=C.dtype, device=C.device).detach().reshape(-1)
        if copy:
            C, R = C.clone(), R.clone()
        C, R = C.contiguous(), R.contiguous()

        if C.ndim != 2 or C.shape[0] != R.numel():
            raise ValueError("expected centers [m, d] and radii [m].")
        if C.shape[0] == 0 or C.shape[1] == 0:
            raise ValueError("a problem must contain at least one ball in dimension d >= 1.")
        if torch.any(R < 0):
            raise ValueError("ball radii must be nonnegative.")

        if original_indices is None:
            ids = torch.arange(R.numel(), device=C.device)
        else:
            ids = torch.as_tensor(original_indices, device=C.device).long().flatten()
            if ids.numel() != R.numel():
                raise ValueError("original_indices must have length m.")

        _set(self, centers=C, radii=R, original_indices=ids.contiguous())

    @classmethod
    def from_tensors(
        cls, centers: TensorLike, radii: TensorLike, **kwargs
    ) -> "MEBProblem":
        return cls(centers, radii, **kwargs)

    @property
    def m(self) -> int:
        return self.radii.numel()

    @property
    def d(self) -> int:
        return self.centers.shape[1]

    @property
    def dtype(self) -> torch.dtype:
        return self.centers.dtype

    @property
    def device(self) -> torch.device:
        return self.centers.device

    def to(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "MEBProblem":
        device = torch.device(device) if device is not None else self.device
        dtype = dtype or self.dtype
        return MEBProblem(
            self.centers.to(device=device, dtype=dtype),
            self.radii.to(device=device, dtype=dtype),
            original_indices=self.original_indices.to(device=device),
            copy=False,
        )

    def subset(self, indices: TensorLike) -> "MEBProblem":
        idx = _idx(indices, self.device)
        return MEBProblem(
            self.centers.index_select(0, idx),
            self.radii.index_select(0, idx),
            original_indices=self.original_indices.index_select(0, idx),
            copy=False,
        )

    def restrict(
        self,
        indices: TensorLike,
        state: Optional["MEBState"] = None,
    ) -> Union["MEBProblem", Tuple["MEBProblem", "MEBState"]]:
        idx = _idx(indices, self.device)
        problem = self.subset(idx)
        return problem if state is None else (problem, state.subset(idx))

    def distances(
        self,
        c: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        C = self.centers if dtype is None else self.centers.to(dtype=dtype)
        c = c.to(device=self.device, dtype=C.dtype)
        return torch.linalg.vector_norm(C - c, dim=1)

    def slack(
        self, state: "MEBState", *, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        R = self.radii if dtype is None else self.radii.to(dtype=dtype)
        r = state.r.to(device=self.device, dtype=R.dtype)
        return r - R - self.distances(state.c, dtype=dtype)

    def violation(
        self,
        state: "MEBState",
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return -self.slack(state, dtype=dtype)

    def screen(
        self,
        state: "MEBState",
        tol_red: float,
        min_keep: int = 1,
    ) -> torch.Tensor:
        slack = self.slack(state)
        keep = torch.nonzero(slack <= tol_red, as_tuple=False).flatten()
        if keep.numel() < min_keep:
            keep = torch.topk(-slack, k=min(min_keep, self.m)).indices
        return torch.sort(torch.unique(keep)).values

    def violated_indices(
        self,
        state: "MEBState",
        tol: float,
    ) -> torch.Tensor:
        return torch.nonzero(self.violation(state) > tol, as_tuple=False).flatten()


@dataclass(frozen=True)
class MEBState:
    """Primal-dual variables ``(r, c, u, v)`` and error vector ``(alpha, beta)``."""

    r: torch.Tensor
    c: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor

    def __init__(
        self,
        r: TensorLike,
        c: TensorLike,
        u: TensorLike,
        v: TensorLike,
        *,
        alpha: Optional[TensorLike] = None,
        beta: Optional[TensorLike] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        c = torch.as_tensor(c, dtype=dtype, device=device).detach().flatten()
        r = torch.as_tensor(r, dtype=c.dtype, device=c.device).detach().reshape(())
        u = torch.as_tensor(u, dtype=c.dtype, device=c.device).detach().flatten()
        v = torch.as_tensor(v, dtype=c.dtype, device=c.device).detach()
        alpha = (
            torch.zeros_like(r)
            if alpha is None
            else torch.as_tensor(alpha, dtype=c.dtype, device=c.device).reshape(())
        )
        beta = (
            torch.zeros_like(c)
            if beta is None
            else torch.as_tensor(beta, dtype=c.dtype, device=c.device).flatten()
        )

        if v.shape != (u.numel(), c.numel()):
            raise ValueError("expected u [m], v [m, d], c [d].")
        if beta.shape != c.shape:
            raise ValueError("beta must have shape [d].")

        _set(
            self,
            r=r.contiguous(),
            c=c.contiguous(),
            u=u.contiguous(),
            v=v.contiguous(),
            alpha=alpha.contiguous(),
            beta=beta.contiguous(),
        )

    @classmethod
    def zeros(
        cls,
        problem: MEBProblem,
        *,
        r: TensorLike = 0.0,
        c: Optional[TensorLike] = None,
    ) -> "MEBState":
        c = (
            torch.zeros(problem.d, dtype=problem.dtype, device=problem.device)
            if c is None
            else c
        )
        u = torch.zeros(problem.m, dtype=problem.dtype, device=problem.device)
        v = torch.zeros(
            problem.m, problem.d, dtype=problem.dtype, device=problem.device
        )
        return cls(r, c, u, v)

    @classmethod
    def initial_feasible(cls, problem: MEBProblem) -> "MEBState":
        c = problem.centers.mean(dim=0)
        r = torch.max(problem.radii + problem.distances(c))
        return cls.zeros(problem, r=r, c=c)

    @property
    def m(self) -> int:
        return self.u.numel()

    @property
    def d(self) -> int:
        return self.c.numel()

    @property
    def dtype(self) -> torch.dtype:
        return self.c.dtype

    @property
    def device(self) -> torch.device:
        return self.c.device

    def validate_for(self, problem: MEBProblem) -> None:
        if (self.m, self.d, self.dtype, self.device) != (
            problem.m,
            problem.d,
            problem.dtype,
            problem.device,
        ):
            raise ValueError(
                "state and problem have incompatible shape, dtype, or device."
            )

    def to(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "MEBState":
        device = torch.device(device) if device is not None else self.device
        dtype = dtype or self.dtype
        return MEBState(
            self.r.to(device=device, dtype=dtype),
            self.c.to(device=device, dtype=dtype),
            self.u.to(device=device, dtype=dtype),
            self.v.to(device=device, dtype=dtype),
            alpha=self.alpha.to(device=device, dtype=dtype),
            beta=self.beta.to(device=device, dtype=dtype),
        )

    def primal_vector(self) -> torch.Tensor:
        return torch.cat((self.r.reshape(1), self.c))

    def auxiliary_vector(self) -> torch.Tensor:
        return torch.cat((self.alpha.reshape(1), self.beta))

    def with_primal(self, r: TensorLike, c: TensorLike) -> "MEBState":
        return MEBState(r, c, self.u, self.v, alpha=self.alpha, beta=self.beta)

    def with_dual(self, u: TensorLike, v: TensorLike) -> "MEBState":
        return MEBState(self.r, self.c, u, v, alpha=self.alpha, beta=self.beta)

    def subset(self, indices: TensorLike) -> "MEBState":
        idx = _idx(indices, self.device)
        return MEBState(
            self.r,
            self.c,
            self.u.index_select(0, idx),
            self.v.index_select(0, idx),
            alpha=self.alpha,
            beta=self.beta,
        )

    def align_constraints(self, source: MEBProblem, target: MEBProblem) -> "MEBState":
        u = torch.zeros(target.m, dtype=self.dtype, device=self.device)
        v = torch.zeros(target.m, target.d, dtype=self.dtype, device=self.device)
        order = torch.argsort(source.original_indices)
        source_ids = source.original_indices.index_select(0, order)
        pos = torch.searchsorted(source_ids, target.original_indices)
        pos_safe = pos.clamp(max=max(source.m - 1, 0))
        found = (pos < source.m) & (
            source_ids.index_select(0, pos_safe) == target.original_indices
        )
        rows = torch.nonzero(found, as_tuple=False).flatten()
        if rows.numel():
            src_rows = order.index_select(0, pos.index_select(0, rows))
            u[rows] = self.u.index_select(0, src_rows)
            v[rows] = self.v.index_select(0, src_rows)
        return MEBState(self.r, self.c, u, v, alpha=self.alpha, beta=self.beta)
