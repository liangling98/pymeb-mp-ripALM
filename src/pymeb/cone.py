"""Second-order cone projection and derivative products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union, overload

import torch


@dataclass(frozen=True)
class SOCProjection:
    """Projection data for batched second-order cones."""

    value: torch.Tensor
    point_t: torch.Tensor
    point_x: torch.Tensor
    t: torch.Tensor
    x: torch.Tensor
    norm_x: torch.Tensor
    unit_x: torch.Tensor
    inside: torch.Tensor
    negative: torch.Tensor
    middle: torch.Tensor
    middle_indices: Optional[torch.Tensor] = None
    middle_jacobian: Optional[torch.Tensor] = None

    @property
    def boundary(self) -> torch.Tensor:
        return self.middle


@overload
def project_soc(
    points: torch.Tensor,
    *,
    return_info: Literal[True],
    return_jacobian: bool = False,
    eps: Optional[float] = None,
) -> SOCProjection:
    ...


@overload
def project_soc(
    points: torch.Tensor,
    *,
    return_info: Literal[False] = False,
    return_jacobian: bool = False,
    eps: Optional[float] = None,
) -> torch.Tensor:
    ...


@torch.no_grad()
def project_soc(
    points: torch.Tensor,
    *,
    return_info: bool = False,
    return_jacobian: bool = False,
    eps: Optional[float] = None,
) -> Union[torch.Tensor, SOCProjection]:
    """Project packed rows ``(t, x)`` onto ``{(t, x): ||x|| <= t}``."""

    z = torch.as_tensor(points)
    if z.ndim < 1 or z.shape[-1] < 2:
        raise ValueError("expected SOC points with shape [..., d + 1].")
    if not torch.is_floating_point(z):
        raise TypeError("SOC projection requires floating point tensors.")

    t = z[..., 0]
    x_vec = z[..., 1:]
    dtype = x_vec.dtype
    device = x_vec.device
    tiny = eps if eps is not None else torch.finfo(dtype).tiny

    norm_x = torch.linalg.vector_norm(x_vec, dim=-1)
    safe_norm = norm_x.clamp_min(tiny)
    unit_x = x_vec / safe_norm.unsqueeze(-1)

    inside = norm_x <= t
    negative = norm_x <= -t
    middle = ~(inside | negative)

    scale = 0.5 * (norm_x + t)
    boundary_x = scale.unsqueeze(-1) * unit_x
    projected_t = torch.where(
        inside, t, torch.where(negative, torch.zeros_like(t), scale)
    )
    projected_x = torch.where(
        inside.unsqueeze(-1),
        x_vec,
        torch.where(negative.unsqueeze(-1), torch.zeros_like(x_vec), boundary_x),
    )
    value = torch.cat((projected_t.unsqueeze(-1), projected_x), dim=-1)

    middle_indices = None
    middle_jacobian = None
    if return_jacobian:
        d = x_vec.shape[-1]
        n = d + 1
        flat_middle = middle.reshape(-1)
        middle_indices_t = torch.nonzero(flat_middle, as_tuple=False).flatten()
        q_mid = unit_x.reshape(-1, d).index_select(0, middle_indices_t)
        tau_mid = (t / safe_norm).reshape(-1).index_select(0, middle_indices_t)

        middle_jacobian_t = torch.zeros(
            middle_indices_t.numel(), n, n, dtype=dtype, device=device
        )
        if middle_indices_t.numel():
            eye_d = torch.eye(d, dtype=dtype, device=device).expand(
                middle_indices_t.numel(), d, d
            )
            mid_xx = 0.5 * (
                (1.0 + tau_mid)[:, None, None] * eye_d
                - tau_mid[:, None, None] * q_mid[:, :, None] * q_mid[:, None, :]
            )
            middle_jacobian_t[:, 0, 0] = 0.5
            middle_jacobian_t[:, 0, 1:] = 0.5 * q_mid
            middle_jacobian_t[:, 1:, 0] = 0.5 * q_mid
            middle_jacobian_t[:, 1:, 1:] = mid_xx
        middle_indices = middle_indices_t
        middle_jacobian = middle_jacobian_t

    if not return_info:
        if return_jacobian:
            raise ValueError("return_info=True is required for derivative outputs.")
        return value

    return SOCProjection(
        value=value,
        point_t=t,
        point_x=x_vec,
        t=projected_t,
        x=projected_x,
        norm_x=norm_x,
        unit_x=unit_x,
        inside=inside,
        negative=negative,
        middle=middle,
        middle_indices=middle_indices,
        middle_jacobian=middle_jacobian,
    )


@torch.no_grad()
def soc_jvp(info: SOCProjection, directions: torch.Tensor) -> torch.Tensor:
    """Apply the selected SOC projection generalized Jacobian."""

    dz = torch.as_tensor(directions, dtype=info.value.dtype, device=info.value.device)
    if dz.shape != info.value.shape:
        raise ValueError("directions must have the same shape as info.value.")

    dt = dz[..., 0]
    dx = dz[..., 1:]
    q = info.unit_x
    r = info.norm_x.clamp_min(torch.finfo(info.value.dtype).tiny)
    tau = info.point_t / r
    qdx = (q * dx).sum(dim=-1)
    pdx = dx - q * qdx.unsqueeze(-1)

    middle_t = 0.5 * (dt + qdx)
    middle_x = 0.5 * (q * dt.unsqueeze(-1) + dx + tau.unsqueeze(-1) * pdx)
    middle = torch.cat((middle_t.unsqueeze(-1), middle_x), dim=-1)
    zero = torch.zeros_like(dz)

    return torch.where(
        info.inside.unsqueeze(-1),
        dz,
        torch.where(info.negative.unsqueeze(-1), zero, middle),
    )
