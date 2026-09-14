"""Shared type aliases and tiny helpers."""

from __future__ import annotations

from typing import Sequence, Union

import torch

ScalarLike = Union[float, int]
TensorLike = Union[
    torch.Tensor,
    ScalarLike,
    Sequence[ScalarLike],
    Sequence[Sequence[ScalarLike]],
]


def _idx(indices: TensorLike, device: torch.device) -> torch.Tensor:
    idx = torch.as_tensor(indices, device=device)
    if idx.dtype == torch.bool:
        idx = torch.nonzero(idx, as_tuple=False).flatten()
    return idx.long().flatten()


def _set(obj: object, **values: torch.Tensor) -> None:
    for name, value in values.items():
        object.__setattr__(obj, name, value)
