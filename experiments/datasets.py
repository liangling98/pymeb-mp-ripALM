"""Deterministic MEB test instances from Liang, Sun, and Toh (2021)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from pymeb import MEBProblem


DATASET_CACHE_DIR = Path(__file__).resolve().parent / ".dataset_cache"


def paper_meb_problem(
    m: int,
    d: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[Union[str, torch.device]] = None,
) -> MEBProblem:
    """Generate the paper's deterministic MEB instance for any ``m`` and ``d``."""

    if m <= 0 or d <= 0:
        raise ValueError("m and d must be positive.")

    target_device = torch.device(device) if device is not None else None
    values = _load_or_generate_paper_values(m, d, dtype=dtype)
    if target_device is not None:
        values = values.to(device=target_device)
    return MEBProblem(values[:, 1:], values[:, 0], copy=False)


def _load_or_generate_paper_values(
    m: int,
    d: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    path = _paper_cache_path(m, d, dtype)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)

    values = _paper_sequence(m * (d + 1), dtype=dtype, device="cpu").reshape(m, d + 1)
    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(values, tmp_path)
    tmp_path.replace(path)
    return values


def _paper_cache_path(m: int, d: int, dtype: torch.dtype) -> Path:
    dtype_name = str(dtype).removeprefix("torch.")
    return DATASET_CACHE_DIR / f"paper_meb_m{m}_d{d}_{dtype_name}.pt"


def _paper_sequence(
    n: int,
    *,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    p = 7
    values = torch.empty(n, dtype=dtype, device=device)
    for i in range(n):
        p = (445 * p + 1) % 4096
        values[i] = p / 40.96
    return values
