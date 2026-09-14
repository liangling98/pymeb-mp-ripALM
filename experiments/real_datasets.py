"""Reproducible real-data MEB instances for the accompanying paper.

The point datasets use zero radii.  FEMNIST uses natural writer metadata to
construct nonzero-radius balls whose centers are group means and whose radii
are the maximum within-group distances.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import torch

from pymeb import MEBProblem


REAL_CACHE_DIR = Path(__file__).resolve().parent / ".dataset_cache" / "real"
HIGGS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/"
    "HIGGS.csv.gz"
)
FEMNIST_REPO = "flwrlabs/femnist"
FEMNIST_FILE = "data/train-00000-of-00001.parquet"


@dataclass(frozen=True)
class RealMEBInstance:
    """A real-data MEB problem and its reproducibility metadata."""

    name: str
    problem: MEBProblem
    metadata: dict[str, Any]


def fashion_mnist_problem(
    *,
    limit: int = 60_000,
    seed: int = 20260825,
    download: bool = False,
    dtype: torch.dtype = torch.float64,
) -> RealMEBInstance:
    """Return a deterministic Fashion-MNIST point instance in 784 dimensions."""

    from torchvision.datasets import FashionMNIST

    root = REAL_CACHE_DIR / "fashion_mnist"
    dataset = FashionMNIST(root=root, train=True, download=download)
    total = len(dataset)
    if not (1 <= limit <= total):
        raise ValueError(f"limit must be in [1, {total}]")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:limit]
    centers = dataset.data.index_select(0, indices).reshape(limit, -1)
    centers = centers.to(dtype=dtype).div_(255.0)
    radii = torch.zeros(limit, dtype=dtype)
    labels = dataset.targets.index_select(0, indices)
    counts = torch.bincount(labels, minlength=10)
    return RealMEBInstance(
        name="fashion_mnist_points",
        problem=MEBProblem(centers, radii, copy=False),
        metadata={
            "source": "Fashion-MNIST official training split",
            "source_url": "https://github.com/zalandoresearch/fashion-mnist",
            "raw_samples": limit,
            "sampling": f"fixed torch.randperm seed={seed}",
            "preprocessing": "flatten 28x28 images; divide pixels by 255; radii=0",
            "class_counts": counts.tolist(),
        },
    )


def higgs_problem(
    *,
    limit: int = 100_000,
    download: bool = False,
    dtype: torch.dtype = torch.float64,
) -> RealMEBInstance:
    """Return a standardized point instance from the first ``limit`` HIGGS rows.

    The compressed UCI file is streamed and closed after the requested prefix,
    so constructing a subset does not require storing the full 11-million-row
    dataset.  The parsed float64 subset is cached locally.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    cache_path = REAL_CACHE_DIR / "higgs" / f"higgs_first_{limit}_float64.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        centers = payload["centers"].to(dtype=dtype)
        metadata = dict(payload["metadata"])
    else:
        if not download:
            raise FileNotFoundError(
                f"{cache_path} is absent; pass download=True to stream it from UCI"
            )
        centers, metadata = _stream_higgs(limit)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        torch.save({"centers": centers, "metadata": metadata}, tmp_path)
        tmp_path.replace(cache_path)
        centers = centers.to(dtype=dtype)

    radii = torch.zeros(centers.shape[0], dtype=dtype)
    return RealMEBInstance(
        name="higgs_points",
        problem=MEBProblem(centers, radii, copy=False),
        metadata=metadata,
    )


def femnist_ball_problem(
    *,
    max_rows: int = 100_000,
    grouping: Literal["writer", "writer_character"] = "writer_character",
    min_group_size: int = 2,
    max_groups: int | None = None,
    download: bool = False,
    dtype: torch.dtype = torch.float64,
) -> RealMEBInstance:
    """Construct FEMNIST writer-summary balls in the 784-dimensional pixel space."""

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive when supplied")
    if grouping not in ("writer", "writer_character"):
        raise ValueError("unsupported FEMNIST grouping")

    group_suffix = "all" if max_groups is None else str(max_groups)
    cache_path = (
        REAL_CACHE_DIR
        / "femnist"
        / f"balls_{grouping}_rows{max_rows}_min{min_group_size}_groups{group_suffix}.pt"
    )
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        centers = payload["centers"].to(dtype=dtype)
        radii = payload["radii"].to(dtype=dtype)
        metadata = dict(payload["metadata"])
        verbose_labels = metadata.pop("selected_group_labels", None)
        if verbose_labels is not None:
            metadata["selected_group_label_count"] = len(verbose_labels)
            metadata["selected_group_label_sample"] = verbose_labels[:10]
    else:
        parquet_path = _femnist_parquet_path(download=download)
        pixels, group_ids, group_labels = _read_femnist_prefix(
            parquet_path,
            max_rows=max_rows,
            grouping=grouping,
        )
        centers, radii, group_sizes, selected_labels = group_ball_tensors(
            pixels,
            group_ids,
            group_labels=group_labels,
            min_group_size=min_group_size,
            max_groups=max_groups,
            dtype=torch.float64,
        )
        metadata = {
            "source": "FEMNIST writer-partitioned training data",
            "source_url": "https://huggingface.co/datasets/flwrlabs/femnist",
            "upstream": "LEAF FEMNIST",
            "raw_samples": int(pixels.shape[0]),
            "grouping": grouping,
            "min_group_size": min_group_size,
            "groups": int(centers.shape[0]),
            "group_size_min": int(group_sizes.min()),
            "group_size_median": float(group_sizes.to(torch.float64).median()),
            "group_size_max": int(group_sizes.max()),
            "preprocessing": (
                "flatten 28x28 images; divide pixels by 255; center=group mean; "
                "radius=max Euclidean distance to the group mean"
            ),
            "selected_group_label_count": len(selected_labels),
            "selected_group_label_sample": selected_labels[:10],
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        torch.save(
            {"centers": centers, "radii": radii, "metadata": metadata},
            tmp_path,
        )
        tmp_path.replace(cache_path)
        centers = centers.to(dtype=dtype)
        radii = radii.to(dtype=dtype)

    return RealMEBInstance(
        name=f"femnist_{grouping}_balls",
        problem=MEBProblem(centers, radii, copy=False),
        metadata=metadata,
    )


def standardize_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize columns, mapping constant columns to zero."""

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have shape [m, d] with m > 0")
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False)
    safe_scale = torch.where(scale > torch.finfo(features.dtype).eps, scale, torch.ones_like(scale))
    return (features - mean) / safe_scale, mean, safe_scale


def group_ball_tensors(
    pixels: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    group_labels: list[str] | None = None,
    min_group_size: int = 1,
    max_groups: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Return centroid/max-distance balls for integer-labeled point groups."""

    if pixels.ndim != 2 or group_ids.ndim != 1 or pixels.shape[0] != group_ids.numel():
        raise ValueError("pixels and group_ids must have shapes [n, d] and [n]")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")

    order = torch.argsort(group_ids)
    sorted_ids = group_ids.index_select(0, order)
    unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    eligible = torch.nonzero(counts >= min_group_size, as_tuple=False).flatten()
    if max_groups is not None:
        eligible = eligible[:max_groups]
    if eligible.numel() == 0:
        raise ValueError("no groups satisfy min_group_size")

    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(dim=0)))
    centers: list[torch.Tensor] = []
    radii: list[torch.Tensor] = []
    selected_sizes: list[torch.Tensor] = []
    selected_labels: list[str] = []
    for position in eligible.tolist():
        start = int(offsets[position])
        end = int(offsets[position + 1])
        rows = order[start:end]
        points = pixels.index_select(0, rows).to(dtype=dtype).div(255.0)
        center = points.mean(dim=0)
        radius = torch.linalg.vector_norm(points - center, dim=1).max()
        centers.append(center)
        radii.append(radius)
        selected_sizes.append(counts[position])
        group_id = int(unique_ids[position])
        selected_labels.append(
            group_labels[group_id] if group_labels is not None else str(group_id)
        )

    return (
        torch.stack(centers),
        torch.stack(radii),
        torch.stack(selected_sizes),
        selected_labels,
    )


def _stream_higgs(limit: int) -> tuple[torch.Tensor, dict[str, Any]]:
    import numpy as np
    import pandas as pd
    import requests

    with requests.get(HIGGS_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as stream:
            frame = pd.read_csv(stream, header=None, nrows=limit, dtype=np.float64)
    if frame.shape != (limit, 29):
        raise RuntimeError(f"expected {(limit, 29)} HIGGS values, got {frame.shape}")
    labels = torch.from_numpy(frame.iloc[:, 0].to_numpy(copy=True)).to(torch.int64)
    raw = torch.from_numpy(frame.iloc[:, 1:].to_numpy(copy=True)).to(torch.float64)
    centers, mean, scale = standardize_features(raw)
    return centers.contiguous(), {
        "source": "UCI HIGGS",
        "source_url": HIGGS_URL,
        "raw_samples": limit,
        "sampling": "first rows of the official compressed CSV stream",
        "preprocessing": "drop class label; standardize each feature on the selected prefix; radii=0",
        "positive_labels": int(labels.sum()),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
    }


def _femnist_parquet_path(*, download: bool) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir = REAL_CACHE_DIR / "femnist" / "source"
    candidate = local_dir / FEMNIST_FILE
    if candidate.exists():
        return candidate
    legacy_candidate = REAL_CACHE_DIR.parent / "femnist" / FEMNIST_FILE
    if legacy_candidate.exists():
        return legacy_candidate
    if not download:
        raise FileNotFoundError(
            f"{candidate} is absent; pass download=True to fetch FEMNIST"
        )
    downloaded = hf_hub_download(
        repo_id=FEMNIST_REPO,
        filename=FEMNIST_FILE,
        repo_type="dataset",
        local_dir=local_dir,
    )
    return Path(downloaded)


def _read_femnist_prefix(
    path: Path,
    *,
    max_rows: int,
    grouping: Literal["writer", "writer_character"],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    pixels = torch.empty(max_rows, 28 * 28, dtype=torch.uint8)
    keys: list[str] = []
    count = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=min(4096, max_rows),
        columns=["image", "writer_id", "character"],
    ):
        image_bytes = batch.column(0).field("bytes").to_pylist()
        writers = batch.column(1).to_pylist()
        characters = batch.column(2).to_pylist()
        for encoded, writer, character in zip(image_bytes, writers, characters):
            if count == max_rows:
                break
            if encoded is None:
                raise RuntimeError("FEMNIST row has no embedded image bytes")
            with Image.open(BytesIO(encoded)) as image:
                array = np.asarray(image.convert("L"), dtype=np.uint8)
            if array.shape != (28, 28):
                raise RuntimeError(f"expected a 28x28 FEMNIST image, got {array.shape}")
            pixels[count].copy_(torch.from_numpy(array.copy()).reshape(-1))
            key = str(writer) if grouping == "writer" else f"{writer}:{int(character)}"
            keys.append(key)
            count += 1
        if count == max_rows:
            break
    pixels = pixels[:count]
    if count == 0:
        raise RuntimeError("no FEMNIST rows were read")

    label_to_id: dict[str, int] = {}
    group_ids = torch.empty(count, dtype=torch.long)
    labels: list[str] = []
    for row, key in enumerate(keys):
        group_id = label_to_id.get(key)
        if group_id is None:
            group_id = len(labels)
            label_to_id[key] = group_id
            labels.append(key)
        group_ids[row] = group_id
    return pixels, group_ids, labels
