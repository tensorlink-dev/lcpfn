"""Utilities for managing pretrained LC-PFN checkpoints."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Dict

import requests  # type: ignore[import-untyped]

DEFAULT_MODEL_NAME = "EMSIZE512_NLAYERS12_NBUCKETS1000"
_BASE_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/lcpfn"

_MODEL_FILES: Dict[str, str] = {
    "EMSIZE512_NLAYERS12_NBUCKETS1000": "pfn_EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000.pt",
    "EMSIZE512_NLAYERS6_NBUCKETS1000": "pfn_EPOCH1000_EMSIZE512_NLAYERS6_NBUCKETS1000.pt",
}


def _weights_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "trained_models"


def _download_file(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as handle:
        handle.write(response.content)


def _decompress_gzip(archive: Path, destination: Path) -> None:
    with gzip.open(archive, "rb") as gz, open(destination, "wb") as out:
        shutil.copyfileobj(gz, out)


def ensure_weights(name: str) -> Path:
    """Ensure that the checkpoint for ``name`` exists locally and return its path."""

    if name not in _MODEL_FILES:
        raise ValueError(f"Unknown pretrained LC-PFN model: {name!r}")

    weights_dir = _weights_dir()
    weights_path = weights_dir / _MODEL_FILES[name]
    archive_path = weights_path.with_suffix(weights_path.suffix + ".gz")

    if weights_path.exists():
        return weights_path

    if archive_path.exists():
        _decompress_gzip(archive_path, weights_path)
        return weights_path

    url = f"{_BASE_URL}/{archive_path.name}"
    _download_file(url, archive_path)
    _decompress_gzip(archive_path, weights_path)
    return weights_path


__all__ = ["DEFAULT_MODEL_NAME", "ensure_weights"]
