"""LC-PFN – Learning Curve Prior-Fitted Networks."""

from __future__ import annotations

from importlib import metadata

from lcpfn.inference import load_pretrained, predict_curve
from lcpfn.train import MetaTrainConfig, train_meta

try:  # pragma: no cover - importlib metadata fallback for editable installs
    __version__ = metadata.version("lcpfn")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "load_pretrained",
    "predict_curve",
    "MetaTrainConfig",
    "train_meta",
]
