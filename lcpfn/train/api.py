"""High-level meta-training API for LC-PFN."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import torch
from torch import Tensor, nn

from lcpfn.models import distributions as bar_distribution
from lcpfn.models import encoders
from lcpfn.priors import utils as putils
from lcpfn.train import loop
from lcpfn.utils import (
    get_cosine_schedule_with_warmup,
    get_uniform_single_eval_pos_sampler,
)


class BatchGenerator(Protocol):
    """Protocol describing the signature expected from curve batch generators."""

    def __call__(
        self,
        batch_size: int,
        seq_len: int,
        num_features: int,
        *,
        hyperparameters: Optional[Dict[str, Any]] = None,
        single_eval_pos: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return a tuple ``(style, x, y)`` matching the original LC-PFN API."""


@dataclass
class MetaTrainConfig:
    """Configuration for meta-training the LC-PFN prior."""

    seq_len: int = 100
    emsize: int = 512
    nlayers: int = 12
    num_borders: int = 1_000
    lr: float = 1e-4
    batch_size: int = 100
    epochs: int = 1_000
    nhid: Optional[int] = None
    nhead: Optional[int] = None
    warmup_ratio: float = 0.25
    steps_per_epoch: int = 100
    device: str | torch.device = "cpu"
    train_mixed_precision: bool = False
    extra_prior_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_loop_kwargs(self) -> Dict[str, Any]:
        """Translate this dataclass to the argument structure expected by ``loop.train``."""

        nhid = self.nhid if self.nhid is not None else self.emsize * 2
        nhead = self.nhead if self.nhead is not None else max(1, self.emsize // 128)
        warmup_epochs = max(1, int(self.epochs * self.warmup_ratio))
        return {
            "emsize": self.emsize,
            "nhid": nhid,
            "nlayers": self.nlayers,
            "nhead": nhead,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "warmup_epochs": warmup_epochs,
            "scheduler": get_cosine_schedule_with_warmup,
            "steps_per_epoch": self.steps_per_epoch,
            "train_mixed_precision": self.train_mixed_precision,
            "gpu_device": str(self.device),
        }


def _resolve_git_revision() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:  # pragma: no cover - git may not be available during tests
        return None


def _save_checkpoint(
    model: nn.Module,
    config: MetaTrainConfig,
    checkpoint_path: Path,
    *,
    model_args: Dict[str, Any],
    criterion_state: Dict[str, Any],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "model_args": model_args,
        "criterion": criterion_state,
        "metadata": {
            "version": torch.__version__,
            "git_revision": _resolve_git_revision(),
        },
    }
    if extra_metadata:
        payload["metadata"].update(extra_metadata)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def train_meta(
    get_batch_func: BatchGenerator,
    config: Optional[MetaTrainConfig] = None,
    *,
    checkpoint_path: Optional[Path] = None,
) -> nn.Module:
    """Meta-train an LC-PFN model.

    Parameters
    ----------
    get_batch_func:
        Callable returning batches of synthetic or recorded learning curves. The callable must
        follow the signature required by the original repository.
    config:
        Training configuration. If omitted :class:`MetaTrainConfig` defaults are used.
    checkpoint_path:
        Optional path where the resulting checkpoint should be saved. The file contains the
        model state dict, configuration, and metadata for reproducibility.
    """

    config = config or MetaTrainConfig()
    priordataloader_class = putils.get_batch_to_dataloader(get_batch_func)
    num_features = 1

    ys = get_batch_func(
        10_000,
        config.seq_len,
        num_features,
        hyperparameters=config.extra_prior_kwargs,
        single_eval_pos=config.seq_len,
    )

    bucket_limits = bar_distribution.get_bucket_limits(config.num_borders, ys=ys[2])

    criterion = bar_distribution.FullSupportBarDistribution(bucket_limits)

    training_kwargs = config.to_loop_kwargs()
    training_kwargs.update(
        {
            "priordataloader_class": priordataloader_class,
            "criterion": criterion,
            "encoder_generator": lambda in_dim, out_dim: nn.Sequential(
                encoders.Normalize(0.0, 101.0),
                encoders.Normalize(0.5, (1 / 12) ** 0.5),
                encoders.Linear(in_dim, out_dim),
            ),
            "y_encoder_generator": encoders.get_normalized_uniform_encoder(
                encoders.Linear
            ),
            "extra_prior_kwargs_dict": {
                "num_features": num_features,
                "hyperparameters": {**config.extra_prior_kwargs},
            },
            "bptt": config.seq_len,
            "single_eval_pos_gen": get_uniform_single_eval_pos_sampler(
                config.seq_len, min_len=1
            ),
            "aggregate_k_gradients": 1,
            "nhid": training_kwargs["nhid"],
            "steps_per_epoch": config.steps_per_epoch,
        }
    )

    model = loop.train(**training_kwargs)

    if checkpoint_path is not None:
        _save_checkpoint(
            model,
            config,
            Path(checkpoint_path),
            model_args={
                "emsize": config.emsize,
                "nlayers": config.nlayers,
                "nhead": training_kwargs["nhead"],
                "nhid": training_kwargs["nhid"],
                "num_borders": config.num_borders,
            },
            criterion_state={"bucket_limits": bucket_limits.cpu()},
            extra_metadata={"backend": "loop"},
        )

    return model


__all__ = ["MetaTrainConfig", "train_meta", "BatchGenerator"]
