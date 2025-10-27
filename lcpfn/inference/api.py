"""User-facing inference helpers for LC-PFN."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import torch
from torch import Tensor, nn

from .resources import DEFAULT_MODEL_NAME, ensure_weights
from ..models import distributions as bar_distribution
from ..models import encoders, positional_encodings
from ..models.lcpfn import LCPFN
from ..models.transformer import TransformerModel


def _prepare_tensor(tensor: Tensor, *, name: str) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}.")
    if tensor.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D tensor, received shape {tuple(tensor.shape)}."
        )
    return tensor.to(dtype=torch.float32)


def _coerce_model(
    obj: Union[nn.Module, dict], *, map_location: Union[str, torch.device]
) -> nn.Module:
    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict) and "state_dict" in obj and "model_args" in obj:
        bucket_limits = obj.get("criterion", {}).get("bucket_limits")
        if bucket_limits is None:
            raise ValueError(
                "Checkpoint is missing bucket limit information required for inference."
            )
        if not isinstance(bucket_limits, Tensor):
            bucket_limits = torch.tensor(bucket_limits, dtype=torch.float32)
        bucket_limits = bucket_limits.to(dtype=torch.float32, device=map_location)
        model_args = obj["model_args"]
        emsize = int(model_args.get("emsize", 512))
        nlayers = int(model_args.get("nlayers", 12))
        nhead = int(model_args.get("nhead", max(1, emsize // 128)))
        nhid = int(model_args.get("nhid", emsize * 2))
        n_out = len(bucket_limits) - 1
        encoder = nn.Sequential(
            encoders.Normalize(0.0, 101.0),
            encoders.Normalize(0.5, (1 / 12) ** 0.5),
            encoders.Linear(1, emsize),
        )
        y_encoder = encoders.get_normalized_uniform_encoder(encoders.Linear)(1, emsize)
        model = TransformerModel(
            encoder=encoder,
            n_out=n_out,
            ninp=emsize,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
            dropout=float(model_args.get("dropout", 0.0)),
            y_encoder=y_encoder,
            pos_encoder=positional_encodings.NoPositionalEncoding(emsize),
        )
        model.load_state_dict(obj["state_dict"])
        model.criterion = bar_distribution.FullSupportBarDistribution(bucket_limits)
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected a torch.nn.Module or a dict checkpoint."
        )

    model.to(map_location)
    model.eval()
    return model


def load_pretrained(
    identifier: Union[str, Path] = "default",
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> LCPFN:
    """Load a pretrained LC-PFN model.

    Parameters
    ----------
    identifier:
        Either ``"default"`` to load the canonical checkpoint, an alias from the official
        model zoo (e.g. ``"EMSIZE512_NLAYERS6_NBUCKETS1000"``), or a filesystem path to a
        checkpoint generated via :func:`lcpfn.train.api.train_meta`.
    map_location:
        Device mapping passed to :func:`torch.load` to ensure CPU compatibility.
    """

    if isinstance(identifier, Path):
        checkpoint_path = identifier
    elif identifier == "default":
        checkpoint_path = ensure_weights(DEFAULT_MODEL_NAME)
    else:
        candidate = Path(identifier)
        if candidate.exists() or candidate.suffix:
            checkpoint_path = candidate
        else:
            checkpoint_path = ensure_weights(str(identifier))

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = _coerce_model(checkpoint, map_location=map_location)
    return LCPFN(model)


def _monotonic_cummax(values: Tensor) -> Tensor:
    return torch.cummax(values, dim=0)[0]


def predict_curve(
    model: Union[LCPFN, nn.Module],
    steps_seen: Tensor,
    acc_seen: Tensor,
    future_steps: Tensor,
    *,
    enforce_monotonic: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Predict the continuation of a learning curve.

    Parameters
    ----------
    model:
        Model returned by :func:`load_pretrained` or an equivalent :class:`~torch.nn.Module`
        with a ``criterion`` attribute implementing ``mean`` and ``quantile``.
    steps_seen:
        1D tensor of observed training steps.
    acc_seen:
        1D tensor of observed accuracies. Values should lie in ``[0, 1]`` and represent the
        cumulative best accuracy.
    future_steps:
        1D tensor of future steps for which predictions are required.
    enforce_monotonic:
        If ``True`` (default), the observed accuracies are transformed into a non-decreasing
        sequence using :func:`torch.cummax`.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Mean and an uncertainty estimate (half-width of the 68.2% central interval) for each
        requested future step.
    """

    if not isinstance(model, LCPFN):
        model = LCPFN(model)

    steps_seen = _prepare_tensor(steps_seen, name="steps_seen")
    acc_seen = _prepare_tensor(acc_seen, name="acc_seen")
    future_steps = _prepare_tensor(future_steps, name="future_steps")

    if steps_seen.shape != acc_seen.shape:
        raise ValueError("steps_seen and acc_seen must share the same shape.")

    if enforce_monotonic:
        acc_seen = _monotonic_cummax(acc_seen)

    logits = model(x_train=steps_seen, y_train=acc_seen, x_test=future_steps)
    criterion = getattr(model.model, "criterion", None)
    if criterion is None:
        raise AttributeError(
            "Model is missing a 'criterion' attribute required for inference."
        )

    pred_mean = criterion.mean(logits).squeeze(-1)
    if hasattr(criterion, "quantile"):
        quantiles = criterion.quantile(logits, center_prob=0.682)
        pred_std = 0.5 * (quantiles[..., 1] - quantiles[..., 0]).squeeze(-1)
    else:
        pred_std = torch.zeros_like(pred_mean)

    return pred_mean, pred_std


__all__ = ["load_pretrained", "predict_curve"]
