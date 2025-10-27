"""Wrapper utilities around the Transformer model used for LC-PFN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor, nn

from lcpfn.utils import identity_normalizer


@dataclass
class CurveBatch:
    """Container describing a single meta-test instance."""

    x_train: Tensor
    y_train: Tensor
    x_test: Tensor


class LCPFN(nn.Module):
    """Thin wrapper that provides convenience utilities around the transformer model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()

    def check_input(
        self,
        x_train: Tensor,
        x_test: Tensor,
        y_train: Tensor,
        y_test: Optional[Tensor] = None,
    ) -> None:
        if torch.any(x_train < 0) or torch.any(x_test < 0):
            raise ValueError("Learning curve steps must be non-negative.")
        if torch.any((0 > y_train) | (y_train > 1)):
            raise ValueError(
                "Observed accuracies must lie in [0, 1]. Use cumulative best accuracy to enforce monotonicity."
            )
        if y_test is not None and torch.any((0 > y_test) | (y_test > 1)):
            raise ValueError("Target accuracies must lie in [0, 1].")

    @torch.no_grad()
    def predict_mean(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor,
        *,
        normalizer: Tuple = identity_normalizer(),
    ) -> Tensor:
        y_train_norm = normalizer[0](y_train)
        logits = self(x_train=x_train, y_train=y_train_norm, x_test=x_test)
        return normalizer[1](self.model.criterion.mean(logits))

    @torch.no_grad()
    def predict_quantiles(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor,
        qs: Iterable[float],
        *,
        normalizer: Tuple = identity_normalizer(),
    ) -> Tensor:
        y_train_norm = normalizer[0](y_train)
        logits = self(x_train=x_train, y_train=y_train_norm, x_test=x_test)
        return normalizer[1](
            torch.cat([self.model.criterion.icdf(logits, q) for q in qs], dim=1)
        )

    @torch.no_grad()
    def nll_loss(
        self, x_train: Tensor, y_train: Tensor, x_test: Tensor, y_test: Tensor
    ) -> Tensor:
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test)
        return self.model.criterion(logits, y_test)

    def forward(self, *, x_train: Tensor, y_train: Tensor, x_test: Tensor) -> Tensor:
        self.check_input(x_train, x_test, y_train)
        single_eval_pos = x_train.shape[0]
        x = torch.cat([x_train, x_test], dim=0).unsqueeze(1)
        y = y_train.unsqueeze(1)
        return self.model((x, y), single_eval_pos=single_eval_pos)


__all__ = ["LCPFN", "CurveBatch"]
