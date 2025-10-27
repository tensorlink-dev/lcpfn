import pytest

torch = pytest.importorskip("torch")
from torch import nn

from lcpfn.inference import predict_curve
from lcpfn.models import distributions, encoders, positional_encodings, transformer


def _toy_model() -> nn.Module:
    emsize = 32
    encoder = nn.Sequential(
        encoders.Normalize(0.0, 101.0),
        encoders.Normalize(0.5, (1 / 12) ** 0.5),
        encoders.Linear(1, emsize),
    )
    y_encoder = encoders.get_normalized_uniform_encoder(encoders.Linear)(1, emsize)
    model = transformer.TransformerModel(
        encoder=encoder,
        n_out=16,
        ninp=emsize,
        nhead=4,
        nhid=64,
        nlayers=2,
        dropout=0.0,
        y_encoder=y_encoder,
        pos_encoder=positional_encodings.NoPositionalEncoding(emsize),
    )
    bucket_limits = torch.linspace(0, 1, 17)
    model.criterion = distributions.FullSupportBarDistribution(bucket_limits)
    return model


def test_predict_curve_shapes():
    model = _toy_model()
    steps_seen = torch.arange(5, dtype=torch.float32)
    acc_seen = torch.tensor([0.2, 0.3, 0.45, 0.55, 0.6], dtype=torch.float32)
    future_steps = torch.tensor([6.0, 7.0, 8.0], dtype=torch.float32)

    pred_mean, pred_std = predict_curve(model, steps_seen, acc_seen, future_steps)

    assert pred_mean.shape == future_steps.shape
    assert pred_std.shape == future_steps.shape
    assert torch.isfinite(pred_mean).all()
    assert torch.isfinite(pred_std).all()
