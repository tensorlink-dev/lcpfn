import pytest

torch = pytest.importorskip("torch")

from lcpfn.train import MetaTrainConfig, train_meta


def _synthetic_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    *,
    hyperparameters=None,
    single_eval_pos: int | None = None,
    **_: int,
):
    steps = torch.arange(1, seq_len + 1, dtype=torch.float32)
    x = steps.view(seq_len, 1, 1).repeat(1, batch_size, num_features)
    progress = steps.float() / seq_len
    y = torch.sigmoid(5 * (progress - 0.5)).unsqueeze(1).repeat(1, batch_size)
    return x, y, y.clone()


def test_meta_training_smoke(tmp_path):
    config = MetaTrainConfig(
        seq_len=8,
        emsize=32,
        nlayers=2,
        num_borders=16,
        lr=1e-3,
        batch_size=4,
        epochs=1,
        steps_per_epoch=2,
        device="cpu",
    )

    model = train_meta(_synthetic_batch, config, checkpoint_path=tmp_path / "ckpt.pt")
    assert hasattr(model, "criterion")
    assert (tmp_path / "ckpt.pt").exists()
