# LC-PFN

Learning Curve Prior-Fitted Networks (LC-PFN) provide amortized Bayesian extrapolation of
learning curves using a transformer trained on synthetic or historical curves. This
repository packages the original research code into a modern Python library with a clean
inference API, reproducible training utilities, and CI-ready tooling.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

LC-PFN targets Python 3.10–3.12 and only requires CPU dependencies by default. CUDA is
optional and only used when available.

## Quickstart

```python
import torch
from lcpfn import load_pretrained, predict_curve

model = load_pretrained("default")
steps_seen = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
acc_seen = torch.tensor([0.32, 0.45, 0.51, 0.55, 0.57], dtype=torch.float32)
future_steps = torch.tensor([10, 20, 30], dtype=torch.float32)

pred_mean, pred_std = predict_curve(model, steps_seen, acc_seen, future_steps)
print(pred_mean, pred_std)
```

Observed accuracies should be monotonically non-decreasing and normalized to ``[0, 1]``.
We recommend using the cumulative best validation accuracy when building the inputs.

## Meta-training your own prior

```python
import torch
from lcpfn.train import MetaTrainConfig, train_meta


def synthetic_batch(batch_size: int, seq_len: int, num_features: int, **_) -> tuple:
    steps = torch.arange(1, seq_len + 1, dtype=torch.float32)
    x = steps.view(seq_len, 1, 1).repeat(1, batch_size, num_features)
    progress = steps / seq_len
    y = torch.sigmoid(4 * (progress - 0.5)).unsqueeze(1).repeat(1, batch_size)
    return x, y, y.clone()

config = MetaTrainConfig(seq_len=64, emsize=256, nlayers=6, num_borders=256)
model = train_meta(synthetic_batch, config, checkpoint_path="checkpoints/lcpfn.pt")
```

Checkpoints contain the trained transformer state dict, the training configuration, and
metadata (library version and git revision) for reproducibility.

## Development workflow

* Format and lint with ``ruff`` and ``black``.
* Run ``pytest`` for unit tests.
* ``mypy`` validates the public API type hints.
* CI is configured via GitHub Actions under ``.github/workflows/ci.yml``.

## Documentation

Rendered documentation lives in ``docs/`` (powered by MkDocs Material). The main entry
points are:

* ``docs/index.md`` – project overview
* ``docs/inference.md`` – how to run inference
* ``docs/training.md`` – reproducing meta-training
* ``docs/api_reference.md`` – auto-generated API summary

To build the docs locally:

```bash
mkdocs serve
```

## License and citation

LC-PFN is released under the MIT License. If you use the project in your research please
cite:

```
@inproceedings{adriaensens2023lcpfn,
  title={Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks},
  author={Adriaensen, Steven and Rakotoarison, Herilalaina and M{"u}ller, Samuel and Hutter, Frank},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=xgTV6rmH6n}
}
```
