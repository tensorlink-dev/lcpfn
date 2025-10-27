# Meta-training

Meta-training fits the LC-PFN transformer to a distribution over learning curves. Use
:func:`lcpfn.train.train_meta` with a :class:`lcpfn.train.MetaTrainConfig` instance.

```python
from lcpfn.train import MetaTrainConfig, train_meta

config = MetaTrainConfig(seq_len=64, emsize=256, nlayers=6, num_borders=256)
model = train_meta(my_curve_sampler, config, checkpoint_path="checkpoints/lcpfn.pt")
```

The saved checkpoint stores the model state dict, the training configuration, and the git
revision (if available) for reproducibility.

## Required batch sampler

``my_curve_sampler`` must follow the original PFN API and return ``(x, y, target_y)`` where:

* ``x`` – tensor shaped ``(seq_len, batch_size, num_features)``
* ``y`` – noisy curve observations
* ``target_y`` – noise-free targets used for computing the loss

The helper :func:`lcpfn.data.domhan_prior.create_get_batch_func` reproduces the synthetic
curves from the original paper.
