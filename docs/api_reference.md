# API Reference

## ``lcpfn`` top-level

* :func:`lcpfn.load_pretrained` – load packaged or user-provided checkpoints.
* :func:`lcpfn.predict_curve` – run inference on partial learning curves.
* :class:`lcpfn.train.MetaTrainConfig` – dataclass configuring meta-training.
* :func:`lcpfn.train.train_meta` – launch meta-training for custom priors.

## Subpackages

* ``lcpfn.models`` – transformer, encoders, and criterion utilities.
* ``lcpfn.data`` – curve priors and sampling helpers.
* ``lcpfn.inference`` – inference helpers.
* ``lcpfn.train`` – training loop and configuration.
