# Inference

```python
import torch
from lcpfn import load_pretrained, predict_curve

model = load_pretrained("default")
steps_seen = torch.arange(5, dtype=torch.float32)
acc_seen = torch.tensor([0.2, 0.4, 0.5, 0.55, 0.6])
future_steps = torch.tensor([8.0, 16.0, 32.0])

pred_mean, pred_std = predict_curve(model, steps_seen, acc_seen, future_steps)
```

## Input requirements

* ``steps_seen`` – observed training steps as a 1D tensor.
* ``acc_seen`` – monotonically non-decreasing validation or accuracy measurements in
  ``[0, 1]``.
* ``future_steps`` – steps to extrapolate.

Set ``enforce_monotonic=False`` if you already compute a cumulative best curve. The model
runs entirely on CPU unless you explicitly move it to GPU.
