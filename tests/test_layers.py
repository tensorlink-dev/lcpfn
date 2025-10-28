from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcpfn.models.layers import TransformerEncoderLayer


def test_transformer_encoder_layer_supports_is_causal_keyword():
    layer = TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)
    x = torch.randn(2, 5, 16)
    causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(5)

    # Direct invocation should accept the is_causal keyword without raising.
    layer(x, is_causal=False)
    layer(x, src_mask=causal_mask, is_causal=True)

    # The layer should also integrate with torch.nn.TransformerEncoder which
    # forwards the is_causal flag during execution.
    encoder = torch.nn.TransformerEncoder(layer, num_layers=1)
    encoder(x)
