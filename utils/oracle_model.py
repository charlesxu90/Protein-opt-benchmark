"""
Learned fitness oracle (GGS / LatProtRL `BaseCNN`).

Ported verbatim (architecture-wise) from GGS `ggs/models/predictors.py`
(Kirjner et al., ICML 2024) so that oracle checkpoints are interchangeable
with the reference implementation. The model is length-agnostic: a global
max-pool over the sequence dimension lets one class serve AAV (28), PAB1 (75),
CreiLOV (119) and GFP (237) without changes.

Encoding convention: integer-encoded sequences with `make_one_hot=True`, or
pre-one-hot tensors of shape (N, L, n_tokens) with `make_one_hot=False`.
Alphabet matches GGS: "ARNDCQEGHILKMFPSTWYV".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# GGS alphabet ordering (configs/train_predictor.yaml: data.alphabet)
ALPHABET = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}
N_TOKENS = len(ALPHABET)


class LengthMaxPool1D(nn.Module):
    """Linear projection + activation followed by max-pool over length."""

    def __init__(self, in_dim: int, out_dim: int, linear: bool = False,
                 activation: str = "relu"):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)
        if activation == "relu":
            self.act_fn = lambda x: F.relu(x)
        elif activation == "leakyrelu":
            self.act_fn = nn.LeakyReLU()
        elif activation == "softplus":
            self.act_fn = nn.Softplus()
        elif activation == "sigmoid":
            self.act_fn = nn.Sigmoid()
        else:
            raise NotImplementedError(activation)

    def forward(self, x):
        if self.linear:
            x = self.act_fn(self.layer(x))
        return torch.max(x, dim=1)[0]


class BaseCNN(nn.Module):
    """GGS/LatProtRL CNN fitness predictor. Outputs a scalar fitness per sequence."""

    def __init__(
        self,
        n_tokens: int = N_TOKENS,
        kernel_size: int = 5,
        input_size: int = 256,
        dropout: float = 0.0,
        make_one_hot: bool = True,
        activation: str = "relu",
        linear: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(
            linear=linear, in_dim=input_size, out_dim=input_size * 2,
            activation=activation,
        )
        self.decoder = nn.Linear(input_size * 2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self._make_one_hot = make_one_hot

    def forward(self, x):
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        return self.decoder(x).squeeze(1)


# =============================================================================
# Encoding helpers
# =============================================================================

def encode_int(sequences, seq_len: int = None) -> torch.Tensor:
    """Integer-encode sequences to a (N, L) long tensor for `make_one_hot=True`."""
    if seq_len is None:
        seq_len = len(sequences[0])
    out = torch.zeros(len(sequences), seq_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:seq_len]):
            out[i, j] = AA_TO_IDX.get(aa, 0)
    return out
