"""PyTorch building blocks for the CVAE.

These replace the former Keras helper layers. Note that in the TF graph
``FeatureLossLayer`` was instantiated but never connected to any tensor
(``model.py:64`` built it and discarded the result; ``model.py:117`` did the
same), and ``SampleLayer`` was commented out (``model.py:80-82,98-99``). The
active loss path was ``model.add_loss(list(feature_losses))`` plus ``kl*0``.
So the only behavior worth porting is the per-feature reconstruction loss and
the reparameterization, which now live on the features and the model module.

The text encoder/decoder mirror the Keras
``Bidirectional(LSTM(return_sequences=False))`` encoder and the
``RepeatVector → LSTM(return_sequences=True) → TimeDistributed(Dense)`` decoder
in ``features.py:610-625``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """z = mu + exp(log_sigma/2) * eps, eps ~ N(0,1).

    Mirrors ``CVAE.sample_z`` (``model.py:56-59``). TF draws a single
    ``(latent_dim,)`` eps and broadcasts it across the batch; we draw a
    per-row ``(batch, latent_dim)`` eps (standard reparameterization). With KL
    weight 0 the posterior is unregularized (``log_sigma`` drifts very negative,
    so ``z ≈ mu``) and generation samples the fitted BGM on ``mu`` rather than
    ``z``, so the difference is immaterial — documented in
    ``docs/migration/pytorch_backend_design.md``.
    """
    eps = torch.randn_like(mu)
    return mu + torch.exp(log_sigma / 2.0) * eps


class TextEncoder(nn.Module):
    """BiLSTM over a one-hot char sequence → concatenated final hidden state.

    Equivalent to Keras ``Bidirectional(LSTM(rnn_units, return_sequences=False))``
    (``features.py:612``): forward and backward final hidden states concatenated
    → ``(batch, 2*rnn_units)``.
    """

    def __init__(self, vocab_size: int, rnn_units: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=rnn_units,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, vocab)
        _, (h_n, _) = self.lstm(x)
        # h_n: (2, B, rnn_units) for a 1-layer bidirectional LSTM.
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)


class TextDecoder(nn.Module):
    """RepeatVector → LSTM(return_sequences=True) → TimeDistributed(Linear).

    Mirrors ``features.py:618-623``. Takes the shared-decoder output vector and
    expands it to a length-``text_max_len`` sequence of vocab logits.
    """

    def __init__(self, in_features: int, text_max_len: int, rnn_units: int, vocab_size: int):
        super().__init__()
        self.text_max_len = text_max_len
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=rnn_units,
            batch_first=True,
        )
        self.linear = nn.Linear(rnn_units, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, in_features)
        x = x.unsqueeze(1).expand(-1, self.text_max_len, -1)  # RepeatVector
        seq, _ = self.lstm(x)  # (B, T, rnn_units)
        return self.linear(seq)  # (B, T, vocab) — linear logits
