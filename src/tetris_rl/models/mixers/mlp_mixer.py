# src/tetris_rl/models/mixers/mlp_mixer.py
from __future__ import annotations

"""
MLPMixer (EAGER)

Consumes TokenStream and returns (B, features_dim).

Tokenizer already applied positional encodings.
Mixer responsibilities:
- (optional) prepend CLS token(s) + prepend CLS type id(s)
- add learned type embeddings (using stream.types)
- run MLP-Mixer blocks:
    - token-mixing MLP (mix across token axis T, per-channel)
    - channel-mixing FFN (mix across channel axis D, per-token)
- pool tokens
- project to features_dim

This implementation is *fully eager*:
- d_model and T_total are provided at construction
- all submodules are built in __init__
- no lazy init / no device-dependent .to(...) inside forward
"""

import torch
from torch import nn

from tetris_rl.config.model.mixer_spec import MLPMixerParams
from tetris_rl.models.api import TokenStream
from tetris_rl.models.layers.cls import prepend_cls
from tetris_rl.models.layers.ffn import FFN, FFNSpec
from tetris_rl.models.layers.pooling import pool_tokens, pooled_dim
from tetris_rl.models.layers.token_type import TokenType, TokenTypeEncoding, TokenTypeEncodingSpec


# ---------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------


class _TokenMixMLP(nn.Module):
    """
    Token-mixing MLP: mixes across token axis T, per channel.

    x: (B,T,D) -> (B,T,D)
    """

    def __init__(self, *, T: int, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}")
        p = float(dropout)
        if p < 0.0:
            raise ValueError("dropout must be >= 0")

        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(T, hidden),
            nn.GELU(),
            nn.Dropout(p) if p > 0.0 else nn.Identity(),
            nn.Linear(hidden, T),
            nn.Dropout(p) if p > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TokenMix expects (B,T,D), got {tuple(x.shape)}")
        B, T, D = x.shape

        h = self.ln(x).transpose(1, 2).reshape(B * D, T)  # (B*D,T)
        h = self.mlp(h).reshape(B, D, T).transpose(1, 2)  # (B,T,D)
        return x + h


class _ChannelMixFFN(nn.Module):
    """
    Channel-mixing FFN: mixes across channel axis D, per token.

    x: (B,T,D) -> (B,T,D)
    """

    def __init__(self, *, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        p = float(dropout)
        if p < 0.0:
            raise ValueError("dropout must be >= 0")

        mult = float(hidden_dim) / float(d_model)
        self.ln = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model, spec=FFNSpec(mult=mult, dropout=p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ChannelMix expects (B,T,D), got {tuple(x.shape)}")
        return x + self.ffn(self.ln(x))


class _MLPMixerBlock(nn.Module):
    def __init__(self, *, T: int, d_model: int, token_hidden: int, channel_hidden: int, dropout: float) -> None:
        super().__init__()
        self.token_mix = _TokenMixMLP(T=T, d_model=d_model, hidden=token_hidden, dropout=dropout)
        self.channel_mix = _ChannelMixFFN(d_model=d_model, hidden_dim=channel_hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mix(self.token_mix(x))


# ---------------------------------------------------------------------
# Mixer
# ---------------------------------------------------------------------


class MLPMixer(nn.Module):
    """
    MLP-Mixer style TokenMixer.

    Contract:
      input:  TokenStream(x=(B,T,D), types=(T,))
      output: (B, features_dim)

    Eager invariants:
      - d_model is fixed
      - T_total (including optional CLS prefix) is fixed
    """

    def __init__(self, *, spec: MLPMixerParams, d_model: int, T_total: int) -> None:
        super().__init__()
        self.spec = spec

        D = int(d_model)
        T = int(T_total)

        if D <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if T <= 0:
            raise ValueError(f"T_total must be > 0, got {T_total}")

        if int(spec.features_dim) <= 0:
            raise ValueError(f"features_dim must be > 0, got {spec.features_dim}")
        if int(spec.n_layers) <= 0:
            raise ValueError(f"n_layers must be > 0, got {spec.n_layers}")
        if int(spec.token_mlp_dim) <= 0:
            raise ValueError("token_mlp_dim must be > 0")
        if int(spec.channel_mlp_dim) <= 0:
            raise ValueError("channel_mlp_dim must be > 0")
        if float(spec.dropout) < 0.0:
            raise ValueError("dropout must be >= 0")

        if bool(spec.use_cls):
            if int(spec.num_cls_tokens) <= 0:
                raise ValueError("num_cls_tokens must be > 0 when use_cls=True")
        else:
            if str(spec.pool).startswith("cls"):
                raise ValueError("cls* pooling requires use_cls=True")

        self._d_model = D
        self._T_total = T

        k = int(spec.num_cls_tokens) if bool(spec.use_cls) else 0
        self._num_cls = k

        # global token type universe (authoritative)
        n_types = int(len(TokenType))
        cls_type_id = int(TokenType.CLS)

        self.type_enc = TokenTypeEncoding(
            d_model=D,
            spec=TokenTypeEncodingSpec(n_types=n_types, dropout=float(spec.dropout)),
        )

        if k > 0:
            self.cls_token = nn.Parameter(torch.zeros((1, k, D), dtype=torch.float32))
        else:
            self.cls_token = None

        self.input_ln = nn.LayerNorm(D) if bool(spec.pre_ln_input) else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                _MLPMixerBlock(
                    T=T,
                    d_model=D,
                    token_hidden=int(spec.token_mlp_dim),
                    channel_hidden=int(spec.channel_mlp_dim),
                    dropout=float(spec.dropout),
                )
                for _ in range(int(spec.n_layers))
            ]
        )

        pooled_width = pooled_dim(kind=spec.pool, T=T, D=D, num_cls_tokens=k)
        F = int(spec.features_dim)
        self.out_proj = nn.Identity() if pooled_width == F else nn.Linear(pooled_width, F, bias=True)

        self._cls_type_id = cls_type_id

    def forward(self, *, stream: TokenStream) -> torch.Tensor:
        x = stream.x
        types = stream.types

        if x.dim() != 3:
            raise ValueError(f"TokenStream.x must be (B,T,D), got {tuple(x.shape)}")
        if types.dim() != 1:
            raise ValueError(f"TokenStream.types must be (T,), got {tuple(types.shape)}")
        if int(types.shape[0]) != int(x.shape[1]):
            raise ValueError(f"type length mismatch: types T={int(types.shape[0])} vs x T={int(x.shape[1])}")

        _B, T0, D = x.shape
        if int(D) != int(self._d_model):
            raise ValueError(f"d_model mismatch: mixer expects {self._d_model}, got {D}")

        expected_T0 = int(self._T_total - self._num_cls)
        if int(T0) != expected_T0:
            raise ValueError(
                f"token count mismatch: mixer expects T0={expected_T0} "
                f"(T_total={self._T_total}, cls={self._num_cls}), got {T0}"
            )

        if self._num_cls > 0:
            assert self.cls_token is not None
            x, types = prepend_cls(x=x, types=types, cls=self.cls_token, cls_type_id=int(self._cls_type_id))

        x = x + self.type_enc(types=types)
        x = self.input_ln(x)

        for blk in self.blocks:
            x = blk(x)

        pooled = pool_tokens(x=x, kind=self.spec.pool, num_cls_tokens=self._num_cls)

        out = self.out_proj(pooled)
        if out.dim() != 2:
            raise ValueError(f"MLPMixer must return (B,F), got {tuple(out.shape)}")
        return out


__all__ = ["MLPMixer"]
