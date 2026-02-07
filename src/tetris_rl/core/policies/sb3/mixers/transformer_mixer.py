# src/tetris_rl/core/policies/sb3/mixers/transformer_mixer.py
from __future__ import annotations

"""
TransformerMixer (EAGER)

Consumes a TokenStream and returns a fixed-size feature vector (B, features_dim).

Tokenizer responsibilities (upstream):
- build environment-derived tokens only (board + optional specials)
- apply positional encodings inside the tokenizer
- emit TokenStream(x=(B,T,D), types=(T,))

Mixer responsibilities (this module):
- optionally prepend one or more CLS tokens (mixer-owned)
- add learned token-type embeddings based on stream.types
- run a stack of Transformer encoder blocks
- pool tokens to (B,H) using PoolKind
- project to (B, features_dim)

This implementation is *fully eager*:
- d_model and T_total are provided at construction
- all submodules are built in __init__
- no lazy init / no device-dependent .to(...) inside forward
"""

import torch
from torch import nn

from tetris_rl.core.policies.sb3.mixers.config import TransformerMixerParams
from tetris_rl.core.policies.sb3.api import TokenStream
from tetris_rl.core.policies.sb3.layers.cls import prepend_cls
from tetris_rl.core.policies.sb3.layers.ffn import FFNSpec
from tetris_rl.core.policies.sb3.layers.pooling import pool_tokens, pooled_dim
from tetris_rl.core.policies.sb3.layers.token_type import TokenType, TokenTypeEncoding, TokenTypeEncodingSpec
from tetris_rl.core.policies.sb3.layers.transformer import TransformerBlock, TransformerBlockSpec


class TransformerMixer(nn.Module):
    """
    Transformer-style TokenMixer.

    Contract:
      input:  TokenStream(x=(B,T,D), types=(T,))
      output: (B, features_dim)

    Eager invariants:
      - d_model is fixed
      - T_total (including optional CLS prefix) is fixed
    """

    def __init__(self, *, spec: TransformerMixerParams, d_model: int, T_total: int) -> None:
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
        if int(spec.n_heads) <= 0:
            raise ValueError(f"n_heads must be > 0, got {spec.n_heads}")
        if float(spec.mlp_ratio) <= 0.0:
            raise ValueError(f"mlp_ratio must be > 0, got {spec.mlp_ratio}")
        if float(spec.dropout) < 0.0 or float(spec.attn_dropout) < 0.0 or float(spec.resid_dropout) < 0.0:
            raise ValueError("dropout, attn_dropout, resid_dropout must be >= 0")

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

        blk_spec = TransformerBlockSpec(
            n_heads=int(spec.n_heads),
            attn_dropout=float(spec.attn_dropout),
            resid_dropout=float(spec.resid_dropout),
            ffn=FFNSpec(mult=float(spec.mlp_ratio), dropout=float(spec.dropout)),
        )
        self.blocks = nn.ModuleList([TransformerBlock(d_model=D, spec=blk_spec) for _ in range(int(spec.n_layers))])

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
            raise ValueError(f"TransformerMixer must return (B,F), got {tuple(out.shape)}")
        return out


__all__ = ["TransformerMixer"]


