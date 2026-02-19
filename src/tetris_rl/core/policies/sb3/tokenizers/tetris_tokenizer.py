# src/tetris_rl/core/policies/sb3/tokenizers/tetris_tokenizer.py
"""
TetrisTokenizer (Option B: CLS lives in the mixer)

Implements Tokenizer Protocol:
  (SpatialFeatures, Specials) -> TokenStream

Tokenizer responsibilities:
- Build token stream of environment-derived tokens only (no CLS).
- Extract board tokens according to layout:
    - row | column | patch | row_column
- Embed board tokens to D via one of:
    - linear projection
    - conv1d stripe embedder (row/column/row_column only)
    - discrete_pattern embedding (binary-only; row/column/patch)
- Add positional encodings INSIDE tokenizer (row/col positions).
- Optionally add Specials as tokens (active_kind, next_kind) via embedding table(s).
- Expose deterministic token counts AFTER construction (depends on SpatialSpec + layout + specials flags).

Mixer responsibilities:
- Optionally prepend CLS (+ its types)
- Add type embeddings (using stream.types)
- Mix + pool -> (B,F)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from tetris_rl.core.policies.sb3.api import SpatialFeatures, SpatialSpec, Specials, TokenStream, TokenStreamSpec
from tetris_rl.core.policies.sb3.layers.token_type import TokenType
from tetris_rl.core.policies.sb3.tokenizers.config import (
    BoardEmbeddingConfig,
    BoardEmbedType,
    ColumnLayoutParams,
    Conv1DEmbedParams,
    LayoutConfig,
    PatchLayoutParams,
    RowColumnLayoutParams,
    RowLayoutParams,
)
from tetris_rl.core.policies.sb3.tokenizers.embeddings.conv1d import Conv1DEmbedder
from tetris_rl.core.policies.sb3.tokenizers.layout.patch import PatchTokenizer

# Not user-configurable: keep tables small and predictable.
_MAX_PATTERN_BITS: int = 12


class TetrisTokenizer(nn.Module):
    """
    Important invariants (Tetris-specific):
      - Token counts are deterministic from:
          * SpatialSpec (H,W,C, is_discrete_binary)
          * layout (incl patch params)
          * add_active_token / add_next_token
      - NO lazy parameter creation: all parameters are allocated in __init__.
    """

    def __init__(
            self,
            *,
            d_model: int,
            layout: LayoutConfig,
            board_embedding: BoardEmbeddingConfig,
            add_active_token: bool = True,
            add_next_token: bool = False,
            # NEW: whether ACTIVE/NEXT share one kind embedding table
            share_kind_embedding: bool = True,
            # injected by factory if add_active_token or add_next_token
            n_kinds: Optional[int] = None,
            # injected by orchestrator: post-(preproc+stem) spatial spec
            in_spec: SpatialSpec,
    ) -> None:
        super().__init__()

        self.d_model = int(d_model)
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")

        self.in_spec = in_spec
        self.H = int(in_spec.h)
        self.W = int(in_spec.w)
        self.C = int(in_spec.c)
        if self.H <= 0 or self.W <= 0 or self.C <= 0:
            raise ValueError(f"invalid SpatialSpec: (H,W,C)=({self.H},{self.W},{self.C})")

        # Normalize discriminator strings defensively (even though schema uses Literals).
        self.layout_cfg = LayoutConfig(
            type=str(layout.type).strip().lower(),  # type: ignore[arg-type]
            params=layout.params,
        )
        self.embed_cfg = BoardEmbeddingConfig(
            type=str(board_embedding.type).strip().lower(),  # type: ignore[arg-type]
            params=board_embedding.params,
        )

        if self.layout_cfg.type not in {"row", "column", "patch", "row_column"}:
            raise ValueError(f"unknown layout: {self.layout_cfg.type!r}")
        if self.embed_cfg.type not in {"linear", "conv1d", "discrete_pattern"}:
            raise ValueError(f"unknown board_embedding type: {self.embed_cfg.type!r}")

        self.add_active_token = bool(add_active_token)
        self.add_next_token = bool(add_next_token)
        self.share_kind_embedding = bool(share_kind_embedding)

        if self.add_active_token or self.add_next_token:
            if n_kinds is None or int(n_kinds) <= 0:
                raise ValueError(
                    "n_kinds must be provided (>0) when add_active_token or add_next_token is enabled "
                    "(inject from env/assets in factory)"
                )
            self.n_kinds = int(n_kinds)
        else:
            self.n_kinds = None

        # Static compatibility checks
        if self.embed_cfg.type == "conv1d" and self.layout_cfg.type == "patch":
            raise ValueError("board_embedding.type='conv1d' does not support layout='patch'")

        # ------------------------------------------------------------------
        # Deterministic token counts (used by mixer construction)
        # ------------------------------------------------------------------
        self._n_board_tokens = int(self._compute_n_board_tokens())
        self._n_special_tokens = int((1 if self.add_active_token else 0) + (1 if self.add_next_token else 0))
        self._stream_spec = TokenStreamSpec(T=self._n_board_tokens + self._n_special_tokens, d_model=self.d_model)

        # ------------------------------------------------------------------
        # Positional embeddings (ONLY allocate what we might use)
        # ------------------------------------------------------------------
        layout_kind = self.layout_cfg.type
        needs_row_pos = False
        needs_col_pos = False
        row_pos_size = int(self.H)
        col_pos_size = int(self.W)
        self._patch_use_row_pos: bool = False
        self._patch_use_col_pos: bool = False

        def _resolve_switch(v: object, *, auto_default: bool) -> bool:
            if isinstance(v, str):
                s = v.strip().lower()
                if s == "auto":
                    return bool(auto_default)
                if s in {"true", "1", "yes", "on"}:
                    return True
                if s in {"false", "0", "no", "off"}:
                    return False
                raise ValueError(f"invalid positional switch value: {v!r}")
            return bool(v)

        if layout_kind == "row":
            params = self.layout_cfg.params
            if not isinstance(params, RowLayoutParams):
                raise TypeError(f"layout='row' requires RowLayoutParams, got {type(params).__name__}")
            needs_row_pos = _resolve_switch(getattr(params, "use_row_pos", "auto"), auto_default=True)
            needs_col_pos = False

        elif layout_kind == "column":
            params = self.layout_cfg.params
            if not isinstance(params, ColumnLayoutParams):
                raise TypeError(f"layout='column' requires ColumnLayoutParams, got {type(params).__name__}")
            needs_row_pos = False
            needs_col_pos = _resolve_switch(getattr(params, "use_col_pos", "auto"), auto_default=True)

        elif layout_kind == "row_column":
            params = self.layout_cfg.params
            if not isinstance(params, RowColumnLayoutParams):
                raise TypeError(f"layout='row_column' requires RowColumnLayoutParams, got {type(params).__name__}")
            needs_row_pos = _resolve_switch(getattr(params, "use_row_pos", "auto"), auto_default=True)
            needs_col_pos = _resolve_switch(getattr(params, "use_col_pos", "auto"), auto_default=True)

        if layout_kind == "patch":
            params = self.layout_cfg.params
            if not isinstance(params, PatchLayoutParams):
                raise TypeError(f"layout='patch' requires PatchLayoutParams, got {type(params).__name__}")
            ph = int(params.patch_h)
            pw = int(params.patch_w)
            sh = int(params.stride_h) if params.stride_h is not None else int(params.patch_h)
            sw = int(params.stride_w) if params.stride_w is not None else int(params.patch_w)
            if ph <= 0 or pw <= 0:
                raise ValueError("patch_h/patch_w must be > 0")
            if sh <= 0 or sw <= 0:
                raise ValueError("stride_h/stride_w must be > 0")
            if ph > self.H or pw > self.W:
                raise ValueError(
                    f"patch ({ph},{pw}) cannot exceed grid ({self.H},{self.W})"
                )
            n_h = int(((self.H - ph) // sh) + 1)
            n_w = int(((self.W - pw) // sw) + 1)
            row_pos_size = n_h
            col_pos_size = n_w
            self._patch_use_row_pos = _resolve_switch(
                getattr(params, "use_row_pos", "auto"),
                auto_default=(n_h > 1),
            )
            self._patch_use_col_pos = _resolve_switch(
                getattr(params, "use_col_pos", "auto"),
                auto_default=(n_w > 1),
            )
            needs_row_pos = bool(self._patch_use_row_pos)
            needs_col_pos = bool(self._patch_use_col_pos)

        self._row_pos: Optional[nn.Embedding] = nn.Embedding(int(row_pos_size), self.d_model) if needs_row_pos else None
        self._col_pos: Optional[nn.Embedding] = nn.Embedding(int(col_pos_size), self.d_model) if needs_col_pos else None

        # ------------------------------------------------------------------
        # Specials embeddings (shared or separate)
        # ------------------------------------------------------------------
        self.kind_emb_shared: Optional[nn.Embedding] = None
        self.kind_emb_active: Optional[nn.Embedding] = None
        self.kind_emb_next: Optional[nn.Embedding] = None

        if self.add_active_token or self.add_next_token:
            assert self.n_kinds is not None
            if self.share_kind_embedding:
                self.kind_emb_shared = nn.Embedding(int(self.n_kinds), self.d_model)
            else:
                if self.add_active_token:
                    self.kind_emb_active = nn.Embedding(int(self.n_kinds), self.d_model)
                if self.add_next_token:
                    self.kind_emb_next = nn.Embedding(int(self.n_kinds), self.d_model)

        # ------------------------------------------------------------------
        # Board embedding modules (NO lazy creation)
        # ------------------------------------------------------------------
        self._proj_row: Optional[nn.Linear] = None
        self._proj_col: Optional[nn.Linear] = None
        self._proj_patch: Optional[nn.Linear] = None

        self._conv_row: Optional[Conv1DEmbedder] = None
        self._conv_col: Optional[Conv1DEmbedder] = None

        self._pattern_row: Optional[nn.Embedding] = None
        self._pattern_col: Optional[nn.Embedding] = None
        self._pattern_patch: Optional[nn.Embedding] = None

        self._patch_tok: Optional[PatchTokenizer] = None

        emb_type = self.embed_cfg.type

        # --- patch tokenizer (if needed) ---
        if layout_kind == "patch":
            params = self.layout_cfg.params
            if not isinstance(params, PatchLayoutParams):
                raise TypeError(f"layout='patch' requires PatchLayoutParams, got {type(params).__name__}")
            self._patch_tok = PatchTokenizer(
                patch_h=params.patch_h,
                patch_w=params.patch_w,
                stride_h=params.stride_h,
                stride_w=params.stride_w,
            )

        # --- linear projections (deterministic from SpatialSpec + layout params) ---
        if emb_type == "linear":
            if layout_kind in {"row", "row_column"}:
                f_row = self.W * self.C
                self._proj_row = nn.Linear(int(f_row), self.d_model, bias=True)
            if layout_kind in {"column", "row_column"}:
                f_col = self.H * self.C
                self._proj_col = nn.Linear(int(f_col), self.d_model, bias=True)
            if layout_kind == "patch":
                assert self._patch_tok is not None
                params = self.layout_cfg.params
                assert isinstance(params, PatchLayoutParams)
                f_patch = int(params.patch_h) * int(params.patch_w) * self.C
                self._proj_patch = nn.Linear(int(f_patch), self.d_model, bias=True)

        # --- conv1d embedders (deterministic from C + conv spec) ---
        if emb_type == "conv1d":
            params = self.embed_cfg.params
            if not isinstance(params, Conv1DEmbedParams):
                raise RuntimeError(
                    f"board_embedding.type='conv1d' requires Conv1DEmbedParams, got {type(params).__name__}"
                )

            if layout_kind in {"row", "row_column"}:
                self._conv_row = Conv1DEmbedder(
                    in_channels=self.C,
                    d_model=self.d_model,
                    params=params,
                )
            if layout_kind in {"column", "row_column"}:
                self._conv_col = Conv1DEmbedder(
                    in_channels=self.C,
                    d_model=self.d_model,
                    params=params,
                )

        # --- discrete pattern tables (deterministic from layout + bits) ---
        if emb_type == "discrete_pattern":
            # requires binary spatial features and C==1 (fail fast)
            if not bool(in_spec.is_discrete_binary):
                raise ValueError("board_embedding.type='discrete_pattern' requires SpatialSpec.is_discrete_binary=True")
            if self.C != 1:
                raise ValueError(f"discrete_pattern requires C==1, got C={self.C}")

            if layout_kind in {"row", "row_column"}:
                self._pattern_row = self._make_pattern_table(n_bits=self.W)
            if layout_kind in {"column", "row_column"}:
                self._pattern_col = self._make_pattern_table(n_bits=self.H)
            if layout_kind == "patch":
                params = self.layout_cfg.params
                if not isinstance(params, PatchLayoutParams):
                    raise TypeError(f"layout='patch' requires PatchLayoutParams, got {type(params).__name__}")
                n_bits = int(params.patch_h) * int(params.patch_w)
                self._pattern_patch = self._make_pattern_table(n_bits=n_bits)

    # ------------------------------------------------------------------
    # Public metadata
    # ------------------------------------------------------------------

    def stream_spec(self) -> TokenStreamSpec:
        return self._stream_spec

    @property
    def n_board_tokens(self) -> int:
        return int(self._n_board_tokens)

    @property
    def n_special_tokens(self) -> int:
        return int(self._n_special_tokens)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *, spatial: SpatialFeatures, specials: Specials) -> TokenStream:
        x = spatial.x
        if x.dim() != 4:
            raise ValueError(f"SpatialFeatures.x must be (B,H,W,C), got {tuple(x.shape)}")

        B = int(x.shape[0])
        H = int(x.shape[1])
        W = int(x.shape[2])
        C = int(x.shape[3])
        if H != self.H or W != self.W or C != self.C:
            raise ValueError(
                f"spatial shape changed: expected (H,W,C)=({self.H},{self.W},{self.C}), got ({H},{W},{C})"
            )

        # 1) Board tokens + types
        board_x, board_types = self._embed_board(spatial=spatial)  # (B,Tb,D), (Tb,)

        tokens: list[torch.Tensor] = [board_x]
        types: list[torch.Tensor] = [board_types]

        # 2) Optional specials tokens (end-append)
        if self.add_active_token:
            active_idx = self._as_batched_int64(specials.active_kind, B)
            if self.share_kind_embedding:
                if self.kind_emb_shared is None:
                    raise RuntimeError("kind_emb_shared is None but share_kind_embedding=True")
                active_tok = self.kind_emb_shared(active_idx).unsqueeze(1)  # (B,1,D)
            else:
                if self.kind_emb_active is None:
                    raise RuntimeError("kind_emb_active is None but share_kind_embedding=False and add_active_token=True")
                active_tok = self.kind_emb_active(active_idx).unsqueeze(1)  # (B,1,D)

            tokens.append(active_tok)
            types.append(torch.full((1,), int(TokenType.ACTIVE), device=board_x.device, dtype=torch.int64))

        if self.add_next_token and specials.next_kind is not None:
            next_idx = self._as_batched_int64(specials.next_kind, B)
            if self.share_kind_embedding:
                if self.kind_emb_shared is None:
                    raise RuntimeError("kind_emb_shared is None but share_kind_embedding=True")
                next_tok = self.kind_emb_shared(next_idx).unsqueeze(1)  # (B,1,D)
            else:
                if self.kind_emb_next is None:
                    raise RuntimeError("kind_emb_next is None but share_kind_embedding=False and add_next_token=True")
                next_tok = self.kind_emb_next(next_idx).unsqueeze(1)  # (B,1,D)

            tokens.append(next_tok)
            types.append(torch.full((1,), int(TokenType.NEXT), device=board_x.device, dtype=torch.int64))

        out_x = torch.cat(tokens, dim=1)  # (B,T,D)
        out_types = torch.cat(types, dim=0)  # (T,)

        # sanity: keep counts honest (helps catch bugs in patch math)
        expected_T = int(self.stream_spec().T)
        if int(out_types.shape[0]) != expected_T:
            raise RuntimeError(f"token count mismatch: expected T={expected_T}, got T={int(out_types.shape[0])}")

        return TokenStream(x=out_x, types=out_types)

    # ------------------------------------------------------------------
    # Token count math
    # ------------------------------------------------------------------

    def _compute_n_board_tokens(self) -> int:
        layout = self.layout_cfg.type
        H = int(self.H)
        W = int(self.W)

        if layout == "row":
            return H
        if layout == "column":
            return W
        if layout == "row_column":
            return H + W
        if layout == "patch":
            params = self.layout_cfg.params
            if not isinstance(params, PatchLayoutParams):
                raise TypeError(f"layout='patch' requires PatchLayoutParams, got {type(params).__name__}")
            ph = int(params.patch_h)
            pw = int(params.patch_w)
            sh = int(params.stride_h) if params.stride_h is not None else ph
            sw = int(params.stride_w) if params.stride_w is not None else pw
            if ph <= 0 or pw <= 0:
                raise ValueError("patch_h/patch_w must be > 0")
            if sh <= 0 or sw <= 0:
                raise ValueError("stride_h/stride_w must be > 0")
            if ph > H or pw > W:
                return 0
            n_h = ((H - ph) // sh) + 1
            n_w = ((W - pw) // sw) + 1
            return int(n_h * n_w)

        raise ValueError(f"unknown layout: {layout!r}")

    # ------------------------------------------------------------------
    # Board embedding
    # ------------------------------------------------------------------

    def _embed_board(self, *, spatial: SpatialFeatures) -> tuple[torch.Tensor, torch.Tensor]:
        layout = self.layout_cfg.type
        emb_type = self.embed_cfg.type

        if layout == "row":
            emb = self._embed_rows(spatial=spatial, emb_type=emb_type)  # (B,H,D)
            types = torch.full((emb.shape[1],), int(TokenType.ROW), device=emb.device, dtype=torch.int64)
            return emb, types

        if layout == "column":
            emb = self._embed_cols(spatial=spatial, emb_type=emb_type)  # (B,W,D)
            types = torch.full((emb.shape[1],), int(TokenType.COL), device=emb.device, dtype=torch.int64)
            return emb, types

        if layout == "patch":
            emb = self._embed_patches(spatial=spatial, emb_type=emb_type)  # (B,T,D)
            types = torch.full((emb.shape[1],), int(TokenType.PATCH), device=emb.device, dtype=torch.int64)
            return emb, types

        if layout == "row_column":
            rows = self._embed_rows(spatial=spatial, emb_type=emb_type)  # (B,H,D)
            cols = self._embed_cols(spatial=spatial, emb_type=emb_type)  # (B,W,D)
            emb = torch.cat([rows, cols], dim=1)  # (B,H+W,D)
            types = torch.cat(
                [
                    torch.full((rows.shape[1],), int(TokenType.ROW), device=emb.device, dtype=torch.int64),
                    torch.full((cols.shape[1],), int(TokenType.COL), device=emb.device, dtype=torch.int64),
                ],
                dim=0,
            )
            return emb, types

        raise ValueError(f"unknown layout: {layout!r}")

    # ------------------------------------------------------------------
    # Row / Column
    # ------------------------------------------------------------------

    def _embed_rows(self, *, spatial: SpatialFeatures, emb_type: BoardEmbedType) -> torch.Tensor:
        x = spatial.x
        B, H, W, C = x.shape

        if emb_type == "conv1d":
            if self._conv_row is None:
                raise RuntimeError("conv1d row embedder is None (bad init/config)")
            stripes = x.reshape(B, H, W, C)  # (B,T=H,L=W,C)
            kind = "row" if self._conv_row.params.padding == "tetris" else None
            emb = self._conv_row(stripes, kind=kind)  # (B,H,D)



        elif emb_type == "linear":
            if self._proj_row is None:
                raise RuntimeError("linear row projection is None (bad init/config)")
            tok = x.reshape(B, H, W * C)  # (B,H,W*C)
            emb = self._proj_row(tok)  # (B,H,D)

        elif emb_type == "discrete_pattern":
            if self._pattern_row is None:
                raise RuntimeError("pattern row table is None (bad init/config)")
            emb = self._embed_discrete_patterns_from_rows(spatial=spatial, tab=self._pattern_row)  # (B,H,D)

        else:
            raise ValueError(f"unknown board_embedding type: {emb_type!r}")

        if self._row_pos is None:
            return emb
        pos = self._row_pos(torch.arange(H, device=x.device, dtype=torch.int64))  # (H,D)
        return emb + pos.unsqueeze(0)

    def _embed_cols(self, *, spatial: SpatialFeatures, emb_type: BoardEmbedType) -> torch.Tensor:
        x = spatial.x
        B, H, W, C = x.shape

        if emb_type == "conv1d":
            if self._conv_col is None:
                raise RuntimeError("conv1d col embedder is None (bad init/config)")
            stripes = x.permute(0, 2, 1, 3).contiguous()  # (B,T=W,L=H,C)
            kind = "col" if self._conv_col.params.padding == "tetris" else None
            emb = self._conv_col(stripes, kind=kind)  # (B,W,D)


        elif emb_type == "linear":
            if self._proj_col is None:
                raise RuntimeError("linear col projection is None (bad init/config)")
            tok = x.permute(0, 2, 1, 3).contiguous().reshape(B, W, H * C)  # (B,W,H*C)
            emb = self._proj_col(tok)  # (B,W,D)

        elif emb_type == "discrete_pattern":
            if self._pattern_col is None:
                raise RuntimeError("pattern col table is None (bad init/config)")
            emb = self._embed_discrete_patterns_from_cols(spatial=spatial, tab=self._pattern_col)  # (B,W,D)

        else:
            raise ValueError(f"unknown board_embedding type: {emb_type!r}")

        if self._col_pos is None:
            return emb
        pos = self._col_pos(torch.arange(W, device=x.device, dtype=torch.int64))  # (W,D)
        return emb + pos.unsqueeze(0)

    # ------------------------------------------------------------------
    # Patches
    # ------------------------------------------------------------------

    def _embed_patches(self, *, spatial: SpatialFeatures, emb_type: BoardEmbedType) -> torch.Tensor:
        if emb_type == "conv1d":
            raise ValueError("conv1d embedding does not support patch layout")

        if self._patch_tok is None:
            raise RuntimeError("patch tokenizer is None but layout='patch'")

        tokens, pos_h, pos_w = self._patch_tok(spatial=spatial)  # (B,T,F_raw), (T,), (T,)
        if pos_h is None or pos_w is None:
            raise RuntimeError("patch layout must return pos_h and pos_w")

        if emb_type == "linear":
            if self._proj_patch is None:
                raise RuntimeError("linear patch projection is None (bad init/config)")
            emb = self._proj_patch(tokens)  # (B,T,D)

        elif emb_type == "discrete_pattern":
            if self._pattern_patch is None:
                raise RuntimeError("pattern patch table is None (bad init/config)")
            bits = (tokens > 0.5).to(torch.int64)  # (B,T,F_raw)
            emb = self._embed_pattern_bits(bits, tab=self._pattern_patch)  # (B,T,D)

        else:
            raise ValueError(f"unknown board_embedding type: {emb_type!r}")

        # add token-grid positional embeddings per patch index (when enabled)
        pos: Optional[torch.Tensor] = None
        if self._row_pos is not None:
            row_pos_n = int(self._row_pos.num_embeddings)
            row_tab = self._row_pos(torch.arange(row_pos_n, device=emb.device, dtype=torch.int64))  # (T_h,D)
            pos = row_tab[pos_h]
        if self._col_pos is not None:
            col_pos_n = int(self._col_pos.num_embeddings)
            col_tab = self._col_pos(torch.arange(col_pos_n, device=emb.device, dtype=torch.int64))  # (T_w,D)
            pos = col_tab[pos_w] if pos is None else (pos + col_tab[pos_w])
        if pos is not None:
            emb = emb + pos.unsqueeze(0)  # (B,T,D)
        return emb

    # ------------------------------------------------------------------
    # Discrete patterns (binary-only)
    # ------------------------------------------------------------------

    @staticmethod
    def _pattern_ids(bits: torch.Tensor) -> torch.Tensor:
        # bits: (B,T,n) int64 {0,1}
        n = int(bits.shape[-1])
        weights = (2 ** torch.arange(n, device=bits.device, dtype=torch.int64)).view(1, 1, n)
        return (bits * weights).sum(dim=2)  # (B,T)

    def _make_pattern_table(self, *, n_bits: int) -> nn.Embedding:
        n = int(n_bits)
        if n <= 0:
            raise ValueError("pattern bits must be > 0")
        if n > _MAX_PATTERN_BITS:
            raise ValueError(f"pattern bits={n} exceeds max_bits={_MAX_PATTERN_BITS} (increase constant if desired)")
        return nn.Embedding(1 << n, self.d_model)

    def _embed_pattern_bits(self, bits: torch.Tensor, *, tab: nn.Embedding) -> torch.Tensor:
        if bits.dtype != torch.int64:
            bits = bits.to(torch.int64)
        if torch.any((bits < 0) | (bits > 1)):
            raise ValueError("bits must be in {0,1}")
        ids = self._pattern_ids(bits)  # (B,T)
        return tab(ids)  # (B,T,D)

    def _embed_discrete_patterns_from_rows(self, *, spatial: SpatialFeatures, tab: nn.Embedding) -> torch.Tensor:
        x = spatial.x
        B, H, W, C = x.shape
        if not spatial.is_discrete_binary:
            raise ValueError("discrete_pattern requires spatial.is_discrete_binary=True")
        if C != 1:
            raise ValueError(f"discrete_pattern requires C==1, got C={C}")
        bits = (x[..., 0] > 0.5).to(torch.int64).reshape(B, H, W)  # (B,H,W)
        return self._embed_pattern_bits(bits, tab=tab)  # (B,H,D)

    def _embed_discrete_patterns_from_cols(self, *, spatial: SpatialFeatures, tab: nn.Embedding) -> torch.Tensor:
        x = spatial.x
        B, H, W, C = x.shape
        if not spatial.is_discrete_binary:
            raise ValueError("discrete_pattern requires spatial.is_discrete_binary=True")
        if C != 1:
            raise ValueError(f"discrete_pattern requires C==1, got C={C}")
        bits = (x[..., 0] > 0.5).to(torch.int64).permute(0, 2, 1).contiguous()  # (B,W,H)
        return self._embed_pattern_bits(bits, tab=tab)  # (B,W,D)

    # ------------------------------------------------------------------
    # Small helper
    # ------------------------------------------------------------------

    @staticmethod
    def _as_batched_int64(t: torch.Tensor, B: int) -> torch.Tensor:
        """
        Accepts:
          - scalar kind
          - (B,) kind
          - (B,K) one-hot kind (SB3 preprocessing)
        Returns (B,) int64 indices.
        """
        if t.dim() == 2:
            t = torch.argmax(t, dim=1)
        elif t.dim() == 0:
            t = t.view(1).expand(B)

        if t.dim() != 1 or int(t.shape[0]) != B:
            raise ValueError(f"expected kind tensor shape scalar/(B,)/(B,K), got {tuple(t.shape)} with B={B}")
        return t.to(dtype=torch.int64)


__all__ = ["TetrisTokenizer"]


