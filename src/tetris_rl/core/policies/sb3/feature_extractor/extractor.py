# src/tetris_rl/core/policies/sb3/feature_extractor/extractor.py
"""
SB3 Feature Extractor Orchestrator (and builder).

This is the ONLY class that Stable-Baselines3 interacts with.

Responsibilities
---------------
- Build the configured model pipeline from config + registries (catalog.py).
- Route observations through exactly one base feature route:
    A) token route   : preproc -> (optional stem) -> tokenizer -> token_mixer -> (B, F_base)
    B) spatial route : preproc -> (optional stem) -> spatial_head            -> (B, F_base)
- Optionally apply a feature augmenter:
    (B, F_base) + Specials -> (B, F_final)
- Enforce shape contracts and return (B, F_final).

IMPORTANT SB3 CONTRACT
----------------------
BaseFeaturesExtractor.features_dim MUST equal the actual returned feature dimension,
i.e. AFTER augmentation (F_final).
"""

from __future__ import annotations

from typing import Optional

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from tetris_rl.core.policies.sb3.api import (
    BoardSpec,
    SpatialPreprocessor,
    SpatialSpec,
    SpatialStem,
)
from tetris_rl.core.policies.sb3.feature_augmenters.config import FeatureAugmenterConfig
from tetris_rl.core.policies.sb3.mixers.config import MixerConfig
from tetris_rl.core.policies.sb3.spatial.config import SpatialPreprocessorConfig, StemConfig
from tetris_rl.core.policies.sb3.spatial_heads.config import SpatialHeadConfig
from tetris_rl.core.policies.sb3.tokenizers.config import TokenizerConfig
from tetris_rl.core.policies.sb3.types import NULL_COMPONENT_TAGS

from .builders import (
    build_feature_augmenter,
    build_spatial_head,
    build_spatial_preprocessor,
    build_stem,
    build_token_mixer,
    build_tokenizer,
    infer_feature_augmenter_extra_dim,
    resolve_spatial_head_features_dim,
)
from .validators import (
    check_spatial,
    check_specials,
    check_stream,
    infer_grid_hw_from_obs_space,
)


class TetrisFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        *,
        spatial_preprocessor: SpatialPreprocessorConfig,
        stem: Optional[StemConfig],
        # --- token encoder branch ---
        tokenizer: Optional[TokenizerConfig] = None,
        mixer: Optional[MixerConfig] = None,
        # --- spatial encoder branch ---
        spatial_head: Optional[SpatialHeadConfig] = None,
        # --- optional post-augment ---
        feature_augmenter: Optional[FeatureAugmenterConfig] = None,
        # injected from env/assets when needed (tokenizer specials tokens; augmenters; etc.)
        n_kinds: Optional[int] = None,
    ) -> None:
        # SB3 requires BaseFeaturesExtractor.features_dim at construction time.
        # We'll set a placeholder and overwrite self._features_dim once we know F_final.
        super().__init__(observation_space, features_dim=1)

        # ------------------------------------------------------------------
        # Routing invariants (static)
        # ------------------------------------------------------------------
        using_token_route = (tokenizer is not None) or (mixer is not None)
        using_spatial_route = spatial_head is not None

        if using_token_route and using_spatial_route:
            raise ValueError("cannot configure both token route and spatial route")

        if using_token_route:
            if tokenizer is None or mixer is None:
                raise ValueError("token route requires BOTH tokenizer and mixer")
        else:
            if spatial_head is None:
                raise ValueError("must configure either token route (tokenizer+mixer) or spatial_head")

        # ------------------------------------------------------------------
        # Board / geometry (known at init)
        # ------------------------------------------------------------------
        grid_h, grid_w = infer_grid_hw_from_obs_space(observation_space)
        self._board_spec = BoardSpec(h=grid_h, w=grid_w)

        # ------------------------------------------------------------------
        # 1) Spatial preprocessor (built at init)
        # ------------------------------------------------------------------
        self.spatial_preprocessor: SpatialPreprocessor = build_spatial_preprocessor(cfg=spatial_preprocessor)
        self._spatial_spec_pre: SpatialSpec = self.spatial_preprocessor.out_spec(board=self._board_spec)

        # ------------------------------------------------------------------
        # 2) Optional stem (spatial -> spatial) (built at init)
        # ------------------------------------------------------------------
        self.stem: SpatialStem | None = (
            build_stem(cfg=stem, in_channels=int(self._spatial_spec_pre.c)) if stem is not None else None
        )
        self._spatial_spec: SpatialSpec = (
            self.stem.out_spec(in_spec=self._spatial_spec_pre) if self.stem is not None else self._spatial_spec_pre
        )

        # ------------------------------------------------------------------
        # 3) Base encoder branch (built at init)
        # ------------------------------------------------------------------
        self.tokenizer: Optional[nn.Module] = None
        self.token_mixer: Optional[nn.Module] = None
        self.spatial_head: Optional[nn.Module] = None

        # Base encoder output dimension (before augmentation)
        F_base: int

        if using_token_route:
            assert tokenizer is not None
            assert mixer is not None

            tok = build_tokenizer(cfg=tokenizer, n_kinds=n_kinds, in_spec=self._spatial_spec)
            self.tokenizer = tok

            tok_spec = tok.stream_spec()
            T_stream = int(tok_spec.T)
            D = int(tok_spec.d_model)

            use_cls = bool(getattr(mixer.params, "use_cls", True))
            num_cls = int(getattr(mixer.params, "num_cls_tokens", 1))
            if num_cls < 0:
                raise ValueError("num_cls_tokens must be >= 0")

            T_total = T_stream + (num_cls if use_cls else 0)

            self.token_mixer = build_token_mixer(cfg=mixer, d_model=D, T_total=T_total)

            # Prefer explicit out_dim on the mixer; fallback to d_model for CLS-style mixers.
            out_dim = int(getattr(self.token_mixer, "out_dim", 0))
            F_base = out_dim if out_dim > 0 else int(D)

        else:
            assert spatial_head is not None

            # Pattern "B": SpatialHeadConfig params own base output dim.
            F_base = resolve_spatial_head_features_dim(cfg=spatial_head, in_spec=self._spatial_spec)

            self.spatial_head = build_spatial_head(
                cfg=spatial_head,
                in_spec=self._spatial_spec,
                features_dim=int(F_base),  # base dim, NOT post-augment dim
            )

            # Prefer explicit out_dim if present; otherwise trust the config.
            out_dim = int(getattr(self.spatial_head, "out_dim", 0))
            if out_dim > 0 and out_dim != int(F_base):
                raise ValueError(
                    f"spatial_head out_dim mismatch: built out_dim={out_dim} vs cfg.params.features_dim={F_base}"
                )

        if int(F_base) <= 0:
            raise ValueError(f"F_base must be > 0, got {F_base}")

        # ------------------------------------------------------------------
        # 4) Optional feature augmenter (built at init; NOT lazy)
        # ------------------------------------------------------------------
        self._n_kinds = n_kinds
        self.feature_augmenter: Optional[nn.Module] = None

        extra = 0
        if feature_augmenter is not None:
            t = str(feature_augmenter.type).strip().lower()
            if t in NULL_COMPONENT_TAGS:
                feature_augmenter = None
        if feature_augmenter is not None:
            extra = infer_feature_augmenter_extra_dim(cfg=feature_augmenter, n_kinds=self._n_kinds)
            if extra < 0:
                raise ValueError(f"feature augmenter extra dim must be >= 0, got {extra}")

        F_final = int(F_base) + int(extra)
        if F_final <= 0:
            raise ValueError(f"final features_dim must be > 0, got {F_final}")

        if feature_augmenter is not None:
            self.feature_augmenter = build_feature_augmenter(
                cfg=feature_augmenter,
                n_kinds=self._n_kinds,        # injected (NOT in params)
                in_dim=int(F_base),           # base features coming in
                features_dim=int(F_final),    # SB3 must see post-augment dim
            )

        # ------------------------------------------------------------------
        # 5) SB3 contract: set final features_dim (POST-AUGMENT)
        # ------------------------------------------------------------------
        self._final_features_dim: int = int(F_final)
        self._features_dim = int(F_final)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # 1) preproc
        spatial, specials = self.spatial_preprocessor(observations=observations)
        check_spatial(spatial)
        check_specials(specials)

        # 2) optional stem
        if self.stem is not None:
            spatial = self.stem(spatial)
            check_spatial(spatial)

        # 3) base route
        if self.tokenizer is not None:
            assert self.token_mixer is not None
            stream = self.tokenizer(spatial=spatial, specials=specials)
            check_stream(stream)
            feats = self.token_mixer(stream=stream)
        else:
            assert self.spatial_head is not None
            feats = self.spatial_head(spatial=spatial, specials=specials)

        if feats.dim() != 2:
            raise ValueError(f"base route must return (B,F), got {tuple(feats.shape)}")

        # 4) optional augmenter
        if self.feature_augmenter is not None:
            feats = self.feature_augmenter(features=feats, specials=specials)

        # 5) SB3 contract
        if feats.dim() != 2 or int(feats.shape[1]) != int(self._final_features_dim):
            raise ValueError(
                f"feature extractor must return (B,features_dim={self._final_features_dim}), got {tuple(feats.shape)}"
            )
        return feats


__all__ = ["TetrisFeatureExtractor"]


