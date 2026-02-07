# src/tetris_rl/policies/sb3/feature_extractor.py
from __future__ import annotations

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

import inspect
from typing import Optional, cast, Any

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from pydantic import BaseModel

from tetris_rl.policies.sb3.feature_augmenters.config import FeatureAugmenterConfig
from tetris_rl.policies.sb3.mixers.config import MixerConfig
from tetris_rl.policies.sb3.spatial_heads.config import SpatialHeadConfig
from tetris_rl.policies.sb3.spatial.config import SpatialPreprocessorConfig, StemConfig
from tetris_rl.policies.sb3.tokenizers.config import TokenizerConfig
from tetris_rl.policies.sb3.api import BoardSpec, Specials, SpatialFeatures, SpatialSpec, TokenStream
from tetris_rl.policies.sb3.catalog import (
    FEATURE_AUGMENTER_REGISTRY,
    SPATIAL_HEAD_REGISTRY,
    SPATIAL_PREPROCESSOR_REGISTRY,
    STEM_REGISTRY,
    TOKEN_MIXER_REGISTRY,
)
from tetris_rl.policies.sb3.tokenizers.tetris_tokenizer import TetrisTokenizer


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
        grid_h, grid_w = _infer_grid_hw_from_obs_space(observation_space)
        self._board_spec = BoardSpec(h=grid_h, w=grid_w)

        # ------------------------------------------------------------------
        # 1) Spatial preprocessor (built at init)
        # ------------------------------------------------------------------
        self.spatial_preprocessor = _build_spatial_preprocessor(cfg=spatial_preprocessor)
        self._spatial_spec_pre: SpatialSpec = self.spatial_preprocessor.out_spec(board=self._board_spec)

        # ------------------------------------------------------------------
        # 2) Optional stem (spatial -> spatial) (built at init)
        # ------------------------------------------------------------------
        self.stem: Optional[nn.Module] = (
            _build_stem(cfg=stem, in_channels=int(self._spatial_spec_pre.c)) if stem is not None else None
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

            tok = _build_tokenizer(cfg=tokenizer, n_kinds=n_kinds, in_spec=self._spatial_spec)
            self.tokenizer = tok

            tok_spec = tok.stream_spec()
            T_stream = int(tok_spec.T)
            D = int(tok_spec.d_model)

            use_cls = bool(getattr(mixer.params, "use_cls", True))
            num_cls = int(getattr(mixer.params, "num_cls_tokens", 1))
            if num_cls < 0:
                raise ValueError("num_cls_tokens must be >= 0")

            T_total = T_stream + (num_cls if use_cls else 0)

            self.token_mixer = _build_token_mixer(cfg=mixer, d_model=D, T_total=T_total)

            # Prefer explicit out_dim on the mixer; fallback to d_model for CLS-style mixers.
            out_dim = int(getattr(self.token_mixer, "out_dim", 0))
            F_base = out_dim if out_dim > 0 else int(D)

        else:
            assert spatial_head is not None

            # Pattern "B": SpatialHeadConfig params own base output dim.
            F_base = int(getattr(spatial_head.params, "features_dim", 0))
            if F_base <= 0:
                raise ValueError(f"spatial_head.params.features_dim must be > 0, got {F_base}")

            self.spatial_head = _build_spatial_head(
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
            if t in {"none", "null"}:
                feature_augmenter = None
        if feature_augmenter is not None:
            extra = _infer_feature_augmenter_extra_dim(cfg=feature_augmenter, n_kinds=self._n_kinds)
            if extra < 0:
                raise ValueError(f"feature augmenter extra dim must be >= 0, got {extra}")

        F_final = int(F_base) + int(extra)
        if F_final <= 0:
            raise ValueError(f"final features_dim must be > 0, got {F_final}")

        if feature_augmenter is not None:
            self.feature_augmenter = _build_feature_augmenter(
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
        _check_spatial(spatial)
        _check_specials(specials)

        # 2) optional stem
        if self.stem is not None:
            spatial = self.stem(spatial)
            _check_spatial(spatial)

        # 3) base route
        if self.tokenizer is not None:
            assert self.token_mixer is not None
            stream = self.tokenizer(spatial=spatial, specials=specials)
            _check_stream(stream)
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


# ---------------------------------------------------------------------
# Builders (small, boring, registry-backed)
# ---------------------------------------------------------------------


def _infer_grid_hw_from_obs_space(observation_space: spaces.Space) -> tuple[int, int]:
    if not isinstance(observation_space, spaces.Dict):
        raise TypeError(f"expected spaces.Dict observation_space, got {type(observation_space)!r}")
    if "grid" not in observation_space.spaces:
        raise KeyError("observation_space missing key 'grid'")
    sp = observation_space.spaces["grid"]
    if not isinstance(sp, spaces.Box):
        raise TypeError(f"obs['grid'] must be spaces.Box, got {type(sp)!r}")
    if sp.shape is None or len(sp.shape) != 2:
        raise ValueError(f"obs['grid'] must be shape (H,W), got {sp.shape!r}")
    H, W = int(sp.shape[0]), int(sp.shape[1])
    if H <= 0 or W <= 0:
        raise ValueError(f"invalid grid shape from obs_space: (H,W)=({H},{W})")
    return H, W


def _build_spatial_preprocessor(*, cfg: SpatialPreprocessorConfig) -> nn.Module:
    key = str(cfg.type).strip().lower()
    cls = SPATIAL_PREPROCESSOR_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown spatial_preprocessor type: {cfg.type!r}")

    params = cfg.params
    if params is None:
        return cast(nn.Module, cls())
    if isinstance(params, dict):
        return cast(nn.Module, cls(**params))
    return cast(nn.Module, cls(params=params))


def _build_stem(*, cfg: StemConfig, in_channels: int) -> nn.Module:
    cls = STEM_REGISTRY[cfg.type]

    if cfg.params is None:
        # preset stem
        return cast(nn.Module, cls(in_channels=int(in_channels)))

    # configurable stem
    return cast(nn.Module, cls(in_channels=int(in_channels), spec=cfg.params))


def _build_tokenizer(*, cfg: TokenizerConfig, n_kinds: Optional[int], in_spec: SpatialSpec) -> TetrisTokenizer:
    return TetrisTokenizer(
        d_model=int(cfg.d_model),
        layout=cfg.layout,
        board_embedding=cfg.board_embedding,
        add_active_token=bool(cfg.add_active_token),
        add_next_token=bool(cfg.add_next_token),
        share_kind_embedding=bool(getattr(cfg, "share_kind_embedding", True)),
        n_kinds=n_kinds,
        in_spec=in_spec,
    )


def _build_token_mixer(*, cfg: MixerConfig, d_model: int, T_total: int) -> nn.Module:
    key = str(cfg.type).strip().lower()
    cls = TOKEN_MIXER_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown mixer type: {cfg.type!r}")

    sig = inspect.signature(cls.__init__)

    kwargs: dict[str, Any] = {"spec": cfg.params}
    if "d_model" in sig.parameters:
        kwargs["d_model"] = int(d_model)
    if "T_total" in sig.parameters:
        kwargs["T_total"] = int(T_total)

    return cast(nn.Module, cls(**kwargs))


def _build_spatial_head(*, cfg: SpatialHeadConfig, in_spec: SpatialSpec, features_dim: int) -> nn.Module:
    key = str(cfg.type).strip().lower()
    cls = SPATIAL_HEAD_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown spatial_head type: {cfg.type!r}")

    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {"spec": cfg.params}

    # Standardized geometry injection (heads may accept any subset)
    if "in_h" in sig.parameters:
        kwargs["in_h"] = int(in_spec.h)
    if "in_w" in sig.parameters:
        kwargs["in_w"] = int(in_spec.w)
    if "in_channels" in sig.parameters:
        kwargs["in_channels"] = int(in_spec.c)

    # Pattern "B": output dim is configured in params but injected as a separate arg.
    if "features_dim" in sig.parameters:
        kwargs["features_dim"] = int(features_dim)

    return cast(nn.Module, cls(**kwargs))


def _infer_feature_augmenter_extra_dim(*, cfg: FeatureAugmenterConfig, n_kinds: Optional[int]) -> int:
    """
    Infer how many dimensions the augmenter ADDS on top of base features (F_base).

    This is needed to set SB3's features_dim to the post-augment dim at init time.
    """
    if cfg.params is None:
        raise ValueError("feature_augmenter.params must not be None when enabled")

    p = cfg.params

    # onehot_concat: adds K or 2K (depending on flags)
    if cfg.type == "onehot_concat":
        if n_kinds is None or int(n_kinds) <= 0:
            raise ValueError("feature_augmenter requires n_kinds injection from env/assets")
        K = int(n_kinds)
        use_active = bool(getattr(p, "use_active", True))
        use_next = bool(getattr(p, "use_next", False))
        return (K if use_active else 0) + (K if use_next else 0)

    # mlp_joint: adds out_dim (or 0)
    if cfg.type == "mlp_joint":
        out_dim = int(getattr(p, "out_dim", 0))
        use_active = bool(getattr(p, "use_active", True))
        use_next = bool(getattr(p, "use_next", False))
        if (not use_active) and (not use_next):
            return 0
        return max(0, out_dim)

    # mlp_split: adds used dims (active+next) based on out_dim_total split and enabled flags
    if cfg.type == "mlp_split":
        total = int(getattr(p, "out_dim_total", 0))
        if total <= 0:
            return 0

        da = getattr(p, "out_dim_active", None)
        dn = getattr(p, "out_dim_next", None)

        # Match SplitMLPAugmenter._resolve_dims behavior
        if da is None and dn is None:
            da_i = total // 2
            dn_i = total - da_i
        elif da is None:
            dn_i = int(dn)
            da_i = total - dn_i
        elif dn is None:
            da_i = int(da)
            dn_i = total - da_i
        else:
            da_i = int(da)
            dn_i = int(dn)

        if da_i < 0 or dn_i < 0 or (da_i + dn_i != total):
            raise ValueError(
                f"invalid mlp_split dims: total={total} active={da!r} next={dn!r} -> {da_i}+{dn_i}"
            )

        use_active = bool(getattr(p, "use_active", True))
        use_next = bool(getattr(p, "use_next", False))

        used = 0
        if use_active:
            used += da_i
        if use_next:
            used += dn_i
        return max(0, int(used))

    raise ValueError(f"unsupported feature_augmenter type for dim inference: {cfg.type!r}")


def _maybe_set_param_features_dim(params: Any, *, features_dim: int) -> Any:
    """
    If params has a `features_dim` field, overwrite it.

    Rationale:
    - YAML should NOT need to specify features_dim for the augmenter.
    - Typed params / back-compat may still include it.
    """
    if params is None:
        return None
    if not hasattr(params, "features_dim"):
        return params

    try:
        cur = int(getattr(params, "features_dim"))
        if cur == int(features_dim):
            return params
        if isinstance(params, BaseModel):
            return params.model_copy(update={"features_dim": int(features_dim)})
        d = dict(getattr(params, "__dict__", {}))
        d["features_dim"] = int(features_dim)
        return type(params)(**d)
    except Exception:
        return params


def _build_feature_augmenter(
    *,
    cfg: FeatureAugmenterConfig,
    n_kinds: Optional[int],
    in_dim: int,
    features_dim: int,
) -> nn.Module:
    key = str(cfg.type).strip().lower()
    if key in {"", "none", "null"}:
        raise ValueError("feature_augmenter type is disabled but builder was called")

    cls = FEATURE_AUGMENTER_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown feature_augmenter type: {cfg.type!r}")
    if cfg.params is None:
        raise ValueError("feature_augmenter params must not be None when enabled")

    params = _maybe_set_param_features_dim(cfg.params, features_dim=int(features_dim))
    sig = inspect.signature(cls.__init__)

    # Support multiple init styles:
    #   __init__(*, spec=...)
    #   __init__(*, params=..., in_dim=..., features_dim=..., n_kinds=...)
    kwargs: dict[str, Any] = {}
    if "spec" in sig.parameters:
        kwargs["spec"] = params
    elif "params" in sig.parameters:
        kwargs["params"] = params
    else:
        raise TypeError(f"{cls.__name__}.__init__ must accept 'spec' or 'params'")

    if "in_dim" in sig.parameters:
        kwargs["in_dim"] = int(in_dim)
    if "features_dim" in sig.parameters:
        kwargs["features_dim"] = int(features_dim)

    # n_kinds is injected (NOT in params anymore)
    if "n_kinds" in sig.parameters:
        if n_kinds is None or int(n_kinds) <= 0:
            raise ValueError("feature_augmenter requires n_kinds injection from env/assets")
        kwargs["n_kinds"] = int(n_kinds)

    return cast(nn.Module, cls(**kwargs))


# ---------------------------------------------------------------------
# Runtime validators
# ---------------------------------------------------------------------


def _check_spatial(spatial: SpatialFeatures) -> None:
    x = spatial.x
    if x.dim() != 4:
        raise ValueError(f"SpatialFeatures.x must be (B,H,W,C), got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise ValueError("SpatialFeatures.x must be floating point (e.g. float32)")


def _check_specials(specials: Specials) -> None:
    if specials.active_kind.dim() not in (0, 1, 2):
        raise ValueError(f"Specials.active_kind must be scalar/(B,)/(B,K), got {tuple(specials.active_kind.shape)}")
    if specials.next_kind is not None and specials.next_kind.dim() not in (0, 1, 2):
        raise ValueError(f"Specials.next_kind must be scalar/(B,)/(B,K), got {tuple(specials.next_kind.shape)}")


def _check_stream(stream: TokenStream) -> None:
    x = stream.x
    types = stream.types
    if x.dim() != 3:
        raise ValueError(f"TokenStream.x must be (B,T,D), got {tuple(x.shape)}")
    if types.dim() != 1:
        raise ValueError(f"TokenStream.types must be (T,), got {tuple(types.shape)}")
    if int(types.shape[0]) != int(x.shape[1]):
        raise ValueError(f"type length mismatch: types T={int(types.shape[0])} vs x T={int(x.shape[1])}")


__all__ = ["TetrisFeatureExtractor"]


