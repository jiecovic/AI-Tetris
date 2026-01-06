# src/tetris_rl/config/resolve.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Type, TypeVar, cast

from tetris_rl.config.schema_types import (
    get_mapping,
    get_str,
    require_mapping,
    require_mapping_strict,
)
from tetris_rl.config.snapshot import load_yaml
from tetris_rl.utils.paths import repo_root as find_repo_root
from tetris_rl.utils.logging import setup_logger

LOG = setup_logger(name="tetris_rl.config.resolve", use_rich=True, level="info")

T = TypeVar("T")


# =============================================================================
# Spec-file include resolution (specs.train/env/model)
# =============================================================================

def _resolve_path(*, raw: str, cfg_path: Path, repo: Path) -> Path:
    s = str(raw).strip().strip('"').strip("'")
    if not s:
        raise ValueError("empty path")

    p = Path(s)

    if p.is_absolute():
        return p.resolve()

    # QoL: avoid repo/configs/configs/... if the user wrote "configs/..." already
    if p.parts and p.parts[0].lower() == "configs":
        p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path()

    cand_repo = (repo / p).resolve()
    if cand_repo.exists():
        return cand_repo

    return (cfg_path.parent / p).resolve()


def _unwrap_single_root(*, loaded: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    """
    Allow spec files to either be:
      - wrapped: {key: {...}}
      - bare:    {...}
    """
    if key in loaded and isinstance(loaded[key], dict):
        return require_mapping_strict(loaded[key], where=f"{where}.{key}")
    return require_mapping_strict(loaded, where=where)


# =============================================================================
# Generic dataclass hydration utilities
# =============================================================================

def _ensure_mapping(obj: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(f"{where} must be a mapping, got {type(obj)!r}")
    return cast(dict[str, Any], obj)


def _dc_from_mapping(dc: Type[T], obj: Any, *, where: str) -> T:
    if isinstance(obj, dc):
        return obj
    m = _ensure_mapping(obj, where=where)
    try:
        return dc(**m)  # type: ignore[misc]
    except TypeError as e:
        raise TypeError(f"{where}: failed to build {dc.__name__} from {m!r}: {e}") from e


def _hydrate_tagged_params(
        *,
        type_value: Any,
        params_value: Any,
        params_registry: Mapping[str, Type[Any]],
        where: str,
) -> Any:
    """
    Build the correct params dataclass for a tagged-union section.

      section:
        type: <tag>
        params: {...}

    -> params_registry[tag](**params)

    Notes:
    - params may be missing/None => treated as {}
    - if params already is instance of expected dataclass => returned as-is
    """
    tag = str(type_value).strip().lower()
    if not tag:
        raise ValueError(f"{where}.type must be a non-empty string")

    try:
        params_cls = params_registry[tag]
    except KeyError as e:
        known = ", ".join(sorted(params_registry.keys()))
        raise KeyError(f"{where}.type unknown: {tag!r}. known: [{known}]") from e

    if params_value is None:
        params_value = {}

    if isinstance(params_value, params_cls):
        return params_value

    m = _ensure_mapping(params_value, where=f"{where}.params")
    try:
        return params_cls(**m)
    except TypeError as e:
        raise TypeError(f"{where}.params: failed to build {params_cls.__name__} from {m!r}: {e}") from e


def _normalize_col_collapse_backcompat(params: Any) -> Any:
    """
    Back-compat for ColumnCollapseParams:
      - allow YAML to pass `pooling: ...` (older configs)
      - map it to `pool` if `pool` is left default / not explicitly set
    """
    # Avoid importing dataclass type here; duck-type the fields.
    try:
        pooling = getattr(params, "pooling", None)
        pool = getattr(params, "pool", None)
        if pooling and isinstance(pooling, str):
            p = str(pooling).strip().lower()
            if p in {"avg", "max", "avgmax"}:
                # If user still has pooling, prefer it unless pool was explicitly changed.
                # Dataclasses are frozen; rebuild with updated pool.
                # Safe: ColumnCollapseParams ctor signature matches these names.
                if pool == "avg":  # default in our params class
                    return type(params)(**{**params.__dict__, "pool": p, "pooling": pooling})
    except Exception:
        pass
    return params


# =============================================================================
# Model feature_extractor hydration (single place, all spec modules)
# =============================================================================

def _hydrate_model_feature_extractor_config(*, root: dict[str, Any]) -> None:
    """
    Mutates cfg in-place:
      cfg.model.feature_extractor becomes "SB3 features_extractor_kwargs"
      where all typed sub-sections are dataclass instances.

    This is what prevents:
      AttributeError: 'dict' object has no attribute 'type'
    inside the feature extractor builders.
    """
    model = root.get("model")
    if model is None:
        return
    if not isinstance(model, dict):
        raise TypeError("cfg.model must be a mapping")

    fe = model.get("feature_extractor")
    if fe is None:
        return
    if not isinstance(fe, dict):
        raise TypeError("cfg.model.feature_extractor must be a mapping")

    # --- imports of YOUR spec dataclasses/registries ---
    from tetris_rl.config.model.spatial_spec import (
        SpatialPreprocessorConfig,
        StemConfig,
        CNNStemParams,
    )

    from tetris_rl.config.model.tokenizer_spec import (
        LayoutConfig,
        BoardEmbeddingConfig,
        TokenizerSpec,
        TOKENIZER_LAYOUT_PARAMS_REGISTRY,
        TOKENIZER_BOARD_EMBED_PARAMS_REGISTRY,
    )

    from tetris_rl.config.model.mixer_spec import (
        MixerConfig,
        MIXER_PARAMS_REGISTRY,
    )

    from tetris_rl.config.model.spatial_head_spec import (
        SpatialHeadConfig,
        SPATIAL_HEAD_PARAMS_REGISTRY,
    )

    from tetris_rl.config.model.feature_augmenter_spec import (
        FeatureAugmenterConfig,
        FEATURE_AUGMENTER_PARAMS_REGISTRY,
    )

    # ------------------------------------------------------------------
    # spatial_preprocessor (required)
    # ------------------------------------------------------------------
    sp = fe.get("spatial_preprocessor")
    if sp is None:
        raise KeyError("cfg.model.feature_extractor.spatial_preprocessor missing")
    fe["spatial_preprocessor"] = _dc_from_mapping(
        SpatialPreprocessorConfig,
        sp,
        where="cfg.model.feature_extractor.spatial_preprocessor",
    )

    # ------------------------------------------------------------------
    # stem (optional)
    # ------------------------------------------------------------------
    stem = fe.get("stem", None)
    if stem is not None:
        stem_m = _ensure_mapping(stem, where="cfg.model.feature_extractor.stem")
        stem_type = str(stem_m.get("type", "")).strip().lower()
        if not stem_type or stem_type == "none":
            fe["stem"] = None
        else:
            # only type='cnn' has structured params
            if stem_type == "cnn":
                params = stem_m.get("params", None)
                if params is None:
                    raise KeyError("cfg.model.feature_extractor.stem.params missing for stem.type='cnn'")
                stem_m = dict(stem_m)
                stem_m["params"] = _dc_from_mapping(
                    CNNStemParams,
                    params,
                    where="cfg.model.feature_extractor.stem.params",
                )
            else:
                # fixed preset stem should have params omitted/None
                # if user supplied params, let StemConfig(**...) raise (strict)
                pass

            fe["stem"] = _dc_from_mapping(
                StemConfig,
                stem_m,
                where="cfg.model.feature_extractor.stem",
            )

    # ------------------------------------------------------------------
    # encoder (required): token OR spatial
    # ------------------------------------------------------------------
    enc = fe.get("encoder")
    if enc is None:
        raise KeyError("cfg.model.feature_extractor.encoder missing")
    enc_m = _ensure_mapping(enc, where="cfg.model.feature_extractor.encoder")

    enc_type = str(enc_m.get("type", "")).strip().lower()
    if enc_type not in {"token", "spatial"}:
        raise ValueError("cfg.model.feature_extractor.encoder.type must be 'token' or 'spatial'")

    if enc_type == "token":
        # ===== tokenizer (required) =====
        tok = enc_m.get("tokenizer")
        if tok is None:
            raise KeyError("cfg.model.feature_extractor.encoder.tokenizer missing for token encoder")
        tok_m = _ensure_mapping(tok, where="cfg.model.feature_extractor.encoder.tokenizer")

        # layout tagged union
        layout = tok_m.get("layout")
        if layout is None:
            raise KeyError("cfg.model.feature_extractor.encoder.tokenizer.layout missing")
        layout_m = _ensure_mapping(layout, where="cfg.model.feature_extractor.encoder.tokenizer.layout")

        layout_type = layout_m.get("type", None)
        layout_params = _hydrate_tagged_params(
            type_value=layout_type,
            params_value=layout_m.get("params", None),
            params_registry=cast(Mapping[str, Type[Any]], TOKENIZER_LAYOUT_PARAMS_REGISTRY),
            where="cfg.model.feature_extractor.encoder.tokenizer.layout",
        )
        layout_dc = LayoutConfig(type=cast(Any, str(layout_type).strip().lower()), params=layout_params)

        # board_embedding tagged union
        be = tok_m.get("board_embedding")
        if be is None:
            raise KeyError("cfg.model.feature_extractor.encoder.tokenizer.board_embedding missing")
        be_m = _ensure_mapping(be, where="cfg.model.feature_extractor.encoder.tokenizer.board_embedding")

        be_type = be_m.get("type", None)
        be_params = _hydrate_tagged_params(
            type_value=be_type,
            params_value=be_m.get("params", None),
            params_registry=cast(Mapping[str, Type[Any]], TOKENIZER_BOARD_EMBED_PARAMS_REGISTRY),
            where="cfg.model.feature_extractor.encoder.tokenizer.board_embedding",
        )
        be_dc = BoardEmbeddingConfig(type=cast(Any, str(be_type).strip().lower()), params=be_params)

        # build TokenizerSpec (top-level typed)
        tok_spec = TokenizerSpec(
            d_model=int(tok_m.get("d_model")),
            layout=layout_dc,
            board_embedding=be_dc,
            add_active_token=bool(tok_m.get("add_active_token", True)),
            add_next_token=bool(tok_m.get("add_next_token", False)),
            share_kind_embedding=bool(tok_m.get("share_kind_embedding", True)),
        )

        enc_m["tokenizer"] = tok_spec

        # ===== mixer tagged union (required) =====
        mix = enc_m.get("mixer")
        if mix is None:
            raise KeyError("cfg.model.feature_extractor.encoder.mixer missing for token encoder")
        mix_m = _ensure_mapping(mix, where="cfg.model.feature_extractor.encoder.mixer")

        mix_type = str(mix_m.get("type", "")).strip().lower()
        mix_params_raw = mix_m.get("params", None)

        mix_params = _hydrate_tagged_params(
            type_value=mix_type,
            params_value=mix_params_raw,
            params_registry=cast(Mapping[str, Type[Any]], MIXER_PARAMS_REGISTRY),
            where="cfg.model.feature_extractor.encoder.mixer",
        )
        enc_m["mixer"] = MixerConfig(type=cast(Any, mix_type), params=mix_params)

        # spatial_head must be absent
        enc_m.pop("spatial_head", None)

    else:
        # ===== spatial encoder =====
        sh = enc_m.get("spatial_head")
        if sh is None:
            raise KeyError("cfg.model.feature_extractor.encoder.spatial_head missing for spatial encoder")
        sh_m = _ensure_mapping(sh, where="cfg.model.feature_extractor.encoder.spatial_head")

        sh_type = str(sh_m.get("type", "")).strip().lower()
        sh_params_raw = sh_m.get("params", None)

        # Pattern "B": features_dim is NOT inside params anymore; it is passed separately.
        sh_features_dim_raw = sh_m.get("features_dim", None)
        if sh_features_dim_raw is None:
            raise KeyError("cfg.model.feature_extractor.encoder.spatial_head.features_dim missing (pattern B)")

        sh_params = _hydrate_tagged_params(
            type_value=sh_type,
            params_value=sh_params_raw,
            params_registry=cast(Mapping[str, Type[Any]], SPATIAL_HEAD_PARAMS_REGISTRY),
            where="cfg.model.feature_extractor.encoder.spatial_head",
        )

        # Back-compat fixups for col_collapse configs:
        sh_params = _normalize_col_collapse_backcompat(sh_params)

        enc_m["spatial_head"] = SpatialHeadConfig(
            type=cast(Any, sh_type),
            features_dim=int(sh_features_dim_raw),
            params=sh_params,
        )

        # tokenizer/mixer must be absent
        enc_m.pop("tokenizer", None)
        enc_m.pop("mixer", None)

    fe["encoder"] = enc_m

    # ------------------------------------------------------------------
    # feature_augmenter (optional)
    # ------------------------------------------------------------------
    aug = fe.get("feature_augmenter", None)
    if aug is None:
        fe["feature_augmenter"] = None
    else:
        aug_m = _ensure_mapping(aug, where="cfg.model.feature_extractor.feature_augmenter")
        aug_type_raw = aug_m.get("type", None)

        aug_type = "" if aug_type_raw is None else str(aug_type_raw).strip().lower()

        # Treat null/none/empty as "disabled"
        if aug_type in {"", "none", "null"}:
            if aug_m.get("params", None) not in (None, {}):
                LOG.warning(
                    "feature_augmenter disabled (type=%r) but params were provided; ignoring params.",
                    aug_type_raw,
                )
            fe["feature_augmenter"] = None
        else:
            aug_params_raw = aug_m.get("params", None)

            aug_params = _hydrate_tagged_params(
                type_value=aug_type,
                params_value=aug_params_raw,
                params_registry=cast(Mapping[str, Type[Any]], FEATURE_AUGMENTER_PARAMS_REGISTRY),
                where="cfg.model.feature_extractor.feature_augmenter",
            )
            fe["feature_augmenter"] = FeatureAugmenterConfig(type=cast(Any, aug_type), params=aug_params)

    model["feature_extractor"] = fe
    root["model"] = model


# =============================================================================
# Public entry
# =============================================================================

def resolve_config(*, cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    root = require_mapping(cfg, where="cfg")
    specs = get_mapping(root, "specs", default=None, where="cfg.specs")

    if specs is not None:
        repo = find_repo_root()

        def _load_spec(key: str) -> None:
            if key in root and root[key] is not None:
                return

            raw = get_str(specs, key, default="", where=f"cfg.specs.{key}").strip()
            if not raw:
                return

            p = _resolve_path(raw=raw, cfg_path=cfg_path, repo=repo)
            if not p.is_file():
                raise FileNotFoundError(f"specs.{key} path not found: {p}")

            loaded_any = load_yaml(p)
            loaded = require_mapping_strict(loaded_any, where=f"specs.{key} ({p})")
            root[key] = _unwrap_single_root(loaded=loaded, key=key, where=f"specs.{key} ({p})")

        _load_spec("train")
        _load_spec("env")
        _load_spec("model")

    # Always hydrate model.feature_extractor if present
    _hydrate_model_feature_extractor_config(root=root)

    return root


__all__ = ["resolve_config"]
