# src/tetris_rl/config/instantiate.py
from __future__ import annotations

"""
Config instantiation helpers.

This module turns YAML-ish "{type, params}" specs into concrete Python objects
via simple registries.

Two constructor styles are supported:

(A) kwargs-style constructors (classic):
    class Foo:
        def __init__(self, *, a: int, b: float = 0.0): ...

    YAML:
      foo:
        type: foo
        params: {a: 1, b: 0.2}

    Registry:
      {"foo": Foo}

    Call performed:
      Foo(**injected, **params)

(B) spec-style constructors (preferred for model components):
    class Bar:
        def __init__(self, *, spec: BarParams): ...

    YAML:
      bar:
        type: bar
        params: { ... fields of BarParams ... }

    Registry:
      {"bar": Bar}

    Call performed (when ctor accepts "spec"):
      Bar(**injected, spec=<params_obj>)

Key rules (STRICT):
- injected kwargs are derived by code and must not be overrideable via params.
- Overlap between injected keys and params keys is forbidden (kwargs-style).
- If the registry entry is a pre-built instance (non-callable), it is returned
  directly ONLY when no injected kwargs and empty params are provided.
- Returning None is treated as an error (catches mis-registered procedures).

About "params" values:
- params may be a mapping (dict-like) OR a dataclass instance.
- If params is a dataclass and ctor accepts "spec": pass it as spec=...
- If params is a dataclass and ctor does NOT accept "spec": expand it via
  dataclasses.asdict(...) into kwargs.
"""

import inspect
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Mapping, MutableMapping, TypeVar

from tetris_rl.config.schema_types import ComponentSpec, as_component_spec

T = TypeVar("T")

# Special tags meaning "disabled component" (instantiate() returns None).
_NULL_TYPES = {"none", "null", "off", "disabled"}


def _callable_name(obj: Any) -> str:
    try:
        return str(getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj))))
    except Exception:
        return repr(obj)


def _signature_str(fn: Any) -> str:
    try:
        return f" signature={inspect.signature(fn)!s}"
    except Exception:
        return ""


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """
    True if fn signature contains a parameter named `name`.

    Note: we intentionally do NOT treat **kwargs as accepting everything here,
    because in SB3-style factories you typically want explicitness.
    """
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    return name in sig.parameters


def _normalize_params(params_any: Any, *, where: str) -> Any:
    """
    Normalize spec.params into one of:
      - {}                  (when None)
      - dict(...)           (when mapping)
      - dataclass instance  (left as-is)
      - otherwise: error
    """
    if params_any is None:
        return {}
    if is_dataclass(params_any):
        return params_any
    if isinstance(params_any, Mapping):
        return dict(params_any)
    raise TypeError(f"{where}.params must be a mapping or a dataclass, got {type(params_any)!r}")


def instantiate(
    *,
    spec_obj: Any,
    registry: Mapping[str, Callable[..., Any] | Any],
    where: str,
    injected: MutableMapping[str, Any] | None = None,
) -> Any:
    """
    Instantiate a component defined as {type, params} using a provided registry.

    See module docstring for conventions and supported constructor styles.
    """
    spec: ComponentSpec = as_component_spec(spec_obj, where=where)

    # ------------------------------------------------------------
    # Special-case: disabled component (type: none | null | off)
    # ------------------------------------------------------------
    if spec.type in _NULL_TYPES:
        params_any = _normalize_params(spec.params, where=where)
        inj = dict(injected) if injected else {}

        has_params = (is_dataclass(params_any)) or (isinstance(params_any, dict) and bool(params_any))
        if inj or has_params:
            raise TypeError(
                f"{where} type={spec.type!r} does not accept params or injected kwargs "
                f"(got injected={list(inj.keys())!r}, params={spec.params!r})"
            )
        return None

    try:
        ctor_or_obj = registry[spec.type]
    except KeyError as e:
        known = ", ".join(sorted(str(k) for k in registry.keys()))
        raise KeyError(f"unknown {where}.type {spec.type!r}. known: [{known}]") from e

    inj = dict(injected) if injected else {}
    params_any = _normalize_params(spec.params, where=where)

    # Allow pre-built instances in registry only when no kwargs are provided.
    if not callable(ctor_or_obj):
        # params_any is {} or dataclass/dict; forbid any non-empty call context
        has_params = (is_dataclass(params_any)) or (isinstance(params_any, dict) and bool(params_any))
        if inj or has_params:
            raise TypeError(
                f"{where} type={spec.type!r} resolved to a non-callable registry entry "
                f"(type={type(ctor_or_obj).__name__}), but params/injected were provided: "
                f"injected={list(inj.keys())!r} params_type={type(params_any).__name__}"
            )
        return ctor_or_obj

    ctor = ctor_or_obj

    # ------------------------------------------------------------------
    # Dataclass params
    # ------------------------------------------------------------------
    if is_dataclass(params_any):
        kwargs: dict[str, Any] = dict(inj)

        # If ctor supports spec=..., prefer passing the dataclass as spec directly.
        if _accepts_kwarg(ctor, "spec"):
            kwargs["spec"] = params_any
        else:
            # Fall back: expand dataclass into kwargs-style.
            kwargs.update(asdict(params_any))

        try:
            result = ctor(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"failed to construct {where} type={spec.type!r} using ctor={_callable_name(ctor)!r}{_signature_str(ctor)} "
                f"with dataclass params={params_any!r} injected={list(inj.keys())!r}: {e}"
            ) from e

        if result is None:
            raise TypeError(
                f"{where} type={spec.type!r} constructed via ctor={_callable_name(ctor)!r}{_signature_str(ctor)} "
                "returned None. Registry entries must be factories/constructors that RETURN an object."
            )
        return result

    # ------------------------------------------------------------------
    # Dict params (classic)
    # ------------------------------------------------------------------
    assert isinstance(params_any, dict)
    params: dict[str, Any] = params_any

    overlap = sorted(set(inj.keys()) & set(params.keys()))
    if overlap:
        overlap_s = ", ".join(overlap)
        raise TypeError(
            f"{where} type={spec.type!r} has injected/params key overlap: [{overlap_s}]. "
            f"Remove these keys from {where}.params; they are injected/derived."
        )

    kwargs2: dict[str, Any] = dict(inj)

    # If ctor has a 'spec' kwarg, prefer spec-style call unless user already provided "spec".
    if _accepts_kwarg(ctor, "spec") and "spec" not in params:
        kwargs2["spec"] = params
    else:
        kwargs2.update(params)

    try:
        result = ctor(**kwargs2)
    except TypeError as e:
        raise TypeError(
            f"failed to construct {where} type={spec.type!r} using ctor={_callable_name(ctor)!r}{_signature_str(ctor)} "
            f"with params={spec.params!r} injected={list(inj.keys())!r}: {e}"
        ) from e

    if result is None:
        raise TypeError(
            f"{where} type={spec.type!r} constructed via ctor={_callable_name(ctor)!r}{_signature_str(ctor)} "
            "returned None. Registry entries must be factories/constructors that RETURN an object. "
            "If you intended to register a warmup function directly, wrap it in a factory, e.g. "
            "lambda **_params: warmup_fn  (or implement a class with __call__)."
        )

    return result


def instantiate_tagged_params(
    *,
    section_obj: Any,
    params_registry: Mapping[str, Callable[..., Any] | Any],
    wrapper_ctor: Callable[..., T],
    where: str,
) -> T:
    """
    Instantiate a nested {type, params} section into a wrapper dataclass/object that
    takes (type=<tag>, params=<params_instance>).

    Use-case: specs where "params" is a UNION and the union choice is determined by "type",
    e.g. LayoutConfig(type=..., params=RowLayoutParams|PatchLayoutParams|...)

    Example:
      section_obj = {"type": "patch", "params": {"patch_h": 2, "patch_w": 2}}
      params_registry maps "patch" -> PatchLayoutParams
      wrapper_ctor is LayoutConfig

      -> LayoutConfig(type="patch", params=PatchLayoutParams(patch_h=2, patch_w=2))
    """
    spec: ComponentSpec = as_component_spec(section_obj, where=where)

    params_obj = instantiate(
        spec_obj={"type": spec.type, "params": spec.params},
        registry=params_registry,
        where=f"{where}.params",
        injected=None,
    )

    try:
        return wrapper_ctor(type=spec.type, params=params_obj)
    except TypeError as e:
        raise TypeError(
            f"failed to construct wrapper for {where} using ctor={_callable_name(wrapper_ctor)!r}{_signature_str(wrapper_ctor)} "
            f"with type={spec.type!r} params_cls={type(params_obj).__name__}: {e}"
        ) from e


__all__ = ["instantiate", "instantiate_tagged_params"]