# src/tetris_rl/config/instantiate.py
from __future__ import annotations

"""
Config instantiation helpers.

This module turns Pydantic "{type, params}" specs into concrete Python objects
via simple registries.

Constructor styles:

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
      Foo(**injected, **params_model_dump)

(B) spec-style constructors (preferred):
    class Bar:
        def __init__(self, *, spec: BarParams): ...

    YAML:
      bar:
        type: bar
        params: { ... fields of BarParams ... }

    Registry:
      {"bar": Bar}

    Call performed:
      Bar(**injected, spec=<params_model>)

Key rules (STRICT):
- injected kwargs are derived by code and must not be overrideable via params.
- Overlap between injected keys and params keys is forbidden (kwargs-style).
- Returning None is treated as an error (catches mis-registered procedures).
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

from pydantic import BaseModel

# Special tags meaning "disabled component" (instantiate() returns None).
_NULL_TYPES = {"none", "null", "off", "disabled"}


@dataclass(frozen=True)
class ComponentSpec:
    type: str
    params: Any


def as_component_spec(obj: Any, *, where: str) -> ComponentSpec:
    if isinstance(obj, BaseModel):
        try:
            t = getattr(obj, "type")
            params = getattr(obj, "params", None)
        except Exception as e:
            raise TypeError(f"{where} must expose .type and .params, got {type(obj)!r}") from e
    else:
        raise TypeError(f"{where} must be a Pydantic model with keys {{type, params}}, got {type(obj)!r}")

    if t is None:
        t = "null"

    s = str(t).strip().lower()
    if not s:
        raise TypeError(f"{where}.type must be a non-empty string, got {t!r}")

    return ComponentSpec(type=s, params=params)


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
      - {}            (when None)
      - BaseModel     (left as-is)
      - otherwise: error
    """
    if params_any is None:
        return {}
    if isinstance(params_any, BaseModel):
        return params_any
    raise TypeError(f"{where}.params must be a Pydantic model, got {type(params_any)!r}")


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
        if inj or isinstance(params_any, BaseModel):
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
    ctor = ctor_or_obj

    # ------------------------------------------------------------------
    # BaseModel params
    # ------------------------------------------------------------------
    params_any = _normalize_params(spec.params, where=where)
    if isinstance(params_any, BaseModel):
        kwargs: dict[str, Any] = dict(inj)

        # If ctor supports spec=..., prefer passing the model as spec directly.
        if _accepts_kwarg(ctor, "spec"):
            kwargs["spec"] = params_any
        else:
            # Fall back: expand model into kwargs-style.
            params_dict = params_any.model_dump()
            overlap = sorted(set(inj.keys()) & set(params_dict.keys()))
            if overlap:
                overlap_s = ", ".join(overlap)
                raise TypeError(
                    f"{where} type={spec.type!r} has injected/params key overlap: [{overlap_s}]. "
                    f"Remove these keys from {where}.params; they are injected/derived."
                )
            kwargs.update(params_dict)

        try:
            result = ctor(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"failed to construct {where} type={spec.type!r} using ctor={_callable_name(ctor)!r}{_signature_str(ctor)} "
                f"with params={params_any!r} injected={list(inj.keys())!r}: {e}"
            ) from e

        if result is None:
            raise TypeError(
                f"{where} type={spec.type!r} constructed via ctor={_callable_name(ctor)!r}{_signature_str(ctor)} "
                "returned None. Registry entries must be factories/constructors that RETURN an object."
            )
        return result

    # ------------------------------------------------------------------
    raise TypeError(f"{where}.params must be a Pydantic model, got {type(params_any)!r}")


__all__ = ["instantiate"]
