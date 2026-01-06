# src/tetris_rl/config/schema_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Iterable, Optional, TypeVar

T = TypeVar("T")

# -----------------------------------------------------------------------------
# Core type checks
# -----------------------------------------------------------------------------

def require_mapping(obj: Any, *, where: str) -> dict[str, Any]:
    """
    Optional mapping:
      - None -> {}
      - mapping -> materialized dict[str, Any]
    """
    if obj is None:
        return {}
    if not isinstance(obj, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(obj)!r}")
    return {str(k): v for k, v in obj.items()}


def require_mapping_strict(
    obj: Any,
    *,
    where: str,
    allowed_keys: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """
    Required mapping:
      - None -> error
      - mapping -> materialized dict[str, Any]

    If allowed_keys is provided, reject unknown keys.
    """
    if obj is None:
        raise TypeError(f"{where} must be a mapping, got None")
    if not isinstance(obj, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(obj)!r}")

    out = {str(k): v for k, v in obj.items()}

    if allowed_keys is not None:
        allowed = set(allowed_keys)
        unknown = set(out.keys()) - allowed
        if unknown:
            raise KeyError(
                f"{where} contains unknown keys: {sorted(unknown)}; "
                f"allowed keys are: {sorted(allowed)}"
            )

    return out


def mget(m: Mapping[str, Any], k: str, default: Any) -> Any:
    try:
        return m.get(k, default)
    except Exception:
        return default


def require_submapping(cfg: Mapping[str, Any], key: str, *, where: str) -> dict[str, Any]:
    """
    Read cfg[key] as an optional mapping (or {} if missing/None).
    """
    if not isinstance(cfg, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(cfg)!r}")

    v = cfg.get(key, {})
    if v is None:
        return {}
    if not isinstance(v, Mapping):
        raise TypeError(f"{where}.{key} must be a mapping, got {type(v)!r}")
    return {str(k): vv for k, vv in v.items()}


# -----------------------------------------------------------------------------
# Scalar coercions
# -----------------------------------------------------------------------------


def as_int(x: Any, default: int) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def as_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def as_bool(x: Any, default: bool) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def as_str(x: Any, default: str) -> str:
    if x is None:
        return str(default)
    try:
        s = str(x)
    except Exception:
        return str(default)
    s2 = s.strip()
    return s2 if s2 else str(default)


def clamp_prob(p: float) -> float:
    p = float(p)
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


# -----------------------------------------------------------------------------
# Mapping getters (canonical style used by parsers)
# -----------------------------------------------------------------------------


def get_int(m: Mapping[str, Any], k: str, *, default: int, where: str) -> int:
    if not isinstance(m, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(m)!r}")
    return as_int(m.get(k, default), int(default))


def get_float(m: Mapping[str, Any], k: str, *, default: float, where: str) -> float:
    if not isinstance(m, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(m)!r}")
    return as_float(m.get(k, default), float(default))


def get_bool(m: Mapping[str, Any], k: str, *, default: bool, where: str) -> bool:
    if not isinstance(m, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(m)!r}")
    return as_bool(m.get(k, default), bool(default))


def get_str(m: Mapping[str, Any], k: str, *, default: str, where: str) -> str:
    if not isinstance(m, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(m)!r}")
    return as_str(m.get(k, default), str(default))


def get_mapping(
    m: Mapping[str, Any],
    k: str,
    *,
    default: Mapping[str, Any] | None = None,
    where: str,
) -> dict[str, Any]:
    """
    Read m[k] as a mapping (or default / {}), materialize to dict[str, Any].
    """
    if not isinstance(m, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(m)!r}")

    v = m.get(k, None)
    if v is None:
        v = {} if default is None else default

    if not isinstance(v, Mapping):
        raise TypeError(f"{where}.{k} must be a mapping, got {type(v)!r}")

    return {str(kk): vv for kk, vv in v.items()}


# -----------------------------------------------------------------------------
# Tiny structured helpers
# -----------------------------------------------------------------------------


def ensure_dict_str_any(x: Any, *, where: str) -> dict[str, Any]:
    if x is None:
        return {}
    if not isinstance(x, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(x)!r}")
    return {str(k): v for k, v in x.items()}


# -----------------------------------------------------------------------------
# Standard component shape {type, params}
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSpec:
    type: str
    params: dict[str, Any]


def as_component_spec(obj: Any, *, where: str) -> ComponentSpec:
    if not isinstance(obj, Mapping):
        raise TypeError(f"{where} must be a mapping with keys {{type, params}}, got {type(obj)!r}")

    t = obj.get("type", None)

    # Allow YAML null -> disabled component.
    # We normalize None to the literal tag "null" so downstream code can handle it
    # consistently (e.g. instantiate() can treat type in {"null","none","off",...}).
    if t is None:
        t = "null"

    if not isinstance(t, str) or not t.strip():
        raise TypeError(f"{where}.type must be a non-empty string, got {t!r}")

    params = obj.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise TypeError(f"{where}.params must be a mapping, got {type(params)!r}")

    return ComponentSpec(type=str(t).strip(), params={str(k): v for k, v in params.items()})
