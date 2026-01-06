# src/tetris_rl/config/parsing.py
from __future__ import annotations

from typing import Any, Iterable, Mapping, cast

def as_literal(value: str, allowed: Iterable[str], *, where: str) -> str:
    """
    Runtime validate a string against an allowed set, normalize to lowercase,
    and return it as `str` (caller then cast()s to the desired Literal type).
    """
    v = str(value).strip().lower()
    allowed_set = {str(x).strip().lower() for x in allowed}
    if v not in allowed_set:
        raise ValueError(f"{where} must be one of {sorted(allowed_set)}, got {value!r}")
    return v

def ensure_str_keys(m: Mapping[Any, Any], *, where: str) -> dict[str, Any]:
    """
    Convert Mapping[Any, Any] -> dict[str, Any] (strict check).
    Useful before **kwargs into dataclasses.
    """
    out: dict[str, Any] = {}
    for k, v in m.items():
        if not isinstance(k, str):
            raise TypeError(f"{where} must have str keys, got {k!r} ({type(k).__name__})")
        out[k] = v
    return out

def cast_literal(tp: type[Any], value: str) -> Any:
    # convenience wrapper so call-sites stay short
    return cast(tp, value)
