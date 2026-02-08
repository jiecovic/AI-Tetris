# src/tetris_rl/core/config/typed_params.py
from __future__ import annotations

from typing import Any, Mapping, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def normalize_type(value: Any, *, where: str) -> str:
    s = str(value).strip().lower()
    if not s:
        raise ValueError(f"{where}.type must be a non-empty string")
    return s


def parse_typed_params(
    *,
    type_value: Any,
    params_value: Any,
    registry: Mapping[str, Type[T]],
    where: str,
) -> tuple[str, T]:
    tag = normalize_type(type_value, where=where)
    try:
        params_cls = registry[tag]
    except KeyError as e:
        known = ", ".join(sorted(registry.keys()))
        raise KeyError(f"{where}.type unknown: {tag!r}. known: [{known}]") from e

    if params_value is None:
        params_value = {}

    if isinstance(params_value, params_cls):
        return tag, params_value

    params = params_cls.model_validate(params_value)
    return tag, params

