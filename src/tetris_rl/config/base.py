from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

