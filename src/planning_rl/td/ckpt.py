# src/planning_rl/td/ckpt.py
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict

import torch

from planning_rl.ga.utils import to_jsonable


def save_td_checkpoint(
    *,
    path: Path,
    meta: Dict[str, Any],
    cfg: Dict[str, Any],
    policy_state: Dict[str, Any],
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any] | None,
    stats: list[Dict[str, Any]] | None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(to_jsonable(meta), indent=2))
        zf.writestr("cfg.json", json.dumps(to_jsonable(cfg), indent=2))
        zf.writestr("policy_state.json", json.dumps(to_jsonable(policy_state), indent=2))

        buf = io.BytesIO()
        torch.save(model_state, buf)
        zf.writestr("model_state.pt", buf.getvalue())

        if optimizer_state is not None:
            buf = io.BytesIO()
            torch.save(optimizer_state, buf)
            zf.writestr("optimizer_state.pt", buf.getvalue())

        if stats is not None:
            zf.writestr("stats.json", json.dumps(to_jsonable(stats), indent=2))

    return out_path


def load_td_checkpoint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("meta.json") as fh:
            meta = json.load(fh)
        with zf.open("cfg.json") as fh:
            cfg = json.load(fh)
        with zf.open("policy_state.json") as fh:
            policy_state = json.load(fh)

        with zf.open("model_state.pt") as fh:
            model_state = torch.load(fh, map_location="cpu")

        optimizer_state = None
        if "optimizer_state.pt" in zf.namelist():
            with zf.open("optimizer_state.pt") as fh:
                optimizer_state = torch.load(fh, map_location="cpu")

        stats = None
        if "stats.json" in zf.namelist():
            with zf.open("stats.json") as fh:
                stats = json.load(fh)

    return {
        "meta": meta,
        "cfg": cfg,
        "policy_state": policy_state,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "stats": stats,
    }


__all__ = ["load_td_checkpoint", "save_td_checkpoint"]
