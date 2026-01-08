# src/tetris_rl/utils/paths.py
from __future__ import annotations

from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file():
            return p
    return None


def repo_root() -> Path:
    """
    Return the repository root by searching upwards for pyproject.toml.
    """
    here = Path(__file__).resolve()
    root = _find_repo_root(here.parent)
    if root is None:
        raise FileNotFoundError("Could not locate repo root (pyproject.toml not found).")
    return root


def assets_dir() -> Path:
    """
    Return repo_root/assets (must exist).
    """
    p = repo_root() / "assets"
    if not p.is_dir():
        raise FileNotFoundError(f"Assets directory not found: {p}")
    return p


def pieces_dir() -> Path:
    """
    Return repo_root/assets/pieces (must exist).
    """
    p = assets_dir() / "pieces"
    if not p.is_dir():
        raise FileNotFoundError(f"Pieces directory not found: {p}")
    return p


# ---------------------------------------------------------------------------
# ADDITIONS (used by cli/watch.py)
# ---------------------------------------------------------------------------

def relpath(path: Path, *, base: Path) -> str:
    """
    Safe relative path helper for logging.
    Falls back to absolute if relative_to fails.
    """
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(path)


def resolve_run_dir(repo: Path, run: str) -> Path:
    """
    Resolve a run identifier into a concrete directory.

    Accepted:
      - absolute path
      - relative path
      - bare run name -> repo/experiments/<run>
      - experiments/<run>

    This matches how training + watch refer to runs.
    """
    r = run.strip().strip('"').strip("'")
    if not r:
        raise ValueError("run must be non-empty")

    p = Path(r)

    if p.is_absolute():
        out = p
    else:
        if p.parts and p.parts[0] == "experiments":
            out = (repo / p).resolve()
        elif len(p.parts) > 1:
            out = (repo / p).resolve()
        else:
            out = (repo / "experiments" / p).resolve()

    if not out.exists():
        raise FileNotFoundError(f"run dir not found: {out}")
    if not out.is_dir():
        raise NotADirectoryError(f"run path is not a directory: {out}")
    return out


def resolve_repo_or_cfg_path(*, raw: str, repo: Path, cfg_path: Path) -> Path:
    """
    Resolve a path string that may be:
      - absolute
      - repo-relative (including "configs/...")
      - cfg-relative (relative to the YAML file)

    Tries multiple candidates in a deterministic order.
    """
    s = str(raw).strip().strip('"').strip("'")
    if not s:
        raise ValueError("empty path")

    p_raw = Path(s)

    if p_raw.is_absolute():
        return p_raw.resolve()

    candidates: list[Path] = []

    # 1) repo-relative as written
    candidates.append((repo / p_raw).resolve())

    # 2) if it starts with "configs/", also try stripping that prefix
    #    (supports old QoL without breaking canonical "configs/..." usage)
    if p_raw.parts and p_raw.parts[0].lower() == "configs":
        p_stripped = Path(*p_raw.parts[1:]) if len(p_raw.parts) > 1 else Path()
        candidates.append((repo / p_stripped).resolve())

    # 3) cfg-relative as written
    candidates.append((cfg_path.parent / p_raw).resolve())

    # 4) cfg-relative stripped (if it started with configs/)
    if p_raw.parts and p_raw.parts[0].lower() == "configs":
        p_stripped = Path(*p_raw.parts[1:]) if len(p_raw.parts) > 1 else Path()
        candidates.append((cfg_path.parent / p_stripped).resolve())

    for cand in candidates:
        if cand.is_file():
            return cand

    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(f"path not found. tried:\n{tried}")
