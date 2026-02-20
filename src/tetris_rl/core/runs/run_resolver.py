# src/tetris_rl/core/runs/run_resolver.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from planning_rl.ga import GAAlgorithm
from tetris_rl.core.config.io import (
    load_experiment_config,
    load_imitation_config,
    load_yaml,
    to_plain_dict,
)
from tetris_rl.core.policies.planning_policies.heuristic_policy import HeuristicPlanningPolicy
from tetris_rl.core.policies.sb3.config import SB3PolicyConfig
from tetris_rl.core.policies.spec import HeuristicSearch
from tetris_rl.core.runs.checkpoints.checkpoint_manifest import resolve_checkpoint_from_manifest
from tetris_rl.core.runs.run_io import choose_config_path
from tetris_rl.core.training.config import PolicySourceConfig
from tetris_rl.core.utils.paths import repo_root, resolve_run_dir


@dataclass(frozen=True)
class RunSpec:
    run_dir: Path
    cfg_path: Path
    cfg_raw: dict[str, Any]
    cfg_plain: dict[str, Any]
    algo_type: str
    exp_cfg: Any | None
    algo_cfg: Any | None
    env_train_cfg: Any
    env_eval_cfg: Any
    ga_features: list[str] | None = None
    ga_search: HeuristicSearch | None = None


@dataclass(frozen=True)
class InferenceArtifact:
    kind: Literal["sb3_ckpt", "ga_policy", "ga_zip", "imitation_ckpt", "td_ckpt"]
    path: Path
    note: str | None = None


@dataclass(frozen=True)
class PolicyBootstrap:
    source: Path
    policy_cfg: SB3PolicyConfig
    checkpoint: Path | None
    preferred_algo: str | None
    note: str


def load_run_spec(run_name: str | Path) -> RunSpec:
    repo = repo_root()
    run_dir = resolve_run_dir(repo, str(run_name))
    cfg_path = choose_config_path(run_dir)
    cfg_raw = load_yaml(cfg_path)
    algo_type = str(cfg_raw.get("algo", {}).get("type", "")).strip().lower()

    exp_cfg = None
    cfg_plain = cfg_raw
    algo_cfg = None
    env_train_cfg = cfg_raw.get("env_train", None) or cfg_raw.get("env", None)
    env_eval_cfg = cfg_raw.get("env_eval", None) or env_train_cfg
    ga_features: list[str] | None = None
    ga_search: HeuristicSearch | None = None

    if algo_type == "imitation":
        exp_cfg = load_imitation_config(cfg_path)
        cfg_plain = to_plain_dict(exp_cfg)
        env_train_cfg = exp_cfg.env_train
        env_eval_cfg = exp_cfg.env_eval
        algo_cfg = exp_cfg.algo
        algo_type = str(algo_cfg.type).strip().lower()
    elif algo_type in {"ga", "td"}:
        policy_cfg = cfg_raw.get("policy", {}) or {}
        features = policy_cfg.get("features", []) or []
        if not isinstance(features, list) or not features:
            raise ValueError("policy.features must be a non-empty list for planning runs")
        search_cfg = policy_cfg.get("search", {}) or {}
        if not isinstance(search_cfg, dict):
            raise TypeError("policy.search must be a mapping for planning runs")
        ga_features = list(features)
        ga_search = HeuristicSearch.model_validate(search_cfg)

        if not isinstance(env_train_cfg, dict):
            raise TypeError("env_train must be a mapping for planning runs")
        if not isinstance(env_eval_cfg, dict):
            raise TypeError("env_eval must be a mapping for planning runs")
    elif algo_type in {"ppo", "maskable_ppo"}:
        exp_cfg = load_experiment_config(cfg_path)
        cfg_plain = to_plain_dict(exp_cfg)
        env_train_cfg = exp_cfg.env_train
        env_eval_cfg = exp_cfg.env_eval
        algo_cfg = exp_cfg.algo
        algo_type = str(algo_cfg.type).strip().lower()
    else:
        raise ValueError("unknown config shape: expected algo.type in {ppo,maskable_ppo,imitation,ga,td}")

    return RunSpec(
        run_dir=run_dir,
        cfg_path=cfg_path,
        cfg_raw=cfg_raw,
        cfg_plain=cfg_plain,
        algo_type=algo_type,
        exp_cfg=exp_cfg,
        algo_cfg=algo_cfg,
        env_train_cfg=env_train_cfg,
        env_eval_cfg=env_eval_cfg,
        ga_features=ga_features,
        ga_search=ga_search,
    )


def resolve_env_cfg(*, spec: RunSpec, which_env: str) -> dict[str, Any]:
    which_env = str(which_env).strip().lower()
    env_cfg = spec.env_train_cfg if which_env == "train" else spec.env_eval_cfg
    if spec.exp_cfg is not None:
        return env_cfg.model_dump(mode="json")
    if not isinstance(env_cfg, dict):
        raise TypeError("env config must be a mapping")
    return dict(env_cfg)


def _find_latest_ga_ckpt(run_dir: Path) -> Path | None:
    ckpt_dir = Path(run_dir) / "checkpoints"
    latest = ckpt_dir / "latest.zip"
    if latest.is_file():
        return latest
    candidates = sorted(ckpt_dir.glob("ga_gen_*.zip"))
    if candidates:
        return candidates[-1]
    return None


def expected_checkpoint_path(*, run_dir: Path, which: str) -> Path:
    ckpt_dir = Path(run_dir) / "checkpoints"
    w = str(which).strip().lower()
    if w == "latest":
        name = "latest.zip"
    elif w in {"best", "reward"}:
        name = "best_reward.zip"
    elif w == "lines":
        name = "best_lines.zip"
    elif w in {"survival", "len", "length", "time"}:
        name = "best_survival.zip"
    elif w == "final":
        name = "final.zip"
    else:
        raise ValueError(f"unknown checkpoint selector: {which!r}")
    return ckpt_dir / name


def _resolve_policy_cfg_from_exp_policy(*, policy_obj: Any) -> SB3PolicyConfig:
    if isinstance(policy_obj, SB3PolicyConfig):
        return policy_obj
    if isinstance(policy_obj, PolicySourceConfig):
        nested = resolve_policy_bootstrap(
            source=str(policy_obj.source),
            which=str(policy_obj.which),
        )
        return nested.policy_cfg
    raise TypeError(f"unsupported policy object in source run: {type(policy_obj)!r}")


def resolve_policy_bootstrap(*, source: str, which: str = "latest") -> PolicyBootstrap:
    """
    Resolve a policy bootstrap source into:
      - policy config (required)
      - optional checkpoint path
      - optional preferred loader algo

    Supported source kinds:
      - run dir path/name
      - YAML path (full config with `policy` or bare policy config)
    """
    src_raw = str(source).strip()
    if not src_raw:
        raise ValueError("policy.source must be non-empty")

    repo = repo_root()
    src = Path(src_raw).expanduser()
    if src.is_absolute():
        src_path = src.resolve()
    else:
        src_path = (repo / src).resolve()

    spec: RunSpec | None = None
    try:
        spec = load_run_spec(src_raw)
    except Exception:
        spec = None

    if spec is not None:
        if spec.algo_type in {"ga", "td"}:
            raise ValueError(f"policy.source does not support planning runs (algo={spec.algo_type!r}): {spec.run_dir}")
        if spec.exp_cfg is None:
            raise ValueError(f"run config has no typed experiment config: {spec.run_dir}")
        policy_obj = getattr(spec.exp_cfg, "policy", None)
        if policy_obj is None:
            raise ValueError(f"source run has no top-level policy config: {spec.run_dir}")
        policy_cfg = _resolve_policy_cfg_from_exp_policy(policy_obj=policy_obj)

        try:
            ckpt = resolve_checkpoint_from_manifest(run_dir=spec.run_dir, which=str(which))
        except Exception:
            ckpt = expected_checkpoint_path(run_dir=spec.run_dir, which=str(which))
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"policy checkpoint not found: {ckpt} (run={spec.run_dir}, which={which!r})"
            )
        preferred_algo = "maskable_ppo" if spec.algo_type == "imitation" else spec.algo_type
        return PolicyBootstrap(
            source=spec.run_dir,
            policy_cfg=policy_cfg,
            checkpoint=ckpt,
            preferred_algo=preferred_algo,
            note=f"run:{spec.run_dir.name} which={str(which).strip().lower()}",
        )

    if not src_path.is_file():
        raise FileNotFoundError(f"policy source not found: {src_path}")

    if src_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            "policy.source must be a run dir or YAML path (full config or policy config): "
            f"{src_path}"
        )

    raw = load_yaml(src_path)
    policy_raw = raw.get("policy") if isinstance(raw.get("policy"), dict) else raw
    policy_cfg = SB3PolicyConfig.model_validate(policy_raw)
    return PolicyBootstrap(
        source=src_path,
        policy_cfg=policy_cfg,
        checkpoint=None,
        preferred_algo=None,
        note=f"config:{src_path.name}",
    )


def resolve_inference_artifact(*, spec: RunSpec, which: str) -> InferenceArtifact:
    if spec.algo_type == "imitation":
        path = resolve_checkpoint_from_manifest(run_dir=spec.run_dir, which=str(which))
        return InferenceArtifact(kind="imitation_ckpt", path=path)

    if spec.algo_type == "td":
        path = resolve_checkpoint_from_manifest(run_dir=spec.run_dir, which=str(which))
        return InferenceArtifact(kind="td_ckpt", path=path)

    if spec.algo_type != "ga":
        path = resolve_checkpoint_from_manifest(run_dir=spec.run_dir, which=str(which))
        return InferenceArtifact(kind="sb3_ckpt", path=path)

    ga_ckpt = _find_latest_ga_ckpt(spec.run_dir)
    if ga_ckpt is not None:
        return InferenceArtifact(kind="ga_zip", path=ga_ckpt)

    policy_path = spec.run_dir / "best_policy.yaml"
    if policy_path.is_file():
        return InferenceArtifact(kind="ga_policy", path=policy_path)

    intermediate_path = spec.run_dir / "intermediate_best_policy.yaml"
    if intermediate_path.is_file():
        return InferenceArtifact(kind="ga_policy", path=intermediate_path, note="intermediate")

    raise FileNotFoundError(
        "GA run missing checkpoints/latest.zip and policy specs.\n"
        f"run_dir={spec.run_dir}"
    )


def load_ga_policy_from_artifact(
    *,
    spec: RunSpec,
    artifact: InferenceArtifact,
    env: Any,
) -> HeuristicPlanningPolicy:
    if artifact.kind == "ga_policy":
        return HeuristicPlanningPolicy.from_yaml(artifact.path)

    if artifact.kind != "ga_zip":
        raise ValueError(f"unsupported artifact kind for GA: {artifact.kind}")
    if spec.ga_features is None or spec.ga_search is None:
        raise ValueError("GA run spec missing features/search")

    ga_stub = HeuristicPlanningPolicy(features=spec.ga_features, search=spec.ga_search)
    algo = GAAlgorithm.load(artifact.path, policy=ga_stub, env=env)
    algo.policy.set_params(algo.best_weights.tolist())
    return algo.policy


__all__ = [
    "expected_checkpoint_path",
    "InferenceArtifact",
    "PolicyBootstrap",
    "RunSpec",
    "load_ga_policy_from_artifact",
    "load_run_spec",
    "resolve_policy_bootstrap",
    "resolve_env_cfg",
    "resolve_inference_artifact",
]
