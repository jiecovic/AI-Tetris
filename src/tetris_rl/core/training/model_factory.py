# src/tetris_rl/core/training/model_factory.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from tetris_rl.core.policies.sb3.config import SB3PolicyConfig
from tetris_rl.core.policies.sb3.feature_extractor import TetrisFeatureExtractor
from tetris_rl.core.runs.config import RunConfig
from tetris_rl.core.training.config import AlgoConfig
from tetris_rl.core.utils.logging import setup_logger
from tetris_rl.core.utils.model_params import (
    build_algo_kwargs,
    parse_activation_fn,
    parse_net_arch,
)

LOG = setup_logger(
    name="tetris_rl.model_factory",
    use_rich=True,
    level="info",
)


def _infer_kind_vocab_size_from_obs_space(observation_space: spaces.Space) -> int:
    if not isinstance(observation_space, spaces.Dict):
        raise TypeError(f"expected spaces.Dict observation_space, got {type(observation_space)!r}")
    if "active_kind" not in observation_space.spaces:
        raise KeyError("observation_space missing key 'active_kind'")
    sp = observation_space.spaces["active_kind"]
    if not isinstance(sp, spaces.Discrete):
        raise TypeError(f"obs['active_kind'] must be spaces.Discrete, got {type(sp)!r}")
    k = int(sp.n)
    if k < 1:
        raise ValueError(f"invalid Discrete(K) for kinds: K={k}")
    return k


def build_policy_kwargs_from_cfg(
    *,
    policy_cfg: SB3PolicyConfig,
    observation_space: spaces.Space,
) -> Dict[str, Any]:
    """
    SB3 policy_kwargs:
      - feature extractor wiring
      - net_arch
      - activation_fn
    """
    policy_kwargs = dict(policy_cfg.policy_kwargs or {})
    net_arch = parse_net_arch(policy_kwargs.get("net_arch", policy_cfg.net_arch))
    activation_fn = parse_activation_fn(policy_kwargs.get("activation_fn", policy_cfg.activation_fn or "gelu"))

    fe_cfg = policy_cfg.feature_extractor

    tokenizer = None
    mixer = None
    spatial_head = None

    if fe_cfg.encoder.type == "token":
        tokenizer = fe_cfg.encoder.tokenizer
        mixer = fe_cfg.encoder.mixer
    elif fe_cfg.encoder.type == "spatial":
        spatial_head = fe_cfg.encoder.spatial_head
    else:
        raise ValueError("model.feature_extractor.encoder.type must be 'token' or 'spatial'")

    n_kinds = _infer_kind_vocab_size_from_obs_space(observation_space)

    # NOTE:
    # The feature extractor derives its own output dim from the built branch:
    #   - token route: mixer.params.features_dim
    #   - spatial route: spatial_head.params.features_dim
    features_extractor_kwargs: Dict[str, Any] = dict(
        spatial_preprocessor=fe_cfg.spatial_preprocessor,
        stem=fe_cfg.stem,
        tokenizer=tokenizer,
        mixer=mixer,
        spatial_head=spatial_head,
        feature_augmenter=fe_cfg.feature_augmenter,
        n_kinds=n_kinds,
    )

    return dict(
        features_extractor_class=TetrisFeatureExtractor,
        features_extractor_kwargs=features_extractor_kwargs,
        net_arch=net_arch,
        activation_fn=activation_fn,
    )


def build_policy_from_cfg(
    *,
    policy_cfg: SB3PolicyConfig,
    policy_backend: str,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    device: str = "cpu",
):
    policy_kwargs = build_policy_kwargs_from_cfg(
        policy_cfg=policy_cfg,
        observation_space=observation_space,
    )

    backend = str(policy_backend).strip().lower()
    if backend == "maskable_ppo":
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

        Policy = MaskableActorCriticPolicy
    elif backend == "ppo":
        Policy = ActorCriticPolicy
    else:
        raise ValueError(f"unsupported policy_backend for imitation: {backend!r}")

    def _lr_schedule(_progress: float) -> float:
        return 0.0

    policy = Policy(
        observation_space,
        action_space,
        _lr_schedule,
        **policy_kwargs,
    )
    return policy.to(device)


def make_model_from_cfg(
    *,
    cfg: SB3PolicyConfig,
    algo_cfg: AlgoConfig,
    run_cfg: RunConfig,
    vec_env: Any,
    tensorboard_log: Path | None,
):
    algo_type = str(algo_cfg.type).strip().lower()
    algo_params = algo_cfg.params or {}

    policy_kwargs = build_policy_kwargs_from_cfg(
        policy_cfg=cfg,
        observation_space=vec_env.observation_space,
    )

    device = str(run_cfg.device).strip() or "auto"

    # ------------------------------------------------------------------
    # PPO / MaskablePPO
    # ------------------------------------------------------------------
    if algo_type == "maskable_ppo":
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        from sb3_contrib.ppo_mask import MaskablePPO

        model_kwargs = build_algo_kwargs(
            algo_cls=MaskablePPO,
            raw=algo_params,
            seed=run_cfg.seed,
            where="algo.params",
        )
        model = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=vec_env,
            device=device,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        setattr(model, "_tetris_algo_type", algo_type)
        return model

    if algo_type == "ppo":
        model_kwargs = build_algo_kwargs(
            algo_cls=PPO,
            raw=algo_params,
            seed=run_cfg.seed,
            where="algo.params",
        )
        model = PPO(
            policy=ActorCriticPolicy,
            env=vec_env,
            device=device,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        setattr(model, "_tetris_algo_type", algo_type)
        return model

    raise ValueError(f"unsupported algo type: {algo_type!r}")

