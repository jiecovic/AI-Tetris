# src/tetris_rl/training/model_factory.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy

from tetris_rl.config.model_spec import ModelConfig
from tetris_rl.config.run_spec import RunSpec
from tetris_rl.config.train_spec import TrainSpec
from tetris_rl.models.feature_extractor import TetrisFeatureExtractor
from tetris_rl.utils.logging import setup_logger
from tetris_rl.utils.model_params import (
    build_algo_kwargs,
    net_arch_for_dqn,
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
    model_cfg: ModelConfig,
    observation_space: spaces.Space,
) -> Dict[str, Any]:
    """
    SB3 policy_kwargs:
      - feature extractor wiring
      - net_arch
      - activation_fn
    """
    policy_cfg = dict(model_cfg.policy_kwargs or {})
    net_arch = parse_net_arch(policy_cfg.get("net_arch", model_cfg.net_arch))
    activation_fn = parse_activation_fn(policy_cfg.get("activation_fn", model_cfg.activation_fn or "gelu"))

    fe_cfg = model_cfg.feature_extractor

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


def make_model_from_cfg(
    *,
    cfg: ModelConfig,
    train_spec: TrainSpec,
    run_spec: RunSpec,
    vec_env: Any,
    tensorboard_log: Path | None,
):
    algo_type = str(train_spec.rl.algo.type).strip().lower()
    algo_params = train_spec.rl.algo.params or {}

    policy_kwargs = build_policy_kwargs_from_cfg(
        model_cfg=cfg,
        observation_space=vec_env.observation_space,
    )

    device = str(run_spec.device).strip() or "auto"

    # ------------------------------------------------------------------
    # DQN
    # ------------------------------------------------------------------
    if algo_type == "dqn":
        if not isinstance(vec_env.action_space, spaces.Discrete):
            raise TypeError("DQN requires Discrete action space")

        model_kwargs = build_algo_kwargs(
            algo_cls=DQN,
            raw=algo_params,
            seed=run_spec.seed,
            where="train.rl.algo.params",
        )

        pk = dict(policy_kwargs)
        pk["net_arch"] = net_arch_for_dqn(pk.get("net_arch"))

        model = DQN(
            policy=DQNPolicy,
            env=vec_env,
            device=device,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            policy_kwargs=pk,
            **model_kwargs,
        )
        model._tetris_algo_type = "dqn"
        return model

    # ------------------------------------------------------------------
    # PPO / MaskablePPO
    # ------------------------------------------------------------------
    if algo_type in {"ppo", "maskable_ppo"}:
        if algo_type == "maskable_ppo":
            from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
            from sb3_contrib.ppo_mask import MaskablePPO

            Algo = MaskablePPO
            Policy = MaskableActorCriticPolicy
        else:
            Algo = PPO
            Policy = ActorCriticPolicy

        model_kwargs = build_algo_kwargs(
            algo_cls=Algo,
            raw=algo_params,
            seed=run_spec.seed,
            where="train.rl.algo.params",
        )

        model = Algo(
            policy=Policy,
            env=vec_env,
            device=device,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        model._tetris_algo_type = algo_type
        return model

    raise ValueError(f"unsupported algo type: {algo_type!r}")
