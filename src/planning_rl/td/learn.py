# src/planning_rl/td/learn.py
from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Deque, Sequence

import numpy as np
import torch

from planning_rl.callbacks import PlanningCallback

if TYPE_CHECKING:
    from planning_rl.td.algorithm import TDAlgorithm
from planning_rl.logging import ScalarLogger
from planning_rl.td.config import TDConfig
from planning_rl.td.features import extract_features
from planning_rl.td.utils import episode_seed


def _compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros((rewards.shape[1],), dtype=np.float32)
    for t in range(rewards.shape[0] - 1, -1, -1):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
    return advantages


def learn_td(
    *,
    algo: TDAlgorithm,
    envs: Sequence[Any],
    features: Sequence[str],
    cfg: TDConfig,
    callback: PlanningCallback | None = None,
    logger: ScalarLogger | None = None,
) -> None:
    cb = callback

    n_envs = max(1, int(cfg.n_envs))
    if len(envs) != n_envs:
        raise ValueError("envs length must match cfg.n_envs")

    device = algo.device
    model = algo.policy.value_model
    model.train()
    if algo.optimizer is None:
        raise RuntimeError("TD algorithm missing optimizer")

    target_tau = float(getattr(cfg, "target_tau", 0.0))
    target_update_every = max(1, int(getattr(cfg, "target_update_every", 1)))
    target_model = None
    if target_tau > 0.0:
        target_model = deepcopy(model)
        target_model.eval()
        for p in target_model.parameters():
            p.requires_grad_(False)

    gamma = float(cfg.gamma)
    gae_lambda = float(cfg.gae_lambda)
    grad_clip = float(cfg.grad_clip)
    clip_range_vf = float(cfg.clip_range_vf)
    batch_size = max(1, int(cfg.batch_size))
    n_epochs = max(1, int(cfg.n_epochs))
    rollout_steps = max(1, int(cfg.rollout_steps))
    total_steps_target = max(0, int(cfg.total_timesteps))

    feature_clear_mode = str(getattr(cfg, "feature_clear_mode", "auto")).strip().lower()
    if feature_clear_mode == "auto":
        env_mode = None
        for env in envs:
            mode = getattr(env, "feature_clear_mode", None)
            if mode is None:
                continue
            mode = str(mode).strip().lower()
            if mode in {"pre", "lock", "pre_clear", "before"}:
                mode = "lock"
            elif mode in {"post", "clear", "post_clear", "after"}:
                mode = "post"
            else:
                continue
            if env_mode is None:
                env_mode = mode
            elif env_mode != mode:
                raise ValueError("TD envs disagree on feature_clear_mode")
        feature_clear_mode = env_mode or "post"

    if feature_clear_mode in {"pre", "lock", "pre_clear", "before"}:
        pre_clear = True
    elif feature_clear_mode in {"post", "clear", "post_clear", "after"}:
        pre_clear = False
    else:
        raise ValueError(
            "feature_clear_mode must be 'auto', 'pre', or 'post' "
            f"(got {feature_clear_mode!r})"
        )

    line_feature_names = {"complete_lines", "complete_lines_norm", "lines", "lines_norm"}
    line_feature_idx = [
        i for i, name in enumerate(features) if str(name).strip().lower() in line_feature_names
    ]

    def _features_for_action(*, env: Any, action: Any) -> np.ndarray:
        phi = extract_features(
            env=env,
            features=features,
            action=action,
            pre_clear=pre_clear,
        )
        if not pre_clear and line_feature_idx:
            # Post-clear grids have zero full rows; keep line counts from lock grids.
            line_names = [features[i] for i in line_feature_idx]
            line_vals = extract_features(
                env=env,
                features=line_names,
                action=action,
                pre_clear=True,
            )
            for idx, val in zip(line_feature_idx, line_vals):
                phi[idx] = float(val)
        return phi

    rng = np.random.default_rng(int(cfg.seed))

    ep_returns = [0.0 for _ in range(n_envs)]
    ep_steps = [0 for _ in range(n_envs)]
    ep_idx = [0 for _ in range(n_envs)]
    stats_window = max(1, int(cfg.stats_window))
    ep_ret_hist: Deque[float] = deque(maxlen=stats_window)
    ep_len_hist: Deque[int] = deque(maxlen=stats_window)

    for i, env in enumerate(envs):
        seed = episode_seed(base_seed=int(cfg.seed), env_idx=int(i), episode_idx=0)
        env.reset(seed=int(seed))

    while int(algo.num_timesteps) < int(total_steps_target):
        rollout_features: list[list[float]] = []
        rollout_values: list[list[float]] = []
        rollout_rewards: list[list[float]] = []
        rollout_dones: list[list[float]] = []
        rollout_next_values: list[list[float]] = []

        for _ in range(int(rollout_steps)):
            if int(algo.num_timesteps) >= int(total_steps_target):
                break

            features_batch: list[np.ndarray] = []
            actions_batch: list[Any] = []
            for env in envs:
                action = algo.policy.predict(env=env)
                phi = _features_for_action(env=env, action=action)
                features_batch.append(phi)
                actions_batch.append(action)

            feats_arr = np.asarray(features_batch, dtype=np.float32)
            with torch.no_grad():
                v = model(torch.tensor(feats_arr, device=device)).detach().cpu().numpy()
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.shape[0] != n_envs:
                if v.shape[0] == 1:
                    v = np.repeat(v, n_envs)
                else:
                    raise RuntimeError(
                        f"TD value shape mismatch: expected ({n_envs},), got {v.shape}"
                    )

            rewards_batch: list[float] = []
            dones_batch: list[float] = []
            next_features_batch: list[np.ndarray] = []

            for env_idx, env in enumerate(envs):
                action = actions_batch[env_idx]
                _obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated) or bool(truncated)

                ep_returns[env_idx] += float(reward)
                ep_steps[env_idx] += 1
                rewards_batch.append(float(reward))
                dones_batch.append(1.0 if done else 0.0)

                if done:
                    next_phi = np.zeros((len(features),), dtype=np.float32)
                else:
                    next_action = algo.policy.predict(env=env)
                    next_phi = _features_for_action(env=env, action=next_action)
                next_features_batch.append(next_phi)

                algo.num_timesteps += 1
                if cb is not None:
                    cb.on_event(event="step", num_timesteps=int(algo.num_timesteps))

                if done:
                    ep_ret_hist.append(float(ep_returns[env_idx]))
                    ep_len_hist.append(int(ep_steps[env_idx]))
                    ep_returns[env_idx] = 0.0
                    ep_steps[env_idx] = 0
                    ep_idx[env_idx] += 1
                    if int(algo.num_timesteps) < int(total_steps_target):
                        env.reset()

            next_feats_arr = np.asarray(next_features_batch, dtype=np.float32)
            with torch.no_grad():
                target = target_model if target_model is not None else model
                next_v = target(torch.tensor(next_feats_arr, device=device)).detach().cpu().numpy()
            next_v = np.asarray(next_v, dtype=np.float32).reshape(-1)
            if next_v.shape[0] != n_envs:
                if next_v.shape[0] == 1:
                    next_v = np.repeat(next_v, n_envs)
                else:
                    raise RuntimeError(
                        f"TD next value shape mismatch: expected ({n_envs},), got {next_v.shape}"
                    )

            for i, done_flag in enumerate(dones_batch):
                if done_flag > 0.0:
                    next_v[i] = 0.0

            rollout_features.append(feats_arr)
            rollout_values.append(np.asarray(v, dtype=np.float32))
            rollout_rewards.append(np.asarray(rewards_batch, dtype=np.float32))
            rollout_dones.append(np.asarray(dones_batch, dtype=np.float32))
            rollout_next_values.append(np.asarray(next_v, dtype=np.float32))

            if int(algo.num_timesteps) >= int(total_steps_target):
                break

        if not rollout_features:
            break

        features_arr = np.asarray(rollout_features, dtype=np.float32)
        values_arr = np.asarray(rollout_values, dtype=np.float32)
        rewards_arr = np.asarray(rollout_rewards, dtype=np.float32)
        dones_arr = np.asarray(rollout_dones, dtype=np.float32)
        next_values_arr = np.asarray(rollout_next_values, dtype=np.float32)

        advantages = _compute_gae(
            rewards=rewards_arr,
            values=values_arr,
            next_values=next_values_arr,
            dones=dones_arr,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        adv_norm = str(getattr(cfg, "advantage_norm", "none")).strip().lower()
        if adv_norm not in {"none", "off", "false", "0", "scale", "std"}:
            raise ValueError(f"advantage_norm must be 'none' or 'scale' (got {adv_norm!r})")
        if adv_norm in {"scale", "std"}:
            # Scale-only normalization keeps targets unbiased for a bias-free value model.
            adv_std = float(np.std(advantages))
            if adv_std > 0.0:
                advantages = advantages / adv_std
        targets = advantages + values_arr

        feats_flat = features_arr.reshape((-1, features_arr.shape[-1]))
        targets_flat = targets.reshape(-1)
        values_flat = values_arr.reshape(-1)

        idx = np.arange(targets_flat.shape[0])
        rng.shuffle(idx)
        last_loss = 0.0
        update_step = 0
        for _ in range(int(n_epochs)):
            rng.shuffle(idx)
            for start in range(0, len(idx), batch_size):
                mb = idx[start : start + batch_size]
                feats_mb = feats_flat[mb]
                target_mb = targets_flat[mb]
                v_old_mb = values_flat[mb]

                v_pred = model(torch.tensor(feats_mb, device=device)).reshape(-1)
                target_t = torch.tensor(target_mb, dtype=torch.float32, device=device)
                v_old_t = torch.tensor(v_old_mb, dtype=torch.float32, device=device)

                if clip_range_vf > 0.0:
                    v_pred_clipped = v_old_t + torch.clamp(
                        v_pred - v_old_t,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                    loss_unclipped = (v_pred - target_t) ** 2
                    loss_clipped = (v_pred_clipped - target_t) ** 2
                    loss = torch.mean(torch.max(loss_unclipped, loss_clipped))
                else:
                    loss = torch.mean((v_pred - target_t) ** 2)

                algo.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                algo.optimizer.step()
                last_loss = float(loss.detach().cpu().item())
                update_step += 1
                if target_model is not None and (update_step % target_update_every == 0):
                    with torch.no_grad():
                        for p_t, p in zip(target_model.parameters(), model.parameters()):
                            p_t.data.mul_(1.0 - target_tau).add_(p.data, alpha=target_tau)

        algo.policy.sync_from_model()
        algo.stats.append(
            {
                "num_timesteps": int(algo.num_timesteps),
                "loss": float(last_loss),
                "adv_mean": float(np.mean(advantages)),
                "adv_std": float(np.std(advantages)),
                "target_mean": float(np.mean(targets)),
                "ep_ret_mean": float(np.mean(ep_ret_hist)) if ep_ret_hist else None,
                "ep_len_mean": float(np.mean(ep_len_hist)) if ep_len_hist else None,
            }
        )
        if logger is not None:
            step = int(algo.num_timesteps)
            logger.log_scalar("train/mean_td_loss", float(last_loss), step)
            logger.log_scalar("train/adv_mean", float(np.mean(advantages)), step)
            logger.log_scalar("train/adv_std", float(np.std(advantages)), step)
            logger.log_scalar("train/target_mean", float(np.mean(targets)), step)
            if ep_ret_hist:
                logger.log_scalar("train/ep_ret_mean", float(np.mean(ep_ret_hist)), step)
            if ep_len_hist:
                logger.log_scalar("train/ep_len_mean", float(np.mean(ep_len_hist)), step)

    _ = cb


__all__ = ["learn_td"]
