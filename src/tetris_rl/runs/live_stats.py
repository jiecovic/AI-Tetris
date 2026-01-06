# src/tetris_rl/runs/live_stats.py
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass(frozen=True)
class StepWindowSummary:
    steps: int
    avg_reward: float
    sum_lines: int

    # correctness: illegal rate over steps (0..1)
    avg_illegal: float

    # debug signals (kept for now; may or may not be shown in HUD)
    sum_masked: int
    sum_redrot: int

    avg_score_delta: float
    avg_episode_len: float

    # normalized joint action entropy over action_id in [0,1]
    action_entropy: float

    # live (current episode) diagnostics
    cur_episode_reward: float
    last_step_reward: float


class StepWindow:
    """
    Sliding window over env steps (not episodes).

    Tracks (rolling, per-step):
      - reward average
      - cleared lines sum
      - illegal rate (env truth)
      - masked count (debug)
      - redundant rotation count (debug)
      - score delta average
      - action entropy (normalized) over requested action_id

    Also tracks (live, current episode):
      - current episode accumulated reward
      - last step reward
    """

    def __init__(self, *, capacity: int, episode_capacity: int | None = None) -> None:
        self.capacity = max(0, int(capacity))

        if episode_capacity is None:
            episode_capacity = min(max(0, int(self.capacity)), 200)
        self.episode_capacity = max(0, int(episode_capacity))

        self._r: Deque[float] = deque()
        self._lines: Deque[int] = deque()
        self._illegal: Deque[int] = deque()

        self._masked: Deque[int] = deque()
        self._redrot: Deque[int] = deque()
        self._score_delta: Deque[float] = deque()

        # Requested-action tracking (for entropy); action_id is joint id (rot*w + col).
        self._action_id: Deque[int] = deque()
        self._action_dim: Optional[int] = None  # latest (assumed constant in a run)

        self._ep_lens: Deque[int] = deque()
        self._cur_ep_steps: int = 0

        self._cur_ep_reward: float = 0.0
        self._last_step_reward: float = 0.0

    def clear(self) -> None:
        self._r.clear()
        self._lines.clear()
        self._illegal.clear()
        self._masked.clear()
        self._redrot.clear()
        self._score_delta.clear()
        self._action_id.clear()
        self._action_dim = None

        self._ep_lens.clear()
        self._cur_ep_steps = 0

        self._cur_ep_reward = 0.0
        self._last_step_reward = 0.0

    def reset_episode(self) -> None:
        self._cur_ep_steps = 0
        self._cur_ep_reward = 0.0
        self._last_step_reward = 0.0

    def push(
            self,
            *,
            step_reward: float,
            cleared_lines: int,
            illegal: int,
            masked: int,
            redrot: int,
            score_delta: float = 0.0,
            action_id: Optional[int] = None,
            action_dim: Optional[int] = None,
            episode_done: bool = False,
    ) -> None:
        # Track episode length regardless of step-window capacity.
        self._cur_ep_steps += 1

        sr = float(step_reward)
        self._last_step_reward = sr
        self._cur_ep_reward += sr

        if bool(episode_done):
            if self.episode_capacity > 0:
                self._ep_lens.append(int(self._cur_ep_steps))
                while len(self._ep_lens) > self.episode_capacity:
                    self._ep_lens.popleft()
            self._cur_ep_steps = 0
            self._cur_ep_reward = 0.0
            self._last_step_reward = 0.0

        if self.capacity <= 0:
            return

        self._r.append(sr)
        self._lines.append(int(cleared_lines))
        self._illegal.append(1 if int(illegal) != 0 else 0)

        self._masked.append(1 if int(masked) != 0 else 0)
        self._redrot.append(1 if int(redrot) != 0 else 0)
        self._score_delta.append(float(score_delta))

        if action_dim is not None:
            try:
                ad = int(action_dim)
                if ad > 1:
                    self._action_dim = ad
            except Exception:
                pass

        if action_id is not None:
            try:
                self._action_id.append(int(action_id))
            except Exception:
                self._action_id.append(-1)
        else:
            self._action_id.append(-1)

        while len(self._r) > self.capacity:
            self._r.popleft()
            self._lines.popleft()
            self._illegal.popleft()
            self._masked.popleft()
            self._redrot.popleft()
            self._score_delta.popleft()
            self._action_id.popleft()

    @staticmethod
    def _normalized_entropy(action_ids: Deque[int], *, action_dim: Optional[int]) -> float:
        n = int(len(action_ids))
        ad = int(action_dim or 0)
        if n <= 0 or ad <= 1:
            return 0.0

        counts: dict[int, int] = {}
        for a in action_ids:
            aa = int(a)
            if aa < 0:
                continue
            counts[aa] = counts.get(aa, 0) + 1

        if not counts:
            return 0.0

        denom = float(sum(counts.values()))
        if denom <= 0.0:
            return 0.0

        h = 0.0
        for c in counts.values():
            p = float(c) / denom
            h -= p * math.log(p)

        hmax = math.log(float(ad))
        if hmax <= 0.0:
            return 0.0

        out = h / hmax
        if out < 0.0:
            return 0.0
        if out > 1.0:
            return 1.0
        return float(out)

    def summary(self) -> StepWindowSummary:
        n = int(len(self._r))

        avg_ep = 0.0
        if len(self._ep_lens) > 0:
            avg_ep = float(sum(self._ep_lens)) / float(len(self._ep_lens))

        ent = self._normalized_entropy(self._action_id, action_dim=self._action_dim)

        if n <= 0:
            return StepWindowSummary(
                steps=0,
                avg_reward=0.0,
                sum_lines=0,
                avg_illegal=0.0,
                sum_masked=0,
                sum_redrot=0,
                avg_score_delta=0.0,
                avg_episode_len=float(avg_ep),
                action_entropy=float(ent),
                cur_episode_reward=float(self._cur_ep_reward),
                last_step_reward=float(self._last_step_reward),
            )

        s_r = float(sum(self._r))
        s_sd = float(sum(self._score_delta))
        denom = float(n)

        s_illegal = int(sum(self._illegal))
        avg_illegal = float(s_illegal) / denom

        return StepWindowSummary(
            steps=int(n),
            avg_reward=float(s_r / denom),
            sum_lines=int(sum(self._lines)),
            avg_illegal=float(avg_illegal),
            sum_masked=int(sum(self._masked)),
            sum_redrot=int(sum(self._redrot)),
            avg_score_delta=float(s_sd / denom),
            avg_episode_len=float(avg_ep),
            action_entropy=float(ent),
            cur_episode_reward=float(self._cur_ep_reward),
            last_step_reward=float(self._last_step_reward),
        )
