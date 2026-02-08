# src/tetris_rl/core/runtime/hud_text.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HudSnapshot:
    run_name: str

    mode: str
    ckpt_name: str
    paused: bool
    seed: int
    reload_every_s: float
    reloads: int
    last_reload_age_s: float

    episode_idx: int
    episode_step: int
    episode_reward: float
    last_step_reward: float

    # keep for now (game window also shows it, but harmless)
    next_kind: str
    piece_rule: str

    # rolling-window averages (implicit per-step)
    win_capacity: int
    win_steps: int
    win_avg_r: float
    win_avg_lines: float
    win_illegal_pct: float
    win_avg_score: float
    win_avg_ep_len: float  # rolling average episode length (in steps)

    # rolling-window action diversity (normalized joint entropy in [0,1])
    win_action_entropy: float


class HudFormatter:
    def __init__(self, *, window_steps: int) -> None:
        self.window_steps = max(0, int(window_steps))

    @staticmethod
    def _yn(b: bool) -> str:
        return "YES" if b else "no"

    @staticmethod
    def _fmt_reload_every(r: float) -> str:
        r = float(r)
        if r <= 0:
            return "off"
        if r.is_integer():
            return f"{int(r)}s"
        return f"{r:g}s"

    @staticmethod
    def _fmt_age(age_s: float) -> str:
        a = float(age_s)
        if a == float("inf"):
            return "never"
        if a < 0:
            a = 0.0
        if a < 60:
            return f"{a:0.0f}s"
        m = int(a // 60)
        s = int(a - 60 * m)
        if m < 60:
            return f"{m}m {s:02d}s"
        h = int(m // 60)
        m2 = int(m - 60 * h)
        return f"{h}h {m2:02d}m"

    @staticmethod
    def _safe(s: str, fallback: str = "?") -> str:
        ss = str(s).strip()
        return ss if ss else fallback

    def format_lines(self, s: HudSnapshot) -> list[str]:
        lines: list[str] = []

        # Status
        lines.append("# Status")
        lines.append(f"Run: {self._safe(s.run_name)} | Checkpoint: {self._safe(s.ckpt_name)}")
        lines.append(f"Mode: {self._safe(s.mode)} | Auto-reload: {self._fmt_reload_every(s.reload_every_s)}")
        lines.append(f"Last reload: {self._fmt_age(s.last_reload_age_s)} ago | Paused: {self._yn(bool(s.paused))}")
        lines.append("")

        # Game
        lines.append("# Game")
        lines.append(f"Rule: {self._safe(s.piece_rule)} | Seed: {int(s.seed)}")
        lines.append("")

        # Episode
        lines.append("# Episode")
        lines.append(f"Episode: {int(s.episode_idx)} | Step: {int(s.episode_step)}")
        lines.append(f"Ep reward: {float(s.episode_reward):+0.2f} | Last r: {float(s.last_step_reward):+0.2f}")
        lines.append("")

        # Recent performance
        cap = int(s.win_capacity)
        lines.append(f"# Recent performance (per-step, last {cap} steps)")
        lines.append(f"Samples: {max(0, int(s.win_steps))} | Avg ep len: {float(s.win_avg_ep_len):0.1f}")
        lines.append(f"Avg reward: {float(s.win_avg_r):+0.3f} | Avg score: {float(s.win_avg_score):0.1f}")
        lines.append(f"Illegal%: {float(s.win_illegal_pct):0.3f}% | Action entropy: {float(s.win_action_entropy):0.3f}")
        lines.append(f"Avg lines: {float(s.win_avg_lines):0.4f}")
        lines.append("")

        # No duplicate controls here (renderer has a controls panel).
        return lines[:24]

    def format_text(self, s: HudSnapshot) -> str:
        return "\n".join(self.format_lines(s))
