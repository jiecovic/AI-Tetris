# src/tetris_rl/core/runtime/speed_control.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional
import time


@dataclass
class SpeedControl:
    """
    Decouples simulation stepping from rendering.

    render_fps_cap:
      - caps the render loop (Clock.tick)
      - keeps input responsive and avoids busy spinning

    step_ms semantics:
      >0 : fixed ms between steps
       0 : one step per rendered frame (step rate ~= render fps cap)
      <0 : uncapped sim (run as many steps as possible; renders still capped)
    """
    render_fps_cap: int = 60
    step_ms: int = 120
    step_ms_delta: int = 5

    def clamp(self) -> None:
        self.render_fps_cap = max(1, int(self.render_fps_cap))
        # step_ms can be negative (uncapped), otherwise >=0
        if self.step_ms >= 0:
            self.step_ms = int(self.step_ms)

    def is_uncapped(self) -> bool:
        return int(self.step_ms) < 0

    def is_frame_locked(self) -> bool:
        return int(self.step_ms) == 0

    def interval_ms(self) -> Optional[int]:
        """
        Returns:
          None => frame-locked (1 step per render frame)
          0    => uncapped
          >0   => fixed ms interval
        """
        sm = int(self.step_ms)
        if sm < 0:
            return 0
        if sm == 0:
            return None
        return max(1, sm)

    def target_sps(self) -> Optional[float]:
        """
        Returns:
          None => uncapped (no target)
          float => desired steps/sec
        """
        sm = int(self.step_ms)
        if sm < 0:
            return None
        if sm == 0:
            return float(max(1, int(self.render_fps_cap)))
        return 1000.0 / float(max(1, sm))

    def handle_slower(self) -> None:
        # '['
        if self.is_uncapped():
            # exit uncapped -> frame-locked first
            self.step_ms = 0
            return
        # if frame-locked or ms-mode: make slower by adding ms
        if self.is_frame_locked():
            # frame-locked slower => go to small ms-mode
            self.step_ms = max(1, self.step_ms_delta)
        else:
            self.step_ms = int(self.step_ms) + int(self.step_ms_delta)

    def handle_faster(self) -> None:
        # ']'
        if self.is_uncapped():
            # already fastest
            return
        if self.is_frame_locked():
            # already at frame-locked
            return
        # ms-mode: reduce towards 0
        self.step_ms = max(0, int(self.step_ms) - int(self.step_ms_delta))

    def handle_max(self) -> None:
        # '='
        self.step_ms = -1

    def label(self) -> str:
        sm = int(self.step_ms)
        if sm < 0:
            return "max"
        if sm == 0:
            return f"{max(1, int(self.render_fps_cap))} (frame)"
        return f"{1000.0 / float(max(1, sm)):.1f} ({sm}ms)"


class RateMeter:
    """
    Sliding-window rate meter for events (steps or frames).
    """
    def __init__(self, *, window: int = 60) -> None:
        self._t: Deque[float] = deque(maxlen=max(2, int(window)))

    def tick(self, now_s: Optional[float] = None) -> None:
        self._t.append(time.perf_counter() if now_s is None else float(now_s))

    def rate_hz(self) -> float:
        if len(self._t) < 2:
            return 0.0
        dt = self._t[-1] - self._t[0]
        if dt <= 1e-12:
            return 0.0
        return float(len(self._t) - 1) / dt

    def reset(self) -> None:
        self._t.clear()

    def trim_to_last(self, n: int = 2) -> None:
        keep = max(0, int(n))
        if keep <= 0 or len(self._t) <= keep:
            return
        last = list(self._t)[-keep:]
        self._t.clear()
        self._t.extend(last)
