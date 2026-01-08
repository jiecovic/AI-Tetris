# src/tetris_rl/metrics/stats_accumulator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _as_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _as_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    try:
        return bool(x)
    except Exception:
        return None


@dataclass
class StatsAccumulatorConfig:
    log_action_histograms: bool = False
    hist_max_samples: int = 4096
    log_total_means: bool = False


class StatsAccumulator:
    """
    Shared per-step stats accumulator.

    Contract (strict):
      info["tf"]   : Mapping with per-step transition features
      info["game"] : Mapping with per-step KPIs
      info["ui"]   : Optional Mapping (used only for action histograms)

    This class is intentionally dumb and stable:
      - ingests infos
      - accumulates sums/counts/rates/max
      - emits flattened metric keys (tf/*, game/*)
    """

    def __init__(self, *, cfg: Optional[StatsAccumulatorConfig] = None) -> None:
        self.cfg = cfg or StatsAccumulatorConfig()

        self._n_steps: int = 0

        # --- tf means ---
        self._sum_cleared_lines = 0.0
        self._cnt_cleared_lines = 0

        self._sum_delta_holes = 0.0
        self._cnt_delta_holes = 0

        self._sum_delta_max_height = 0.0
        self._cnt_delta_max_height = 0

        self._sum_delta_bumpiness = 0.0
        self._cnt_delta_bumpiness = 0

        self._sum_delta_agg_height = 0.0
        self._cnt_delta_agg_height = 0

        self._sum_placed_cells_cleared = 0.0
        self._cnt_placed_cells_cleared = 0

        # --- tf absolutes means ---
        self._sum_holes = 0.0
        self._cnt_holes = 0

        self._sum_max_height = 0.0
        self._cnt_max_height = 0

        self._sum_bumpiness = 0.0
        self._cnt_bumpiness = 0

        self._sum_agg_height = 0.0
        self._cnt_agg_height = 0

        # --- tf rates ---
        self._sum_all_cells_cleared = 0.0
        self._cnt_all_cells_cleared = 0

        self._sum_invalid_action = 0.0
        self._cnt_invalid_action = 0

        self._sum_redundant_rotation = 0.0
        self._cnt_redundant_rotation = 0

        self._sum_game_over = 0.0
        self._cnt_game_over = 0

        # --- game means/max ---
        self._sum_score = 0.0
        self._cnt_score = 0

        self._sum_delta_score = 0.0
        self._cnt_delta_score = 0

        self._sum_level = 0.0
        self._cnt_level = 0
        self._max_level: Optional[float] = None

        self._sum_lines_total = 0.0
        self._cnt_lines_total = 0

        # derived
        self._sum_lines_for_rate = 0.0

        # --- optional action histograms (debug-only) ---
        self._log_hists = bool(self.cfg.log_action_histograms)
        self._hist_max = max(0, int(self.cfg.hist_max_samples))
        self._req_rot: list[int] = []
        self._req_col: list[int] = []
        self._used_rot: list[int] = []
        self._used_col: list[int] = []

    def reset(self) -> None:
        self.__init__(cfg=self.cfg)

    @property
    def steps(self) -> int:
        return int(self._n_steps)

    def _cap_append(self, buf: list[int], v: Optional[int]) -> None:
        if not self._log_hists or self._hist_max <= 0 or v is None:
            return
        if len(buf) >= self._hist_max:
            return
        buf.append(int(v))

    def ingest_info(self, info: Mapping[str, Any]) -> None:
        self._n_steps += 1

        tf = info.get("tf")
        game = info.get("game")
        ui = info.get("ui")

        if not isinstance(tf, Mapping) or not isinstance(game, Mapping):
            # strict contract: ignore malformed infos
            return

        # --- tf means ---
        v = _as_float(tf.get("cleared_lines"))
        if v is not None:
            self._sum_cleared_lines += float(v)
            self._cnt_cleared_lines += 1
            self._sum_lines_for_rate += float(v)

        v = _as_float(tf.get("delta_holes"))
        if v is not None:
            self._sum_delta_holes += float(v)
            self._cnt_delta_holes += 1

        v = _as_float(tf.get("delta_max_height"))
        if v is not None:
            self._sum_delta_max_height += float(v)
            self._cnt_delta_max_height += 1

        v = _as_float(tf.get("delta_bumpiness"))
        if v is not None:
            self._sum_delta_bumpiness += float(v)
            self._cnt_delta_bumpiness += 1

        v = _as_float(tf.get("delta_agg_height"))
        if v is not None:
            self._sum_delta_agg_height += float(v)
            self._cnt_delta_agg_height += 1

        v = _as_float(tf.get("placed_cells_cleared"))
        if v is not None:
            self._sum_placed_cells_cleared += float(v)
            self._cnt_placed_cells_cleared += 1

        # --- tf absolutes means ---
        v = _as_float(tf.get("holes"))
        if v is not None:
            self._sum_holes += float(v)
            self._cnt_holes += 1

        v = _as_float(tf.get("max_height"))
        if v is not None:
            self._sum_max_height += float(v)
            self._cnt_max_height += 1

        v = _as_float(tf.get("bumpiness"))
        if v is not None:
            self._sum_bumpiness += float(v)
            self._cnt_bumpiness += 1

        v = _as_float(tf.get("agg_height"))
        if v is not None:
            self._sum_agg_height += float(v)
            self._cnt_agg_height += 1

        # --- tf rates ---
        b = _as_bool(tf.get("placed_all_cells_cleared"))
        if b is not None:
            self._sum_all_cells_cleared += 1.0 if b else 0.0
            self._cnt_all_cells_cleared += 1

        b = _as_bool(tf.get("invalid_action"))
        if b is not None:
            self._sum_invalid_action += 1.0 if b else 0.0
            self._cnt_invalid_action += 1

        b = _as_bool(tf.get("redundant_rotation"))
        if b is not None:
            self._sum_redundant_rotation += 1.0 if b else 0.0
            self._cnt_redundant_rotation += 1

        b = _as_bool(tf.get("game_over"))
        if b is not None:
            self._sum_game_over += 1.0 if b else 0.0
            self._cnt_game_over += 1

        # --- game means/max ---
        v = _as_float(game.get("score"))
        if v is not None:
            self._sum_score += float(v)
            self._cnt_score += 1

        v = _as_float(game.get("delta_score"))
        if v is not None:
            self._sum_delta_score += float(v)
            self._cnt_delta_score += 1

        v = _as_float(game.get("level"))
        if v is not None:
            self._sum_level += float(v)
            self._cnt_level += 1
            if self._max_level is None or float(v) > float(self._max_level):
                self._max_level = float(v)

        v = _as_float(game.get("lines_total"))
        if v is not None:
            self._sum_lines_total += float(v)
            self._cnt_lines_total += 1

        # --- optional debug histograms (ui only) ---
        if self._log_hists and isinstance(ui, Mapping):
            self._cap_append(self._req_rot, _as_int(ui.get("requested_rotation")))
            self._cap_append(self._req_col, _as_int(ui.get("requested_column")))
            self._cap_append(self._used_rot, _as_int(ui.get("used_rotation")))
            self._cap_append(self._used_col, _as_int(ui.get("used_column")))

    def ingest_infos(self, infos: Iterable[Mapping[str, Any]]) -> None:
        for info in infos:
            if isinstance(info, Mapping):
                self.ingest_info(info)

    def _mean(self, s: float, n: int) -> Optional[float]:
        if n <= 0:
            return None
        return float(s / float(n))

    def _rate(self, s: float, n: int) -> Optional[float]:
        if n <= 0:
            return None
        return float(s / float(n))

    def summarize(self) -> Dict[str, float]:
        out: Dict[str, float] = {}

        # tf means
        m = self._mean(self._sum_cleared_lines, self._cnt_cleared_lines)
        if m is not None:
            out["tf/cleared_lines_mean"] = m

        m = self._mean(self._sum_delta_holes, self._cnt_delta_holes)
        if m is not None:
            out["tf/delta_holes_mean"] = m

        m = self._mean(self._sum_delta_max_height, self._cnt_delta_max_height)
        if m is not None:
            out["tf/delta_max_height_mean"] = m

        m = self._mean(self._sum_delta_bumpiness, self._cnt_delta_bumpiness)
        if m is not None:
            out["tf/delta_bumpiness_mean"] = m

        m = self._mean(self._sum_delta_agg_height, self._cnt_delta_agg_height)
        if m is not None:
            out["tf/delta_agg_height_mean"] = m

        m = self._mean(self._sum_placed_cells_cleared, self._cnt_placed_cells_cleared)
        if m is not None:
            out["tf/placed_cells_cleared_mean"] = m

        # tf rates
        r = self._rate(self._sum_all_cells_cleared, self._cnt_all_cells_cleared)
        if r is not None:
            out["tf/placed_all_cells_cleared_rate"] = r

        r = self._rate(self._sum_invalid_action, self._cnt_invalid_action)
        if r is not None:
            out["tf/invalid_action_rate"] = r

        r = self._rate(self._sum_redundant_rotation, self._cnt_redundant_rotation)
        if r is not None:
            out["tf/redundant_rotation_rate"] = r

        r = self._rate(self._sum_game_over, self._cnt_game_over)
        if r is not None:
            out["tf/game_over_rate"] = r

        # tf absolutes
        m = self._mean(self._sum_holes, self._cnt_holes)
        if m is not None:
            out["tf/holes_mean"] = m
        m = self._mean(self._sum_max_height, self._cnt_max_height)
        if m is not None:
            out["tf/max_height_mean"] = m
        m = self._mean(self._sum_bumpiness, self._cnt_bumpiness)
        if m is not None:
            out["tf/bumpiness_mean"] = m
        m = self._mean(self._sum_agg_height, self._cnt_agg_height)
        if m is not None:
            out["tf/agg_height_mean"] = m

        # game means/max
        m = self._mean(self._sum_score, self._cnt_score)
        if m is not None:
            out["game/score_mean"] = m

        m = self._mean(self._sum_delta_score, self._cnt_delta_score)
        if m is not None:
            out["game/delta_score_mean"] = m
            if self._n_steps > 0:
                out["game/score_per_step"] = float(self._sum_delta_score / float(self._n_steps))

        m = self._mean(self._sum_level, self._cnt_level)
        if m is not None:
            out["game/level_mean"] = m
        if self._max_level is not None:
            out["game/level_max"] = float(self._max_level)

        if self._n_steps > 0:
            out["game/lines_per_step"] = float(self._sum_lines_for_rate / float(self._n_steps))

        if bool(self.cfg.log_total_means):
            m = self._mean(self._sum_lines_total, self._cnt_lines_total)
            if m is not None:
                out["game/lines_total_mean"] = m

        return out

    def histograms(self) -> Dict[str, np.ndarray]:
        if not self._log_hists:
            return {}
        out: Dict[str, np.ndarray] = {}
        if self._req_rot:
            out["actions/requested_rotation"] = np.asarray(self._req_rot, dtype=np.int64)
        if self._req_col:
            out["actions/requested_column"] = np.asarray(self._req_col, dtype=np.int64)
        if self._used_rot:
            out["actions/used_rotation"] = np.asarray(self._used_rot, dtype=np.int64)
        if self._used_col:
            out["actions/used_column"] = np.asarray(self._used_col, dtype=np.int64)
        return out
