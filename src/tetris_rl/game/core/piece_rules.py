# src/tetris_rl/game/core/piece_rules.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tetris_rl.game.core.constants import CLASSIC_NUM_PIECES


class PieceRule(ABC):
    """
    Piece selection rule interface.

    Lifecycle:
      - reset(rng=..., kinds=...) is called once per episode/reset
      - next_piece(...) is called whenever the engine needs a new preview piece

    Notes:
      - The RNG is env-owned and injected (Gymnasium-style).
      - Rules may be stateful (store rng/kinds) but should not create their own RNG streams.
    """

    @abstractmethod
    def reset(self, *, rng: np.random.Generator, kinds: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def next_piece(self, *, locked_kind: str | None, preview_kind: str | None) -> str:
        raise NotImplementedError


@dataclass
class UniformPieceRule(PieceRule):
    """
    Uniform piece selection from the available kinds, driven by an injected RNG.

    This is the default training-friendly rule.
    """

    _rng: np.random.Generator | None = None
    _kinds: tuple[str, ...] = ()

    def reset(self, *, rng: np.random.Generator, kinds: Sequence[str]) -> None:
        self._rng = rng
        self._kinds = tuple(str(k) for k in kinds)
        if not self._kinds:
            raise ValueError("UniformPieceRule requires non-empty kinds")

    def next_piece(self, *, locked_kind: str | None, preview_kind: str | None) -> str:
        if self._rng is None or not self._kinds:
            raise RuntimeError("UniformPieceRule.reset() must be called before next_piece()")
        i = int(self._rng.integers(0, len(self._kinds)))
        return self._kinds[i]


@dataclass
class GameBoyOrPieceRule(PieceRule):
    """
    Game Boy-style accept/reject rule (Hard Drop description).

    It repeatedly samples a candidate piece and rejects it if:
        (locked_idx | preview_idx | candidate_idx) == 7

    This rule is defined for the classic 7 tetromino set only.
    """

    _rng: np.random.Generator | None = None
    _kinds: tuple[str, ...] = ()
    _kind_to_idx: dict[str, int] | None = None

    def reset(self, *, rng: np.random.Generator, kinds: Sequence[str]) -> None:
        self._rng = rng
        self._kinds = tuple(str(k) for k in kinds)
        if not self._kinds:
            raise ValueError("GameBoyOrPieceRule requires non-empty kinds")

        if len(self._kinds) != int(CLASSIC_NUM_PIECES):
            raise ValueError(
                "GameBoyOrPieceRule requires the classic 7 tetromino set "
                f"(got {len(self._kinds)} kinds). Use UniformPieceRule for custom PieceSets."
            )

        self._kind_to_idx = {k: i for i, k in enumerate(self._kinds)}

    def _idx(self, kind: str | None) -> int:
        if kind is None or self._kind_to_idx is None:
            raise RuntimeError("GameBoyOrPieceRule: kind mapping not initialized")
        v = self._kind_to_idx.get(str(kind))
        if v is None:
            raise KeyError(
                f"unknown kind {kind!r} for GameBoyOrPieceRule (kinds={list(self._kinds)!r})"
            )
        return int(v)

    def next_piece(self, *, locked_kind: str | None, preview_kind: str | None) -> str:
        if self._rng is None or not self._kinds or self._kind_to_idx is None:
            raise RuntimeError("GameBoyOrPieceRule.reset() must be called before next_piece()")

        li = self._idx(locked_kind)
        pi = self._idx(preview_kind)

        gb_or_reject_value = 7

        while True:
            ci = int(self._rng.integers(0, int(CLASSIC_NUM_PIECES)))
            if (li | pi | ci) == gb_or_reject_value:
                continue
            return self._kinds[ci]


@dataclass
class BagPieceRule(PieceRule):
    """
    K-bag randomizer (generalization of 7-bag).

    Parameters:
      - bag_copies: how many copies of each kind are placed into a bag before shuffling.
          * bag_copies=1 -> classic 7-bag for tetrominoes.
          * bag_copies>1 -> larger bag: N copies of each piece per bag.

    Notes:
      - Ignores locked_kind/preview_kind; randomness is bag-state-only.
      - Deterministic w.r.t. injected RNG.
    """

    bag_copies: int = 1

    _rng: np.random.Generator | None = None
    _kinds: tuple[str, ...] = ()
    _bag: list[str] = None  # type: ignore[assignment]

    def reset(self, *, rng: np.random.Generator, kinds: Sequence[str]) -> None:
        self._rng = rng
        self._kinds = tuple(str(k) for k in kinds)
        if not self._kinds:
            raise ValueError("BagPieceRule requires non-empty kinds")
        if int(self.bag_copies) <= 0:
            raise ValueError(f"BagPieceRule.bag_copies must be >= 1 (got {self.bag_copies})")
        self._bag = []
        self._refill()

    def _refill(self) -> None:
        if self._rng is None or not self._kinds:
            raise RuntimeError("BagPieceRule.reset() must be called before _refill()")
        # Build bag with N copies of each kind, then shuffle in-place.
        self._bag = [k for k in self._kinds for _ in range(int(self.bag_copies))]
        self._rng.shuffle(self._bag)

    def next_piece(self, *, locked_kind: str | None, preview_kind: str | None) -> str:
        if self._rng is None or not self._kinds:
            raise RuntimeError("BagPieceRule.reset() must be called before next_piece()")
        if not self._bag:
            self._refill()
        # Pop from end (cheaper than pop(0))
        return self._bag.pop()
