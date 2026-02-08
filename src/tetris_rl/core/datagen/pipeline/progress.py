# src/tetris_rl/core/datagen/pipeline/progress.py
from __future__ import annotations

import warnings
from threading import Event, Thread
from typing import Any, Dict, Optional, Tuple

# NOTE:
# This module must be Ctrl+C-safe on Windows with ProcessPoolExecutor(spawn).
# Therefore we must NOT use multiprocessing.Manager proxies for the progress queue.
# Use a spawn-context mp.Queue (or SimpleQueue) owned by the parent process instead.

# -----------------------------------------------------------------------------
# Renderer choice
# -----------------------------------------------------------------------------
# We want the nice Rich bar style, but also want reliable descriptions.
# tqdm.rich is experimental and emits warnings; we suppress those warnings here.
try:
    from tqdm.rich import tqdm as _tqdm  # type: ignore

    try:
        from tqdm.rich import TqdmExperimentalWarning  # type: ignore

        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    except Exception:
        pass
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm as _tqdm  # type: ignore

# -----------------------------------------------------------------------------
# Progress tuning
# -----------------------------------------------------------------------------

# Parent progress-consumer queue poll timeout / drain cadence (seconds).
PROGRESS_DRAIN_POLL_S: float = 0.5

# Force tqdm to refresh aggressively (otherwise it will batch redraws and "jump").
# If this becomes too expensive, raise MININTERVAL (e.g. 0.05) first.
TQDM_MININTERVAL_S: float = 0.0
TQDM_MAXINTERVAL_S: float = 0.0
TQDM_MINITERS: int = 1

# Queue maxsize: bounded so producers never block forever on shutdown.
# We drop progress events if full (best-effort UI only).
PROGRESS_QUEUE_MAXSIZE: int = 10_000


def _desc(slot_id: int, sid: Optional[int]) -> str:
    if sid is None:
        return f"[w{int(slot_id)}] idle"
    return f"[w{int(slot_id)}] shard_{int(sid):04d}"


def _set_desc(bar: Any, text: str) -> None:
    """
    tqdm.rich is a bit finicky; try the variants it supports.
    """
    try:
        bar.set_description_str(str(text))
        return
    except Exception:
        pass
    try:
        bar.set_description(str(text))
        return
    except Exception:
        pass
    try:
        bar.desc = str(text)
    except Exception:
        pass


def _force_refresh(bar: Any) -> None:
    try:
        bar.refresh()
    except Exception:
        pass


def _ensure_slot_started(
        *,
        slot_id: int,
        sid: int,
        total: int,
        slot_bars: Dict[int, Any],
        slot_state: Dict[int, Tuple[int, int, int]],  # slot -> (sid, done, total)
        refresh: bool,
) -> None:
    slot_id = int(slot_id)
    sid = int(sid)
    total = int(total)

    prev = slot_state.get(slot_id)
    if prev is not None and int(prev[0]) == sid and int(prev[2]) == total:
        return

    slot_state[slot_id] = (sid, 0, total)
    bar = slot_bars.get(slot_id)
    if bar is None:
        return

    # reset progress for this slot
    try:
        bar.reset(total=total)
    except Exception:
        try:
            bar.total = total
            bar.n = 0
        except Exception:
            pass

    _set_desc(bar, _desc(slot_id, sid))
    if refresh:
        _force_refresh(bar)


def _handle_event(
        *,
        ev: Any,
        shard_bar: Any,
        slot_bars: Dict[int, Any],
        slot_state: Dict[int, Tuple[int, int, int]],
        refresh: bool,
) -> None:
    if not ev:
        return

    kind = ev[0]

    if kind == "start":
        _, slot_id, sid, total = ev
        _ensure_slot_started(
            slot_id=int(slot_id),
            sid=int(sid),
            total=int(total),
            slot_bars=slot_bars,
            slot_state=slot_state,
            refresh=refresh,
        )
        return

    if kind == "progress":
        _, slot_id, sid, done, total = ev
        slot_id = int(slot_id)
        sid = int(sid)
        done = int(done)
        total = int(total)

        # robust even if "start" is missed
        _ensure_slot_started(
            slot_id=slot_id,
            sid=sid,
            total=total,
            slot_bars=slot_bars,
            slot_state=slot_state,
            refresh=refresh,
        )

        prev = slot_state.get(slot_id)
        prev_done = int(prev[1]) if prev is not None and int(prev[0]) == sid else 0
        delta = max(0, done - prev_done)
        slot_state[slot_id] = (sid, done, total)

        bar = slot_bars.get(slot_id)
        if bar is not None and delta:
            bar.update(int(delta))
            if refresh:
                _force_refresh(bar)
        return

    if kind == "done":
        _, slot_id, sid, total = ev
        slot_id = int(slot_id)
        sid = int(sid)
        total = int(total)

        _ensure_slot_started(
            slot_id=slot_id,
            sid=sid,
            total=total,
            slot_bars=slot_bars,
            slot_state=slot_state,
            refresh=refresh,
        )

        prev = slot_state.get(slot_id)
        prev_done = int(prev[1]) if prev is not None and int(prev[0]) == sid else 0
        delta = max(0, total - prev_done)
        slot_state[slot_id] = (sid, total, total)

        bar = slot_bars.get(slot_id)
        if bar is not None and delta:
            bar.update(int(delta))

        # back to idle
        if bar is not None:
            _set_desc(bar, _desc(slot_id, None))
            if refresh:
                _force_refresh(bar)

        shard_bar.update(1)
        if refresh:
            _force_refresh(shard_bar)
        return

    # unknown kind -> ignore


def _consume_loop(
        *,
        q: Any,
        stop_evt: Event,
        shard_bar: Any,
        slot_bars: Dict[int, Any],
        slot_state: Dict[int, Tuple[int, int, int]],
        poll_s: float,
) -> None:
    timeout = float(max(0.001, poll_s))

    while not stop_evt.is_set():
        try:
            ev = q.get(timeout=timeout)
        except Exception:
            continue

        # sentinel to wake up / exit cleanly
        if ev is None:
            break

        try:
            _handle_event(
                ev=ev,
                shard_bar=shard_bar,
                slot_bars=slot_bars,
                slot_state=slot_state,
                refresh=True,
            )
        except Exception:
            continue

    # best-effort final refresh
    _force_refresh(shard_bar)
    for b in slot_bars.values():
        _force_refresh(b)


def _drain_queue(
        *,
        q: Any,
        shard_bar: Any,
        slot_bars: Dict[int, Any],
        slot_state: Dict[int, Tuple[int, int, int]],
        poll_s: float,
        max_iters: int = 200,
) -> None:
    timeout = float(max(0.001, poll_s))

    for _ in range(int(max_iters)):
        try:
            ev = q.get(timeout=timeout)
        except Exception:
            return

        if ev is None:
            return

        try:
            _handle_event(
                ev=ev,
                shard_bar=shard_bar,
                slot_bars=slot_bars,
                slot_state=slot_state,
                refresh=False,
            )
        except Exception:
            continue


class MultiWorkerProgress:
    """
    Multi-line progress UI:

      line 0: shard completion (0..num_shards)
      line 1..N: per-worker-slot intra-shard sample progress

    Contract:
      - self.queue must be pickleable for ProcessPoolExecutor(spawn) workers.
      - Must not rely on multiprocessing.Manager proxies (breaks Ctrl+C on Windows).
    """

    def __init__(
            self,
            *,
            total_shards: int,
            shard_steps: int,
            num_slots: int,
            already_done: int = 0,
            poll_s: float = PROGRESS_DRAIN_POLL_S,
            queue_maxsize: int = PROGRESS_QUEUE_MAXSIZE,
    ) -> None:
        self.total_shards = int(total_shards)
        self.shard_steps = int(shard_steps)
        self.num_slots = max(1, int(num_slots))
        self.already_done = max(0, int(already_done))
        self.poll_s = float(poll_s)
        self.queue_maxsize = int(queue_maxsize)

        self._q: Any = None
        self._stop_evt = Event()
        self._thread: Optional[Thread] = None

        self._shard_bar: Any = None
        self._slot_bars: Dict[int, Any] = {}
        self._slot_state: Dict[int, Tuple[int, int, int]] = {}

    @property
    def queue(self) -> Any:
        if self._q is None:
            raise RuntimeError("progress queue not initialized (did you enter the context?)")
        return self._q

    def __enter__(self) -> "MultiWorkerProgress":
        import multiprocessing as mp

        ctx = mp.get_context("spawn")

        # IMPORTANT: mp.Queue is NOT a manager proxy; it's safe under Ctrl+C.
        # Bounded queue: UI only; we prefer dropping events to blocking workers.
        self._q = ctx.Queue(maxsize=max(1, int(self.queue_maxsize)))

        self._shard_bar = _tqdm(
            total=self.total_shards,
            desc="[datagen] shards",
            unit="shard",
            position=0,
            leave=True,
            mininterval=float(TQDM_MININTERVAL_S),
            maxinterval=float(TQDM_MAXINTERVAL_S),
            miniters=int(TQDM_MINITERS),
        )
        if self.already_done:
            try:
                self._shard_bar.update(int(self.already_done))
            except Exception:
                pass

        self._slot_bars = {}
        for slot_id in range(self.num_slots):
            b = _tqdm(
                total=self.shard_steps,
                desc=_desc(int(slot_id), None),
                unit="samp",
                position=1 + int(slot_id),
                leave=True,
                mininterval=float(TQDM_MININTERVAL_S),
                maxinterval=float(TQDM_MAXINTERVAL_S),
                miniters=int(TQDM_MINITERS),
            )
            self._slot_bars[int(slot_id)] = b

        self._slot_state = {}

        self._thread = Thread(
            target=_consume_loop,
            kwargs=dict(
                q=self._q,
                stop_evt=self._stop_evt,
                shard_bar=self._shard_bar,
                slot_bars=self._slot_bars,
                slot_state=self._slot_state,
                poll_s=float(self.poll_s),
            ),
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Signal consumer to stop and wake it up even if blocked in q.get(timeout=...)
        self._stop_evt.set()
        try:
            if self._q is not None:
                try:
                    self._q.put_nowait(None)  # sentinel
                except Exception:
                    pass
        except Exception:
            pass

        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass

        # Drain remaining events without refreshing to avoid flicker
        try:
            if self._q is not None:
                _drain_queue(
                    q=self._q,
                    shard_bar=self._shard_bar,
                    slot_bars=self._slot_bars,
                    slot_state=self._slot_state,
                    poll_s=float(self.poll_s),
                )
        except Exception:
            pass

        try:
            if self._shard_bar is not None:
                self._shard_bar.close()
        except Exception:
            pass

        for b in list(self._slot_bars.values()):
            try:
                b.close()
            except Exception:
                pass

        # Best-effort mp.Queue cleanup (prevents join-thread hangs on Windows)
        try:
            if self._q is not None:
                try:
                    self._q.close()
                except Exception:
                    pass
                try:
                    # Avoid atexit join deadlocks if workers are mid-put when parent dies.
                    self._q.cancel_join_thread()
                except Exception:
                    pass
        except Exception:
            pass
