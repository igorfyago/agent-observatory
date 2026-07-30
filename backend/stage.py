"""One shared stage: every run plays to every viewer.

The old model was request/response: a question came in, and the SSE stream
went back on that one request. This module inverts it. The observatory runs
itself (backend/autopilot.py picks the scenarios) and every open browser
watches the same performance over GET /api/live. A visitor can take the stage
at any time: their question preempts an autopilot run, never another
visitor's.

Why one stage and not a run per viewer: the box runs on one personal API key
behind backend/spend.py. A shared stage means N viewers cost the same as one,
and the room going quiet costs nothing at all (the autopilot only performs
for a non-empty room).

Everything here runs on the event loop; there is no cross-thread traffic, so
plain attributes are enough and no locks are needed.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from datetime import datetime, timezone

import registry
import runner
import spend
import store

RUN_TIMEOUT_S = 600          # a wedged model call must not hold the stage
TRACE_TRIES = 4              # LangSmith ingestion lags a run by a few seconds


class StageBusy(Exception):
    """A visitor run is on stage; the new submission has to wait."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NothingPending(Exception):
    """A resume arrived but no interrupt is waiting for one."""


# --------------------------------------------------------------- the hub ----

_subscribers: dict[int, asyncio.Queue] = {}
_next_sub_id = 0

# The run currently on stage, or None. While set it accumulates every frame,
# so a viewer who arrives mid-run gets the whole story instantly.
current: dict | None = None

# An interrupt waiting for a decision: {agent, thread, source, auto_at, payload}.
pending_interrupt: dict | None = None

# Set by the guard refusing a run; the autopilot reads it and backs off.
blocked_until: float = 0.0

_task: asyncio.Task | None = None
_last_run_end: float = 0.0


def viewers() -> int:
    return len(_subscribers)


def busy() -> bool:
    return _task is not None and not _task.done()


def last_run_end() -> float:
    return _last_run_end


def publish(frame: dict) -> None:
    """Fan one frame out to every viewer (and into the current run's record).

    Status traffic and late trace links stay out of the record: a replayed
    archive should contain exactly the run, nothing that happened around it.
    """
    if current is not None and frame.get("type") not in ("autopilot", "stats", "trace"):
        current["frames"].append(frame)
    for q in list(_subscribers.values()):
        q.put_nowait(frame)


def subscribe() -> tuple[int, asyncio.Queue]:
    """Register a viewer. The caller owns unsubscribe(sub_id)."""
    global _next_sub_id
    _next_sub_id += 1
    q: asyncio.Queue = asyncio.Queue()
    _subscribers[_next_sub_id] = q
    return _next_sub_id, q


def unsubscribe(sub_id: int) -> None:
    _subscribers.pop(sub_id, None)


def hello() -> dict:
    """The first frame every viewer receives: enough to draw the whole page."""
    return {
        "type": "hello",
        "stage": {k: v for k, v in current.items() if k != "frames"} if current else None,
        "pending_interrupt": pending_interrupt,
        "stats": store.stage_stats(),
        "spend": spend.status(),
        "mode": runner.mode(),
    }


def snapshot_frames() -> list[dict]:
    """The current run so far, for a viewer who arrived mid-performance."""
    return list(current["frames"]) if current else []


# ------------------------------------------------------------ submitting ----

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def submit(agent_id: str, question: str, source: str, ip: str,
                 title: str = "", resume: dict | None = None,
                 thread: str | None = None,
                 checkpoint_id: str | None = None) -> dict:
    """Put a run on stage. Returns its meta, or raises StageBusy.

    Preemption rule, the whole social contract of the stage: a visitor
    preempts the autopilot, nobody preempts a visitor.
    """
    global _task, pending_interrupt, current

    agent = registry.get(agent_id)
    if agent is None or agent.kind != "text":
        raise ValueError(f"no text agent '{agent_id}'")

    if resume is not None:
        if pending_interrupt is None or pending_interrupt["thread"] != thread:
            raise NothingPending()
        pending_interrupt = None

    if busy():
        if current and current.get("source") == "autopilot" and source != "autopilot":
            _task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(_task, 15)
        else:
            raise StageBusy("a visitor's run is on stage right now, give it a moment")

    meta = {
        "run_id": uuid.uuid4().hex[:10],
        "agent_id": agent.id,
        "source": source,
        "title": title,
        "question": question if resume is None else f"decision: {resume.get('action', '')}",
        "started_at": _now_iso(),
        "thread_id": thread or "",
    }
    # The stage is taken the moment submit returns, not when the task gets
    # its first slice of the loop: viewers and busy() must never see a gap.
    current = {**meta, "frames": []}
    _task = asyncio.get_running_loop().create_task(
        _perform(agent, meta, question, ip, resume, thread, checkpoint_id))
    return meta


async def wait_idle(timeout: float | None = None) -> None:
    """Block until the stage is free. The autopilot paces itself with this."""
    if _task is not None and not _task.done():
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(_task), timeout)


async def _perform(agent, meta: dict, question: str, ip: str,
                   resume: dict | None, thread: str | None,
                   checkpoint_id: str | None) -> None:
    """Run one performance start to finish, then leave the stage clean.

    All bookkeeping lives in the finally block, so a cancelled (preempted)
    run still archives what it managed to do and still frees the stage.
    """
    global current, pending_interrupt, blocked_until, _last_run_end

    t0 = time.perf_counter()
    outcome = "ok"
    tokens_in = tokens_out = tool_calls = 0
    cost = 0.0
    thread_seen = thread or ""
    interrupted = False

    publish({"type": "stage", "state": "running",
             **{k: v for k, v in meta.items()}})

    async def consume() -> None:
        nonlocal outcome, tokens_in, tokens_out, tool_calls, cost
        nonlocal thread_seen, interrupted
        async for frame in runner.stream(agent, question, thread=thread,
                                         resume=resume,
                                         checkpoint_id=checkpoint_id):
            kind = frame.get("type")
            if kind == "run_start":
                thread_seen = frame.get("thread", thread_seen)
            elif kind == "usage":
                tokens_in += int(frame.get("input_tokens") or 0)
                tokens_out += int(frame.get("output_tokens") or 0)
                cost += float(frame.get("cost_usd") or 0.0)
            elif kind == "tool_end":
                tool_calls += 1
            elif kind == "interrupt":
                interrupted = True
                outcome = "interrupt"
                # Autopilot runs approve themselves after a visible pause;
                # the frame carries the deadline so every viewer sees the
                # same countdown and can beat it with a click.
                if meta["source"] == "autopilot":
                    frame = {**frame,
                             "auto_approve_at": time.time() + approve_delay_s()}
                pending_interrupt_set(agent.id, frame)
            elif kind == "error":
                outcome = "error"
            publish(frame)

    try:
        with spend.guard(ip):
            await asyncio.wait_for(consume(), RUN_TIMEOUT_S)
    except spend.SpendRefused as refusal:
        outcome = "refused"
        blocked_until = time.time() + (refusal.retry_after or 300)
        publish({"type": "error", "message": refusal.message, **refusal.as_dict()})
    except asyncio.TimeoutError:
        outcome = "error"
        publish({"type": "error",
                 "message": f"run exceeded {RUN_TIMEOUT_S}s and was stopped"})
    except asyncio.CancelledError:
        outcome = "preempted"
        publish({"type": "run_end", "agent": agent.id, "thread": thread_seen,
                 "answer": "", "note": "a visitor took the stage"})
        raise
    except Exception as exc:
        outcome = "error"
        publish({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        meta_final = {**meta, "outcome": outcome, "latency_ms": latency_ms,
                      "tokens_in": tokens_in, "tokens_out": tokens_out,
                      "cost_usd": round(cost, 6), "tool_calls": tool_calls,
                      "thread_id": thread_seen}
        frames = current["frames"] if current else []
        current = None
        _last_run_end = time.time()
        try:
            if outcome != "refused":
                run_id = store.save_stage_run(meta_final, frames)
                store.log_run(agent.id, "text", meta["question"], tool_calls,
                              tokens_in + tokens_out, latency_ms,
                              "ok" if outcome in ("ok", "interrupt") else "error")
                root = next((f.get("root_run_id") for f in frames
                             if f.get("root_run_id")), "")
                if root and os.environ.get("LANGSMITH_API_KEY"):
                    asyncio.get_running_loop().create_task(
                        _fetch_trace_url(run_id, root))
        except Exception:
            pass  # archiving must never take the stage down
        publish({"type": "stage", "state": "idle", "outcome": outcome,
                 "stats": store.stage_stats(), "spend": spend.status()})


def pending_interrupt_set(agent_id: str, frame: dict) -> None:
    global pending_interrupt
    pending_interrupt = {
        "agent": agent_id,
        "thread": frame.get("thread", ""),
        "source": current["source"] if current else "visitor",
        "auto_approve_at": frame.get("auto_approve_at"),
        "payload": frame.get("payload"),
    }


def approve_delay_s() -> float:
    return float(os.environ.get("OBS_APPROVE_DELAY_S", "8"))


async def _fetch_trace_url(run_id: int, root_run_id: str) -> None:
    """Best effort: ask LangSmith for the run's URL once it has ingested it."""
    try:
        from langsmith import Client
    except ImportError:
        return
    for attempt in range(TRACE_TRIES):
        await asyncio.sleep(2 + attempt * 3)
        try:
            url = await asyncio.to_thread(
                lambda: Client().read_run(root_run_id).url)
        except Exception:
            continue
        if url:
            store.set_stage_trace(run_id, url)
            publish({"type": "trace", "run_id": run_id, "url": url})
            return
