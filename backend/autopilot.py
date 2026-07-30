"""The autopilot: the observatory performs without being asked.

Nobody should have to invent a question to see the agents think. As long as
at least one browser is watching, this loop keeps putting real runs on the
stage, cycling through scenarios chosen so that every LangGraph mechanism in
the registry gets its moment: the parallel fan-out, the tool loop repairing
its own SQL, the reflection loop, the regime router, the interrupt.

Cost discipline, in order:
  - an empty room costs nothing (no viewers, no runs, unless OBS_AUTOPILOT_
    EMPTY_ROOM=1)
  - every run still passes through backend/spend.py's caps like any visitor
  - a refusal from the guard backs the autopilot off for exactly as long as
    the guard asked

The loop never outranks people: it waits while a visitor holds the stage, and
stage.submit lets a visitor cancel an autopilot run mid-flight.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import time

import stage

# Each scenario names the LangGraph mechanism it exists to light up ("shows"),
# and the UI repeats that line in the ribbon, so a viewer always knows what
# they are looking at. Questions lean on the observatory's own live telemetry
# and source where possible: real questions over real data, every cycle.
SCENARIOS = [
    {
        "agent": "pipeline",
        "title": "five roles, one graph",
        "shows": "parallel fan-out, a join, and a critic loop on conditional edges",
        "question": ("Should a production agent team let a critic send work "
                     "back, or trust the first draft? Argue it concretely."),
    },
    {
        "agent": "sql",
        "title": "the tool loop repairs itself",
        "shows": "a create_agent tool loop reading SQL errors and fixing its own query",
        "question": ("Which agent on this observatory had the highest error "
                     "rate over the last 7 days, and how does it compare to "
                     "the busiest agent? Show the SQL."),
    },
    {
        "agent": "research",
        "title": "a loop inside a loop",
        "shows": "an inner tool loop nested in an outer reflection loop, both on conditional edges",
        "question": ("How busy has this observatory been in the last 24 hours "
                     "versus its 30-day average, and which agents drive the "
                     "difference? Numbers, not vibes."),
    },
    {
        "agent": "repo",
        "title": "the app explains its own source",
        "shows": "retrieval as a tool the model calls at will, with file citations",
        "question": ("Trace one token's journey from a LangGraph node to a lit "
                     "edge in the browser DAG. Name the files and functions on "
                     "the way."),
    },
    {
        "agent": "pipeline",
        "title": "five roles, one graph",
        "shows": "parallel fan-out, a join, and a critic loop on conditional edges",
        "question": ("A team is torn between one strong agent with many tools "
                     "and five narrow agents in a graph. What breaks first in "
                     "each design as the task grows?"),
    },
    {
        "agent": "brief",
        "title": "structured output, no prose",
        "shows": "with_structured_output forcing a Pydantic schema instead of free text",
        "question": "Is SPY pinned by dealers into Friday opex?",
    },
    {
        "agent": "sql",
        "title": "the tool loop repairs itself",
        "shows": "a create_agent tool loop reading SQL errors and fixing its own query",
        "question": ("Break down this observatory's runs per agent per day for "
                     "the last week. Which agent is getting more reliable and "
                     "which is not? Show the SQL."),
    },
    {
        "agent": "analyst",
        "title": "the flagship: router, specialists, gate, human",
        "shows": "a Command router, three parallel sub-agents, a critic gate and interrupt()",
        "question": "SPY",
    },
]


def enabled() -> bool:
    return os.environ.get("OBS_AUTOPILOT", "1") not in ("0", "false", "off")


def rest_s() -> float:
    """Quiet time between runs. Long enough to read an answer, short enough
    that the room never feels dead."""
    return float(os.environ.get("OBS_AUTOPILOT_REST_S", "25"))


def first_run_delay_s() -> float:
    """How fast the first run starts once someone is actually watching."""
    return float(os.environ.get("OBS_AUTOPILOT_FIRST_S", "3"))


def performs_to_empty_room() -> bool:
    return os.environ.get("OBS_AUTOPILOT_EMPTY_ROOM", "0") in ("1", "true", "on")


def runnable(scenario: dict, mode: dict) -> bool:
    """Demo mode (no key) can only stage the pipeline: it has scripted output.
    Everything else needs a real model behind it."""
    return mode.get("llm") != "demo" or scenario["agent"] == "pipeline"


_status: dict = {"state": "starting"}


def status() -> dict:
    return {**_status, "enabled": enabled(), "viewers": stage.viewers()}


def _announce(state: str, **extra) -> None:
    """Only material changes go out; a chatty status channel would drown the
    frames people actually came to watch."""
    global _status
    new = {"state": state, **extra}
    if new != _status:
        _status = new
        stage.publish({"type": "autopilot", **status()})


async def loop() -> None:
    """The performance loop. Started once from the app's lifespan."""
    import runner

    idx = 0
    await asyncio.sleep(2)      # let uvicorn finish booting before any run

    while True:
        try:
            if not enabled():
                _announce("off")
                await asyncio.sleep(5)
                continue

            if stage.viewers() == 0 and not performs_to_empty_room():
                _announce("empty_room")
                await asyncio.sleep(1.5)
                continue

            pending = stage.pending_interrupt
            if stage.busy() or (pending and pending.get("source") == "autopilot"):
                # A visitor holds the stage, or one of our own runs is parked
                # at its approval gate. A VISITOR'S parked interrupt does not
                # stop the show: their thread stays resumable in the
                # checkpointer while other scenarios play.
                await _handle_pending_approval()
                await asyncio.sleep(1)
                continue

            if time.time() < stage.blocked_until:
                _announce("resting", reason="spend guard",
                          next_at=stage.blocked_until)
                await asyncio.sleep(min(30, stage.blocked_until - time.time()))
                continue

            # Pace: quick first run for a fresh room, quiet gap otherwise.
            gap = rest_s() if stage.last_run_end() else first_run_delay_s()
            next_at = max(stage.last_run_end() + gap,
                          time.time() + first_run_delay_s())

            scenario = None
            mode = runner.mode()
            for _ in range(len(SCENARIOS)):
                candidate = SCENARIOS[idx % len(SCENARIOS)]
                idx += 1
                if runnable(candidate, mode):
                    scenario = candidate
                    break
            if scenario is None:
                _announce("nothing_runnable")
                await asyncio.sleep(30)
                continue

            _announce("countdown", next_at=next_at,
                      title=scenario["title"], agent=scenario["agent"],
                      shows=scenario["shows"], question=scenario["question"])
            while time.time() < next_at:
                if stage.busy() or stage.viewers() == 0 and not performs_to_empty_room():
                    break               # room emptied or a visitor stepped up
                await asyncio.sleep(0.5)
            if stage.busy() or (stage.viewers() == 0 and not performs_to_empty_room()):
                continue

            try:
                await stage.submit(scenario["agent"], scenario["question"],
                                   source="autopilot", ip="stage:autopilot",
                                   title=scenario["title"])
            except stage.StageBusy:
                continue                # a visitor beat us to it: their stage
            _announce("running", title=scenario["title"],
                      agent=scenario["agent"], shows=scenario["shows"])
            await stage.wait_idle(timeout=stage.RUN_TIMEOUT_S + 30)
            await _handle_pending_approval()

        except asyncio.CancelledError:
            raise
        except Exception:
            # One bad cycle must not kill the performance forever.
            await asyncio.sleep(5)


async def _handle_pending_approval() -> None:
    """Approve our own parked run once the countdown expires.

    The pause is the point: interrupt() left the graph checkpointed and
    waiting, and any viewer can override with the buttons before the
    deadline. If nobody does, the show must go on.
    """
    pending = stage.pending_interrupt
    if not pending or pending.get("source") != "autopilot":
        return
    deadline = pending.get("auto_approve_at") or 0
    while time.time() < deadline:
        if stage.pending_interrupt is not pending:
            return                      # a human decided first
        await asyncio.sleep(0.5)
    if stage.pending_interrupt is not pending or stage.busy():
        return
    with contextlib.suppress(stage.StageBusy, stage.NothingPending):
        await stage.submit(pending["agent"], "", source="autopilot",
                           ip="stage:autopilot", title="approving the memo",
                           resume={"action": "approve",
                                   "notes": "approved after the countdown"},
                           thread=pending["thread"])
        await stage.wait_idle(timeout=stage.RUN_TIMEOUT_S + 30)
