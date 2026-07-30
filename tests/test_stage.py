"""The stage: one shared performance, preemption rules, and the archive.

What is pinned here is the social contract and the honesty of the record:
a visitor preempts the autopilot but never another visitor, a viewer who
arrives mid-run gets the whole run so far, every finished run lands whole in
the archive, and the demo pipeline emits the full frame vocabulary the UI is
built on (state deltas, checkpoints, routes) without any API key at all.
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import graph as pipeline_module  # noqa: E402
import registry  # noqa: E402
import runner  # noqa: E402
import stage  # noqa: E402
import store  # noqa: E402


@pytest.fixture()
def obs(tmp_path, monkeypatch):
    """Demo mode, throwaway data + spend DBs, instant demo tokens, clean stage."""
    monkeypatch.setenv("OBS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OBS_SPEND_DB", str(tmp_path / "spend.db"))
    monkeypatch.setenv("OBS_COOLDOWN_S", "0")
    monkeypatch.setenv("OBS_MAX_CONCURRENT", "4")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(pipeline_module, "DEMO_TOKEN_DELAY", 0.0)

    import spend
    spend._reset_runtime()

    stage._subscribers.clear()
    stage.current = None
    stage.pending_interrupt = None
    stage.blocked_until = 0.0
    stage._task = None
    stage._last_run_end = 0.0
    yield
    spend._reset_runtime()


# ---------------------------------------------------------------- archive ---

def test_archive_roundtrip(obs):
    meta = {"started_at": "2026-07-30T10:00:00+00:00", "agent_id": "pipeline",
            "source": "autopilot", "title": "five roles", "question": "q",
            "outcome": "ok", "latency_ms": 1234, "tokens_in": 100,
            "tokens_out": 50, "cost_usd": 0.0021, "tool_calls": 3,
            "thread_id": "t1", "trace_url": ""}
    frames = [{"type": "run_start"}, {"type": "run_end", "answer": "done"}]

    run_id = store.save_stage_run(meta, frames)
    got = store.get_stage_run(run_id)
    assert got["agent_id"] == "pipeline"
    assert got["frames"] == frames

    listed = store.list_stage_runs()
    assert listed[0]["id"] == run_id
    assert "frames" not in listed[0]          # summaries stay light

    assert store.latest_stage_run()["id"] == run_id

    stats = store.stage_stats()
    assert stats["runs_archived"] == 1
    assert stats["tool_calls_today"] >= 0     # date filter must not explode


def test_trace_url_lands_after_the_fact(obs):
    run_id = store.save_stage_run({"agent_id": "pipeline"}, [])
    store.set_stage_trace(run_id, "https://smith.langchain.com/r/abc")
    assert store.get_stage_run(run_id)["trace_url"].endswith("/r/abc")


# ------------------------------------------------------------ demo frames ---

def _run_pipeline_frames():
    agent = registry.get("pipeline")

    async def collect():
        return [f async for f in runner.stream(agent, "graphs or chains?")]

    return asyncio.run(collect())


def test_demo_pipeline_emits_the_full_frame_vocabulary(obs):
    frames = _run_pipeline_frames()
    types = [f["type"] for f in frames]

    assert types[0] == "run_start"
    assert types[-1] == "run_end"
    assert "node_start" in types and "node_end" in types and "token" in types
    assert "route" in types                   # the critic decided something

    # State deltas ride node_end: the supervisor's plan must be in one.
    updates = [f.get("update", {}) for f in frames if f["type"] == "node_end"]
    assert any("plan" in u for u in updates)

    # The checkpointer saw every superstep and the strip has real entries.
    cps = [f for f in frames if f["type"] == "checkpoints"]
    assert cps and len(cps[0]["items"]) >= 3
    assert all(c["checkpoint_id"] for c in cps[0]["items"])


def test_demo_critic_rejects_then_approves(obs):
    """The demo choreography IS the feature: the revise loop must fire."""
    frames = _run_pipeline_frames()
    routes = [f for f in frames if f["type"] == "route"]
    decisions = [r["decision"] for r in routes]
    assert "revise" in decisions and "approve" in decisions


# ------------------------------------------------------------- the stage ---

async def _settled():
    await stage.wait_idle(timeout=30)
    # give the finally-block bookkeeping a beat to archive and go idle
    for _ in range(100):
        if not stage.busy():
            return
        await asyncio.sleep(0.05)


def test_mid_run_viewer_gets_the_whole_story(obs, monkeypatch):
    monkeypatch.setattr(pipeline_module, "DEMO_TOKEN_DELAY", 0.01)

    async def scenario():
        await stage.submit("pipeline", "graphs or chains?",
                           source="autopilot", ip="test:ap", title="t")
        await asyncio.sleep(0.4)              # run is mid-flight now
        assert stage.busy()
        snap = stage.snapshot_frames()
        assert any(f["type"] == "run_start" for f in snap)

        sub_id, q = stage.subscribe()
        try:
            frame = await asyncio.wait_for(q.get(), timeout=15)
            assert frame["type"]              # live tail flows to the queue
        finally:
            stage.unsubscribe(sub_id)
        await _settled()

    asyncio.run(scenario())
    runs = store.list_stage_runs()
    assert runs and runs[0]["outcome"] == "ok"
    assert runs[0]["source"] == "autopilot"


def test_visitor_preempts_autopilot_but_not_other_visitors(obs, monkeypatch):
    monkeypatch.setattr(pipeline_module, "DEMO_TOKEN_DELAY", 0.01)

    async def scenario():
        await stage.submit("pipeline", "first", source="autopilot",
                           ip="test:ap", title="t")
        await asyncio.sleep(0.3)
        # A visitor walks in: the autopilot run dies mid-flight.
        await stage.submit("pipeline", "second", source="visitor", ip="v1")
        assert stage.current["source"] == "visitor"

        # Another visitor cannot steal the stage from the first one.
        with pytest.raises(stage.StageBusy):
            await stage.submit("pipeline", "third", source="visitor", ip="v2")
        await _settled()

    asyncio.run(scenario())
    runs = store.list_stage_runs()               # newest first
    assert [r["source"] for r in runs] == ["visitor", "autopilot"]
    assert runs[1]["outcome"] == "preempted"
    assert runs[0]["outcome"] == "ok"


def test_resume_with_nothing_pending_is_refused(obs):
    async def scenario():
        with pytest.raises(stage.NothingPending):
            await stage.submit("analyst", "", source="visitor", ip="v1",
                               resume={"action": "approve"}, thread="ghost")

    asyncio.run(scenario())


def test_spend_refusal_backs_the_stage_off(obs, monkeypatch):
    monkeypatch.setenv("OBS_DAILY_USD", "0.00")   # the meter is already spent

    async def scenario():
        await stage.submit("pipeline", "q", source="autopilot", ip="test:ap")
        await _settled()

    asyncio.run(scenario())
    assert stage.blocked_until > time.time()      # autopilot reads this
    # a refused run is not archived: nothing ran
    assert store.list_stage_runs() == []
