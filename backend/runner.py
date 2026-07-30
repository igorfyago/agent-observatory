"""Turn a LangGraph run into the frame stream the UI draws.

This is the heart of the product. Everything the browser shows about a run
comes out of one generic translator over `astream_events`, so a newly
registered agent gets live node highlighting and tool-call inspection with
zero extra wiring.

Frames emitted (each becomes one SSE `data:` line):

    run_start   {agent, mode}
    node_start  {node}
    node_end    {node, update}                  update = the state delta the
                                                node returned, exactly what the
                                                channels merge next superstep
    token       {node, text}                    streamed model output
    tool_start  {node, tool, id, args}          a tool call, with its arguments
    tool_end    {node, tool, id, output, ms}    its result and how long it took
    usage       {node, model, input_tokens,
                 output_tokens, cost_usd}       one model call, banked + priced
    route       {from, to, decision}            a conditional edge that fired
    interrupt   {thread, payload}               human-in-the-loop pause
    checkpoints {thread, items}                 the thread's saved supersteps,
                                                newest first, for time travel
    run_end     {answer}
    error       {message}

Why astream_events and not a hand-rolled emit callback: node and tool events
are already first-class in LangGraph, so deriving them centrally means the
agent modules stay plain graphs with no observability boilerplate. Agents that
need a frame the runner cannot infer (the demo pipeline's canned tokens, an
explicit routing decision) dispatch a custom event named "obs" and it passes
straight through.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import AsyncIterator

import registry
import spend
from langgraph.types import Command

MAX_TOOL_CHARS = 4000


def mode() -> dict:
    """What the run will actually use, so the UI can be honest about it."""
    return {
        "llm": "openai" if os.environ.get("OPENAI_API_KEY") else "demo",
        "langsmith": bool(
            os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
        ),
        "desk": bool(os.environ.get("DESK_API_URL", "https://desk.b4rruf3t.com")),
    }


def _clip(value) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str)
        except Exception:
            text = str(value)
    return text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS] + " ...(truncated)"


MAX_STATE_CHARS = 600


def _state_delta(update) -> dict | None:
    """A node's returned update, clipped per key so the state inspector stays
    light. Message lists collapse to a count: the tokens already streamed."""
    if not isinstance(update, dict):
        return None
    out = {}
    for key, value in list(update.items())[:12]:
        if isinstance(value, list) and value and type(value[0]).__name__.endswith("Message"):
            out[key] = f"+{len(value)} message(s)"
            continue
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        out[key] = text if len(text) <= MAX_STATE_CHARS else text[:MAX_STATE_CHARS] + " ..."
    return out or None


async def checkpoint_frame(agent: registry.Agent, thread: str) -> dict | None:
    """List the thread's saved supersteps for the time-travel strip.

    Newest first, matching aget_state_history. None when the agent has no
    checkpointer or the thread is unknown (e.g. after a restart: InMemorySaver
    starts empty, and the archive replay covers watching old runs instead).
    """
    try:
        compiled = agent.build()
        config = {"configurable": {"thread_id": thread}}
        items = []
        async for snap in compiled.aget_state_history(config):
            meta = snap.metadata or {}
            items.append({
                "checkpoint_id": (snap.config.get("configurable") or {}).get("checkpoint_id", ""),
                "step": meta.get("step", -1),
                "next": list(snap.next or ()),
                "wrote": sorted((meta.get("writes") or {}).keys())
                if isinstance(meta.get("writes"), dict) else [],
            })
            if len(items) >= 40:
                break
        if not items:
            return None
        return {"type": "checkpoints", "thread": thread, "agent": agent.id,
                "items": items}
    except Exception:
        return None  # a missing strip must never break a run


async def stream(agent: registry.Agent, question: str,
                 thread: str | None = None,
                 resume: dict | None = None,
                 checkpoint_id: str | None = None) -> AsyncIterator[dict]:
    """Yield UI frames for one run of `agent`.

    Three ways in, all the same stream out:
      - fresh run:            question set
      - resumed interrupt:    thread + resume set (Command(resume=...))
      - time travel:          thread + checkpoint_id set. Input None tells
        LangGraph to re-execute from that saved superstep, which is the
        checkpointer feature the checkpoint strip exists to show off.
    """
    node_ids = {n["id"] for n in agent.spec.get("nodes", [])}
    thread = thread or uuid.uuid4().hex[:12]
    configurable = {"thread_id": thread}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    config = {"configurable": configurable, "recursion_limit": 80}

    try:
        compiled = agent.build()
    except Exception as exc:
        yield {"type": "error", "message": f"could not build {agent.id}: {exc}"}
        return

    if resume is not None:
        payload = Command(resume=resume)
    elif checkpoint_id:
        payload = None                      # None = resume from the checkpoint
    else:
        payload = agent.make_input(question)

    yield {"type": "run_start", "agent": agent.id, "thread": thread,
           "mode": mode(), **({"replay_from": checkpoint_id} if checkpoint_id else {})}

    final: dict = {}
    tool_started: dict[str, float] = {}
    open_nodes: list[str] = []
    root_id: str | None = None

    try:
        async for ev in compiled.astream_events(payload, config=config, version="v2"):
            kind = ev["event"]

            # The root run's id IS the LangSmith trace id when tracing is on,
            # so surfacing it once lets the stage link every run to its trace.
            if root_id is None and not ev.get("parent_ids"):
                root_id = str(ev.get("run_id") or "")
                if root_id:
                    yield {"type": "trace_id", "root_run_id": root_id}
            name = ev.get("name", "")
            meta = ev.get("metadata") or {}
            node = meta.get("langgraph_node")

            if kind == "on_chain_start" and name in node_ids:
                open_nodes.append(name)
                yield {"type": "node_start", "node": name}

            elif kind == "on_chain_end" and name in node_ids:
                if name in open_nodes:
                    open_nodes.remove(name)
                # The node's returned delta rides along: it is exactly what the
                # state channels merge in the next superstep, so the UI can
                # show reducers doing their work without a second event.
                delta = _state_delta(ev["data"].get("output"))
                yield {"type": "node_end", "node": name,
                       **({"update": delta} if delta else {})}

            elif kind == "on_chat_model_stream":
                chunk = ev["data"].get("chunk")
                text = getattr(chunk, "content", "") or ""
                if isinstance(text, list):  # some providers chunk as blocks
                    text = "".join(
                        b.get("text", "") for b in text if isinstance(b, dict))
                if text:
                    yield {"type": "token", "node": node, "text": text}

            elif kind == "on_tool_start":
                rid = str(ev.get("run_id", ""))
                tool_started[rid] = time.perf_counter()
                yield {
                    "type": "tool_start", "node": node or "tools",
                    "tool": name, "id": rid,
                    "args": _clip(ev["data"].get("input")),
                }

            elif kind == "on_tool_end":
                rid = str(ev.get("run_id", ""))
                started = tool_started.pop(rid, None)
                out = ev["data"].get("output")
                out = getattr(out, "content", out)  # unwrap ToolMessage
                yield {
                    "type": "tool_end", "node": node or "tools",
                    "tool": name, "id": rid,
                    "output": _clip(out),
                    "ms": int((time.perf_counter() - started) * 1000) if started else None,
                }

            elif kind == "on_chat_model_end":
                # Bank the cost of every model call, including the ones inside
                # sub-agents, so the spend guard sees the whole run. The same
                # reading goes to the browser: the cost ticker is the meter,
                # not an estimate of it.
                try:
                    info = spend.record(ev["data"].get("output"))
                    yield {"type": "usage", "node": node, **info}
                except Exception:
                    pass  # metering must never break a run

            elif kind == "on_custom_event" and name == "obs":
                data = ev.get("data") or {}
                if isinstance(data, dict) and data.get("type"):
                    yield data

            elif kind == "on_chain_end" and not ev.get("parent_ids"):
                out = ev["data"].get("output")
                if isinstance(out, dict):
                    final = out

    except Exception as exc:
        for n in reversed(open_nodes):
            yield {"type": "node_end", "node": n}
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        return

    # The thread's saved supersteps, for the checkpoint strip. Emitted before
    # the closing frame so the UI has them the moment the run settles.
    cps = await checkpoint_frame(agent, thread)
    if cps:
        yield cps

    # Human-in-the-loop: the graph parked itself instead of finishing.
    if agent.interactive:
        try:
            snapshot = await compiled.aget_state(config)
            pending = getattr(snapshot, "interrupts", None) or ()
            if pending:
                yield {
                    "type": "interrupt",
                    "thread": thread,
                    "agent": agent.id,
                    "payload": pending[0].value,
                }
                return
        except Exception:
            pass  # no checkpointer, or nothing parked: fall through to run_end

    try:
        answer = agent.extract(final) if final else ""
    except Exception as exc:
        answer = f"(could not extract an answer: {exc})"

    yield {"type": "run_end", "agent": agent.id, "thread": thread, "answer": answer}
