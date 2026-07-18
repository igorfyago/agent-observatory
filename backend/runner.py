"""Turn a LangGraph run into the frame stream the UI draws.

This is the heart of the product. Everything the browser shows about a run
comes out of one generic translator over `astream_events`, so a newly
registered agent gets live node highlighting and tool-call inspection with
zero extra wiring.

Frames emitted (each becomes one SSE `data:` line):

    run_start   {agent, mode}
    node_start  {node}
    node_end    {node}
    token       {node, text}                    streamed model output
    tool_start  {node, tool, id, args}          a tool call, with its arguments
    tool_end    {node, tool, id, output, ms}    its result and how long it took
    route       {from, to, decision}            a conditional edge that fired
    interrupt   {thread, payload}               human-in-the-loop pause
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


async def stream(agent: registry.Agent, question: str,
                 thread: str | None = None,
                 resume: dict | None = None) -> AsyncIterator[dict]:
    """Yield UI frames for one run (or one resumed run) of `agent`."""
    node_ids = {n["id"] for n in agent.spec.get("nodes", [])}
    thread = thread or uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 80}

    try:
        compiled = agent.build()
    except Exception as exc:
        yield {"type": "error", "message": f"could not build {agent.id}: {exc}"}
        return

    payload = Command(resume=resume) if resume is not None else agent.make_input(question)

    yield {"type": "run_start", "agent": agent.id, "thread": thread, "mode": mode()}

    final: dict = {}
    tool_started: dict[str, float] = {}
    open_nodes: list[str] = []

    try:
        async for ev in compiled.astream_events(payload, config=config, version="v2"):
            kind = ev["event"]
            name = ev.get("name", "")
            meta = ev.get("metadata") or {}
            node = meta.get("langgraph_node")

            if kind == "on_chain_start" and name in node_ids:
                open_nodes.append(name)
                yield {"type": "node_start", "node": name}

            elif kind == "on_chain_end" and name in node_ids:
                if name in open_nodes:
                    open_nodes.remove(name)
                yield {"type": "node_end", "node": name}

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
                # sub-agents, so the spend guard sees the whole run.
                try:
                    spend.record(ev["data"].get("output"))
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
