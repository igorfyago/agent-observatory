"""Multi-agent research pipeline on LangGraph.

One of the hosted agents (registry id "pipeline"). Node start/end, model tokens
and tool calls are derived generically by backend/runner.py from astream_events,
so this module carries no observability boilerplate. The two frames the runner
cannot infer, the demo mode's canned tokens and the critic's explicit routing
decision, are dispatched as LangChain custom events named "obs" and pass
straight through to the browser.

Runs in two modes:
  - DEMO  (no OPENAI_API_KEY): canned role-scripts streamed word-by-word,
          so the full pipeline + UI works with zero credentials.
  - LIVE  (OPENAI_API_KEY set): real ChatOpenAI streaming per node.

LangSmith: set LANGCHAIN_TRACING_V2=true + LANGSMITH_API_KEY and every run
is traced automatically (the graph is a normal LangGraph app).
"""
from __future__ import annotations

import asyncio
import os
from typing import TypedDict

from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph import StateGraph, START, END


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class PipelineState(TypedDict, total=False):
    question: str
    plan: str
    research: str
    analysis: str
    draft: str
    critique: str
    approved: bool
    revisions: int


# Graph spec the frontend renders, kept next to the graph so they can't drift.
# Each node carries its x-ray: the design defended in plain words, readable by
# clicking the node while the run streams past it.
GRAPH_SPEC = {
    "nodes": [
        {
            "id": "supervisor", "label": "Supervisor", "role": "plans & delegates",
            "xray": {
                "gotcha": "Two add_edge calls out of one node = both targets run in the same superstep. Parallelism is topology, not threads.",
                "q": ["Bad plan? The critic gate downstream bounces the damage back, max twice."],
            },
        },
        {
            "id": "researcher", "label": "Researcher", "role": "gathers facts",
            "xray": {
                "gotcha": "Runs concurrently with the analyst. Each writes its own state key, so no reducer needed, no collisions possible.",
                "q": ["Two writers on ONE key? Then you need a reducer. See the analyst agent's analyses list."],
            },
        },
        {
            "id": "analyst", "label": "Analyst", "role": "weighs trade-offs",
            "xray": {
                "gotcha": "Same superstep as the researcher, different key, different prompt. Judgment split from fact-finding on purpose.",
                "q": [],
            },
        },
        {
            "id": "writer", "label": "Writer", "role": "drafts answer",
            "xray": {
                "gotcha": "A join: LangGraph will not start it until both branches land. The barrier is free, it is just two edges in.",
                "q": ["Who counts revisions? The writer, so the critic stays a pure judge."],
            },
        },
        {
            "id": "critic", "label": "Critic", "role": "reviews & gates",
            "xray": {
                "gotcha": "add_conditional_edges routes approve to END, revise back to the writer. The loop in the picture is real control flow.",
                "q": ["Infinite loop? The router forces END after 2 revisions, no matter the verdict.",
                      "Cheaper critic model? One line with init_chat_model: judging is easier than writing."],
            },
        },
    ],
    "edges": [
        {"from": "supervisor", "to": "researcher"},
        {"from": "supervisor", "to": "analyst"},
        {"from": "researcher", "to": "writer"},
        {"from": "analyst",    "to": "writer"},
        {"from": "writer",     "to": "critic"},
        {"from": "critic",     "to": "writer", "kind": "loop", "label": "revise"},
        {"from": "critic",     "to": "end",    "label": "approve"},
    ],
    # The state channels, as the inspector explains them while deltas land.
    "state": [
        {"key": "question",  "kind": "overwrite", "note": "the input, written once"},
        {"key": "plan",      "kind": "overwrite", "note": "supervisor's approach"},
        {"key": "research",  "kind": "overwrite", "note": "researcher's findings"},
        {"key": "analysis",  "kind": "overwrite", "note": "analyst's verdict"},
        {"key": "draft",     "kind": "overwrite", "note": "writer's latest pass"},
        {"key": "critique",  "kind": "overwrite", "note": "critic's last review"},
        {"key": "approved",  "kind": "overwrite", "note": "the gate's decision"},
        {"key": "revisions", "kind": "overwrite", "note": "loop bound, counts writer passes"},
    ],
    # The LangGraph surface this graph actually exercises.
    "framework": [
        {"api": "StateGraph(PipelineState)", "note": "typed shared state"},
        {"api": "add_edge fan-out + join", "note": "parallel superstep, free barrier"},
        {"api": "add_conditional_edges", "note": "approve/revise gate in topology"},
        {"api": "astream_events v2", "note": "one listener feeds the whole UI"},
        {"api": "InMemorySaver", "note": "checkpoint per superstep"},
    ],
}


def mode() -> dict:
    return {
        "llm": "openai" if os.environ.get("OPENAI_API_KEY") else "demo",
        "langsmith": bool(
            os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
        ),
    }


# --------------------------------------------------------------------------- #
# Demo scripts (keyless mode)
# --------------------------------------------------------------------------- #

DEMO_SCRIPTS = {
    "supervisor": (
        "Breaking this question down. I need grounded facts and a trade-off view, "
        "so I'm fanning out to two specialists in parallel: the Researcher will "
        "collect the concrete evidence, and the Analyst will weigh the options. "
        "The Writer will merge both into a draft, and the Critic gates it before "
        "anything ships."
    ),
    "researcher": (
        "Collecting the facts. Key findings: the core framework exposes a graph "
        "API with explicit state, checkpointing gives durable conversations and "
        "time travel, and streaming events are first-class, every node emits "
        "start, token, and end signals we can observe. Ecosystem docs confirm "
        "human-in-the-loop interrupts and a CLI for local dev and deployment. "
        "Evidence quality: high, sourced from primary documentation."
    ),
    "analyst": (
        "Weighing trade-offs. Strengths: explicit control flow beats prompt-chained "
        "spaghetti, state is typed and inspectable, and observability comes almost "
        "free. Costs: a steeper learning curve than a single agent loop, and the "
        "graph abstraction is overkill for one-shot tasks. Verdict: for multi-step, "
        "multi-agent work with audit requirements, the structured approach wins "
        "clearly. Confidence: high."
    ),
    "writer": (
        "Drafting the answer. Combining the Researcher's evidence with the "
        "Analyst's verdict: adopt the graph-based architecture for multi-agent "
        "pipelines. It gives explicit, testable control flow, durable state via "
        "checkpointing, and native streaming that powers live observability, "
        "exactly the properties needed for production agents. Reserve simple "
        "prompt loops for one-shot tasks where a graph adds ceremony without value."
    ),
    "critic-revise": (
        "Reviewing the draft. The argument is sound but the recommendation buries "
        "its lede and lacks a concrete next step. Sending it back: lead with the "
        "recommendation, then evidence, and close with a first action the team can "
        "take this week. One revision requested."
    ),
    "writer-revision": (
        "Revising per the Critic. Recommendation first: adopt the graph-based "
        "architecture for all multi-agent pipelines, starting now. Why: typed "
        "state and explicit edges make agent behaviour testable; checkpointing "
        "gives durable, resumable runs; native event streams give the team live "
        "observability out of the box. First action: port one existing pipeline "
        "to a five-node supervisor pattern and wire its event stream into the "
        "monitoring dashboard this week."
    ),
    "critic-approve": (
        "Re-reviewing. The revision leads with the recommendation, grounds it in "
        "the collected evidence, and ends with a concrete, scoped first action. "
        "Quality gate passed, approving for delivery."
    ),
}

DEMO_TOKEN_DELAY = 0.035  # seconds per word, tuned so the UI reads well


# --------------------------------------------------------------------------- #
# LLM streaming (real or demo)
# --------------------------------------------------------------------------- #

async def _stream_llm(node: str, prompt: str, emit) -> str:
    """Stream a real model's answer for this node.

    No custom token events here: the runner already forwards every
    on_chat_model_stream chunk, and emitting both painted every token twice
    in live mode (found on screen: "the the first first draft draft").
    Custom events stay a demo-mode tool, where there is no real stream.
    """
    from llm import get_model

    full = ""
    async for chunk in get_model().astream(prompt):
        text = chunk.content or ""
        if text:
            full += text
    return full


async def _stream_demo(node: str, script_key: str, emit) -> str:
    """Stream a canned script word-by-word so the UI lights up without keys."""
    full = ""
    for word in DEMO_SCRIPTS[script_key].split(" "):
        token = word + " "
        full += token
        await emit({"type": "token", "node": node, "text": token})
        await asyncio.sleep(DEMO_TOKEN_DELAY)
    return full.strip()


async def speak(node: str, prompt: str, script_key: str, emit) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return await _stream_llm(node, prompt, emit)
    return await _stream_demo(node, script_key, emit)


async def _emit(ev: dict) -> None:
    """Push a frame the generic runner cannot derive on its own (demo tokens,
    routing decisions). astream_events surfaces these as on_custom_event."""
    await adispatch_custom_event("obs", ev)


def _emit_of(config) -> callable:
    return _emit


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

async def supervisor(state: PipelineState, config) -> PipelineState:
    emit = _emit_of(config)
    prompt = (
        "You are the Supervisor of a research team (Researcher, Analyst, Writer, "
        f"Critic). In 2-3 sentences, state your plan for answering: {state['question']}"
    )
    plan = await speak("supervisor", prompt, "supervisor", emit)
    return {"plan": plan}


async def researcher(state: PipelineState, config) -> PipelineState:
    emit = _emit_of(config)
    prompt = (
        "You are the Researcher. List the key concrete facts (with confidence "
        "levels) needed to answer the question. One short paragraph or 4 "
        f"tight bullets. Question: {state['question']}\nPlan: {state.get('plan', '')}"
    )
    research = await speak("researcher", prompt, "researcher", emit)
    return {"research": research}


async def analyst(state: PipelineState, config) -> PipelineState:
    emit = _emit_of(config)
    prompt = (
        "You are the Analyst. Weigh the trade-offs and give a verdict with "
        "confidence, in one short paragraph. "
        f"Question: {state['question']}\nPlan: {state.get('plan', '')}"
    )
    analysis = await speak("analyst", prompt, "analyst", emit)
    return {"analysis": analysis}


async def writer(state: PipelineState, config) -> PipelineState:
    emit = _emit_of(config)
    revisions = state.get("revisions", 0)
    if revisions == 0:
        prompt = (
            "You are the Writer. Merge the research and analysis into a concise "
            "answer: one short paragraph, recommendation in the first sentence. "
            f"Question: {state['question']}\n\nResearch: {state.get('research', '')}"
            f"\n\nAnalysis: {state.get('analysis', '')}"
        )
        script = "writer"
    else:
        prompt = (
            "You are the Writer. Revise your draft per the Critic's feedback. "
            "One short paragraph.\n\n"
            f"Draft: {state.get('draft', '')}\n\nCritique: {state.get('critique', '')}"
        )
        script = "writer-revision"
    draft = await speak("writer", prompt, script, emit)
    return {"draft": draft, "revisions": revisions + 1}


async def critic(state: PipelineState, config) -> PipelineState:
    emit = _emit_of(config)
    revisions = state.get("revisions", 0)

    if os.environ.get("OPENAI_API_KEY"):
        prompt = (
            "You are the Critic. Review this draft. Reply starting with exactly "
            "APPROVE or REVISE, then one short paragraph of reasoning.\n\n"
            f"Question: {state['question']}\n\nDraft: {state.get('draft', '')}"
        )
        critique = await speak("critic", prompt, "critic-revise", emit)
        # Never loop forever: force approval after the second draft.
        approved = critique.strip().upper().startswith("APPROVE") or revisions >= 2
    else:
        # Demo choreography: reject the first draft (shows the revise loop
        # lighting up), approve the second.
        script = "critic-approve" if revisions >= 2 else "critic-revise"
        critique = await _stream_demo("critic", script, emit)
        approved = revisions >= 2

    await emit({
        "type": "route",
        "from": "critic",
        "to": "end" if approved else "writer",
        "decision": "approve" if approved else "revise",
    })
    return {"critique": critique, "approved": approved}


def route_after_critic(state: PipelineState) -> str:
    return "approve" if state.get("approved") else "revise"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("supervisor", supervisor)
    g.add_node("researcher", researcher)
    g.add_node("analyst", analyst)
    g.add_node("writer", writer)
    g.add_node("critic", critic)

    g.add_edge(START, "supervisor")
    # Parallel fan-out; writer is a join, it waits for both branches.
    g.add_edge("supervisor", "researcher")
    g.add_edge("supervisor", "analyst")
    g.add_edge("researcher", "writer")
    g.add_edge("analyst", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges(
        "critic", route_after_critic, {"revise": "writer", "approve": END}
    )
    # The checkpointer saves state at every superstep, which powers the
    # checkpoint strip and the re-run-from-step buttons in the UI. In-memory
    # is the right weight here: archived frame replays cover history across
    # restarts, and threads themselves are throwaway performances.
    from langgraph.checkpoint.memory import InMemorySaver
    return g.compile(checkpointer=InMemorySaver())


# --------------------------------------------------------------------------- #
# Registry contract (see backend/agents/__init__.py)
# --------------------------------------------------------------------------- #

SPEC = GRAPH_SPEC


def build():
    return build_graph()


def make_input(question: str) -> dict:
    return {"question": question, "revisions": 0}


def extract(result: dict) -> str:
    return result.get("draft", "")
