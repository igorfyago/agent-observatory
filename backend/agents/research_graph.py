"""Agent 4 · Research: explicit graph orchestration with two visible loops.

Forked from ai-trading-desk agents/04_research_graph on 2026-07-18.

  plan -> [ researcher <-> tools ] -> draft -> reflect --ok--> finalize
                 ^  inner tool loop      |
                 +---- revise (outer reflection loop, max 2) ----+

The inner loop is the familiar model/tools cycle. The outer loop is the point:
a critic node reads the draft and a CONDITIONAL EDGE routes either to the end
or back into research with the critique injected. Bounded self-revision, and
both loops light up in the DAG.

Desk coupling: positioning_snapshot, wall_map, market_news and ta_signals used
to import common/market, common/signals and common/news in-process. They now
go over HTTP to the desk (backend/desk_client.py) and say so in their output.
The x_pulse tool did NOT come along: it needed common/xpulse plus an XAI key,
and Grok-backed X search is desk-only. sql_query now targets the observatory's
own run log (backend/store.py), not the options schema.

Models are constructed lazily inside build(), so importing this module never
requires an API key.
"""
from __future__ import annotations

import math
import operator
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import desk_client
import store
from llm import get_model

MAX_REVISIONS = 2

SPEC = {
    "nodes": [
        {
            "id": "plan", "label": "Plan", "role": "writes the steps",
            "xray": {
                "concept": (
                    "Plan first, then act. The plan node writes numbered "
                    "steps that map to actual tools, and seeds the message "
                    "list the researcher will loop over."),
                "here": (
                    "Returns `plan` plus the opening messages: a system turn "
                    "carrying the plan and the method rules."),
                "tradeoffs": (
                    "One extra model call before any work. Skipping it makes "
                    "easy runs cheaper and hard runs chaotic; the plan is "
                    "what keeps a five-tool question from wandering."),
                "questions": [
                    "Is the plan actually followed? It travels in the system message of the loop, so every researcher pass re-reads it; drift shows up as the reflect gate bouncing the draft.",
                ],
            },
        },
        {
            "id": "researcher", "label": "Researcher", "role": "runs the tools",
            "xray": {
                "concept": (
                    "The inner loop's model half: bind_tools gives the model "
                    "the toolbelt, and a conditional edge checks whether its "
                    "last message carries tool_calls or a finished thought."),
                "here": (
                    "Loops against desk data, the run-log SQL tool and a "
                    "calculator until the plan's numbers are gathered."),
                "tradeoffs": (
                    "A hand-built loop is more code than create_agent, and "
                    "buys the outer reflection loop a place to re-enter: "
                    "revise pushes critique into the SAME message list and "
                    "the loop continues where it left off."),
                "questions": [
                    "Why hand-roll what create_agent gives for free? Because this loop needs a second entrance (revise) and a custom exit (draft); prebuilt loops trade that control away.",
                    "How does the loop end? The router reads the last message: tool_calls means back to tools, none means the research phase is done and draft takes over.",
                ],
            },
        },
        {
            "id": "tools", "label": "Tools", "role": "desk, sql, math",
            "xray": {
                "concept": (
                    "A ToolNode executes whatever calls the researcher's "
                    "last message carries and appends the results as "
                    "ToolMessages. handle_tool_errors=True turns crashes "
                    "into text the model can react to."),
                "here": (
                    "Six tools: live desk positioning, walls, news, TA "
                    "signals over HTTP, SQL over the observatory's own run "
                    "log, and a calculator so arithmetic never happens in "
                    "the model's head."),
                "tradeoffs": (
                    "Tools that degrade to explicit 'desk unavailable' "
                    "strings cost a branch in every tool, and buy the whole "
                    "graph its honesty: no fabricated market data, ever."),
                "questions": [
                    "What happens when a tool throws? The error becomes a ToolMessage instead of a crash, and the model decides whether to retry differently or route around it.",
                    "Why a calculator tool for a language model? Because token prediction is not arithmetic; the tool makes every number in the report checkable.",
                ],
            },
        },
        {
            "id": "draft", "label": "Draft", "role": "writes the answer",
            "xray": {
                "concept": (
                    "A synthesis node outside the loop: it reads the full "
                    "message history (plan, tool calls, results) and writes "
                    "the answer in one pass."),
                "here": "Thesis first, then the gathered numbers, then risks.",
                "tradeoffs": (
                    "Drafting from the whole history costs input tokens but "
                    "means the writer saw every number the loop fetched, so "
                    "nothing is summarized away before it is used."),
                "questions": [
                    "Why separate drafting from researching? Different jobs want different prompts: gathering rewards restraint, writing rewards commitment. One node doing both does both worse.",
                ],
            },
        },
        {
            "id": "reflect", "label": "Reflect", "role": "judges sufficiency",
            "xray": {
                "concept": (
                    "The outer loop's gate: a structured verdict "
                    "(with_structured_output) on whether every claim in the "
                    "draft is backed by a retrieved number."),
                "here": (
                    "Returns sufficient or a concrete critique. The router "
                    "then picks finalize or revise, and revisions are "
                    "counted so the outer loop is bounded at two."),
                "tradeoffs": (
                    "Self-review by the same model family has blind spots, "
                    "and still catches the common failure: confident prose "
                    "over missing data. The bound keeps the cost of "
                    "disagreement finite."),
                "questions": [
                    "Is a model grading itself meaningful? For completeness checks, yes: 'is there a number behind this sentence' is a mechanical question the judge answers well.",
                    "Why bound at two revisions? Measured value drops off a cliff after the first bounce; the cap converts a philosophical loop into a priced feature.",
                ],
            },
        },
        {
            "id": "revise", "label": "Revise", "role": "injects critique",
            "xray": {
                "concept": (
                    "The outer loop's re-entry: it appends the critique to "
                    "the message list as a new instruction and hands control "
                    "back to the researcher, tools and all."),
                "here": (
                    "The revise edge in the picture is the reflection loop "
                    "made visible: draft, judged, back to research with "
                    "sharper orders."),
                "tradeoffs": (
                    "Re-entering research (not just rewriting) lets a bounced "
                    "draft go fetch the missing number instead of rephrasing "
                    "around the hole. It is the more expensive and the more "
                    "honest repair."),
                "questions": [
                    "Why not have reflect fix the draft itself? A judge that edits stops judging; keeping roles separate keeps the gate's incentives clean.",
                ],
            },
        },
        {
            "id": "finalize", "label": "Finalize", "role": "ships the report",
            "xray": {
                "concept": "The terminal node: promotes the surviving draft to `report`.",
                "here": "What ships is exactly what passed the gate.",
                "tradeoffs": (
                    "A copy into a dedicated key looks redundant and is not: "
                    "`report` set means the gate passed, which downstream "
                    "code can trust without re-deriving."),
                "questions": [
                    "Why a whole node for a copy? Terminal state transitions deserve their own superstep: it is visible in the trace and in the checkpoint history.",
                ],
            },
        },
    ],
    "edges": [
        {"from": "plan",       "to": "researcher"},
        {"from": "researcher", "to": "tools",      "label": "call"},
        {"from": "tools",      "to": "researcher", "kind": "loop", "label": "result"},
        {"from": "researcher", "to": "draft",      "label": "done"},
        {"from": "draft",      "to": "reflect"},
        {"from": "reflect",    "to": "revise",     "label": "insufficient"},
        {"from": "revise",     "to": "researcher", "kind": "loop"},
        {"from": "reflect",    "to": "finalize",   "label": "sufficient"},
        {"from": "finalize",   "to": "end"},
    ],
    "state": [
        {"key": "question",  "kind": "overwrite", "note": "the input"},
        {"key": "plan",      "kind": "overwrite", "note": "numbered steps mapped to tools"},
        {"key": "messages",  "kind": "append",
         "note": "operator.add reducer: the inner loop and every revision share one growing history"},
        {"key": "draft",     "kind": "overwrite", "note": "latest full answer"},
        {"key": "critique",  "kind": "overwrite", "note": "empty means the gate passed"},
        {"key": "revisions", "kind": "overwrite", "note": "outer loop bound"},
        {"key": "report",    "kind": "overwrite", "note": "only set by finalize"},
    ],
    "framework": [
        {"api": "StateGraph with two loops",
         "note": "an inner model/tools cycle nested inside an outer draft/reflect/revise cycle, both as conditional edges"},
        {"api": "model.bind_tools(TOOLS)",
         "note": "the researcher's model half; the router reads tool_calls off its last message"},
        {"api": "ToolNode(handle_tool_errors=True)",
         "note": "tool crashes come back as ToolMessages, so recovery is the model's move, not an exception"},
        {"api": "with_structured_output(Verdict)",
         "note": "the reflect gate returns a typed verdict, so routing reads a bool, not prose"},
        {"api": "Annotated[list, operator.add]",
         "note": "the messages channel is a reducer: both loops append, nothing overwrites"},
        {"api": "InMemorySaver checkpointer",
         "note": "each superstep saved: watch the checkpoint strip grow as the loops run"},
    ],
}


# ---------------------------------------------------------------- tools ----

@tool
def positioning_snapshot(ticker: str) -> str:
    """Latest dealer-positioning snapshot for a ticker: spot, gamma regime,
    net/abs GEX, gamma flip level, IV, VIX, signal score. Fetched live from
    the trading desk over HTTP. Covered tickers: SPY, QQQ, IWM."""
    return desk_client.positioning_snapshot(ticker)


@tool
def wall_map(ticker: str) -> str:
    """Current strongest call/put walls for a ticker: the strikes dealers
    defend. Fetched live from the trading desk over HTTP."""
    return desk_client.wall_map(ticker)


@tool
def market_news(ticker: str) -> str:
    """Latest headlines for a ticker: catalysts, earnings, macro context.
    Fetched live from the trading desk over HTTP."""
    return desk_client.market_news(ticker)


@tool
def ta_signals(ticker: str) -> str:
    """Technical alerts that fired recently for a ticker (market-structure
    breaks, VWAP band touches, Donchian breaks). Fetched live from the
    trading desk over HTTP."""
    return desk_client.ta_signals(ticker)


@tool
def sql_query(sql: str) -> str:
    """Run a read-only SELECT against the observatory's own telemetry DB.
    Table agent_runs (started_at, agent_id, kind, question, tool_calls,
    tokens, latency_ms, outcome) logs every agent run on this host. Use it
    for questions about the observatory itself."""
    if not sql.strip().lower().startswith(("select", "with")):
        return "REJECTED: SELECT only."
    try:
        rows = store.run_readonly(sql)[:40]
        return "\n".join(str(r) for r in rows) or "0 rows"
    except Exception as exc:
        return f"SQL ERROR: {exc}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression (supports sqrt, log, exp, abs, round, min,
    max, pi). Use it for ratios and annualization: never do arithmetic in
    your head."""
    allowed = {"sqrt": math.sqrt, "log": math.log, "exp": math.exp, "abs": abs,
               "round": round, "min": min, "max": max, "pi": math.pi}
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))  # noqa: S307
    except Exception as exc:
        return f"MATH ERROR: {exc}"


TOOLS = [positioning_snapshot, wall_map, market_news, ta_signals, sql_query, calculator]


# ---------------------------------------------------------------- state ----

class ResearchState(TypedDict, total=False):
    question: str
    plan: str
    messages: Annotated[list, operator.add]
    draft: str
    critique: str
    revisions: int
    report: str


class Verdict(BaseModel):
    sufficient: bool = Field(description="Does the draft fully answer the question with numbers?")
    critique: str = Field(description="If not sufficient: what is missing or unsupported")


# ---------------------------------------------------------------- graph ----

def build():
    model = get_model()
    research_model = model.bind_tools(TOOLS)

    async def plan_node(state: ResearchState) -> dict:
        plan = (await model.ainvoke([
            SystemMessage(
                "You are a research lead. Write a numbered 3-5 step data-gathering "
                "plan for the question. Steps must map to available tools: live desk "
                "positioning snapshots, wall maps, news, TA signals, SQL over the "
                "observatory run log, and a calculator. No prose beyond the steps."),
            HumanMessage(state["question"]),
        ])).content
        return {"plan": plan, "messages": [
            SystemMessage(
                "You are a quant researcher. Execute the plan step by step using "
                "tools. Gather ALL numbers before concluding. If a tool reports the "
                "desk is unavailable, say so plainly and never invent a figure. "
                "Plan:\n" + plan),
            HumanMessage(state["question"]),
        ]}

    async def researcher(state: ResearchState) -> dict:
        return {"messages": [await research_model.ainvoke(state["messages"])]}

    def route_research(state: ResearchState) -> Literal["tools", "draft"]:
        return "tools" if state["messages"][-1].tool_calls else "draft"

    async def draft_node(state: ResearchState) -> dict:
        draft = (await model.ainvoke(state["messages"] + [
            HumanMessage("Write the research answer now: direct thesis first, then the "
                         "supporting numbers you gathered, then risks and caveats. "
                         "Cite figures explicitly, no vague claims.")
        ])).content
        return {"draft": draft}

    async def reflect_node(state: ResearchState) -> dict:
        verdict = await model.with_structured_output(Verdict).ainvoke([
            SystemMessage("You are a skeptical desk head reviewing a junior's research "
                          "note. It is sufficient only if every claim is backed by a "
                          "retrieved number, or is honestly marked as unavailable."),
            HumanMessage(f"Question: {state['question']}\n\nDraft:\n{state['draft']}"),
        ])
        return {"critique": "" if verdict.sufficient else verdict.critique,
                "revisions": state.get("revisions", 0) + 1}

    def route_reflection(state: ResearchState) -> Literal["revise", "finalize"]:
        if state.get("critique") and state.get("revisions", 0) <= MAX_REVISIONS:
            return "revise"
        return "finalize"

    async def revise_node(state: ResearchState) -> dict:
        return {"messages": [HumanMessage(
            "Reviewer rejected the draft. Address this critique, gathering any "
            f"missing data with tools before re-answering:\n{state['critique']}")]}

    async def finalize(state: ResearchState) -> dict:
        return {"report": state["draft"]}

    return (
        StateGraph(ResearchState)
        .add_node("plan", plan_node)
        .add_node("researcher", researcher)
        .add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
        .add_node("draft", draft_node)
        .add_node("reflect", reflect_node)
        .add_node("revise", revise_node)
        .add_node("finalize", finalize)
        .add_edge(START, "plan")
        .add_edge("plan", "researcher")
        .add_conditional_edges("researcher", route_research, ["tools", "draft"])
        .add_edge("tools", "researcher")
        .add_edge("draft", "reflect")
        .add_conditional_edges("reflect", route_reflection,
                               {"revise": "revise", "finalize": "finalize"})
        .add_edge("revise", "researcher")
        .add_edge("finalize", END)
        .compile(checkpointer=InMemorySaver())
    )


def make_input(question: str) -> dict:
    return {"question": question, "messages": [], "revisions": 0}


def extract(result: dict) -> str:
    return result.get("report") or result.get("draft", "")
