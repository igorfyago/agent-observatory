"""Agent 1 · Brief: the simplest possible agent.

Forked from ai-trading-desk agents/01_market_brief on 2026-07-18.

Level: one model call, zero tools, zero loops. The only trick is STRUCTURED
OUTPUT: instead of free text the model is forced into a Pydantic schema, so
downstream code can consume the result programmatically.

Desk coupling: the original took a `context` string of live desk data injected
by the desk's web layer. Here that context comes over HTTP from the desk
(backend/desk_client.py) when the question mentions a covered ticker, and is
honestly marked unavailable when the desk is offline.
"""
from __future__ import annotations

import re

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import desk_client
from llm import get_model

SPEC = {
    "nodes": [
        {
            "id": "context", "label": "Context", "role": "pulls desk data",
            "xray": {
                "concept": (
                    "A deterministic data node: no model, just an HTTP call "
                    "to the trading desk when the question names a covered "
                    "ticker. Keeping data fetch out of the model call makes "
                    "the failure mode honest: no data means the state says so."),
                "here": (
                    "Writes `context` with the desk snapshot, or an empty "
                    "string when the desk is offline or no ticker matched. "
                    "The model is told what it does not have."),
                "tradeoffs": (
                    "A fixed fetch is cheaper and simpler than giving the "
                    "model a fetch tool, but it cannot decide to pull more. "
                    "For a one-shot brief that is the right trade; the "
                    "Research agent makes the opposite one."),
                "questions": [
                    "Why is this a graph node at all? So the fetch is visible in the trace with its own timing, and so the picture shows where data enters the run.",
                    "What if the desk is down? The context is empty, the prompt says data is unavailable, and the answer is expected to say so rather than invent numbers.",
                ],
            },
        },
        {
            "id": "brief", "label": "Brief", "role": "structured answer",
            "xray": {
                "concept": (
                    "Structured output: with_structured_output binds a "
                    "Pydantic schema, so the model must return typed fields "
                    "(tickers, intent, confidence), not prose that needs "
                    "parsing."),
                "here": (
                    "One model call returning the Brief schema. The schema is "
                    "the contract: downstream code reads fields, never "
                    "regexes."),
                "tradeoffs": (
                    "A schema constrains style and can clip nuance, but it "
                    "turns the model into a dependable API. Free text is for "
                    "people; structures are for systems."),
                "questions": [
                    "How is the schema enforced? with_structured_output uses the provider's native structured mode, and Pydantic validates the result before the node returns.",
                    "Why does confidence exist in the schema? A field the model must fill is a self-report you can chart and alert on; prose confidence disappears into text.",
                    "When is one call the right architecture? When there is no tool to call and no loop to run: the simplest agent that can be correct should be the one deployed.",
                ],
            },
        },
    ],
    "edges": [{"from": "context", "to": "brief"}],
    "state": [
        {"key": "question", "kind": "overwrite", "note": "the input"},
        {"key": "context",  "kind": "overwrite", "note": "desk data, or empty and honest"},
        {"key": "brief",    "kind": "overwrite", "note": "the validated Pydantic dict"},
        {"key": "answer",   "kind": "overwrite", "note": "the answer field, for the feed"},
    ],
    "framework": [
        {"api": "StateGraph, linear",
         "note": "two nodes, one edge: the graph is small because the problem is"},
        {"api": "with_structured_output(Brief)",
         "note": "Pydantic schema out of the model, validated before the node returns"},
        {"api": "InMemorySaver checkpointer",
         "note": "even a two-node graph gets a checkpoint per superstep"},
    ],
}

COVERED = ("SPY", "QQQ", "IWM")


class Brief(BaseModel):
    """A natural-language question, parsed into structured fields."""

    tickers: list[str] = Field(description="Tickers mentioned or implied, uppercase")
    metrics: list[str] = Field(
        description="Quant concepts involved, e.g. GEX, gamma flip, IV, expected move, OI"
    )
    intent: str = Field(description="One of: positioning, pricing, risk, education, execution")
    horizon: str = Field(description="Time horizon, e.g. intraday, weekly opex, monthly")
    restated_question: str = Field(description="The question restated precisely in desk jargon")
    answer: str = Field(description="A concise, direct answer (3-5 sentences)")
    confidence: float = Field(ge=0, le=1, description="How confident the answer is")


SYSTEM = """You are a sell-side derivatives strategist. Parse the question and
answer it from first principles of dealer positioning (GEX/DEX mechanics, gamma
regimes, walls, charm/vanna flows). Be precise and quantitative where possible.
If the question needs live data you do not have, say what data you would check."""


class BriefState(TypedDict, total=False):
    question: str
    context: str
    answer: str
    brief: dict


def _tickers_in(text: str) -> list[str]:
    upper = text.upper()
    return [t for t in COVERED if re.search(rf"\b{t}\b", upper)]


async def context_node(state: BriefState) -> BriefState:
    """Deterministic data pull: no model involved."""
    found = _tickers_in(state["question"])
    if not found:
        return {"context": ""}
    blocks = [desk_client.positioning_snapshot(t) for t in found]
    return {"context": "\n\n".join(blocks)}


async def brief_node(state: BriefState) -> BriefState:
    system = SYSTEM
    if state.get("context"):
        system += "\n\nLive desk data you may use:\n" + state["context"]
    structured = get_model().with_structured_output(Brief)
    result = await structured.ainvoke(
        [{"role": "system", "content": system},
         {"role": "user", "content": state["question"]}]
    )
    return {"brief": result.model_dump(), "answer": result.answer}


def build():
    from langgraph.checkpoint.memory import InMemorySaver

    g = StateGraph(BriefState)
    g.add_node("context", context_node)
    g.add_node("brief", brief_node)
    g.add_edge(START, "context")
    g.add_edge("context", "brief")
    g.add_edge("brief", END)
    return g.compile(checkpointer=InMemorySaver())


def make_input(question: str) -> dict:
    return {"question": question}


def extract(result: dict) -> str:
    brief = result.get("brief") or {}
    if not brief:
        return result.get("answer", "")
    lines = [
        brief.get("answer", ""),
        "",
        f"tickers: {', '.join(brief.get('tickers') or []) or 'none'}",
        f"metrics: {', '.join(brief.get('metrics') or []) or 'none'}",
        f"intent: {brief.get('intent', '')} · horizon: {brief.get('horizon', '')}",
        f"confidence: {brief.get('confidence', 0):.2f}",
    ]
    return "\n".join(lines)
