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
                "gotcha": "No model here: a plain HTTP fetch as its own node, so data entry is visible in the trace with its own timing.",
                "q": ["Desk down? Context stays empty and the prompt says so. No invented numbers."],
            },
        },
        {
            "id": "brief", "label": "Brief", "role": "structured answer",
            "quiet": True,
            "xray": {
                "gotcha": "with_structured_output forces a Pydantic schema: typed fields out, not prose to parse. The model becomes an API.",
                "q": ["Why a confidence field? A forced field can be charted; confidence in prose disappears."],
            },
        },
    ],
    "edges": [{"from": "context", "to": "brief"}],
    "state": [
        {"key": "question", "kind": "overwrite", "note": "the input"},
        {"key": "context",  "kind": "overwrite", "note": "desk data, or empty"},
        {"key": "brief",    "kind": "overwrite", "note": "validated Pydantic dict"},
        {"key": "answer",   "kind": "overwrite", "note": "the answer field"},
    ],
    "framework": [
        {"api": "StateGraph, linear", "note": "two nodes, one edge"},
        {"api": "with_structured_output(Brief)", "note": "typed answer, validated"},
        {"api": "InMemorySaver", "note": "checkpoint per superstep"},
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
