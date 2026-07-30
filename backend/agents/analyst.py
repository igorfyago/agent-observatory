"""Agent 5 · Analyst: the flagship graph. Everything at once.

Forked from ai-trading-desk agents/05_desk_analyst on 2026-07-18.

  * a deterministic data node (fetch), no model involved
  * a REGIME ROUTER: a different playbook branch per gamma regime
  * PARALLEL FAN-OUT: three specialist sub-agents (positioning, flow, risk)
    run concurrently, each with its own toolbelt, joined before synthesis
  * a SELF-CRITIQUE LOOP: a risk-manager critic can bounce the memo back
  * HUMAN-IN-THE-LOOP: interrupt() pauses the graph before publishing
  * a CHECKPOINTER, so the state survives the pause

Desk coupling, three separate decisions:

1. fetch() used common/market.latest_snapshot + common/news in-process. It now
   makes ONE HTTP call to the desk (backend/desk_client.summary) for snapshot,
   walls and headlines. When the desk is unreachable the router takes a third
   branch, no_data_playbook, which tells the specialists to reason from first
   principles and state plainly that live data is missing. It never fabricates.

2. option_quote/sigma_move used common/market.black_scholes and
   .expected_move. That is textbook math, not desk state, so it is
   reimplemented locally below rather than importing the banned module.

3. publish() wrote a memo file next to the desk source. It now writes into the
   observatory data dir (backend/store.data_dir()), so memos survive restarts
   alongside the rest of the observatory's state.
"""
from __future__ import annotations

import json
import math
import operator
from datetime import datetime
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import desk_client
import store
from llm import get_model

MAX_CRITIQUE_ROUNDS = 2

SPEC = {
    "nodes": [
        {
            "id": "fetch", "label": "Fetch", "role": "desk data, no model",
            "xray": {
                "concept": (
                    "A Command router: the node returns Command(update=..., "
                    "goto=...), writing state AND choosing the branch in one "
                    "move. No model involved; the routing key is data."),
                "here": (
                    "One HTTP call to the trading desk. Positive gamma "
                    "regime routes to the long_gamma playbook, negative to "
                    "short_gamma, desk unreachable to no_data."),
                "tradeoffs": (
                    "Command couples a node to its destinations, which costs "
                    "some of add_conditional_edges' separation. When update "
                    "and route derive from the same fetch, one atomic return "
                    "is clearer than two coordinated functions."),
                "questions": [
                    "Why route on data instead of asking the model? The regime is a fact in the snapshot, not a judgment; spending tokens to rediscover a field would add latency and a failure mode.",
                    "What does the third branch buy? Honesty when the desk is down: the run continues, states plainly that live data is missing, and invents nothing.",
                ],
            },
        },
        {
            "id": "long_gamma", "label": "Long gamma", "role": "playbook",
            "xray": {
                "concept": (
                    "A playbook node: deterministic, writes the regime's "
                    "trading doctrine into state as text the specialists "
                    "must obey."),
                "here": "Dealers long gamma: mean reversion, ranges, pinning.",
                "tradeoffs": (
                    "Doctrine in state instead of in each prompt means one "
                    "place to edit and an audit trail of exactly what the "
                    "specialists were told this run."),
                "questions": [
                    "Why three nodes instead of one playbook function? So the DAG shows which doctrine fired: the lit branch is the routing decision, readable from across the room.",
                ],
            },
        },
        {
            "id": "short_gamma", "label": "Short gamma", "role": "playbook",
            "xray": {
                "concept": "Same mechanism as long_gamma, opposite doctrine.",
                "here": "Dealers short gamma: momentum, breakouts, wider stops.",
                "tradeoffs": "See long_gamma: one doctrine per node, visibly routed.",
                "questions": [
                    "Could the playbooks be a lookup instead of nodes? Functionally yes; visibly no. The branch in the picture is the point: routing you can see beats routing you must infer.",
                ],
            },
        },
        {
            "id": "no_data", "label": "No data", "role": "desk offline",
            "xray": {
                "concept": (
                    "The degraded-mode branch, designed on the same footing "
                    "as the happy paths instead of bolted on as error "
                    "handling."),
                "here": (
                    "Instructs every specialist to reason from first "
                    "principles and mark live positioning unavailable in "
                    "every bullet."),
                "tradeoffs": (
                    "A run without data still costs model calls. The "
                    "alternative, failing the run, teaches viewers nothing; "
                    "an honest degraded answer still demonstrates the "
                    "machine."),
                "questions": [
                    "Why run at all without data? Because the failure mode is the demonstration: the memo that says 'unavailable' in every bullet is proof the agents do not fabricate.",
                ],
            },
        },
        {
            "id": "positioning", "label": "Positioning", "role": "specialist",
            "xray": {
                "concept": (
                    "A sub-agent as a node: create_agent builds a full tool "
                    "loop (its own model turns, its own toolbelt), and the "
                    "outer graph runs it as one step. A graph inside a "
                    "graph."),
                "here": (
                    "Reads the GEX/DEX profile with SQL and sigma_move "
                    "tools; runs concurrently with flow and risk in the "
                    "same superstep."),
                "tradeoffs": (
                    "Sub-agents multiply model calls (three loops instead "
                    "of three prompts) and buy real division of labor: each "
                    "specialist iterates its own tools without stepping on "
                    "the others' context."),
                "questions": [
                    "How do three agents run at once? All three playbook edges land on all three specialists, so LangGraph schedules them in one superstep; the join waits for all.",
                    "How do their outputs merge without clobbering? The analyses channel is Annotated[list, operator.add]: a reducer appends concurrent writes instead of last-write-wins.",
                ],
            },
        },
        {
            "id": "flow", "label": "Flow", "role": "specialist",
            "xray": {
                "concept": "Second parallel sub-agent, same pattern as positioning.",
                "here": "Maps walls and OI concentration against the expected-move band.",
                "tradeoffs": "Same as positioning: paid in calls, repaid in focus.",
                "questions": [
                    "Why not one specialist with all the tools? Three narrow prompts beat one broad one on discipline: each agent's loop stays short, cheap and on-topic.",
                ],
            },
        },
        {
            "id": "risk", "label": "Risk", "role": "specialist",
            "xray": {
                "concept": "Third parallel sub-agent; the only one holding the pricing tool.",
                "here": "Prices candidate structures with Black-Scholes and stresses the thesis.",
                "tradeoffs": (
                    "Giving only this agent option_quote is deliberate tool "
                    "scoping: capability boundaries double as role "
                    "boundaries."),
                "questions": [
                    "Why does tool scoping matter? An agent's toolbelt is its permission set: narrow belts make behavior predictable and reviews of what-can-touch-what possible.",
                ],
            },
        },
        {
            "id": "synthesize", "label": "Synthesize", "role": "head of desk",
            "xray": {
                "concept": (
                    "The join and the decision point: waits for all three "
                    "specialists, then forces the merged view into a typed "
                    "Memo via with_structured_output."),
                "here": (
                    "Resolves disagreements explicitly; on a critique round "
                    "it rewrites against the risk manager's exact objection."),
                "tradeoffs": (
                    "A typed memo (bias, conviction, invalidation) is harder "
                    "to write than prose and infinitely easier to gate, "
                    "store and render. Structure at the boundary pays "
                    "everywhere downstream."),
                "questions": [
                    "What enforces the memo's shape? A Pydantic schema through with_structured_output: conviction is an int 1 to 10 by construction, not by hope.",
                    "How does the join behave if one specialist fails? The superstep fails loudly rather than synthesizing from two thirds of a desk; partial teams produce confident holes.",
                ],
            },
        },
        {
            "id": "risk_review", "label": "Risk review", "role": "critic gate",
            "xray": {
                "concept": (
                    "A machine critic ahead of the human: structured verdict, "
                    "conditional edge back to synthesize on rejection, "
                    "bounded rounds."),
                "here": (
                    "Rejects memos that contradict the playbook, ignore a "
                    "wall, or state figures absent from the analyses."),
                "tradeoffs": (
                    "Every rejection is a full synthesize pass. The bound "
                    "(two rounds) caps the bill; after that the human sees "
                    "the memo with the critique attached, which is its own "
                    "kind of honesty."),
                "questions": [
                    "Why gate before the human instead of after? Machine review is cheap and instant; spending it first means the human's attention, the scarcest input, only sees filtered work.",
                ],
            },
        },
        {
            "id": "human", "label": "Human", "role": "approval pause",
            "xray": {
                "concept": (
                    "interrupt(): the graph checkpoints itself and stops "
                    "cold. Resume arrives later as Command(resume=...), on a "
                    "different request, and the run continues as if no time "
                    "had passed. This is durable execution, not a sleep()."),
                "here": (
                    "Surfaces the memo and waits. Approve goes to publish; "
                    "revise re-enters synthesize with notes; reject ends the "
                    "run. On autopilot runs the observatory approves after a "
                    "countdown unless a viewer decides first."),
                "tradeoffs": (
                    "A pause needs a checkpointer and a resume channel, "
                    "which is real machinery. The alternative, side effects "
                    "without a gate, is how agents end up publishing what "
                    "nobody meant to ship."),
                "questions": [
                    "What exactly survives the pause? The whole thread state at the interrupt, held by the checkpointer under the thread id; resume rehydrates it and applies the decision.",
                    "Why does interrupt() come first in the node? Everything before an interrupt re-runs on resume; putting it first makes the node idempotent by construction.",
                    "What if two people resume the same pause? The stage tracks the pending interrupt and only the first decision is submitted; the second gets a clean 'nothing pending'.",
                ],
            },
        },
        {
            "id": "publish", "label": "Publish", "role": "writes the memo",
            "xray": {
                "concept": (
                    "The side effect, placed strictly after the approval "
                    "gate so it runs exactly once, never on a replayed "
                    "prefix."),
                "here": "Writes the memo JSON into the observatory's data dir.",
                "tradeoffs": (
                    "Files after gates is a discipline, not a framework "
                    "feature: LangGraph guarantees the ordering, the design "
                    "has to choose it."),
                "questions": [
                    "Why is the write not inside the human node? Resume re-runs the interrupting node's prefix; a write before the interrupt could execute twice. After the gate, it cannot.",
                ],
            },
        },
    ],
    "edges": [
        {"from": "fetch", "to": "long_gamma",  "label": "positive"},
        {"from": "fetch", "to": "short_gamma", "label": "negative"},
        {"from": "fetch", "to": "no_data",     "label": "offline"},
        *[{"from": pb, "to": sp}
          for pb in ("long_gamma", "short_gamma", "no_data")
          for sp in ("positioning", "flow", "risk")],
        {"from": "positioning", "to": "synthesize"},
        {"from": "flow",        "to": "synthesize"},
        {"from": "risk",        "to": "synthesize"},
        {"from": "synthesize",  "to": "risk_review"},
        {"from": "risk_review", "to": "synthesize", "kind": "loop", "label": "rejected"},
        {"from": "risk_review", "to": "human",      "label": "passed"},
        {"from": "human",       "to": "publish",    "label": "approve"},
        {"from": "human",       "to": "synthesize", "kind": "loop", "label": "revise"},
        {"from": "publish",     "to": "end"},
    ],
    "state": [
        {"key": "ticker",    "kind": "overwrite", "note": "the input"},
        {"key": "snapshot",  "kind": "overwrite", "note": "desk positioning, or empty offline"},
        {"key": "walls",     "kind": "overwrite", "note": "call/put walls from the desk"},
        {"key": "headlines", "kind": "overwrite", "note": "news context, clipped"},
        {"key": "playbook",  "kind": "overwrite", "note": "the doctrine the router picked"},
        {"key": "analyses",  "kind": "append",
         "note": "operator.add reducer: three concurrent specialists append without clobbering"},
        {"key": "memo",      "kind": "overwrite", "note": "the typed Memo, latest round"},
        {"key": "critique",  "kind": "overwrite", "note": "empty means the gate passed"},
        {"key": "rounds",    "kind": "overwrite", "note": "critique loop bound"},
        {"key": "published_path", "kind": "overwrite", "note": "set only after approval"},
    ],
    "framework": [
        {"api": "Command(update=..., goto=...)",
         "note": "the fetch node writes state and routes in one atomic return"},
        {"api": "parallel sub-agents via create_agent",
         "note": "three full tool-loop agents run as nodes in one superstep: graphs inside the graph"},
        {"api": "add_edge([...], 'synthesize') join",
         "note": "the list-edge is the barrier: synthesis waits for all three specialists"},
        {"api": "Annotated[list, operator.add]",
         "note": "the analyses channel merges concurrent writes by appending, the reducer idea in one line"},
        {"api": "with_structured_output(Memo / Review)",
         "note": "typed memo out of synthesis, typed verdict out of the gate"},
        {"api": "interrupt() + Command(resume=...)",
         "note": "the human pause: checkpointed stop, resumed on a later request with the decision"},
        {"api": "InMemorySaver checkpointer",
         "note": "what makes the pause durable and the checkpoint strip real"},
    ],
}


# ------------------------------------------------------------- toolbelt ----
# Black-Scholes reimplemented locally: textbook math, no desk coupling.

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black_scholes(spot: float, strike: float, dte_days: float, iv: float,
                   kind: str, rate: float = 0.045) -> dict:
    t = max(dte_days, 0.0) / 365.0
    if t <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        intrinsic = max(0.0, spot - strike) if kind == "call" else max(0.0, strike - spot)
        return {"price": round(intrinsic, 4), "delta": 0.0, "gamma": 0.0,
                "vega": 0.0, "theta": 0.0}
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    disc = math.exp(-rate * t)
    if kind.lower() == "call":
        price = spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta = (-spot * pdf * iv / (2 * math.sqrt(t))
                 - rate * strike * disc * _norm_cdf(d2))
    else:
        price = strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
        theta = (-spot * pdf * iv / (2 * math.sqrt(t))
                 + rate * strike * disc * _norm_cdf(-d2))
    return {
        "price": round(price, 4),
        "delta": round(delta, 4),
        "gamma": round(pdf / (spot * iv * math.sqrt(t)), 6),
        "vega": round(spot * pdf * math.sqrt(t) / 100.0, 4),
        "theta": round(theta / 365.0, 4),
    }


@tool
def option_quote(spot: float, strike: float, dte_days: float, iv: float, kind: str) -> str:
    """Black-Scholes price and greeks for one option leg.

    Args:
        spot: underlying price
        strike: option strike
        dte_days: days to expiry
        iv: implied volatility as a decimal (0.18 = 18 vol)
        kind: 'call' or 'put'
    """
    return str(_black_scholes(spot, strike, dte_days, iv, kind))


@tool
def sigma_move(spot: float, iv: float, dte_days: float) -> str:
    """One-sigma expected move in dollars for the given spot, IV and horizon."""
    move = spot * iv * math.sqrt(max(dte_days, 0.0) / 365.0)
    return str({"one_sigma_usd": round(move, 2),
                "low": round(spot - move, 2), "high": round(spot + move, 2)})


@tool
def sql_query(sql: str) -> str:
    """Read-only SELECT on the observatory telemetry DB. Table agent_runs
    (started_at, agent_id, kind, question, tool_calls, tokens, latency_ms,
    outcome) logs every agent run on this host."""
    if not sql.strip().lower().startswith(("select", "with")):
        return "REJECTED: SELECT only."
    try:
        rows = store.run_readonly(sql)[:40]
        return "\n".join(str(r) for r in rows) or "0 rows"
    except Exception as exc:
        return f"SQL ERROR: {exc}"


# ---------------------------------------------------------------- state ----

class DeskState(TypedDict, total=False):
    ticker: str
    snapshot: dict
    walls: dict
    headlines: str
    playbook: str
    analyses: Annotated[list[str], operator.add]
    memo: dict
    critique: str
    rounds: int
    published_path: str


class Memo(BaseModel):
    bias: Literal["bullish", "bearish", "neutral", "two-sided"]
    conviction: int = Field(ge=1, le=10)
    thesis: str = Field(description="2-3 sentence core view grounded in the data")
    trade_idea: str = Field(description="One concrete options structure with strikes/expiry")
    key_levels: list[str] = Field(description="Levels that matter and why")
    invalidation: str = Field(description="What would prove the thesis wrong")
    risks: list[str]


PLAYBOOKS = {
    "long_gamma": (
        "REGIME: dealers LONG gamma, hedging dampens moves. Default playbook: "
        "mean-reversion, range trades between walls, premium selling; pinning "
        "risk into expiry; breakouts need a catalyst to break the walls."
    ),
    "short_gamma": (
        "REGIME: dealers SHORT gamma, hedging amplifies moves. Default playbook: "
        "momentum and breakout bias, long premium, wider stops; watch the flip "
        "level, crossing it accelerates; avoid naked short options."
    ),
    "no_data": (
        "REGIME UNKNOWN: the desk feed is offline, so there is NO live snapshot, "
        "no wall map and no headlines. Reason from first principles only, state "
        "explicitly in every bullet that live positioning is unavailable, and do "
        "NOT invent spot, GEX, IV or wall figures."
    ),
}


# ---------------------------------------------------------------- graph ----

def build(checkpointer=None):
    model = get_model()

    def fetch(state: DeskState) -> Command[Literal["long_gamma", "short_gamma", "no_data"]]:
        """Deterministic data node plus router: no model involved."""
        data = desk_client.summary(state["ticker"])
        if data is None:
            return Command(update={"snapshot": {}, "walls": {}, "headlines": ""},
                           goto="no_data")
        snap = data.get("snapshot") or {}
        goto = "long_gamma" if snap.get("regime") == "positive_gamma" else "short_gamma"
        return Command(
            update={"snapshot": snap, "walls": data.get("walls") or {},
                    "headlines": json.dumps(data.get("headlines") or [], default=str)[:2000]},
            goto=goto,
        )

    def _playbook(key: str):
        def node(state: DeskState) -> dict:
            return {"playbook": PLAYBOOKS[key]}
        return node

    def _specialist(name: str, focus: str, tools: list):
        """Each specialist is a full sub-agent with its own tool loop, run as a node."""
        sub_agent = create_agent(
            model=model, tools=tools,
            system_prompt=(
                f"You are the {name} specialist on an options desk. {focus} "
                "Use your tools for any number you do not have. Be quantitative and "
                "terse: 5-8 bullet points, every bullet backed by a figure. If the "
                "playbook says live data is unavailable, say so instead of guessing."
            ),
        )

        async def node(state: DeskState) -> dict:
            result = await sub_agent.ainvoke(
                {"messages": [{"role": "user", "content": (
                    f"Ticker: {state['ticker']}\nSnapshot: {state.get('snapshot') or 'unavailable'}\n"
                    f"Walls: {state.get('walls') or 'unavailable'}\n"
                    f"Headlines: {state.get('headlines') or 'unavailable'}\n{state['playbook']}"
                )}]},
                config={"recursion_limit": 15},
            )
            return {"analyses": [f"## {name} analysis\n{result['messages'][-1].content}"]}

        return node

    async def synthesize(state: DeskState) -> dict:
        note = (
            f"\n\nA previous memo was rejected by the risk manager for:\n"
            f"{state['critique']}\nFix exactly that." if state.get("critique") else ""
        )
        memo = await model.with_structured_output(Memo).ainvoke([
            {"role": "system", "content":
                "You are the head of desk. Merge the specialists' work into one signal "
                "memo. Resolve disagreements explicitly, do not average them away."},
            {"role": "user", "content":
                f"Ticker: {state['ticker']}\nSnapshot: {state.get('snapshot')}\n\n"
                + "\n\n".join(state.get("analyses", [])[-3:]) + note},
        ])
        return {"memo": memo.model_dump()}

    async def risk_review(state: DeskState) -> dict:
        class Review(BaseModel):
            approved: bool
            critique: str = Field(description="If rejected: the specific unsupported claim")

        review = await model.with_structured_output(Review).ainvoke([
            {"role": "system", "content":
                "You are a hard-nosed risk manager. Reject the memo if the trade idea "
                "contradicts the regime playbook, ignores a wall or flip level, or "
                "states a figure not present in the analyses."},
            {"role": "user", "content":
                f"Playbook: {state['playbook']}\n\nAnalyses:\n"
                + "\n".join(state.get("analyses", [])[-3:])
                + f"\n\nMemo:\n{json.dumps(state['memo'], indent=1)}"},
        ])
        return {"critique": "" if review.approved else review.critique,
                "rounds": state.get("rounds", 0) + 1}

    def route_review(state: DeskState) -> Literal["synthesize", "human"]:
        if state.get("critique") and state.get("rounds", 0) <= MAX_CRITIQUE_ROUNDS:
            return "synthesize"
        return "human"

    def human(state: DeskState) -> Command[Literal["publish", "synthesize"]]:
        """Pause and surface the memo. interrupt() must come first: everything
        before it re-runs on resume."""
        decision = interrupt({
            "memo": state["memo"],
            "instructions": "Reply with {'action': 'approve' | 'revise' | 'reject', 'notes': '...'}",
        })
        action = (decision or {}).get("action", "reject")
        if action == "approve":
            return Command(update={}, goto="publish")
        if action == "revise":
            return Command(
                update={"critique": (decision or {}).get("notes", "Human requested changes"),
                        "rounds": 0},
                goto="synthesize")
        return Command(update={}, goto=END)

    def publish(state: DeskState) -> dict:
        """Side effect AFTER the interrupt: runs exactly once."""
        out = store.data_dir() / "memos"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = out / f"{state['ticker']}-{stamp}.json"
        path.write_text(json.dumps(state["memo"], indent=2))
        return {"published_path": str(path)}

    positioning = _specialist(
        "POSITIONING", "Read the GEX/DEX profile: regime strength, distance to the "
        "gamma flip, concentration versus history.", [sql_query, sigma_move])
    flow = _specialist(
        "FLOW & LEVELS", "Map the battlefield: call/put walls, OI concentration, "
        "expected-move band versus wall spacing.", [sql_query, sigma_move])
    risk = _specialist(
        "RISK", "Price the trade: what do candidate structures cost, what are the "
        "greeks, what kills the trade? Stress the thesis.",
        [option_quote, sigma_move, sql_query])

    builder = (
        StateGraph(DeskState)
        .add_node("fetch", fetch)
        .add_node("long_gamma", _playbook("long_gamma"))
        .add_node("short_gamma", _playbook("short_gamma"))
        .add_node("no_data", _playbook("no_data"))
        .add_node("positioning", positioning)
        .add_node("flow", flow)
        .add_node("risk", risk)
        .add_node("synthesize", synthesize)
        .add_node("risk_review", risk_review)
        .add_node("human", human)
        .add_node("publish", publish)
        .add_edge(START, "fetch")
    )
    for pb in ("long_gamma", "short_gamma", "no_data"):
        for sp in ("positioning", "flow", "risk"):
            builder.add_edge(pb, sp)
    return (
        builder
        .add_edge(["positioning", "flow", "risk"], "synthesize")   # join
        .add_edge("synthesize", "risk_review")
        .add_conditional_edges("risk_review", route_review, ["synthesize", "human"])
        .add_edge("publish", END)
        .compile(checkpointer=checkpointer or InMemorySaver())
    )


def make_input(question: str) -> dict:
    """The analyst takes a ticker, so pull the first plausible one out of the text."""
    words = [w.strip(".,?!").upper() for w in (question or "").split()]
    ticker = next((w for w in words if 1 <= len(w) <= 5 and w.isalpha()), "SPY")
    return {"ticker": ticker, "analyses": [], "rounds": 0}


def extract(result: dict) -> str:
    memo = result.get("memo")
    if not memo:
        return "No memo produced."
    out = json.dumps(memo, indent=2)
    if result.get("published_path"):
        out += f"\n\npublished: {result['published_path']}"
    return out
