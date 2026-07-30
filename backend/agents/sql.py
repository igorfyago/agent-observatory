"""Agent 2 · SQL: text to SQL over the observatory's own run log.

Forked from ai-trading-desk agents/02_text_to_sql on 2026-07-18.

Level: first real AGENT, a tool loop. The model reads the schema, writes SQL,
runs it, reads the result (or the error), and iterates. Failed SQL is not a
crash: the error text goes back as a tool result and the model self-corrects.
That feedback loop is the core agent idea, and it is the most watchable thing
in the observatory because every iteration is a visible tool call.

Desk coupling: the original queried the desk's options-flow schema through
common/db. That schema did NOT come along. This agent now queries the
observatory's own `agent_runs` table (backend/store.py): a log of which hosted
agent ran, how many tools it called, how long it took. An agent reasoning over
agent telemetry, which is on-theme and needs no trading data at all.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool

import store
from llm import get_model

SPEC = {
    "nodes": [
        {
            "id": "model", "label": "Model", "role": "writes SQL",
            "xray": {
                "concept": (
                    "The agent half of the classic tool loop that "
                    "create_agent builds: the model decides, on every pass, "
                    "whether to call a tool or answer. The loop in the "
                    "picture is the prebuilt conditional edge doing that "
                    "routing."),
                "here": (
                    "Reads the schema first (that rule lives in the system "
                    "prompt), writes one SELECT at a time, and reads results "
                    "or errors off the message history to decide the next "
                    "move."),
                "tradeoffs": (
                    "A loop the model steers is flexible but unbounded by "
                    "nature, so the run budget is enforced outside the "
                    "prompt: a recursion limit on the graph and a spend guard "
                    "around the whole run."),
                "questions": [
                    "What actually stops a runaway loop? The config's recursion_limit fails the run before the meter runs away, and the spend guard prices every call it makes on the way.",
                    "Why does the model see raw SQL errors? The error text IS the teaching signal: the model reads it as a ToolMessage and repairs its own query on the next pass, live on this page most days.",
                    "Would a fixed write-then-run chain be safer? Safer and dumber: it cannot recover from its own mistakes, and text-to-SQL mistakes are routine, so self-repair earns its loop.",
                ],
            },
        },
        {
            "id": "tools", "label": "Tools", "role": "schema + query",
            "xray": {
                "concept": (
                    "The tool half of the loop. Tools are plain functions "
                    "under the @tool decorator; their docstrings are the "
                    "interface the model reasons over."),
                "here": (
                    "get_schema returns CREATE TABLE text; run_sql enforces "
                    "SELECT-only, one statement, no DDL keywords, then runs "
                    "against the observatory's own telemetry: the run log "
                    "this very run is being written into."),
                "tradeoffs": (
                    "Allowlisting SELECT is coarse but auditable in one "
                    "screen of code. A SQL parser would be tighter and "
                    "heavier; for a read-only sqlite file, coarse and "
                    "readable wins."),
                "questions": [
                    "What is the blast radius of a hostile query? Read-only checks up front and a sqlite file with telemetry rows: worst case is a slow SELECT, and rows are truncated on the way back.",
                    "Why query the observatory's own runs? Because the data is real and always fresh here: the agent answers questions about the agents around it, including itself.",
                ],
            },
        },
    ],
    "edges": [
        {"from": "model", "to": "tools", "label": "call"},
        {"from": "tools", "to": "model", "kind": "loop", "label": "result"},
        {"from": "model", "to": "end", "label": "answer"},
    ],
    "state": [
        {"key": "messages", "kind": "append",
         "note": "the whole loop is one message list with an add reducer: model turns, tool results, repeat"},
    ],
    "framework": [
        {"api": "create_agent(model, tools)",
         "note": "LangChain's prebuilt agent loop, compiled down to a LangGraph under the hood"},
        {"api": "@tool with docstrings",
         "note": "the docstring is the tool's API: it is what the model reads when choosing"},
        {"api": "ToolMessage error feedback",
         "note": "failed SQL returns as a normal tool result, so self-repair is just the next loop pass"},
        {"api": "checkpointer=InMemorySaver()",
         "note": "create_agent accepts a checkpointer like any graph: the loop's supersteps land in the strip"},
    ],
}

FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create",
             "attach", "pragma", "vacuum")
MAX_ROWS = 50


@tool
def get_schema() -> str:
    """Return the CREATE TABLE statements for every table in the observatory
    database. Always call this before writing any SQL."""
    return store.describe_schema()


@tool
def run_sql(sql: str) -> str:
    """Run one read-only SQL SELECT and return the rows as text.

    Rules: SELECT-only (no DML/DDL), single statement. If the query errors,
    the error message is returned so you can fix the SQL and retry.

    Args:
        sql: A single SELECT statement. Add LIMIT yourself for big scans.
    """
    lowered = sql.strip().lower()
    if not lowered.startswith(("select", "with")):
        return "REJECTED: only SELECT queries are allowed."
    if any(f" {w} " in f" {lowered} " or lowered.startswith(w) for w in FORBIDDEN):
        return "REJECTED: query contains a forbidden keyword (read-only access)."
    if ";" in sql.strip().rstrip(";"):
        return "REJECTED: one statement at a time."
    try:
        rows = store.run_readonly(sql)
    except Exception as exc:  # error goes back to the model so it can self-correct
        return f"SQL ERROR: {exc}"
    if not rows:
        return "OK: query ran, 0 rows."
    shown = rows[:MAX_ROWS]
    body = "\n".join(str(r) for r in shown)
    extra = len(rows) - MAX_ROWS
    suffix = f"\n... ({extra} more rows truncated)" if extra > 0 else ""
    return f"OK ({len(rows)} rows):\n{body}{suffix}"


SYSTEM = """You are a data analyst for an agent-hosting platform. Answer
questions by querying the observatory's own telemetry database.

Method, follow it strictly:
1. Call get_schema first. Never guess column names.
2. Write ONE focused SELECT at a time; prefer aggregates over dumping rows.
3. If a query errors, read the error and fix your SQL. Do not apologize, retry.
4. When you have the numbers, answer in plain language and INCLUDE the final
   SQL you used so the human can verify it.

Notes: agent_runs.kind is 'text' or 'voice'. outcome is 'ok', 'error' or
'timeout'. started_at is an ISO-8601 UTC string, so use substr() or date()
for grouping by day. latency_ms is milliseconds."""

TOOLS = [get_schema, run_sql]


def build():
    from langgraph.checkpoint.memory import InMemorySaver

    return create_agent(model=get_model(), tools=TOOLS, system_prompt=SYSTEM,
                        checkpointer=InMemorySaver())


def make_input(question: str) -> dict:
    return {"messages": [{"role": "user", "content": question}]}


def extract(result: dict) -> str:
    return result["messages"][-1].content
