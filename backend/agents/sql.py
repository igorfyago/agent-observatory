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
                "gotcha": "Failed SQL comes back as a ToolMessage, so the model reads its own error and repairs the query next pass. Watch for it.",
                "q": ["Runaway loop? recursion_limit kills the run, the spend guard prices every pass."],
            },
        },
        {
            "id": "tools", "label": "Tools", "role": "schema + query",
            "xray": {
                "gotcha": "It queries the observatory's own run log, the same table this very run is being written into.",
                "q": ["Hostile SQL? SELECT-only allowlist, one statement, rows truncated. Worst case: a slow read."],
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
         "note": "one list, add reducer: model turns + tool results"},
    ],
    "framework": [
        {"api": "create_agent(model, tools)", "note": "the prebuilt tool loop"},
        {"api": "@tool docstrings", "note": "the docstring IS the tool's API"},
        {"api": "ToolMessage errors", "note": "self-repair is the next pass"},
        {"api": "InMemorySaver", "note": "checkpoint per superstep"},
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
4. When you have the numbers, answer in plain language: one short paragraph,
   then the final SQL in one fenced code block. Nothing else.

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
