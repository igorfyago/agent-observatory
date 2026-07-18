"""Observatory persistence: one sqlite file in a real data dir.

This is the observatory's first durable state. Everything the hosted agents
write (voice bookings, saved quotes, custom personas built in the UI, and the
small demo table the SQL agent queries) lands here and survives a restart.

Data dir resolution, in order:
  1. OBS_DATA_DIR env var (set this to a mounted volume in Docker)
  2. <repo root>/data      (local dev)

The schema is created on demand and is additive only, so an existing file
keeps its rows across deploys.
"""
from __future__ import annotations

import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data"

SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    contact TEXT NOT NULL,
    service TEXT NOT NULL,
    slot TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quotes (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    customer TEXT NOT NULL,
    contact TEXT NOT NULL,
    project TEXT NOT NULL,
    low_usd REAL NOT NULL,
    high_usd REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS custom_personas (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    tagline TEXT NOT NULL,
    voice TEXT NOT NULL,
    instructions TEXT NOT NULL,
    tools_csv TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Demo table for the text-to-SQL agent. Deliberately NOT trading data: the
-- observatory hosts general agents, so the SQL agent reasons over its own
-- run log instead of the desk's options schema.
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,        -- ISO-8601 UTC
    agent_id TEXT NOT NULL,          -- which hosted agent ran
    kind TEXT NOT NULL,              -- 'text' | 'voice'
    question TEXT NOT NULL,
    tool_calls INTEGER NOT NULL,     -- how many tools it invoked
    tokens INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    outcome TEXT NOT NULL            -- 'ok' | 'error' | 'timeout'
);
"""


def data_dir() -> Path:
    d = Path(os.getenv("OBS_DATA_DIR", str(_DEFAULT_DIR))).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "observatory.db"


def get_connection() -> sqlite3.Connection:
    """Open the observatory DB, creating the schema and seed rows if needed."""
    fresh = not db_path().exists()
    conn = sqlite3.connect(db_path())
    conn.executescript(SCHEMA)
    conn.commit()
    if fresh or conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0] == 0:
        _seed_runs(conn)
    return conn


def run_readonly(sql: str) -> list[tuple]:
    """Execute one SELECT and return rows. Callers do their own validation."""
    conn = get_connection()
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def describe_schema() -> str:
    """CREATE TABLE statements for every table, for the SQL agent to read."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(r[0] for r in rows)


_AGENTS = ["brief", "sql", "repo", "research", "analyst", "riley", "quinn"]
_QUESTIONS = [
    "summarise the tool loop", "how does the critic gate work",
    "which agent is slowest", "book a cleaning for Tuesday",
    "quote a 150 sqft kitchen", "explain the revise edge",
    "what does the supervisor delegate", "count runs per agent",
]


def _seed_runs(conn: sqlite3.Connection) -> None:
    """Deterministic synthetic run log, so the SQL agent has something real to
    query on a fresh install with zero setup."""
    rng = random.Random(20260718)
    start = datetime.now(timezone.utc) - timedelta(days=30)
    rows = []
    for i in range(240):
        agent = rng.choice(_AGENTS)
        kind = "voice" if agent in ("riley", "quinn") else "text"
        calls = 0 if agent == "brief" else rng.randint(1, 9)
        outcome = rng.choices(["ok", "error", "timeout"], weights=[92, 6, 2])[0]
        rows.append((
            (start + timedelta(hours=i * 3, minutes=rng.randint(0, 59))).isoformat(),
            agent, kind, rng.choice(_QUESTIONS), calls,
            rng.randint(300, 6000),
            rng.randint(600, 45000),
            outcome,
        ))
    conn.executemany(
        "INSERT INTO agent_runs (started_at, agent_id, kind, question, tool_calls,"
        " tokens, latency_ms, outcome) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def log_run(agent_id: str, kind: str, question: str, tool_calls: int,
            tokens: int, latency_ms: int, outcome: str) -> None:
    """Record a real run. The SQL agent can then query the observatory's own
    history, which is a nice loop: an agent reasoning over agent telemetry."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO agent_runs (started_at, agent_id, kind, question, tool_calls,"
        " tokens, latency_ms, outcome) VALUES (?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), agent_id, kind, question[:500],
         tool_calls, tokens, latency_ms, outcome),
    )
    conn.commit()
    conn.close()
