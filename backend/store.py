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

import json
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

-- Every run that crossed the stage (autopilot or visitor), stored whole:
-- the frame list is the run, so the UI can replay any of them exactly and
-- the landing page always has something real to show. Real runs only, no
-- synthetic rows in this table ever.
CREATE TABLE IF NOT EXISTS stage_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,        -- ISO-8601 UTC
    agent_id TEXT NOT NULL,
    source TEXT NOT NULL,            -- 'autopilot' | 'visitor' | 'replay'
    title TEXT NOT NULL,             -- scenario title, or '' for visitor runs
    question TEXT NOT NULL,
    outcome TEXT NOT NULL,           -- 'ok' | 'error' | 'preempted' | 'interrupt'
    latency_ms INTEGER NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    tool_calls INTEGER NOT NULL,
    thread_id TEXT NOT NULL,
    trace_url TEXT NOT NULL,
    frames TEXT NOT NULL             -- full frame list, JSON
);

-- Trade decisions POSTed by the desk (Marcus, the options voice agent).
-- The raw record is kept whole in `payload` so the UI can replay the run
-- exactly as it happened; the other columns are just the list view.
CREATE TABLE IF NOT EXISTS desk_runs (
    id INTEGER PRIMARY KEY,
    received_at TEXT NOT NULL,       -- ISO-8601 UTC, when it landed here
    agent TEXT NOT NULL,
    ticker TEXT NOT NULL,
    outcome TEXT NOT NULL,
    verdict TEXT NOT NULL,
    armed INTEGER NOT NULL,          -- 0 | 1
    latency_ms INTEGER NOT NULL,
    order_line TEXT NOT NULL,
    payload TEXT NOT NULL            -- the full run record, JSON
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


# ------------------------------------------------------------ stage runs ----

# Rolling window, same reasoning as desk_runs: the newest runs are the ones
# people watch, and an unattended public box must not grow a disk problem.
STAGE_RUNS_CAP = 400

_STAGE_SUMMARY_COLS = ("id", "started_at", "agent_id", "source", "title",
                       "question", "outcome", "latency_ms", "tokens_in",
                       "tokens_out", "cost_usd", "tool_calls", "thread_id",
                       "trace_url")


def save_stage_run(meta: dict, frames: list) -> int:
    """Store one finished stage run whole and return its id."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO stage_runs (started_at, agent_id, source, title,"
            " question, outcome, latency_ms, tokens_in, tokens_out, cost_usd,"
            " tool_calls, thread_id, trace_url, frames)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(meta.get("started_at", "")),
                str(meta.get("agent_id", "")),
                str(meta.get("source", "visitor")),
                str(meta.get("title", ""))[:200],
                str(meta.get("question", ""))[:500],
                str(meta.get("outcome", "ok")),
                int(meta.get("latency_ms") or 0),
                int(meta.get("tokens_in") or 0),
                int(meta.get("tokens_out") or 0),
                float(meta.get("cost_usd") or 0.0),
                int(meta.get("tool_calls") or 0),
                str(meta.get("thread_id", "")),
                str(meta.get("trace_url", "")),
                json.dumps(frames, separators=(",", ":"), default=str),
            ),
        )
        run_id = cur.lastrowid
        conn.execute(
            "DELETE FROM stage_runs WHERE id NOT IN"
            " (SELECT id FROM stage_runs ORDER BY id DESC LIMIT ?)",
            (STAGE_RUNS_CAP,),
        )
        conn.commit()
        return run_id
    finally:
        conn.close()


def set_stage_trace(run_id: int, trace_url: str) -> None:
    """The LangSmith URL arrives a moment after the run is saved."""
    conn = get_connection()
    try:
        conn.execute("UPDATE stage_runs SET trace_url = ? WHERE id = ?",
                     (trace_url, run_id))
        conn.commit()
    finally:
        conn.close()


def list_stage_runs(limit: int = 30, agent_id: str | None = None) -> list[dict]:
    """Newest-first summaries, no frames: the list is light, a click loads one.

    agent_id narrows to one agent: that is how a tab click finds the run it
    should replay for exactly the graph on screen.
    """
    limit = max(1, min(int(limit), 200))
    where = " WHERE agent_id = ?" if agent_id else ""
    params: tuple = (agent_id, limit) if agent_id else (limit,)
    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT {', '.join(_STAGE_SUMMARY_COLS)} FROM stage_runs{where}"
            " ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
    finally:
        conn.close()
    return [dict(zip(_STAGE_SUMMARY_COLS, r)) for r in rows]


def get_stage_run(run_id: int) -> dict | None:
    """One archived run with its full frame list, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT {', '.join(_STAGE_SUMMARY_COLS)}, frames FROM stage_runs"
            " WHERE id = ?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    out = dict(zip(_STAGE_SUMMARY_COLS, row[:-1]))
    try:
        out["frames"] = json.loads(row[-1])
    except json.JSONDecodeError:
        out["frames"] = []
    return out


def latest_stage_run() -> dict | None:
    """The newest archived run, frames included, for the landing replay."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM stage_runs ORDER BY id DESC LIMIT 1"
                           ).fetchone()
    finally:
        conn.close()
    return get_stage_run(row[0]) if row else None


def stage_stats() -> dict:
    """Real aggregates for the landing strip, all from actual stage runs."""
    conn = get_connection()
    try:
        today = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(cost_usd),0), COALESCE(SUM(tool_calls),0),"
            " COALESCE(SUM(tokens_in+tokens_out),0)"
            " FROM stage_runs WHERE started_at >= date('now')",
        ).fetchone()
        total = conn.execute("SELECT COUNT(*) FROM stage_runs").fetchone()
        med = conn.execute(
            "SELECT latency_ms FROM stage_runs WHERE outcome='ok'"
            " ORDER BY latency_ms LIMIT 1 OFFSET"
            " (SELECT COUNT(*) FROM stage_runs WHERE outcome='ok') / 2",
        ).fetchone()
    finally:
        conn.close()
    return {
        "runs_today": int(today[0]),
        "cost_today_usd": round(float(today[1]), 4),
        "tool_calls_today": int(today[2]),
        "tokens_today": int(today[3]),
        "runs_archived": int(total[0]),
        "median_latency_ms": int(med[0]) if med else 0,
    }


# ------------------------------------------------------------- desk runs ----

# The table is a rolling window, not an archive: the newest runs are the ones
# anybody replays, and an unattended public box must not grow a disk problem.
DESK_RUNS_CAP = 2000

_DESK_SUMMARY_COLS = ("id", "received_at", "agent", "ticker", "outcome",
                      "verdict", "armed", "latency_ms", "order_line")


def save_desk_run(record: dict) -> int:
    """Store one desk decision whole, prune past the cap, return its id."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO desk_runs (received_at, agent, ticker, outcome, verdict,"
            " armed, latency_ms, order_line, payload) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                str(record.get("agent", "")),
                str(record.get("ticker", "")),
                str(record.get("outcome", "")),
                str(record.get("verdict", "")),
                1 if record.get("armed") else 0,
                int(record.get("latency_ms") or 0),
                str(record.get("order", ""))[:500],
                json.dumps(record, separators=(",", ":"), default=str),
            ),
        )
        run_id = cur.lastrowid
        conn.execute(
            "DELETE FROM desk_runs WHERE id NOT IN"
            " (SELECT id FROM desk_runs ORDER BY id DESC LIMIT ?)",
            (DESK_RUNS_CAP,),
        )
        conn.commit()
        return run_id
    finally:
        conn.close()


def list_desk_runs(limit: int = 50) -> list[dict]:
    """Newest-first summaries for the run list. No payload: the list refreshes
    every 20 seconds and only a clicked run needs the full record."""
    limit = max(1, min(int(limit), 200))
    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT {', '.join(_DESK_SUMMARY_COLS)} FROM desk_runs"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    out = [dict(zip(_DESK_SUMMARY_COLS, r)) for r in rows]
    for r in out:
        r["armed"] = bool(r["armed"])
    return out


def get_desk_run(run_id: int) -> dict | None:
    """One run with the full stored record parsed back out, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT {', '.join(_DESK_SUMMARY_COLS)}, payload FROM desk_runs"
            " WHERE id = ?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    out = dict(zip(_DESK_SUMMARY_COLS, row[:-1]))
    out["armed"] = bool(out["armed"])
    try:
        out["record"] = json.loads(row[-1])
    except json.JSONDecodeError:
        out["record"] = {}
    return out
