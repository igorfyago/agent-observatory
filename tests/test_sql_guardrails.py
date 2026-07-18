"""The SQL agent's run_sql tool: the LLM is never trusted with write access.

Ported with the agent itself from the trading desk. The tool is the only path
from a model's output to the database, so the rejection rules below are the
whole security boundary, not a nicety.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents import sql as agent_sql  # noqa: E402
import store  # noqa: E402


def run(query: str) -> str:
    return agent_sql.run_sql.invoke({"sql": query})


def test_select_allowed():
    out = run("SELECT agent_id, COUNT(*) FROM agent_runs GROUP BY agent_id")
    assert out.startswith("OK")


def test_cte_allowed():
    out = run("WITH t AS (SELECT agent_id FROM agent_runs) SELECT COUNT(*) FROM t")
    assert out.startswith("OK")


@pytest.mark.parametrize("evil", [
    "INSERT INTO agent_runs (agent_id) VALUES ('HACK')",
    "DELETE FROM agent_runs",
    "DROP TABLE agent_runs",
    "UPDATE agent_runs SET outcome = 'ok'",
    "CREATE TABLE pwned (x INT)",
    "ALTER TABLE agent_runs ADD COLUMN pwned INT",
])
def test_dml_and_ddl_rejected(evil):
    assert run(evil).startswith("REJECTED")


def test_forbidden_keyword_inside_select_rejected():
    assert run("SELECT 1; DROP TABLE agent_runs").startswith("REJECTED")


def test_multi_statement_rejected():
    assert run("SELECT 1; SELECT 2").startswith("REJECTED")


def test_bad_sql_returns_error_for_self_correction():
    """A broken query must come back as text the model can read and retry on,
    never as an exception that kills the run."""
    out = run("SELECT no_such_column FROM agent_runs")
    assert out.startswith("SQL ERROR")


def test_row_cap_truncates_instead_of_dumping_the_table():
    out = run("SELECT * FROM agent_runs")
    assert "truncated" in out


def test_no_write_actually_landed():
    """Belt and braces: after every rejection above, prove the table is clean."""
    rows = store.run_readonly("SELECT COUNT(*) FROM agent_runs WHERE agent_id = 'HACK'")
    assert rows[0][0] == 0
    names = [r[0] for r in store.run_readonly(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    assert "pwned" not in names
