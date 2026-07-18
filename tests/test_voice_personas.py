"""The agency voice personas and their server-side tools.

Riley and Quinn came here from the trading desk, which is now a trading app
only. Their tools are the ones that actually write rows, so they are worth
pinning: a booking that silently fails is worse than one that refuses.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "voice"))

from voice import personas  # noqa: E402


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Point the persona tools at a throwaway database with the real schema."""
    monkeypatch.setenv("OBS_DATA_DIR", str(tmp_path))
    import store

    store.get_connection().close()          # create the schema
    conn = sqlite3.connect(store.db_path())
    yield conn
    conn.close()


def test_every_declared_tool_has_an_implementation():
    for pid, p in personas.PERSONAS.items():
        declared = {t["name"] for t in p["tools"]}
        assert declared == set(p["implementations"]), f"{pid} tool/impl mismatch"


def test_tool_schemas_are_valid_function_declarations():
    for p in personas.PERSONAS.values():
        for t in p["tools"]:
            assert t["type"] == "function" and t["name"] and t["description"]
            assert t["parameters"]["type"] == "object"
            for req in t["parameters"].get("required", []):
                assert req in t["parameters"]["properties"]


def test_the_agency_personas_are_hosted_here():
    assert {"riley", "quinn"} <= set(personas.PERSONAS)


def test_clinic_openings_deterministic():
    a = personas.clinic_openings("Tuesday")
    b = personas.clinic_openings("Tuesday")
    assert a == b
    assert json.loads(a)["open_slots"]


def test_book_appointment_writes_row(db):
    out = json.loads(personas.book_appointment("Test Pat", "p@x.com", "cleaning", "Tue 9:00"))
    assert out["status"] == "booked"
    n = db.execute("SELECT COUNT(*) FROM appointments WHERE patient_name='Test Pat'").fetchone()[0]
    assert n == 1


def test_book_appointment_rejects_unknown_service(db):
    out = json.loads(personas.book_appointment("X", "x@x.com", "surgery", "Tue 9:00"))
    assert "error" in out
    n = db.execute("SELECT COUNT(*) FROM appointments WHERE patient_name='X'").fetchone()[0]
    assert n == 0                      # a refusal must not half-write


def test_estimate_project_math():
    out = json.loads(personas.estimate_project("kitchen", 150, "premium"))
    assert out["estimate_low_usd"] == round(95 * 150 * 1.45, -2)
    assert out["estimate_high_usd"] > out["estimate_low_usd"]
