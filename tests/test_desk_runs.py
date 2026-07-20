"""The desk ingest seam: Marcus's trade decisions landing in the observatory.

This is a write path on a public box, so most of what is pinned here is
refusal: no token configured means nobody writes, a wrong token writes
nothing, an oversized body never reaches the database, and the table is a
rolling window that cannot grow a disk problem.
"""

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))


RECORD = {
    "agent": "marcus",
    "started_at": "2026-07-20T14:00:00+00:00",
    "ticker": "SPY",
    "session": "voice",
    "latency_ms": 412,
    "outcome": "ok",
    "verdict": "partial",
    "armed": False,
    "order": "Nothing on yet. Watching the 750 wall.",
    "nodes": [
        {"node": "structure", "ms": 12, "note": "negative_gamma, score -40"},
        {"node": "tape", "ms": 180, "note": "No setup armed at 748.12"},
        {"node": "decide", "ms": 210, "note": "bearish momentum · board wait"},
        {"node": "order", "ms": 1, "note": "Nothing on yet."},
    ],
    "spec": {
        "nodes": [
            {"id": "structure", "label": "structure", "hint": "GEX regime · flip · walls"},
            {"id": "tape", "label": "tape", "hint": ""},
            {"id": "decide", "label": "decide", "hint": ""},
            {"id": "order", "label": "order", "hint": ""},
        ],
        "edges": [["structure", "tape"], ["tape", "decide"], ["decide", "order"]],
    },
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """The app against a throwaway data dir, with a known ingest token."""
    monkeypatch.setenv("OBS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OBS_INGEST_TOKEN", "sesame")
    from fastapi.testclient import TestClient

    import app as app_module

    return TestClient(app_module.app)


def post(client, record=RECORD, token="sesame"):
    headers = {"X-Desk-Token": token} if token is not None else {}
    return client.post("/api/desk/runs", json=record, headers=headers)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

def test_no_token_header_is_refused(client):
    assert post(client, token=None).status_code == 403


def test_wrong_token_is_refused(client):
    assert post(client, token="not-sesame").status_code == 403


def test_unset_server_token_fails_shut(client, monkeypatch):
    """No OBS_INGEST_TOKEN means no token opens the endpoint, ever. An unset
    var must not degrade into an open write path on a public box."""
    monkeypatch.delenv("OBS_INGEST_TOKEN")
    assert post(client, token="sesame").status_code == 403
    assert post(client, token="").status_code == 403


def test_empty_server_token_fails_shut(client, monkeypatch):
    monkeypatch.setenv("OBS_INGEST_TOKEN", "")
    assert post(client, token="").status_code == 403


def test_refused_posts_store_nothing(client):
    post(client, token="wrong")
    assert client.get("/api/desk/runs").json()["runs"] == []


# --------------------------------------------------------------------------- #
# Ingest and read back
# --------------------------------------------------------------------------- #

def test_valid_record_is_stored_and_replayable(client):
    r = post(client)
    assert r.status_code == 200
    run_id = r.json()["id"]

    got = client.get(f"/api/desk/runs/{run_id}")
    assert got.status_code == 200
    body = got.json()
    assert body["ticker"] == "SPY"
    assert body["verdict"] == "partial"
    assert body["armed"] is False
    assert body["order_line"].startswith("Nothing on yet.")
    # the raw record comes back whole, spec included, so the UI can replay it
    assert body["record"]["nodes"][1]["note"] == "No setup armed at 748.12"
    assert body["record"]["spec"]["edges"][0] == ["structure", "tape"]


def test_list_is_newest_first_and_carries_no_payload(client):
    first = post(client, {**RECORD, "ticker": "SPY"}).json()["id"]
    second = post(client, {**RECORD, "ticker": "QQQ"}).json()["id"]

    runs = client.get("/api/desk/runs?limit=10").json()["runs"]
    assert [r["id"] for r in runs] == [second, first]
    assert runs[0]["ticker"] == "QQQ"
    for r in runs:
        assert "payload" not in r and "record" not in r and "nodes" not in r


def test_get_missing_run_is_404(client):
    assert client.get("/api/desk/runs/999999").status_code == 404


# --------------------------------------------------------------------------- #
# Malformed and oversized records
# --------------------------------------------------------------------------- #

def test_record_without_nodes_is_rejected(client):
    bad = {k: v for k, v in RECORD.items() if k != "nodes"}
    assert post(client, bad).status_code == 400


def test_record_with_blank_agent_is_rejected(client):
    assert post(client, {**RECORD, "agent": "  "}).status_code == 400


def test_oversized_record_is_rejected_before_parsing(client):
    fat = {**RECORD, "order": "x" * 40_000}
    assert post(client, fat).status_code == 413
    assert client.get("/api/desk/runs").json()["runs"] == []


def test_non_utf8_body_is_a_400_not_a_500(client):
    """Found live: a cp1252-encoded middle dot in the body blew up as a
    UnicodeDecodeError before json parsing. Encoding mistakes are the
    caller's problem and must come back as one."""
    r = client.post(
        "/api/desk/runs",
        content=b'{"agent": "marcus \xb7 desk"}',
        headers={"X-Desk-Token": "sesame", "Content-Type": "application/json"},
    )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# The rolling window
# --------------------------------------------------------------------------- #

def test_prune_keeps_only_the_newest(client, monkeypatch):
    import store

    monkeypatch.setattr(store, "DESK_RUNS_CAP", 5)
    ids = [post(client, {**RECORD, "ticker": f"T{i}"}).json()["id"] for i in range(8)]

    runs = client.get("/api/desk/runs?limit=50").json()["runs"]
    assert len(runs) == 5
    assert [r["id"] for r in runs] == list(reversed(ids[-5:]))
    # the oldest three are gone for good
    assert client.get(f"/api/desk/runs/{ids[0]}").status_code == 404


# --------------------------------------------------------------------------- #
# The registry seam
# --------------------------------------------------------------------------- #

def test_marcus_is_listed_without_breaking_the_other_agents(client):
    data = client.get("/api/agents").json()
    by_id = {a["id"]: a for a in data["agents"]}
    assert by_id["marcus"]["kind"] == "desk"
    # text agents keep their spec, voice agents their tools: the picker's contract
    assert "spec" in by_id["pipeline"]
    assert "tools" in by_id["riley"]
    assert data["default"] == "pipeline"
