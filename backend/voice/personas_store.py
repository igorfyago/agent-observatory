"""Custom voice personas: the editable layer of the agent builder.

Forked from ai-trading-desk web/personas_store.py on 2026-07-18.

A custom persona = label + voice + instructions + a selection of tools from
the SAFE ALLOWLIST below (existing, server-side, read-mostly tools harvested
from the built-in personas). Stored in the observatory DB, served next to the
built-ins, talk-to-able immediately.

Persistence: the desk version wrote through common/db. This one uses
backend/store.py, which puts the sqlite file in a real data dir (OBS_DATA_DIR,
default <repo>/data) so custom agents survive a restart or redeploy.

Creation is admin-token-gated: the builder UI is public to look at, but only
the owner can mint new agents on this server.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import store

from . import personas as builtin

# name -> (schema, impl) harvested from the built-in personas
_ALL_TOOLS: dict[str, tuple[dict, callable]] = {}
for _p in builtin.PERSONAS.values():
    for _t in _p["tools"]:
        _ALL_TOOLS[_t["name"]] = (_t, _p["implementations"][_t["name"]])

TOOL_ALLOWLIST = sorted(_ALL_TOOLS)
VOICES = ["marin", "cedar", "sage", "alloy", "ash", "ballad", "coral", "echo",
          "shimmer", "verse"]


def admin_ok(token: str | None) -> bool:
    """True when the caller may mint personas. If OBS_ADMIN_TOKEN is unset,
    creation is closed rather than open: fail shut, not open."""
    expected = os.getenv("OBS_ADMIN_TOKEN")
    return bool(expected) and token == expected


def create(label: str, tagline: str, voice: str, instructions: str,
           tools: list[str]) -> dict:
    pid = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:24]
    if not pid or pid in builtin.PERSONAS:
        raise ValueError("bad or reserved name")
    if voice not in VOICES:
        raise ValueError(f"voice must be one of {VOICES}")
    bad = [t for t in tools if t not in _ALL_TOOLS]
    if bad:
        raise ValueError(f"unknown tools: {bad}")
    if not (40 <= len(instructions) <= 6000):
        raise ValueError("instructions must be 40-6000 chars")
    conn = store.get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO custom_personas VALUES (?,?,?,?,?,?,?)",
        (pid, label[:60], tagline[:120], voice, instructions, ",".join(tools),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return {"id": pid, "label": label}


def list_customs() -> list[dict]:
    conn = store.get_connection()
    rows = conn.execute(
        "SELECT id, label, tagline FROM custom_personas ORDER BY created_at").fetchall()
    conn.close()
    return [{"id": r[0], "label": r[1], "tagline": r[2], "category": "custom"}
            for r in rows]


def resolve(pid: str) -> dict | None:
    """Return a persona dict (same shape as built-ins) for any id."""
    if pid in builtin.PERSONAS:
        return builtin.PERSONAS[pid]
    conn = store.get_connection()
    row = conn.execute(
        "SELECT label, tagline, voice, instructions, tools_csv FROM custom_personas"
        " WHERE id = ?", (pid,)).fetchone()
    conn.close()
    if not row:
        return None
    tool_names = [t for t in row[4].split(",") if t]
    return {
        "label": row[0], "tagline": row[1], "voice": row[2],
        "instructions": row[3] + "\n\n" + builtin.VOICE_STYLE,
        "tools": [_ALL_TOOLS[t][0] for t in tool_names],
        "implementations": {t: _ALL_TOOLS[t][1] for t in tool_names},
    }


def run_custom_tool(pid: str, name: str, arguments: dict) -> str:
    p = resolve(pid)
    impl = (p or {}).get("implementations", {}).get(name)
    if impl is None:
        return json.dumps({"error": f"unknown tool {name} for persona {pid}"})
    try:
        return impl(**arguments)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
