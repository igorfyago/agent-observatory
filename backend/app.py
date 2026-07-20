"""Agent Observatory · FastAPI backend.

A multi-agent host: the registry (backend/registry.py) lists every hosted
agent, the runner (backend/runner.py) turns any of them into a live frame
stream, and the static UI draws the real graph topology plus every tool call
as it happens.

Adding an agent is one registry entry. Nothing here needs to change.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load project-root .env before anything reads an env var.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import registry  # noqa: E402  (needs env loaded first)
import runner  # noqa: E402
import spend  # noqa: E402
import store  # noqa: E402
from voice import personas, personas_store  # noqa: E402

app = FastAPI(title="Agent Observatory")


class ChatIn(BaseModel):
    question: str = ""
    agent: str = registry.DEFAULT_ID


class ResumeIn(BaseModel):
    agent: str
    thread: str
    action: str = "approve"       # approve | revise | reject
    notes: str = ""


class PersonaIn(BaseModel):
    label: str
    tagline: str = ""
    voice: str = "sage"
    instructions: str
    tools: list[str] = []


# --------------------------------------------------------------- registry ---

@app.get("/api/agents")
def list_agents():
    """Every hosted agent: id, label, kind, and for text agents the real graph
    topology the UI renders."""
    return {
        "agents": registry.as_json(),
        "custom": personas_store.list_customs(),
        "default": registry.DEFAULT_ID,
        "mode": runner.mode(),
    }


@app.get("/api/graph")
def graph_spec(agent: str = registry.DEFAULT_ID):
    """The real graph topology for one agent. The UI renders THIS, never a
    hand-drawn copy, so the picture cannot drift from the code."""
    a = registry.get(agent)
    if a is None or a.kind != "text":
        raise HTTPException(404, f"no text agent '{agent}'")
    return {"spec": a.spec, "mode": runner.mode()}


# ------------------------------------------------------------------ chat ----

def _ip(request: Request) -> str:
    """Client IP, trusting Caddy's X-Forwarded-For in front of us."""
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd
            else (request.client.host if request.client else "unknown"))


def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev, default=str)}\n\n"


def _stream_response(agent, question, thread=None, resume=None, ip=None):
    async def gen():
        # The spend guard wraps the WHOLE run, so its concurrency slot is
        # released even if the graph blows up mid-stream.
        try:
            with spend.guard(ip):
                async for frame in runner.stream(agent, question, thread=thread,
                                                 resume=resume):
                    yield _sse(frame)
        except spend.SpendRefused as refusal:
            yield _sse({"type": "error", "message": refusal.message,
                        **refusal.as_dict()})
        except asyncio.CancelledError:  # client hung up
            raise
        except Exception as exc:
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    """Run one agent and stream its nodes, tokens and tool calls as SSE."""
    agent = registry.get(body.agent)
    if agent is None or agent.kind != "text":
        raise HTTPException(404, f"no text agent '{body.agent}'")
    return _stream_response(agent, body.question, ip=_ip(request))


@app.post("/api/resume")
async def resume(body: ResumeIn, request: Request):
    """Resume an agent parked at a human-approval interrupt."""
    agent = registry.get(body.agent)
    if agent is None or not agent.interactive:
        raise HTTPException(404, f"agent '{body.agent}' does not pause for approval")
    return _stream_response(
        agent, "", thread=body.thread,
        resume={"action": body.action, "notes": body.notes},
        ip=_ip(request),
    )


# ----------------------------------------------------------------- voice ----

@app.get("/api/personas/{persona_id}")
def get_persona(persona_id: str):
    """Instructions, voice and tool schemas for one voice agent."""
    p = personas_store.resolve(persona_id)
    if p is None:
        raise HTTPException(404, f"no persona '{persona_id}'")
    return {"id": persona_id, "label": p["label"], "tagline": p.get("tagline", ""),
            "voice": p["voice"], "tools": p["tools"]}


@app.post("/api/personas")
def create_persona(body: PersonaIn, x_admin_token: str | None = Header(default=None)):
    """Mint a custom voice agent. Gated: fails shut when OBS_ADMIN_TOKEN is unset."""
    if not personas_store.admin_ok(x_admin_token):
        raise HTTPException(403, "persona creation is closed on this server")
    try:
        return personas_store.create(body.label, body.tagline, body.voice,
                                     body.instructions, body.tools)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/persona-tools")
def persona_tools():
    """The safe allowlist a custom persona may draw its tools from."""
    return {"tools": personas_store.TOOL_ALLOWLIST, "voices": personas_store.VOICES}


@app.post("/api/tool/{persona_id}")
async def run_persona_tool(persona_id: str, body: dict):
    """Server-side execution of a Realtime function call. Tools never run in
    the browser, so data and side effects stay on this host."""
    name = body.get("name", "")
    args = body.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if persona_id in personas.PERSONAS:
        return {"result": personas.run_tool(persona_id, name, args,
                                            body.get("session", "voice"))}
    return {"result": personas_store.run_custom_tool(persona_id, name, args)}


# ------------------------------------------------------------------ desk ----

# One run record is a handful of nodes and one order line; 32KB is generous.
# Anything bigger is not a run record.
MAX_DESK_RECORD_BYTES = 32 * 1024


@app.post("/api/desk/runs")
async def desk_run_ingest(request: Request,
                          x_desk_token: str | None = Header(default=None)):
    """One trade decision from the desk's graph, POSTed by ai-trading-desk.

    Write-gated and fails shut: with OBS_INGEST_TOKEN unset there is no token
    that opens it, because a public box must not carry an open write endpoint.
    The compose file hands the same token to both services.
    """
    expected = os.environ.get("OBS_INGEST_TOKEN", "")
    if not expected or x_desk_token != expected:
        raise HTTPException(403, "desk ingest is closed on this server")

    raw = await request.body()
    if len(raw) > MAX_DESK_RECORD_BYTES:
        raise HTTPException(413, f"run record over {MAX_DESK_RECORD_BYTES} bytes")
    try:
        record = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError too: a non-UTF-8 body is a caller mistake and
        # deserves a 400, not a stack trace (found live, a cp1252 middle dot).
        raise HTTPException(400, "body is not valid UTF-8 JSON") from exc
    if not isinstance(record, dict):
        raise HTTPException(400, "run record must be a JSON object")
    if (not str(record.get("agent") or "").strip()
            or not str(record.get("started_at") or "").strip()
            or not isinstance(record.get("nodes"), list)):
        raise HTTPException(400, "run record needs agent, started_at and a nodes list")

    return {"id": store.save_desk_run(record)}


@app.get("/api/desk/runs")
def desk_runs_list(limit: int = 50):
    """Newest-first summaries for the run list. Read-only, so it is open."""
    return {"runs": store.list_desk_runs(limit)}


@app.get("/api/desk/runs/{run_id}")
def desk_run_get(run_id: int):
    """The full stored record for one run, for the UI to replay."""
    run = store.get_desk_run(run_id)
    if run is None:
        raise HTTPException(404, f"no desk run {run_id}")
    return run


# ------------------------------------------------------------------ misc ----

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "agents": len(registry.AGENTS),
        "data_dir": str(store.data_dir()),
        "mode": runner.mode(),
    }


@app.get("/api/spend")
def spend_status():
    """Current meter reading: caps, spend so far and the one-word state.

    Read-only, so it stays outside the guard: a visitor who is being refused
    still needs to see why, and asking costs nothing.
    """
    return spend.status()


@app.post("/api/langflow")
async def run_langflow(body: ChatIn):
    """Optional bridge: run the same question through a Langflow-hosted flow.

    Prototype the flow visually in Langflow, then set LANGFLOW_URL,
    LANGFLOW_FLOW_ID and LANGFLOW_API_KEY in .env to wire it in.
    """
    url = os.environ.get("LANGFLOW_URL")
    flow_id = os.environ.get("LANGFLOW_FLOW_ID")
    if not url or not flow_id:
        raise HTTPException(
            status_code=503,
            detail="Langflow not configured, set LANGFLOW_URL and LANGFLOW_FLOW_ID in .env",
        )
    import httpx

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{url}/api/v1/run/{flow_id}",
            headers={"x-api-key": os.environ.get("LANGFLOW_API_KEY", "")},
            json={"input_type": "chat", "output_type": "chat",
                  "input_value": body.question},
        )
    r.raise_for_status()
    data = r.json()
    try:
        text = data["outputs"][0]["outputs"][0]["results"]["message"]["text"]
    except (KeyError, IndexError, TypeError):
        text = json.dumps(data)[:2000]
    return {"answer": text}


# Mounted last so /api/* wins.
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent / "static", html=True),
    name="static",
)
