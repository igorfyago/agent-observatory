"""Agent Observatory · FastAPI backend.

A multi-agent host that runs itself. The registry (backend/registry.py) lists
every hosted agent; the stage (backend/stage.py) is the single shared
performance every browser watches over GET /api/live; the autopilot
(backend/autopilot.py) keeps real runs flowing whenever the room is not
empty, so there is information on screen before anyone types anything.

Adding an agent is one registry entry. Nothing here needs to change.
"""
from __future__ import annotations

import asyncio
import contextlib
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

import autopilot  # noqa: E402  (needs env loaded first)
import registry  # noqa: E402
import runner  # noqa: E402
import spend  # noqa: E402
import stage  # noqa: E402
import store  # noqa: E402
from voice import personas, personas_store  # noqa: E402


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    pilot = asyncio.create_task(autopilot.loop())
    try:
        yield
    finally:
        pilot.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pilot


app = FastAPI(title="Agent Observatory", lifespan=lifespan)


@app.middleware("http")
async def revalidate_frontend(request: Request, call_next):
    """Browsers were caching the frontend on modification-time heuristics, so
    a deploy could leave a visitor on a week-old page for another day, or on a
    mixed one, old HTML with new script, which is worse. no-cache means store
    but revalidate: StaticFiles answers 304 through its ETag, so the steady
    cost is one conditional request per file and a page is never stale.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class ChatIn(BaseModel):
    question: str = ""
    agent: str = registry.DEFAULT_ID


class ResumeIn(BaseModel):
    agent: str
    thread: str
    action: str = "approve"       # approve | revise | reject
    notes: str = ""


class ReplayIn(BaseModel):
    """Time travel: re-run a thread from one of its saved checkpoints."""
    agent: str
    thread: str
    checkpoint_id: str
    step: int = -1                # display only, echoed back in the title


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
        "autopilot": autopilot.status(),
    }


@app.get("/api/graph")
def graph_spec(agent: str = registry.DEFAULT_ID):
    """The real graph topology for one agent. The UI renders THIS, never a
    hand-drawn copy, so the picture cannot drift from the code."""
    a = registry.get(agent)
    if a is None or a.kind != "text":
        raise HTTPException(404, f"no text agent '{agent}'")
    return {"spec": a.spec, "mode": runner.mode()}


@app.get("/api/blueprint")
async def blueprint(agent: str = registry.DEFAULT_ID):
    # async on purpose: a sync endpoint runs in a worker thread, and this is
    # the one caller of Agent.build() that could race the runner's build on
    # the event loop. On the loop, builds serialize and the cache stays sane.
    """How this graph is actually configured: the build() source that wires
    it, and the topology LangGraph itself reports after compiling.

    The source is read off the module with inspect, so the page can never
    show stale code; the compiled section comes from get_graph(), so the
    picture provably matches what runs. This is the part a trace viewer
    cannot show: traces are what happened, this is why it is wired to happen.
    """
    import inspect

    a = registry.get(agent)
    if a is None or a.kind != "text":
        raise HTTPException(404, f"no text agent '{agent}'")

    # Show the function that actually wires nodes and edges. The pipeline
    # module keeps its wiring in build_graph() and exposes a thin build()
    # for the registry contract; the wiring is what the page is for.
    fn = getattr(a.module, "build_graph", None) or a.module.build
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = "# source unavailable in this build"

    compiled_desc: dict = {"nodes": [], "edges": []}
    checkpointer = ""
    try:
        compiled = a.build()
        g = compiled.get_graph()
        compiled_desc = {
            "nodes": sorted(g.nodes.keys()),
            "edges": [{"from": e.source, "to": e.target,
                       "conditional": bool(e.conditional)} for e in g.edges],
        }
        cp = getattr(compiled, "checkpointer", None)
        checkpointer = type(cp).__name__ if cp else "none"
    except Exception as exc:
        # Demo mode cannot construct a model for most agents; the source is
        # still the point, so say what happened instead of failing the panel.
        checkpointer = f"not compiled here: {type(exc).__name__}"

    return {
        "agent": a.id,
        "level": a.level,
        "arrangement": a.arrangement,
        "source": source,
        "compiled": compiled_desc,
        "checkpointer": checkpointer,
        "runtime": {
            "streaming": "astream_events v2, one listener for every agent",
            "recursion_limit": 80,
        },
    }


# ----------------------------------------------------------- the stage -----

def _ip(request: Request) -> str:
    """Client IP, trusting Caddy's X-Forwarded-For in front of us."""
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd
            else (request.client.host if request.client else "unknown"))


def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev, default=str)}\n\n"


@app.get("/api/live")
async def live(request: Request):
    """The one stream every browser watches.

    A new viewer gets a hello frame (stage, stats, spend, mode), then the
    current run replayed from its first frame, then everything live. A ping
    every 15 quiet seconds keeps proxies from folding the connection.
    """
    sub_id, queue = stage.subscribe()

    async def gen():
        try:
            yield _sse(stage.hello())
            yield _sse({"type": "autopilot", **autopilot.status()})
            for frame in stage.snapshot_frames():
                yield _sse(frame)
            while True:
                try:
                    frame = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield _sse({"type": "ping"})
                    continue
                yield _sse(frame)
        finally:
            stage.unsubscribe(sub_id)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat", status_code=202)
async def chat(body: ChatIn, request: Request):
    """Put a visitor's question on the stage. Frames arrive on /api/live."""
    if not body.question.strip():
        raise HTTPException(400, "ask something first")
    try:
        meta = await stage.submit(body.agent, body.question.strip(),
                                  source="visitor", ip=_ip(request))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except stage.StageBusy as busy:
        raise HTTPException(409, busy.message) from busy
    return {"accepted": True, "run": meta}


@app.post("/api/resume", status_code=202)
async def resume(body: ResumeIn, request: Request):
    """Decide a parked interrupt. Beating the autopilot's countdown counts."""
    agent = registry.get(body.agent)
    if agent is None or not agent.interactive:
        raise HTTPException(404, f"agent '{body.agent}' does not pause for approval")
    try:
        meta = await stage.submit(
            body.agent, "", source="visitor", ip=_ip(request),
            title=f"decision: {body.action}",
            resume={"action": body.action, "notes": body.notes},
            thread=body.thread)
    except stage.NothingPending as exc:
        raise HTTPException(409, "nothing is waiting for approval") from exc
    except stage.StageBusy as busy:
        raise HTTPException(409, busy.message) from busy
    return {"accepted": True, "run": meta}


@app.post("/api/replay", status_code=202)
async def replay(body: ReplayIn, request: Request):
    """Time travel: re-execute a thread from one of its saved checkpoints.

    This is LangGraph's checkpointer doing the work: input None plus a
    checkpoint_id means "pick the graph up exactly there and run forward".
    Threads live in memory, so after a restart the strip honestly goes stale
    and this returns 409.
    """
    agent = registry.get(body.agent)
    if agent is None or agent.kind != "text":
        raise HTTPException(404, f"no text agent '{body.agent}'")
    cps = await runner.checkpoint_frame(agent, body.thread)
    known = {c["checkpoint_id"] for c in (cps or {}).get("items", [])}
    if body.checkpoint_id not in known:
        raise HTTPException(
            409, "that checkpoint is gone (threads do not survive a restart); "
                 "watch the archived replay instead")
    step = f"step {body.step}" if body.step >= 0 else "a saved checkpoint"
    try:
        meta = await stage.submit(
            body.agent, "", source="replay", ip=_ip(request),
            title=f"re-run from {step}",
            thread=body.thread, checkpoint_id=body.checkpoint_id)
    except stage.StageBusy as busy:
        raise HTTPException(409, busy.message) from busy
    return {"accepted": True, "run": meta}


@app.get("/api/runs")
def runs_list(limit: int = 30, agent: str = ""):
    """Newest-first archive of everything that crossed the stage. All real.
    Pass agent= to get one agent's runs: tab clicks replay from here."""
    return {"runs": store.list_stage_runs(limit, agent_id=agent or None)}


@app.get("/api/runs/latest")
def runs_latest():
    """The newest archived run whole, for the landing replay."""
    run = store.latest_stage_run()
    if run is None:
        raise HTTPException(404, "no runs archived yet")
    return run


@app.get("/api/runs/{run_id}")
def runs_get(run_id: int):
    """One archived run with its full frame list, for replay."""
    run = store.get_stage_run(run_id)
    if run is None:
        raise HTTPException(404, f"no stage run {run_id}")
    return run


@app.get("/api/stats")
def stats():
    """Real aggregates for the landing strip. Nothing here is synthetic."""
    return {"stats": store.stage_stats(), "spend": spend.status(),
            "autopilot": autopilot.status()}


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
