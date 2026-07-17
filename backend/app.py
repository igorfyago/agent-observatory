"""Agent Observatory — FastAPI backend.

Serves the split-pane UI and streams live pipeline events over SSE:
the LangGraph run pushes node_start / token / node_end / route events into a
queue, and /api/chat drains it to the browser as they happen.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load project-root .env before graph.py reads any env vars.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import graph as pipeline  # noqa: E402  (needs env loaded first)

app = FastAPI(title="Agent Observatory")
GRAPH = pipeline.build_graph()


class ChatIn(BaseModel):
    question: str


@app.get("/api/graph")
def graph_spec():
    """The real graph topology — the UI renders THIS, not a hand-drawn copy."""
    return {"spec": pipeline.GRAPH_SPEC, "mode": pipeline.mode()}


@app.post("/api/chat")
async def chat(body: ChatIn):
    async def gen():
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(ev: dict) -> None:
            await queue.put(ev)

        async def run():
            return await GRAPH.ainvoke(
                {"question": body.question, "revisions": 0},
                config={"configurable": {"emit": emit}, "recursion_limit": 25},
            )

        task = asyncio.create_task(run())
        yield _sse({"type": "run_start", "mode": pipeline.mode()})

        while not (task.done() and queue.empty()):
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=0.2)
            except (asyncio.TimeoutError, TimeoutError):
                continue
            yield _sse(ev)

        if task.exception() is not None:
            yield _sse({"type": "error", "message": str(task.exception())})
        else:
            result = task.result()
            yield _sse({
                "type": "run_end",
                "answer": result.get("draft", ""),
                "revisions": result.get("revisions", 0),
            })

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev)}\n\n"


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
            detail="Langflow not configured — set LANGFLOW_URL and LANGFLOW_FLOW_ID in .env",
        )
    import httpx

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{url}/api/v1/run/{flow_id}",
            headers={"x-api-key": os.environ.get("LANGFLOW_API_KEY", "")},
            json={
                "input_type": "chat",
                "output_type": "chat",
                "input_value": body.question,
            },
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
