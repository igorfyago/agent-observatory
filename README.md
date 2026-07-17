# Agent Observatory

A multi-agent research pipeline with **live observability**: chat with the team
on the left while the real agent graph illuminates node-by-node on the right —
LangSmith-style, top-down — as each agent speaks.

```
                 ┌────────────┐
                 │ Supervisor │            UI (split-pane)
                 └─────┬──────┘            ┌──────────┬──────────────┐
              ┌────────┴────────┐          │          │   ● DAG      │
        ┌─────▼─────┐     ┌─────▼─────┐    │   chat   │   lights up  │
        │ Researcher│     │  Analyst  │    │  stream  │   live       │
        └─────┬─────┘     └─────┬─────┘    │          ├──────────────┤
              └────────┬────────┘          │          │ trace + ms   │
                 ┌─────▼─────┐             └──────────┴──────────────┘
                 │   Writer  │◄─── revise ──┐
                 └─────┬─────┘              │
                 ┌─────▼─────┐──────────────┘
                 │   Critic  │─── approve ──► END
                 └───────────┘
```

## Stack — all four tools

| Tool | Role here |
|------|-----------|
| **LangGraph** | The pipeline itself: `StateGraph` with parallel fan-out (researcher ∥ analyst), a join at writer, and a conditional revise-loop at critic. The UI renders the *actual* compiled topology from `/api/graph`. |
| **LangChain** | Model layer (`ChatOpenAI` streaming) inside each node in live mode. |
| **LangSmith** | Cloud tracing: set the env vars below and every run appears as a trace. The right-hand panel is the same idea, local and live. |
| **Langflow** | Visual prototyping companion: build the same flow visually, then wire it via the `/api/langflow` REST bridge. |

## Run

```bash
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
.venv\Scripts\python -m uvicorn app:app --app-dir backend --port 8321
# open http://localhost:8321
```

Works **keyless out of the box** — demo mode streams scripted agent output so
the full pipeline and UI light up with zero credentials. The demo critic
rejects the first draft on purpose so you can watch the revise loop fire.

## Live mode + tracing

Copy `.env.example` → `.env` and set:

```env
OPENAI_API_KEY=sk-...            # real inference per node
LANGCHAIN_TRACING_V2=true        # + LangSmith tracing
LANGSMITH_API_KEY=lsv2_...
LANGCHAIN_PROJECT=agent-observatory
```

## Langflow bridge (optional)

1. `uv pip install langflow && langflow run` (separate venv — heavy deps)
2. Build a chat flow visually at http://localhost:7860, export its JSON into
   `langflow/` in this repo (the JSON *is* the design artifact)
3. Set `LANGFLOW_URL`, `LANGFLOW_FLOW_ID`, `LANGFLOW_API_KEY` in `.env`
4. `POST /api/langflow {"question": "..."}` runs the same question through the
   visual flow.

## How the live UI works

Each LangGraph node receives an `emit` callback via `config.configurable`.
Nodes emit `node_start` / `token` / `node_end` / `route` events into a queue;
`POST /api/chat` drains the queue to the browser as **SSE**. The frontend is a
single dependency-free page: SVG DAG (nodes glow while active, edges animate
on transition, the revise loop is dashed) plus a top-down trace table with
per-node duration (ms) and token counts.
