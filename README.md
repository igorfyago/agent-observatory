# Agent Observatory

The home for every non-trading agent · **agents you can watch think**.

A multi-agent host: pick an agent from the top tab bar, ask it something, and
watch its real graph light up node by node while every tool call it makes is
shown live with its arguments, its result and how long it took.

```
  +--------------------------------------------------------------+
  | agent observatory   Pipeline Brief SQL Repo Research ...      |  top tab bar
  +--------------------------+-----------------------------------+  = the picker
  |                          |        * DAG lights up            |
  |   chat / token stream    |        (drawn from the REAL       |
  |                          |         compiled topology)        |
  |                          +-----------------------------------+
  |                          |  trace + every tool call:         |
  |                          |  name · args · result · ms        |
  +--------------------------+-----------------------------------+
```

## The hosted agents

| id | kind | what it demonstrates |
|----|------|----------------------|
| `pipeline` | text | Five roles, a parallel fan-out and a critic that sends work back. |
| `brief` | text | One model call, zero tools. Structured output instead of free text. |
| `sql` | text | Writes SQL, runs it, reads the error, fixes itself. The tool loop. |
| `repo` | text | RAG over this observatory's own source, with file citations. |
| `research` | text | A tool loop nested inside a reflection loop. Both are visible. |
| `analyst` | text | Regime router, three parallel specialists, critic gate, human approval. |
| `riley` | voice | AI receptionist for a dental clinic. Books real appointments. |
| `quinn` | voice | AI quoting agent for a renovation company. |

Agents 01-05 and the voice personas were forked from `ai-trading-desk` on
2026-07-18. The trading engine did **not** come with them (see the desk seam
below). Marcus, the options-desk voice agent, deliberately stayed in the desk:
he narrates that repo's deterministic signals engine and belongs with it.

## Adding an agent

One entry in `backend/registry.py`, pointing at a module that exposes four
names:

```python
SPEC                  # dict of nodes/edges, the DAG the UI draws
build()               # -> compiled LangGraph runnable
make_input(question)  # -> dict fed to the graph
extract(result)       # -> str, the final answer
```

`/api/agents`, the tab bar, the DAG and the live streaming all follow from
that. Nothing else needs editing.

## How the live view works

`backend/runner.py` is a single generic translator over LangGraph's
`astream_events`. Because node and tool events are already first-class in the
framework, every hosted agent gets live highlighting and tool inspection with
zero observability boilerplate in the agent itself:

| frame | from |
|-------|------|
| `node_start` / `node_end` | LangGraph chain events, filtered to real graph nodes |
| `token` | `on_chat_model_stream` |
| `tool_start` | `on_tool_start`, carrying the call's arguments |
| `tool_end` | `on_tool_end`, carrying the result and elapsed ms |
| `route` | a custom event an agent dispatches for a conditional edge |
| `interrupt` | the graph parked itself for human approval |

An agent that needs a frame the runner cannot infer dispatches a LangChain
custom event named `obs` and it passes straight through.

The DAG itself is rendered by `backend/static/layout.js`, a pure deterministic
layered-graph layout, from the topology the backend reports. The picture is
generated from the compiled graph, so it cannot drift from the code.

## The desk seam

The trading engine (`common/market`, `common/signals`, `common/tape`,
`common/quotes`, `common/trades`) stays in `ai-trading-desk` and is not
vendored here. Where a ported agent used to reach into it in-process, it now
goes over **HTTP** to the desk's public read API
(`backend/desk_client.py`, calling `GET /api/summary/{ticker}`).

When `DESK_API_URL` is unset or the desk is unreachable, every one of those
tools returns an explicit "desk unavailable" string and the agents are
instructed to say so rather than invent a figure. The `analyst` graph even has
a third router branch, `no_data`, for exactly that case.

## Persistence

`backend/store.py` keeps one sqlite file in a real data dir (`OBS_DATA_DIR`,
default `<repo>/data`): voice bookings, saved quotes, custom personas from the
builder, published memos, and an `agent_runs` telemetry table that the `sql`
and `research` agents query. Mount a volume at `/app/data` in production or
every agent the builder mints is lost on redeploy.

## Spend guard

`backend/spend.py` wraps every run: a per-IP cooldown, a global concurrency
ceiling, and daily/monthly USD caps metered from real token usage. A refused
request comes back as a friendly `error` frame, not a stack trace.

## Run

```bash
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
.venv\Scripts\python -m uvicorn app:app --app-dir backend --port 8321
# open http://localhost:8321
```

Works **keyless** for the `pipeline` agent: demo mode streams scripted output
so the DAG and UI light up with zero credentials, and the demo critic rejects
the first draft on purpose so you can watch the revise loop fire. The other
agents need a real key.

## Live mode + tracing

Copy `.env.example` to `.env` and set:

```env
OPENAI_API_KEY=sk-...            # real inference
LANGCHAIN_TRACING_V2=true        # + LangSmith tracing
LANGSMITH_API_KEY=lsv2_...
LANGCHAIN_PROJECT=agent-observatory
```

See `.env.example` for the data dir, desk seam, admin token and spend caps.

## Langflow bridge (optional)

1. `uv pip install langflow && langflow run` (separate venv, heavy deps)
2. Build a chat flow visually at http://localhost:7860, export its JSON into
   `langflow/` in this repo (the JSON *is* the design artifact)
3. Set `LANGFLOW_URL`, `LANGFLOW_FLOW_ID`, `LANGFLOW_API_KEY` in `.env`
4. `POST /api/langflow {"question": "..."}` runs the same question through the
   visual flow.

## API

| endpoint | does |
|----------|------|
| `GET /api/agents` | the registry: every hosted agent, with graph topology |
| `GET /api/graph?agent=` | one agent's compiled topology |
| `POST /api/chat` | run a text agent, SSE frame stream |
| `POST /api/resume` | resume an agent parked at a human-approval interrupt |
| `GET /api/personas/{id}` | a voice persona's instructions, voice and tool schemas |
| `POST /api/personas` | mint a custom voice agent (admin-token gated) |
| `POST /api/tool/{id}` | server-side execution of a Realtime function call |
| `GET /api/health` | agent count, data dir, mode |
