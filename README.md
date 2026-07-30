# Agent Observatory

The home for every non-trading agent · **agents you can watch think**.

A multi-agent host that runs itself. Open the page and information is already
flowing: the autopilot cycles curated scenarios through real LangGraph agents
whenever the room has viewers, every browser watches the same shared stage
over one SSE channel, and the moment the stage is idle the newest archived
run replays so there is never a dead screen. Nobody has to think of a
question to see the machine work; the composer is there for whoever wants to
take the stage anyway.

```
  +----------------------------------------------------------------+
  | agent observatory        Pipeline Brief SQL Repo Research ...   |
  +---------------------------+------------------------------------+
  | feed: every run's tokens  |  LIVE · the DAG lights up           |
  | (autopilot + visitors,    |  (drawn from the REAL compiled      |
  |  same stage for everyone) |   topology, never hand-drawn)       |
  |                           +------------------------------------+
  | autopilot: next scenario  |  trace | state | checkpoints | x-ray|
  | in 12s · [run now]        |  tools · reducers · time travel ·   |
  | [ask your own question]   |  per-node design rationale          |
  +---------------------------+--- runs · cost · tokens · median ---+
```

## The stage and the autopilot

One run at a time plays to every open browser (`GET /api/live`). The social
contract: a visitor preempts the autopilot mid-run, nobody preempts a
visitor. An empty room costs nothing: the autopilot only performs when at
least one viewer is connected, every run passes the same spend guard as a
visitor's, and a guard refusal backs the loop off exactly as long as the
guard asked.

Each scenario in `backend/autopilot.py` exists to light up one LangGraph
mechanism, and says so on the ribbon while it runs:

| scenario | mechanism on display |
|----------|----------------------|
| five roles, one graph | static-edge parallel fan-out, a join, a critic loop on `add_conditional_edges` |
| the tool loop repairs itself | `create_agent` loop reading SQL errors and fixing its own query |
| a loop inside a loop | an inner tool loop nested in an outer reflection loop |
| the app explains its own source | retrieval as a tool, file citations over this repo |
| structured output, no prose | `with_structured_output` forcing a Pydantic schema |
| the flagship | `Command` router, three parallel sub-agents, critic gate, `interrupt()` |

The SQL and research scenarios query the observatory's own run log, which the
stage itself keeps appending to, so the questions stay fresh forever.

## What the right rail proves

- **trace**: every node with timing, every tool call with args, result and
  ms, every model call with its token count and banked cost.
- **state**: the per-node delta each superstep merged, with the channel's
  reducer named (`overwrite` vs `append`): watch `operator.add` collect three
  concurrent specialists without a collision.
- **checkpoints**: the thread's saved supersteps from the checkpointer,
  newest first, each with a "re-run from here" button. That button is
  LangGraph time travel: input `None` plus a `checkpoint_id`, and the graph
  re-executes forward from that saved state, live on the stage.
- **x-ray**: click any node for the design defended in plain words: the
  concept, the trade it makes, and the hard questions it should be asked.
  Every agent also lists the exact framework surface it exercises.

Interrupts are part of the show: when the analyst parks at its human gate,
autopilot runs approve themselves after a visible countdown, and any viewer
can beat the countdown with the approve/revise/reject buttons. Each run ends
with a deep link to its LangSmith trace when tracing is configured.

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
`astream_events`. Because node, tool and model events are already first-class
in the framework, every hosted agent gets live highlighting, state
inspection and cost metering with zero observability boilerplate in the
agent itself:

| frame | from |
|-------|------|
| `node_start` / `node_end` | LangGraph chain events, filtered to real graph nodes; `node_end` carries the state delta the node returned |
| `token` | `on_chat_model_stream` |
| `tool_start` | `on_tool_start`, carrying the call's arguments |
| `tool_end` | `on_tool_end`, carrying the result and elapsed ms |
| `usage` | `on_chat_model_end`, the call's tokens and cost as the spend meter banked them |
| `route` | a custom event an agent dispatches for a conditional edge |
| `interrupt` | the graph parked itself for human approval |
| `checkpoints` | the thread's saved supersteps from `aget_state_history`, for the time-travel strip |
| `trace_id` | the root run id, which is the LangSmith trace id when tracing is on |

An agent that needs a frame the runner cannot infer dispatches a LangChain
custom event named `obs` and it passes straight through.

`backend/stage.py` fans those frames out to every viewer and records each
finished run whole (meta + frames) into the `stage_runs` archive, which is
what powers the landing replay, the recent-runs list and the stats strip.
Nothing on that strip is synthetic: it is all real runs that crossed this
stage.

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
| `GET /api/live` | the one SSE stream every browser watches: hello, snapshot of the run in flight, then everything live |
| `POST /api/chat` | put a question on the stage (202; frames arrive on /api/live) |
| `POST /api/resume` | decide a parked interrupt: approve, revise, reject |
| `POST /api/replay` | time travel: re-execute a thread from a saved checkpoint |
| `GET /api/runs` (+`/{id}`, `/latest`) | the stage archive, whole runs with frames |
| `GET /api/stats` | real aggregates: runs, cost, tokens, latency, autopilot state |
| `GET /api/agents` | the registry: every hosted agent, topology, x-ray content |
| `GET /api/graph?agent=` | one agent's compiled topology |
| `GET /api/personas/{id}` | a voice persona's instructions, voice and tool schemas |
| `POST /api/personas` | mint a custom voice agent (admin-token gated) |
| `POST /api/tool/{id}` | server-side execution of a Realtime function call |
| `GET /api/health` | agent count, data dir, mode |
