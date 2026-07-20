"""The agent registry: one place that lists every hosted agent.

Adding an agent is ONE entry here plus a module exposing the four names in the
contract (SPEC, build, make_input, extract). Nothing else in the app needs to
change: /api/agents, the UI picker, the DAG and the streaming runner are all
driven off this table.

Graphs are built lazily and cached, so importing this module never needs an
API key and a broken agent cannot take the whole app down at boot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import graph as pipeline
from agents import analyst, brief, repo, research_graph, sql
from voice import personas


@dataclass(frozen=True)
class Agent:
    id: str
    label: str
    kind: str                      # "text" | "voice" | "desk"
    tagline: str
    module: object = None          # text agents: the module holding the contract
    persona: str = ""              # voice agents: key into personas.PERSONAS
    placeholder: str = ""
    interactive: bool = False      # pauses for human approval mid-run
    _cache: dict = field(default_factory=dict, compare=False)

    # -- text agent contract ------------------------------------------------
    @property
    def spec(self) -> dict:
        return getattr(self.module, "SPEC", {"nodes": [], "edges": []})

    def build(self):
        """Compile once, then reuse. Raises if no model key is configured."""
        if "graph" not in self._cache:
            self._cache["graph"] = self.module.build()
        return self._cache["graph"]

    def make_input(self, question: str) -> dict:
        return self.module.make_input(question)

    def extract(self, result: dict) -> str:
        return self.module.extract(result)


AGENTS: list[Agent] = [
    Agent(
        id="pipeline",
        label="Pipeline",
        kind="text",
        tagline="Five roles, a parallel fan-out and a critic that sends work back.",
        module=pipeline,
        placeholder="Should we build multi-agent systems as graphs or as prompt chains?",
    ),
    Agent(
        id="brief",
        label="Brief",
        kind="text",
        tagline="One call, zero tools. Structured output instead of free text.",
        module=brief,
        placeholder="Is SPY pinned by dealers into Friday opex?",
    ),
    Agent(
        id="sql",
        label="SQL",
        kind="text",
        tagline="Writes SQL, runs it, reads the error, fixes itself. The tool loop.",
        module=sql,
        placeholder="Which agent has the highest average latency, and how many runs did it have?",
    ),
    Agent(
        id="repo",
        label="Repo",
        kind="text",
        tagline="RAG over this observatory's own source, with file citations.",
        module=repo,
        placeholder="How does the runner turn LangGraph events into DAG frames?",
    ),
    Agent(
        id="research",
        label="Research",
        kind="text",
        tagline="A tool loop inside a reflection loop. Both are visible.",
        module=research_graph,
        placeholder="How busy has this observatory been over the last week?",
    ),
    Agent(
        id="analyst",
        label="Analyst",
        kind="text",
        tagline="Regime router, three parallel specialists, critic gate, human approval.",
        module=analyst,
        placeholder="SPY",
        interactive=True,
    ),
    Agent(
        id="riley",
        label="Riley",
        kind="voice",
        tagline="AI receptionist for a dental clinic. Books real appointments.",
        persona="riley",
    ),
    Agent(
        id="quinn",
        label="Quinn",
        kind="voice",
        tagline="AI quoting agent for a renovation company.",
        persona="quinn",
    ),
    # Not chat-runnable: Marcus lives on the trading desk and his runs arrive
    # over POST /api/desk/runs whenever someone talks to him. This tab is the
    # replay lane for those recorded decisions.
    Agent(
        id="marcus",
        label="Marcus",
        kind="desk",
        tagline="The desk's options voice. Every trade decision he makes lands here live.",
    ),
]

BY_ID: dict[str, Agent] = {a.id: a for a in AGENTS}
DEFAULT_ID = "pipeline"


def get(agent_id: str) -> Agent | None:
    return BY_ID.get(agent_id)


def as_json() -> list[dict]:
    """What /api/agents returns and the UI picker renders."""
    out = []
    for a in AGENTS:
        entry = {
            "id": a.id,
            "label": a.label,
            "kind": a.kind,
            "tagline": a.tagline,
            "interactive": a.interactive,
        }
        if a.kind == "text":
            entry["spec"] = a.spec
            entry["placeholder"] = a.placeholder
        elif a.kind == "voice":
            p = personas.PERSONAS.get(a.persona, {})
            entry["voice"] = p.get("voice", "")
            entry["tools"] = [t["name"] for t in p.get("tools", [])]
        # kind "desk" carries nothing extra: its data arrives per run over
        # /api/desk/runs, spec included, so nothing here can go stale.
        out.append(entry)
    return out
