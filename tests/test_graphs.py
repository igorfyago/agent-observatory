"""Graph topology: the wiring IS the design, so pin it.

These build the real LangGraph graphs (no LLM call happens at build time) and
assert the structure the observatory draws in its DAG view. If an edge here
disappears, the picture on screen is lying about what the agent does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents import analyst, research_graph  # noqa: E402


def test_research_graph_topology():
    g = research_graph.build().get_graph()
    assert {"plan", "researcher", "tools", "draft", "reflect", "revise",
            "finalize"} <= set(g.nodes)
    edges = {(e.source, e.target) for e in g.edges}
    assert ("tools", "researcher") in edges       # inner tool loop closes
    assert ("revise", "researcher") in edges      # outer reflection loop closes
    assert ("plan", "researcher") in edges


SPECIALISTS = ("positioning", "flow", "risk")
PLAYBOOKS = ("long_gamma", "short_gamma", "no_data")


def test_analyst_graph_topology():
    g = analyst.build().get_graph()
    nodes = set(g.nodes)
    assert {"fetch", "synthesize", "risk_review", "human", "publish"} <= nodes
    assert set(PLAYBOOKS) <= nodes and set(SPECIALISTS) <= nodes

    edges = {(e.source, e.target) for e in g.edges}
    cond = {(e.source, e.target) for e in g.edges if e.conditional}

    # the regime router picks exactly one playbook, including the empty-data one
    for playbook in PLAYBOOKS:
        assert ("fetch", playbook) in cond
    # every playbook fans out to all three specialists, who join at synthesize
    for playbook in PLAYBOOKS:
        for specialist in SPECIALISTS:
            assert (playbook, specialist) in edges
    for specialist in SPECIALISTS:
        assert (specialist, "synthesize") in edges

    # the critique loop can send the memo back, and a human gates publishing
    assert ("synthesize", "risk_review") in edges
    assert ("risk_review", "synthesize") in cond
    assert ("human", "publish") in cond and ("human", "synthesize") in cond


def test_publish_is_only_reachable_through_the_human_gate():
    """The whole point of the HITL node: nothing publishes itself."""
    g = analyst.build().get_graph()
    into_publish = {e.source for e in g.edges if e.target == "publish"}
    assert into_publish == {"human"}


def test_every_registered_agent_has_a_unique_id():
    import registry

    ids = [a.id for a in registry.AGENTS]
    assert len(set(ids)) == len(ids)
    assert {"brief", "sql", "repo", "research", "analyst"} <= set(ids)
