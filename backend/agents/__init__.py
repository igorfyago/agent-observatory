"""Hosted agents.

Every module here exposes the same four names, which is the entire contract
the registry depends on:

    SPEC                 dict with nodes/edges, the DAG the UI draws
    build()              -> compiled LangGraph runnable (cached by the registry)
    make_input(question) -> dict fed to the graph
    extract(result)      -> str, the final answer

Forked from ai-trading-desk agents/ on 2026-07-18. The trading engine did not
come with them: see backend/desk_client.py for the seam.
"""
