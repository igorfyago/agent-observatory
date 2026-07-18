"""Shared model factory: every hosted agent gets its model the same way.

Forked from ai-trading-desk common/llm.py on 2026-07-18.

Change OBS_MODEL in .env to swap providers without touching agent code.
"""
from __future__ import annotations

import os

from langchain.chat_models import init_chat_model

DEFAULT_MODEL = "openai:gpt-4.1"


def have_key() -> bool:
    """True when a real model can be constructed. The UI degrades honestly
    rather than throwing when this is False."""
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def get_model(temperature: float = 0.0):
    """Chat model used by all text agents. Configured via OBS_MODEL env var."""
    return init_chat_model(
        os.getenv("OBS_MODEL", os.getenv("OPENAI_MODEL") or DEFAULT_MODEL),
        temperature=temperature,
    )


def get_embeddings():
    """Embedding model for the RAG agent."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model="text-embedding-3-small")
