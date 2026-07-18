"""Agent 3 · Repo: RAG over a real codebase.

Forked from ai-trading-desk agents/03_repo_interpreter on 2026-07-18.

Level: retrieval-augmented agent. Retrieval is exposed to the model AS A TOOL,
so it decides when and what to search, and can search several times for a
multi-part question, instead of a fixed retrieve-then-answer chain.

Two changes from the desk version:

1. Target repo. The original indexed options-flow-analytics via GEX_REPO_PATH.
   This one indexes the OBSERVATORY'S OWN SOURCE by default, so the agent can
   explain the app you are watching it run in. Point OBS_REPO_PATH somewhere
   else to aim it at another checkout.

2. Vector store. The original persisted to Chroma on disk. Chroma is a heavy
   dependency for a repo this size, so this uses langchain-core's built-in
   InMemoryVectorStore, built lazily on first search and cached for the
   process. Cost: one embedding pass per boot. Benefit: no extra dependency
   and no stale index.
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import get_embeddings, get_model

SPEC = {
    "nodes": [
        {"id": "model", "label": "Model", "role": "asks and answers"},
        {"id": "tools", "label": "Retriever", "role": "semantic search"},
    ],
    "edges": [
        {"from": "model", "to": "tools", "label": "search"},
        {"from": "tools", "to": "model", "kind": "loop", "label": "chunks"},
        {"from": "model", "to": "end", "label": "answer"},
    ],
}

SOURCE_EXTS = {".py", ".js", ".html", ".css", ".md", ".yml", ".yaml", ".toml", ".txt"}
SKIP_DIRS = {".venv", "venv", "node_modules", ".git", ".idea", "dist", "data",
             "__pycache__", ".pytest_cache", "target"}
MAX_FILE_BYTES = 200_000

_RETRIEVER = None


def repo_path() -> Path:
    default = Path(__file__).resolve().parents[2]  # the observatory checkout
    return Path(os.getenv("OBS_REPO_PATH", str(default))).expanduser().resolve()


def load_repo_documents(root: Path) -> list[Document]:
    docs = []
    for path in root.rglob("*"):
        if path.is_dir() or path.suffix.lower() not in SOURCE_EXTS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"file": str(path.relative_to(root)).replace("\\", "/")},
            ))
    return docs


def get_retriever():
    """Build the index once per process, then reuse it."""
    global _RETRIEVER
    if _RETRIEVER is not None:
        return _RETRIEVER
    root = repo_path()
    docs = load_repo_documents(root)
    if not docs:
        raise RuntimeError(f"no indexable source found under {root}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150,
        separators=["\n\n", "\ndef ", "\nclass ", "\n", " ", ""],
    )
    splits = splitter.split_documents(docs)
    store = InMemoryVectorStore.from_documents(splits, get_embeddings())
    _RETRIEVER = store.as_retriever(search_kwargs={"k": 6})
    return _RETRIEVER


@tool
def search_codebase(query: str) -> str:
    """Semantically search this project's source code and docs.

    Returns the most relevant code/doc chunks, each tagged with its file path.
    Call multiple times with different phrasings for multi-part questions.

    Args:
        query: What to look for, e.g. "how tool calls are streamed" or
            "where the agent registry is defined"
    """
    try:
        docs = get_retriever().invoke(query)
    except Exception as exc:
        return f"RETRIEVAL ERROR: {exc}"
    if not docs:
        return "No matches. Try different terminology."
    return "\n\n".join(
        f"--- {d.metadata.get('file', '?')} ---\n{d.page_content}" for d in docs
    )


SYSTEM = """You are the maintainer of the agent observatory: a FastAPI app that
hosts several LangChain and LangGraph agents and streams their node and tool
events to a live DAG in the browser. Answer questions about how it works.

Rules:
- ALWAYS ground answers in retrieved code: call search_codebase before
  answering, and again with new phrasings if the first results fall short.
- Cite file paths for every claim, like (backend/runner.py).
- Quote the key lines when explaining a mechanism.
- If the code genuinely does not answer the question, say so. Never invent code."""

TOOLS = [search_codebase]


def build():
    return create_agent(model=get_model(), tools=TOOLS, system_prompt=SYSTEM)


def make_input(question: str) -> dict:
    return {"messages": [{"role": "user", "content": question}]}


def extract(result: dict) -> str:
    return result["messages"][-1].content
