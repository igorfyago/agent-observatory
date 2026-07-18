"""Shared test setup for the observatory.

Two invariants for the whole suite:
0. No test reaches OpenAI: a dummy key is planted so graphs can be built.
1. No test touches the real spend meter: OBS_SPEND_DB points at a fresh temp
   file per test, so no test can see another one's dollars.
2. No test can email anyone: OBS_SES_FROM is cleared, which puts the alerting
   path in log-only mode unless a test opts in and mocks boto3.
"""

import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

# Building a graph constructs its chat model, which wants a key present. A
# dummy one keeps the topology tests offline: nothing here ever calls out.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")


@pytest.fixture()
def spend(tmp_path, monkeypatch):
    """The spend guard, pointed at a throwaway database with known limits."""
    import spend as module

    monkeypatch.setenv("OBS_SPEND_DB", str(tmp_path / "spend.db"))
    monkeypatch.setenv("OBS_DAILY_USD", "1.00")
    monkeypatch.setenv("OBS_MONTHLY_USD", "10.00")
    monkeypatch.setenv("OBS_COOLDOWN_S", "20")
    monkeypatch.setenv("OBS_MAX_CONCURRENT", "2")
    monkeypatch.delenv("OBS_SES_FROM", raising=False)
    module._reset_runtime()
    yield module
    module._reset_runtime()


@pytest.fixture()
def sent(spend, monkeypatch):
    """Capture alert emails instead of sending them."""
    outbox: list[tuple[str, str]] = []
    monkeypatch.setattr(spend, "_email", lambda subject, body: outbox.append((subject, body)))
    return outbox


def burn(spend, dollars: float, model: str = "gpt-4o-mini") -> None:
    """Record exactly `dollars` of output-token spend."""
    _, price_out = spend.price_for(model)
    spend.record({"input_tokens": 0, "output_tokens": round(dollars * 1_000_000 / price_out)},
                 model=model)
