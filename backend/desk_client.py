"""The desk seam.

The trading engine (common/market, common/signals, common/tape, common/quotes,
common/trades) stays in ai-trading-desk and is NOT vendored here. Where a
ported agent used to reach into those modules in-process, it now goes over
HTTP to the desk's public read API instead.

CHOICE MADE: HTTP call, not a vendored stub. The desk already exposes
GET /api/summary/{ticker} returning snapshot, walls, headlines and TA signals
in one shot, so the observatory can stay honest about where the numbers come
from: they are the desk's, fetched live, attributed in the tool output.

When DESK_API_URL is unset or the desk is unreachable, every tool returns an
explicit "no desk connection" string. It never fabricates a number, and the
agent is told plainly that the data is unavailable so it can say so.
"""
from __future__ import annotations

import json
import os

import httpx

DEFAULT_URL = "https://desk.b4rruf3t.com"
TIMEOUT = 8.0

_UNAVAILABLE = (
    "DESK UNAVAILABLE: no live positioning data is reachable from the "
    "observatory right now. Do not invent numbers. Say plainly that the desk "
    "feed is offline and answer only from first principles."
)


def desk_url() -> str | None:
    """Base URL of the desk, or None when the seam is deliberately open."""
    url = os.getenv("DESK_API_URL", DEFAULT_URL).strip().rstrip("/")
    return url or None


def enabled() -> bool:
    return desk_url() is not None


def summary(ticker: str) -> dict | None:
    """One live snapshot bundle from the desk, or None when unreachable."""
    base = desk_url()
    if not base:
        return None
    try:
        r = httpx.get(f"{base}/api/summary/{ticker.upper()}", timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _section(ticker: str, key: str) -> str:
    data = summary(ticker)
    if data is None:
        return _UNAVAILABLE
    part = data.get(key)
    if not part:
        return f"No {key} reported by the desk for {ticker.upper()}."
    return f"(source: desk {desk_url()} · {key} for {ticker.upper()})\n" + json.dumps(
        part, indent=1, default=str
    )


def positioning_snapshot(ticker: str) -> str:
    """Latest dealer-positioning snapshot, fetched from the desk."""
    return _section(ticker, "snapshot")


def wall_map(ticker: str) -> str:
    """Strongest call/put walls, fetched from the desk."""
    return _section(ticker, "walls")


def market_news(ticker: str) -> str:
    """Latest headlines, fetched from the desk."""
    return _section(ticker, "headlines")


def ta_signals(ticker: str) -> str:
    """Recently fired TA alerts, fetched from the desk."""
    return _section(ticker, "ta_signals")
