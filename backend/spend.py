"""Spend guard: the observatory runs on the owner's own OpenAI key, so every
public request has to be metered, capped and rate limited.

Three independent brakes, in the order check() applies them:

  1. CAPS      a daily and a monthly dollar ceiling, counted from real token
               usage and persisted in SQLite so a restart cannot zero the meter.
  2. COOLDOWN  a per-IP minimum gap between requests, so one visitor cannot
               machine-gun the key.
  3. CAPACITY  a global ceiling on requests in flight, so a crowd cannot either.

Over any of those, check() raises SpendRefused, which carries a friendly
message the UI can render as-is. It is never a stack trace in the user's face.

The owner gets an email (AWS SES if OBS_SES_FROM is set, else a log line) when
spend crosses 50%, 80% and 100% of either cap. Each alert is sent at most once
per day, deduped in the database, and a failing mailer can never break a
request: every send is wrapped.

Wiring, for whoever owns app.py:

    import spend

    try:
        with spend.guard(request.client.host):
            result = await run_the_graph(...)
    except spend.SpendRefused as refusal:
        return JSONResponse(refusal.as_dict(), status_code=refusal.status_code)

    spend.record({"input_tokens": 812, "output_tokens": 240}, model="gpt-4o-mini")

or hand spend.SpendCallback() to LangChain/LangGraph as a callback and let it
record token usage itself.

Environment
-----------
OBS_DAILY_USD        daily cap, default 2.00
OBS_MONTHLY_USD      monthly cap, default 20.00
OBS_COOLDOWN_S       seconds between requests from one IP, default 20
OBS_MAX_CONCURRENT   requests in flight at once, default 2
OBS_SPEND_DB         meter database, default <repo>/data/spend.db
OBS_ALERT_TO         who hears about it, default igor.yago@gmail.com
OBS_SES_FROM         verified SES sender; unset means alerts only get logged
OBS_SES_REGION       SES region, default AWS_REGION or eu-central-1
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("observatory.spend")

# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
# USD per 1,000,000 tokens, as (input, output). THIS IS THE ONE PLACE PRICES
# LIVE: change a number here and every meter, cap and alert follows. Vendor
# prices move, so treat these as a snapshot to check against the pricing page,
# not as gospel. Lookup falls back to the longest matching prefix (so
# "gpt-4o-mini-2024-07-18" prices as "gpt-4o-mini"), then to FALLBACK_PRICE.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":  (0.15,  0.60),
    "gpt-4o":       (2.50, 10.00),
    "gpt-4.1-nano": (0.10,  0.40),
    "gpt-4.1-mini": (0.40,  1.60),
    "gpt-4.1":      (2.00,  8.00),
    "o4-mini":      (1.10,  4.40),
    "o3-mini":      (1.10,  4.40),
}

# Unknown model: price it like the most expensive thing we know, because a
# meter that guesses low is a meter that lets the bill through.
FALLBACK_PRICE = (2.50, 10.00)

DEFAULT_ALERT_TO = "igor.yago@gmail.com"
ALERT_THRESHOLDS = (50, 80, 100)  # percent of a cap


# --------------------------------------------------------------------------- #
# Config, read lazily so tests and redeploys can move it without an import dance
# --------------------------------------------------------------------------- #

def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "") or default))
    except ValueError:
        return default


def daily_cap() -> float:
    return _f("OBS_DAILY_USD", 2.00)


def monthly_cap() -> float:
    return _f("OBS_MONTHLY_USD", 20.00)


def cooldown_s() -> float:
    return _f("OBS_COOLDOWN_S", 20)


def max_concurrent() -> int:
    return max(1, _i("OBS_MAX_CONCURRENT", 2))


def db_path() -> Path:
    override = os.environ.get("OBS_SPEND_DB")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "spend.db"


def default_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,                 -- ISO timestamp (UTC)
    day TEXT NOT NULL,                -- YYYY-MM-DD  (UTC)
    month TEXT NOT NULL,              -- YYYY-MM     (UTC)
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_day ON usage(day);
CREATE INDEX IF NOT EXISTS usage_month ON usage(month);

-- one row per IP: when it was last let through. Persisted so a restart is not
-- a free pass through the cooldown.
CREATE TABLE IF NOT EXISTS hits (
    ip TEXT PRIMARY KEY,
    last_ts REAL NOT NULL
);

-- the dedupe ledger: an alert exists at most once per (day, kind), forever.
CREATE TABLE IF NOT EXISTS alerts (
    day TEXT NOT NULL,
    kind TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    PRIMARY KEY (day, kind)
);
"""

_db_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


@contextmanager
def _db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _keys(now: datetime | None = None) -> tuple[str, str]:
    now = now or _now()
    return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")


# --------------------------------------------------------------------------- #
# Refusal
# --------------------------------------------------------------------------- #

class SpendRefused(Exception):
    """A request the guard will not let through.

    Carries everything the UI needs: a friendly message, a machine-readable
    reason, an HTTP status and, where it makes sense, how long to wait.
    """

    def __init__(self, reason: str, message: str, status_code: int = 429,
                 retry_after: float | None = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.status_code = status_code
        self.retry_after = None if retry_after is None else round(retry_after, 1)

    def as_dict(self) -> dict:
        return {
            "error": "spend_guard",
            "reason": self.reason,
            "message": self.message,
            "retry_after": self.retry_after,
            "status": status(),
        }


# --------------------------------------------------------------------------- #
# Pricing and recording
# --------------------------------------------------------------------------- #

def price_for(model: str | None) -> tuple[float, float]:
    """USD per 1M (input, output) tokens for a model id."""
    name = (model or default_model() or "").strip().lower()
    if name in PRICES:
        return PRICES[name]
    for prefix in sorted(PRICES, key=len, reverse=True):
        if name.startswith(prefix):
            return PRICES[prefix]
    return FALLBACK_PRICE


def cost_of(input_tokens: int, output_tokens: int, model: str | None = None) -> float:
    p_in, p_out = price_for(model)
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def _tokens_from(usage) -> tuple[int, int, str | None]:
    """Normalize the several shapes token usage arrives in.

    Accepts LangChain usage_metadata ({input_tokens, output_tokens}), the
    OpenAI token_usage shape ({prompt_tokens, completion_tokens}), an
    AIMessage carrying .usage_metadata, or a plain (in, out) pair.
    """
    if usage is None:
        return 0, 0, None
    if isinstance(usage, (tuple, list)) and len(usage) == 2:
        return int(usage[0] or 0), int(usage[1] or 0), None
    if not isinstance(usage, dict):
        meta = getattr(usage, "usage_metadata", None)
        if not isinstance(meta, dict):
            return 0, 0, None
        rmeta = getattr(usage, "response_metadata", None)
        model = rmeta.get("model_name") if isinstance(rmeta, dict) else None
        in_tok, out_tok, _ = _tokens_from(meta)
        return in_tok, out_tok, model

    nested = usage.get("token_usage") or usage.get("usage") or usage.get("usage_metadata")
    if nested and isinstance(nested, dict):
        i, o, m = _tokens_from(nested)
        return i, o, m or usage.get("model") or usage.get("model_name")

    def pick(*names):
        for n in names:
            v = usage.get(n)
            if v is not None:
                return int(v)
        return 0

    return (
        pick("input_tokens", "prompt_tokens", "in_tokens"),
        pick("output_tokens", "completion_tokens", "out_tokens"),
        usage.get("model") or usage.get("model_name"),
    )


def record(usage, model: str | None = None) -> dict:
    """Bank the cost of one LLM call and fire any alert it crosses.

    Returns {model, input_tokens, output_tokens, cost_usd}. Zero-token usage
    is still recorded, so the request count stays honest.
    """
    in_tok, out_tok, usage_model = _tokens_from(usage)
    model = model or usage_model or default_model()
    cost = cost_of(in_tok, out_tok, model)
    now = _now()
    day, month = _keys(now)

    with _db_lock, _db() as conn:
        conn.execute(
            "INSERT INTO usage (ts, day, month, model, input_tokens, output_tokens, cost_usd)"
            " VALUES (?,?,?,?,?,?,?)",
            (now.isoformat(), day, month, model, in_tok, out_tok, cost),
        )

    _check_thresholds()
    return {
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost, 6),
    }


def _totals(conn: sqlite3.Connection) -> tuple[float, float, int]:
    day, month = _keys()
    d = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) AS c, COUNT(*) AS n FROM usage WHERE day = ?", (day,)
    ).fetchone()
    m = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) AS c FROM usage WHERE month = ?", (month,)
    ).fetchone()
    return float(d["c"]), float(m["c"]), int(d["n"])


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #

_active_lock = threading.Lock()
_active = 0


def check(ip: str | None = None) -> None:
    """Let a request through, or raise SpendRefused explaining why not.

    Caps are checked against spend already banked: a request cannot be priced
    before it runs, so the last one through may push slightly past the cap.
    That is the intended shape, the next one is refused.

    A request that passes stamps the IP's cooldown clock. A refused one does
    not, so being refused never extends the wait.
    """
    ip = (ip or "unknown").strip() or "unknown"
    cd = cooldown_s()

    with _db_lock, _db() as conn:
        day_usd, month_usd, _ = _totals(conn)
        d_cap, m_cap = daily_cap(), monthly_cap()

        if day_usd >= d_cap:
            _alerts_pending(conn, day_usd, month_usd)
            raise SpendRefused(
                "daily_cap",
                f"the observatory has spent its daily budget of ${d_cap:.2f}. it runs on "
                "one personal API key, so it takes the rest of the day off and resets at "
                "midnight UTC. thanks for looking.",
                status_code=429,
                retry_after=_seconds_to_utc_midnight(),
            )

        if month_usd >= m_cap:
            _alerts_pending(conn, day_usd, month_usd)
            raise SpendRefused(
                "monthly_cap",
                f"the observatory has spent its monthly budget of ${m_cap:.2f} and is "
                "closed for the month. the graph and the demo mode still work.",
                status_code=429,
            )

        if cd > 0:
            row = conn.execute("SELECT last_ts FROM hits WHERE ip = ?", (ip,)).fetchone()
            if row is not None:
                waited = time.time() - float(row["last_ts"])
                if waited < cd:
                    left = cd - waited
                    raise SpendRefused(
                        "cooldown",
                        f"one run at a time please, the agents are thinking. try again in "
                        f"{int(left) + 1}s.",
                        status_code=429,
                        retry_after=left,
                    )

        conn.execute(
            "INSERT INTO hits (ip, last_ts) VALUES (?,?)"
            " ON CONFLICT(ip) DO UPDATE SET last_ts = excluded.last_ts",
            (ip, time.time()),
        )

    _prune_hits()


@contextmanager
def guard(ip: str | None = None):
    """check(ip) plus a slot in the global concurrency ceiling.

    Use around the whole run so the slot is released even if the graph blows up:

        with spend.guard(ip):
            ...
    """
    _acquire_slot()
    try:
        check(ip)
    except BaseException:
        _release_slot()
        raise
    try:
        yield
    finally:
        _release_slot()


def _acquire_slot() -> None:
    global _active
    with _active_lock:
        if _active >= max_concurrent():
            raise SpendRefused(
                "busy",
                "the observatory is already running as many agent teams as it can hold. "
                "give it a few seconds and press run again.",
                status_code=503,
                retry_after=5,
            )
        _active += 1


def _release_slot() -> None:
    global _active
    with _active_lock:
        _active = max(0, _active - 1)


def _seconds_to_utc_midnight() -> float:
    now = _now()
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return (tomorrow.timestamp() + 86400) - now.timestamp()


def _prune_hits() -> None:
    """Keep the cooldown table from growing forever: an IP older than a day
    has no cooldown left to enforce."""
    try:
        with _db() as conn:
            conn.execute("DELETE FROM hits WHERE last_ts < ?", (time.time() - 86400,))
    except sqlite3.Error as exc:  # a full table is not worth failing a request
        log.warning("spend: could not prune hits: %s", exc)


# --------------------------------------------------------------------------- #
# Status, for the UI badge
# --------------------------------------------------------------------------- #

def status() -> dict:
    with _db() as conn:
        day_usd, month_usd, requests = _totals(conn)
    d_cap, m_cap = daily_cap(), monthly_cap()
    day_pct = 100.0 * day_usd / d_cap if d_cap > 0 else 0.0
    month_pct = 100.0 * month_usd / m_cap if m_cap > 0 else 0.0
    worst = max(day_pct, month_pct)
    return {
        "today_usd": round(day_usd, 4),
        "month_usd": round(month_usd, 4),
        "daily_cap_usd": round(d_cap, 2),
        "monthly_cap_usd": round(m_cap, 2),
        "day_pct": round(day_pct, 1),
        "month_pct": round(month_pct, 1),
        "requests_today": requests,
        "cooldown_s": cooldown_s(),
        "max_concurrent": max_concurrent(),
        "in_flight": _active,
        # one word the badge can colour on: green, amber, red
        "state": "blocked" if worst >= 100 else "warn" if worst >= 80 else "ok",
    }


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #

def _check_thresholds() -> None:
    try:
        with _db() as conn:
            day_usd, month_usd, _ = _totals(conn)
            _alerts_pending(conn, day_usd, month_usd)
    except sqlite3.Error as exc:
        log.warning("spend: threshold check failed: %s", exc)


def _alerts_pending(conn: sqlite3.Connection, day_usd: float, month_usd: float) -> None:
    """Claim and send every threshold crossed but not yet announced today.

    The claim is the INSERT: SQLite's primary key on (day, kind) is what makes
    the dedupe atomic, so two concurrent requests crossing 80% at the same
    moment produce one email, not two.
    """
    d_cap, m_cap = daily_cap(), monthly_cap()
    for scope, spent, cap in (("daily", day_usd, d_cap), ("monthly", month_usd, m_cap)):
        if cap <= 0:
            continue
        pct = 100.0 * spent / cap
        claimed = [t for t in ALERT_THRESHOLDS
                   if pct >= t and _claim_alert(conn, f"{scope}-{t}")]
        if claimed:
            # One expensive call can clear several thresholds at once. Claim
            # every one of them so none fires later, but announce only the
            # highest: the owner wants to hear "cap reached", not that plus two
            # stale warnings about a line already behind us.
            if not _send_alert(scope, max(claimed), spent, cap, pct):
                # Delivery failed, so hand the claims back. The claim exists to
                # stop DUPLICATE announcements, not to record an attempt, and an
                # alert nobody received was never announced. Without this, one
                # blip of SES trouble at the moment a cap is crossed silences
                # that threshold for the rest of the day, which is precisely
                # when the owner most needs to hear from it.
                for t in claimed:
                    _release_alert(conn, f"{scope}-{t}")


def _claim_alert(conn: sqlite3.Connection, kind: str) -> bool:
    """True exactly once per (today, kind), ever."""
    day, _ = _keys()
    cur = conn.execute(
        "INSERT OR IGNORE INTO alerts (day, kind, sent_at) VALUES (?,?,?)",
        (day, kind, _now().isoformat()),
    )
    return cur.rowcount > 0


def _release_alert(conn: sqlite3.Connection, kind: str) -> None:
    """Undo a claim whose announcement never reached anyone."""
    day, _ = _keys()
    conn.execute("DELETE FROM alerts WHERE day = ? AND kind = ?", (day, kind))


def _send_alert(scope: str, threshold: int, spent: float, cap: float, pct: float) -> bool:
    """Tell the owner. NEVER raises, whatever goes wrong downstream.

    This runs inside check() and record(), which means it runs inside a visitor's
    request. An alert that breaks the request is worse than an alert that gets
    lost, so the whole thing, formatting included, is wrapped and the log keeps
    a copy of anything that could not be delivered.

    Returns True when the alert was delivered, or deliberately logged because no
    mailer is configured. False means it reached nobody and the caller should
    let the threshold fire again later.
    """
    try:
        # what the body says about blocking follows the real state, not the
        # threshold: a warning email must never claim "nothing is blocked yet"
        # while requests are already being refused.
        tripped = pct >= 100
        subject = (
            f"[observatory] {scope} cap reached: ${spent:.2f} of ${cap:.2f}"
            if threshold >= 100 else
            f"[observatory] {scope} spend at {threshold}%: ${spent:.2f} of ${cap:.2f}"
        )
        st = status()
        body = "\n".join([
            f"agent observatory {scope} spend is at {pct:.0f}% of its cap.",
            "",
            f"  {scope} spent : ${spent:.2f}",
            f"  {scope} cap   : ${cap:.2f}",
            f"  today         : ${st['today_usd']:.2f} of ${st['daily_cap_usd']:.2f}",
            f"  this month    : ${st['month_usd']:.2f} of ${st['monthly_cap_usd']:.2f}",
            f"  requests today: {st['requests_today']}",
            "",
            ("requests are now being refused with a friendly message until the cap resets."
             if tripped else
             "nothing is blocked yet, this is the early warning."),
            "",
            "raise the ceiling with OBS_DAILY_USD / OBS_MONTHLY_USD, or leave it be.",
        ])
        log.warning("spend alert: %s", subject)
        return _email(subject, body)
    except Exception as exc:  # noqa: BLE001 - alerting never breaks a request
        log.warning("spend: alert %s-%s could not be delivered: %s", scope, threshold, exc)
        return False


def _email(subject: str, body: str) -> bool:
    """SES if configured, log line if not. Swallows everything.

    True means it got somewhere: either SES accepted it, or no mailer is
    configured and the log line IS the delivery. False means a mailer was
    configured and it failed, which is the only case worth retrying.
    """
    sender = os.environ.get("OBS_SES_FROM")
    to = os.environ.get("OBS_ALERT_TO", DEFAULT_ALERT_TO)
    if not sender:
        log.warning("spend: no OBS_SES_FROM, alert not emailed:\n%s\n%s", subject, body)
        return True
    try:
        import boto3

        region = os.environ.get("OBS_SES_REGION") or os.environ.get("AWS_REGION") or "eu-central-1"
        boto3.client("ses", region_name=region).send_email(
            Source=sender,
            Destination={"ToAddresses": [to]},
            Message={
                "Subject": {"Data": subject},
                "Body": {"Text": {"Data": body}},
            },
        )
        log.info("spend: alert emailed to %s", to)
        return True
    except Exception as exc:  # noqa: BLE001 - an alert must never break a request
        log.warning("spend: SES send failed (%s), alert logged only:\n%s", exc, body)
        return False


# --------------------------------------------------------------------------- #
# LangChain / LangGraph callback
# --------------------------------------------------------------------------- #

try:  # keep this module importable, and testable, without langchain present
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - exercised only in a bare environment
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


class SpendCallback(BaseCallbackHandler):
    """Hand this to any LangChain call and token usage meters itself.

        llm.astream(prompt, config={"callbacks": [spend.SpendCallback()]})

    Streaming only reports usage when the model is asked for it, so pass
    stream_usage=True to ChatOpenAI, otherwise a streamed call meters as zero.
    """

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: D102
        try:
            for usage, model in _usages_in(response):
                record(usage, model=model)
        except Exception as exc:  # noqa: BLE001 - metering must not break a run
            log.warning("spend: could not record usage: %s", exc)


def _usages_in(response):
    """Pull every (usage, model) pair out of an LLMResult."""
    out = []
    llm_output = getattr(response, "llm_output", None) or {}
    model = llm_output.get("model_name") if isinstance(llm_output, dict) else None

    if isinstance(llm_output, dict) and llm_output.get("token_usage"):
        out.append((llm_output["token_usage"], model))
        return out

    for batch in getattr(response, "generations", []) or []:
        for gen in batch:
            message = getattr(gen, "message", None)
            meta = getattr(message, "usage_metadata", None)
            if meta:
                rmeta = getattr(message, "response_metadata", None) or {}
                out.append((meta, rmeta.get("model_name") or model))
    return out


# --------------------------------------------------------------------------- #
# Test support
# --------------------------------------------------------------------------- #

def _reset_runtime() -> None:
    """Drop in-process state (the concurrency counter). Tests use this, the
    server has no reason to."""
    global _active
    with _active_lock:
        _active = 0
