"""The spend guard: the thing standing between a public URL and a real bill.

Four properties matter more than the rest, and each has a test here:
  cap refusal      over budget, requests are refused, politely
  cooldown refusal one IP cannot machine-gun the key
  alert dedupe     the owner hears about a threshold once a day, not per request
  mailer isolation a broken SES can never break a request
"""

import sqlite3
import time

import pytest

from conftest import burn


# --------------------------------------------------------------------------- #
# Metering
# --------------------------------------------------------------------------- #

def test_cost_is_priced_per_model(spend):
    # gpt-4o-mini at 0.15 / 0.60 per 1M tokens
    assert spend.cost_of(1_000_000, 0, "gpt-4o-mini") == pytest.approx(0.15)
    assert spend.cost_of(0, 1_000_000, "gpt-4o-mini") == pytest.approx(0.60)
    # a dated snapshot prices as its base model
    assert spend.price_for("gpt-4o-mini-2024-07-18") == spend.price_for("gpt-4o-mini")
    # an unknown model is priced high on purpose, never free
    assert spend.price_for("some-new-frontier-model") == spend.FALLBACK_PRICE


def test_record_accepts_every_usage_shape(spend):
    langchain = spend.record({"input_tokens": 1000, "output_tokens": 500}, model="gpt-4o-mini")
    openai = spend.record({"prompt_tokens": 1000, "completion_tokens": 500}, model="gpt-4o-mini")
    assert langchain["cost_usd"] == openai["cost_usd"] > 0
    assert spend.status()["requests_today"] == 2


def test_meter_survives_a_restart(spend):
    burn(spend, 0.40)
    before = spend.status()["today_usd"]
    # a "restart" is a new connection to the same file, which is all the module
    # ever holds, so re-reading proves persistence
    with sqlite3.connect(spend.db_path()) as conn:
        total = conn.execute("SELECT SUM(cost_usd) FROM usage").fetchone()[0]
    assert total == pytest.approx(before, abs=1e-6)
    assert spend.status()["today_usd"] == pytest.approx(before)


# --------------------------------------------------------------------------- #
# Cap refusal
# --------------------------------------------------------------------------- #

def test_under_the_daily_cap_requests_pass(spend):
    burn(spend, 0.50)  # cap is 1.00
    spend.check("1.2.3.4")  # does not raise


def test_over_the_daily_cap_requests_are_refused(spend):
    burn(spend, 1.01)  # cap is 1.00
    with pytest.raises(spend.SpendRefused) as excinfo:
        spend.check("1.2.3.4")
    refusal = excinfo.value
    assert refusal.reason == "daily_cap"
    assert refusal.status_code == 429
    assert "$1.00" in refusal.message
    assert refusal.retry_after > 0


def test_over_the_monthly_cap_requests_are_refused(spend, monkeypatch):
    monkeypatch.setenv("OBS_DAILY_USD", "100.00")  # take the daily cap out of the way
    monkeypatch.setenv("OBS_MONTHLY_USD", "2.00")
    burn(spend, 2.50)
    with pytest.raises(spend.SpendRefused) as excinfo:
        spend.check("1.2.3.4")
    assert excinfo.value.reason == "monthly_cap"


def test_a_refusal_is_serializable_for_the_ui_not_a_stack_trace(spend):
    burn(spend, 1.01)
    with pytest.raises(spend.SpendRefused) as excinfo:
        spend.check("1.2.3.4")
    payload = excinfo.value.as_dict()
    assert payload["reason"] == "daily_cap"
    assert payload["message"] and "Traceback" not in payload["message"]
    assert payload["status"]["state"] == "blocked"
    assert payload["status"]["day_pct"] > 100


def test_status_reports_the_badge_numbers(spend):
    burn(spend, 0.85)
    st = spend.status()
    assert st["daily_cap_usd"] == 1.00
    assert st["monthly_cap_usd"] == 10.00
    assert st["today_usd"] == pytest.approx(0.85, abs=0.001)
    assert st["day_pct"] == pytest.approx(85, abs=0.5)
    assert st["state"] == "warn"


# --------------------------------------------------------------------------- #
# Cooldown and capacity
# --------------------------------------------------------------------------- #

def test_a_second_request_from_one_ip_is_refused_during_the_cooldown(spend):
    spend.check("9.9.9.9")
    with pytest.raises(spend.SpendRefused) as excinfo:
        spend.check("9.9.9.9")
    refusal = excinfo.value
    assert refusal.reason == "cooldown"
    assert 0 < refusal.retry_after <= 20


def test_the_cooldown_is_per_ip(spend):
    spend.check("9.9.9.9")
    spend.check("8.8.8.8")  # a different visitor is not punished


def test_the_cooldown_expires(spend, monkeypatch):
    monkeypatch.setenv("OBS_COOLDOWN_S", "0.05")
    spend.check("9.9.9.9")
    time.sleep(0.06)
    spend.check("9.9.9.9")


def test_being_refused_does_not_extend_the_wait(spend, monkeypatch):
    monkeypatch.setenv("OBS_COOLDOWN_S", "0.20")
    spend.check("9.9.9.9")
    time.sleep(0.10)
    with pytest.raises(spend.SpendRefused):
        spend.check("9.9.9.9")  # a hammered retry must not reset the clock
    time.sleep(0.12)
    spend.check("9.9.9.9")


def test_the_cooldown_survives_a_restart(spend):
    spend.check("9.9.9.9")
    # the module keeps nothing about IPs in memory, so a "fresh process" reading
    # the same file must still see the hit
    with sqlite3.connect(spend.db_path()) as conn:
        rows = conn.execute("SELECT ip FROM hits").fetchall()
    assert [r[0] for r in rows] == ["9.9.9.9"]
    with pytest.raises(spend.SpendRefused):
        spend.check("9.9.9.9")


def test_concurrency_ceiling_refuses_the_overflow(spend, monkeypatch):
    monkeypatch.setenv("OBS_MAX_CONCURRENT", "2")
    monkeypatch.setenv("OBS_COOLDOWN_S", "0")
    with spend.guard("1.1.1.1"):
        with spend.guard("2.2.2.2"):
            assert spend.status()["in_flight"] == 2
            with pytest.raises(spend.SpendRefused) as excinfo:
                with spend.guard("3.3.3.3"):
                    pass
            assert excinfo.value.reason == "busy"
            assert excinfo.value.status_code == 503
    assert spend.status()["in_flight"] == 0


def test_a_crashing_run_releases_its_slot(spend, monkeypatch):
    monkeypatch.setenv("OBS_COOLDOWN_S", "0")
    with pytest.raises(ValueError):
        with spend.guard("1.1.1.1"):
            raise ValueError("the graph blew up")
    assert spend.status()["in_flight"] == 0


def test_a_capped_guard_does_not_leak_a_slot(spend):
    burn(spend, 1.01)
    for _ in range(5):
        with pytest.raises(spend.SpendRefused):
            with spend.guard("1.1.1.1"):
                pass
    assert spend.status()["in_flight"] == 0


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #

def test_crossing_fifty_percent_alerts_once_not_per_request(spend, sent):
    burn(spend, 0.55)  # 55% of the 1.00 cap
    assert [s for s, _ in sent] == [
        "[observatory] daily spend at 50%: $0.55 of $1.00",
    ]
    for _ in range(5):
        burn(spend, 0.01)  # still under 80%, still the same threshold
    assert len(sent) == 1


def test_each_threshold_alerts_once_and_the_cap_says_so(spend, sent):
    burn(spend, 0.55)
    burn(spend, 0.30)  # 85%
    burn(spend, 0.20)  # 105%
    kinds = [s for s, _ in sent]
    assert len(kinds) == 3
    assert "50%" in kinds[0] and "80%" in kinds[1]
    assert "cap reached" in kinds[2]
    assert "refused" in sent[2][1]


def test_one_big_call_clearing_every_threshold_sends_one_email(spend, sent):
    burn(spend, 1.20)  # straight past 50, 80 and 100 in a single call
    assert len(sent) == 1
    subject, body = sent[0]
    assert "cap reached" in subject
    assert "refused" in body          # and it describes the real state
    assert "nothing is blocked" not in body
    # the thresholds it skipped are claimed, so they never fire late
    with sqlite3.connect(spend.db_path()) as conn:
        kinds = sorted(r[0] for r in conn.execute("SELECT kind FROM alerts").fetchall())
    assert kinds == ["daily-100", "daily-50", "daily-80"]


def test_a_tripped_cap_does_not_re_alert_on_every_refusal(spend, sent):
    burn(spend, 1.20)
    assert len(sent) == 1
    for _ in range(5):
        with pytest.raises(spend.SpendRefused):
            spend.check("1.2.3.4")
    assert len(sent) == 1


def test_dedupe_is_persisted_not_in_memory(spend, sent):
    burn(spend, 0.55)
    assert len(sent) == 1
    with sqlite3.connect(spend.db_path()) as conn:
        kinds = [r[0] for r in conn.execute("SELECT kind FROM alerts").fetchall()]
    assert kinds == ["daily-50"]
    # a fresh process re-reads that ledger, so the alert is not sent again
    burn(spend, 0.01)
    assert len(sent) == 1


def test_the_alert_names_the_owner_by_default(spend, monkeypatch):
    captured = {}
    monkeypatch.setenv("OBS_SES_FROM", "observatory@b4rruf3t.com")
    monkeypatch.setattr(spend, "_ses_send", None, raising=False)

    class FakeSES:
        def send_email(self, **kwargs):
            captured.update(kwargs)

    fake_boto3 = type("boto3", (), {"client": staticmethod(lambda *a, **k: FakeSES())})
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake_boto3)

    burn(spend, 0.55)
    assert captured["Destination"]["ToAddresses"] == ["igor.yago@gmail.com"]
    assert captured["Source"] == "observatory@b4rruf3t.com"
    assert "50%" in captured["Message"]["Subject"]["Data"]


# --------------------------------------------------------------------------- #
# A broken mailer is never the user's problem
# --------------------------------------------------------------------------- #

def test_a_failing_ses_cannot_break_a_request(spend, monkeypatch):
    monkeypatch.setenv("OBS_SES_FROM", "observatory@b4rruf3t.com")

    class ExplodingSES:
        def send_email(self, **kwargs):
            raise RuntimeError("MessageRejected: email address is not verified")

    fake_boto3 = type("boto3", (), {"client": staticmethod(lambda *a, **k: ExplodingSES())})
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake_boto3)

    burn(spend, 0.55)          # crosses 50%, the alert path runs and fails
    spend.check("1.2.3.4")     # the request still goes through
    assert spend.status()["today_usd"] == pytest.approx(0.55, abs=0.001)


def test_a_missing_boto3_cannot_break_a_request(spend, monkeypatch):
    monkeypatch.setenv("OBS_SES_FROM", "observatory@b4rruf3t.com")
    monkeypatch.setitem(__import__("sys").modules, "boto3", None)  # import raises
    burn(spend, 0.55)
    spend.check("1.2.3.4")


def test_a_mailer_that_raises_outright_is_still_contained(spend, monkeypatch):
    """Not just a failing SES call: the whole alert path is wrapped, so even a
    mailer that raises before it reaches boto3 cannot reach the visitor."""
    def boom(subject, body):
        raise RuntimeError("the mailer itself is broken")

    monkeypatch.setattr(spend, "_email", boom)

    burn(spend, 1.20)  # records, crosses 50 / 80 / 100, every alert raises
    assert spend.status()["today_usd"] == pytest.approx(1.20, abs=0.001)

    # and the cap it tripped is still enforced, as a refusal and not a 500
    with pytest.raises(spend.SpendRefused) as excinfo:
        spend.check("1.2.3.4")
    assert excinfo.value.reason == "daily_cap"


# --------------------------------------------------------------------------- #
# The LangChain callback
# --------------------------------------------------------------------------- #

class _FakeMessage:
    def __init__(self, usage, model):
        self.usage_metadata = usage
        self.response_metadata = {"model_name": model}


class _FakeGeneration:
    def __init__(self, message):
        self.message = message


class _FakeResult:
    def __init__(self, generations, llm_output=None):
        self.generations = generations
        self.llm_output = llm_output or {}


def test_the_callback_meters_a_langchain_result(spend):
    result = _FakeResult(
        [[_FakeGeneration(_FakeMessage({"input_tokens": 200_000, "output_tokens": 100_000},
                                       "gpt-4o-mini"))]]
    )
    spend.SpendCallback().on_llm_end(result)
    # 200k in at $0.15/M + 100k out at $0.60/M = $0.03 + $0.06
    assert spend.status()["today_usd"] == pytest.approx(0.09, abs=1e-6)


def test_the_callback_meters_the_openai_token_usage_shape(spend):
    result = _FakeResult([[]], llm_output={
        "model_name": "gpt-4o",
        "token_usage": {"prompt_tokens": 200_000, "completion_tokens": 100_000},
    })
    spend.SpendCallback().on_llm_end(result)
    # 200k in at $2.50/M + 100k out at $10.00/M = $0.50 + $1.00
    assert spend.status()["today_usd"] == pytest.approx(1.50, abs=1e-6)


def test_a_junk_response_cannot_break_a_run(spend):
    spend.SpendCallback().on_llm_end(object())  # no generations, no llm_output
    assert spend.status()["today_usd"] == 0
