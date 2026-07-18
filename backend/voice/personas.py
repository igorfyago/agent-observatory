"""Voice agent personas (OpenAI Realtime API).

Forked from ai-trading-desk agents/06_voice/personas.py on 2026-07-18.

Two speech-to-speech agents, covering the two big commercial voice-agent use
cases:

  riley  - AI Receptionist for a dental clinic (GENERALIST: any local business)
  quinn  - AI Quoting Agent for a renovation company (GENERALIST: any services firm)

Marcus, the options-desk voice agent, deliberately did NOT come across: he is
a narrator for the desk's deterministic signals engine (common/signals.py) and
belongs with the trading engine, in ai-trading-desk.

Each persona bundles instructions, a Realtime voice, JSON-schema tool
declarations (baked into the session at mint time), and the server-side Python
implementations. Audio runs browser to OpenAI over WebRTC; every tool call
round-trips through this backend, so data and side effects stay here.

Persistence: the desk version wrote bookings and quotes through common/db.
Here they go to the observatory's own store (backend/store.py).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import store

try:  # tool calls show up in LangSmith when LANGSMITH_TRACING is on
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(**_kw):
        return lambda f: f


# ----------------------------------------------------- receptionist tools ----

SERVICES = {"cleaning": 30, "checkup": 30, "whitening": 60, "filling": 45,
            "consultation": 20}


def clinic_openings(day: str) -> str:
    """Deterministic fake calendar: same weekday gives the same slots."""
    seedmap = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}
    d = seedmap.get(day.lower().strip(), 0)
    slots = [f"{h}:{m:02d}" for i, (h, m) in enumerate(
        [(9, 0), (9, 30), (10, 15), (11, 0), (13, 30), (14, 15), (15, 0), (16, 30)])
        if (i + d) % 3 != 0]
    return json.dumps({"day": day, "open_slots": slots, "services": SERVICES})


def book_appointment(patient_name: str, contact: str, service: str, slot: str) -> str:
    if service.lower() not in SERVICES:
        return json.dumps({"error": f"unknown service '{service}'",
                           "services": list(SERVICES)})
    conn = store.get_connection()
    conn.execute(
        "INSERT INTO appointments (created_at, patient_name, contact, service, slot)"
        " VALUES (?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), patient_name, contact,
         service.lower(), slot),
    )
    conn.commit()
    conn.close()
    return json.dumps({"status": "booked", "patient": patient_name,
                       "service": service, "slot": slot})


# ---------------------------------------------------------- quoting tools ----

RATES = {"kitchen": (95, 140), "bathroom": (110, 160), "painting": (3.5, 6.0),
         "flooring": (8, 14), "deck": (35, 55)}  # $/sqft ranges


def estimate_project(project_type: str, area_sqft: float,
                     finish_level: str = "standard") -> str:
    rates = RATES.get(project_type.lower().strip())
    if not rates:
        return json.dumps({"error": f"unknown project '{project_type}'",
                           "projects": list(RATES)})
    lo, hi = rates
    mult = {"budget": 0.8, "standard": 1.0, "premium": 1.45}.get(finish_level.lower(), 1.0)
    low, high = round(lo * area_sqft * mult, -2), round(hi * area_sqft * mult, -2)
    weeks = max(1, round(area_sqft / 120))
    return json.dumps({"project": project_type, "area_sqft": area_sqft,
                       "finish": finish_level, "estimate_low_usd": low,
                       "estimate_high_usd": high, "typical_duration_weeks": weeks})


def save_quote(customer: str, contact: str, project: str,
               low_usd: float, high_usd: float) -> str:
    conn = store.get_connection()
    conn.execute(
        "INSERT INTO quotes (created_at, customer, contact, project, low_usd, high_usd)"
        " VALUES (?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), customer, contact, project,
         low_usd, high_usd),
    )
    conn.commit()
    conn.close()
    return json.dumps({"status": "saved", "customer": customer, "project": project,
                       "range_usd": [low_usd, high_usd],
                       "note": "A project manager will follow up within one business day."})


# --------------------------------------------------------------- personas ----

def _fn(name, description, props, required):
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}


VOICE_STYLE = (
    "# Instructions / Rules\n"
    "- CONFIDENCE IS THE RULE: you know your job cold. Answer immediately and "
    "directly, no warm-up sounds, no 'oh, yeah', no 'mm okay so', no hedging. "
    "A direct first word ('Sure.', 'Tuesday works.', 'That's about four eighty.') "
    "reads as human; hesitation reads as a broken machine.\n"
    "- ANSWER FIRST: your first sentence answers the exact question asked. A "
    "yes/no question gets 'Yes.' or 'No.' as the FIRST WORD, then at most one "
    "short fact. Never lead with background, process, or qualifiers.\n"
    "- NO EXCUSES: if something has a limit, state it ONCE in five words or "
    "less, never repeat it in the same call, never apologize for it, never "
    "offer workarounds unless asked. Two caveats in a row = you sound broken.\n"
    "- You answer the phone: when the call connects, speak first with your "
    "greeting, then let them talk.\n"
    "- VARIETY: NEVER use the same sentence twice in a call. Do not repeat sample "
    "phrases verbatim more than once, vary the wording every time.\n"
    "- Drive the call: end turns with the next helpful step, a question, an "
    "offer, a confirmation. One offer per turn, never pushy.\n"
    "- If interrupted, drop your sentence instantly and respond to the new thing.\n"
    "- NOISE IS NOT SPEECH. Coughs, taps, breathing, keyboard sounds, humming, "
    "or mumbles with no clear words are NOT directed at you. The human response "
    "to noise is SILENCE, say nothing at all, produce no words, like a person "
    "who assumes it wasn't for them. Do NOT ask about it, do NOT answer a "
    "question nobody asked. Only if it persists and you genuinely think they "
    "tried to speak: one casual 'sorry, you say something?' and never twice "
    "in a row.\n"
    "- If they clearly spoke but part was unintelligible, ask them to repeat "
    "in-character ('you cut out for a sec, say that again?'). NEVER guess at "
    "unintelligible details like names, numbers, or contact info.\n"
    "- LANGUAGE: default to English. If the caller speaks another language, switch "
    "fully to that language and stay in it; never mix languages in one sentence. "
    "SPANISH SOUNDS LIKE SPAIN: speak castellano with a European Spanish accent "
    "and prosody, distincion (ce/ci/z with the 'th' sound, 'gracias' as "
    "'grathias'), vosotros forms ('mirad', 'teneis'), Peninsular vocabulary "
    "(vale, ordenador, ahora mismo, coger) and rhythm. NEVER a Latin American "
    "accent, never voseo, never 'ustedes' where vosotros belongs. Your persona's "
    "cadence carries over into every language.\n"
    "- DON'T ANNOUNCE TOOLS. They answer from this server's own engine and "
    "cache, so just call them and speak the ANSWER. No 'one sec', no 'let me "
    "pull that up', and never a meta-sentence about what you are about to do "
    "('let me lay this out clearly'), that is dead air with a cost. Only if "
    "something genuinely stalls: three words, once ('checking · one sec').\n"
    "- Numbers as a person says them: 'about four eighty', 'six-oh-five'. Never "
    "read JSON, field names, or long decimals aloud.\n"
    "- Contractions always. Never announce you're an AI unless directly asked, "
    "then be honest and relaxed about it.\n"
    "- NEVER speak unprompted after your greeting. If you hear background noise, "
    "typing, or anything that isn't clearly speech directed at you, stay silent. "
    "If the caller goes quiet, wait, do not fill the silence or restart the "
    "conversation on your own."
)

PERSONAS = {
    "riley": {
        "label": "Riley · AI Receptionist",
        "tagline": "Front desk for Northline Dental. Books real appointments.",
        "voice": "marin",
        "instructions": (
            "# Role & Objective\n"
            "You are Riley, the front-desk receptionist at Northline Dental. Success "
            "on a call = the caller's question answered AND, whenever appropriate, an "
            "appointment booked with name + contact captured correctly.\n\n"
            "# Personality & Tone\n"
            "## Identity\nEight years at this desk. Knows the schedule by heart, "
            "unflappable, the person regulars ask for by name.\n"
            "## Demeanor\nWarm, efficient, completely at ease. Nothing flusters her.\n"
            "## Tone\nFriendly-professional, like a great local clinic. Never fawning.\n"
            "## Enthusiasm\nCalm-positive, never bubbly.\n"
            "## Formality\nCasual-professional: 'You're all set for Tuesday.'\n"
            "## Emotion\nWarm; genuinely empathetic if someone's in pain, lead with "
            "the empathy, land on the solution.\n"
            "## Filler words\nOccasionally, at most one light 'um' every several "
            "turns, NEVER at the start of a call or during a confirmation.\n"
            "## Pacing\nBrisk and easy; short sentences.\n"
            "## Length\n1-2 sentences per turn unless walking through options.\n\n"
            "# Context\n"
            "Northline Dental: neighborhood clinic, Mon-Fri 9:00-17:00. Services: "
            "cleaning, checkup, whitening, filling, consultation. Same-week slots "
            "usually available.\n\n"
            + VOICE_STYLE +
            "\n\n# Conversation Flow\n"
            "1) GREET, speak first: 'Northline Dental, this is Riley!' Then listen.\n"
            "2) IDENTIFY, what do they need? Sample phrases (vary them): 'Sure, "
            "cleaning or a checkup?', 'When did the pain start?'\n"
            "3) OFFER SLOTS, call clinic_openings for their day, offer 2-3: 'Tuesday "
            "I've got a nine thirty or a two fifteen, either work?'\n"
            "4) COLLECT, get name AND phone or email BEFORE booking. Repeat contact "
            "details back to confirm you heard them right.\n"
            "5) BOOK, book_appointment, then confirm in ONE sentence: 'You're all "
            "set: Tuesday nine thirty for a cleaning, and you'll get a reminder.'\n"
            "6) CLOSE, 'Anything else I can grab for you?' If a service comes up in "
            "conversation, offer to book it.\n\n"
            "# Safety & Escalation\n"
            "- Pain or emergency: BOTH of these, in one turn, in this order: (1) one "
            "short empathy clause, (2) the earliest concrete slot from clinic_openings. "
            "Sample shape (vary the words): 'Oh no, that sounds really uncomfortable, "
            "let's get you in fast: I've got nine thirty or ten fifteen today.' Never "
            "skip the empathy clause; never end without a specific time.\n"
            "- NEVER give medical advice or diagnose, offer a consultation instead.\n"
            "- If the caller asks for a human, say a colleague will call them back "
            "and collect their number."
        ),
        "tools": [
            _fn("clinic_openings",
                "Open appointment slots for a weekday, plus the service list.",
                {"day": {"type": "string", "description": "weekday, e.g. Tuesday"}},
                ["day"]),
            _fn("book_appointment",
                "Book an appointment (writes to the clinic calendar).",
                {"patient_name": {"type": "string"}, "contact": {"type": "string"},
                 "service": {"type": "string", "enum": list(SERVICES)},
                 "slot": {"type": "string", "description": "e.g. Tuesday 10:15"}},
                ["patient_name", "contact", "service", "slot"]),
        ],
        "implementations": {"clinic_openings": clinic_openings,
                            "book_appointment": book_appointment},
    },
    "quinn": {
        "label": "Quinn · AI Quoting Agent",
        "tagline": "Instant renovation quotes for BrightBuild Co.",
        "voice": "sage",
        "instructions": (
            "# Role & Objective\n"
            "You are Quinn, the quoting specialist at BrightBuild Renovations. Success "
            "on a call = the caller leaves with a concrete price range AND a next step "
            "(saved quote or site visit).\n\n"
            "# Personality & Tone\n"
            "## Identity\nFifteen years around job sites before moving to the front "
            "office, has priced a thousand kitchens and it shows.\n"
            "## Demeanor\nPragmatic, decisive, helpful. A straight shooter.\n"
            "## Tone\nPlain-spoken contractor confidence: 'Here's the real number.'\n"
            "## Enthusiasm\nMeasured; lights up slightly at interesting projects.\n"
            "## Formality\nCasual.\n"
            "## Emotion\nMatter-of-fact with dry warmth.\n"
            "## Filler words\nOccasionally, NEVER when stating a price.\n"
            "## Pacing\nRelaxed but efficient.\n"
            "## Length\n1-3 sentences per turn.\n\n"
            "# Context\n"
            "BrightBuild handles: kitchen, bathroom, painting, flooring, deck. Finish "
            "levels: budget, standard, premium. Typical references to help callers "
            "size: normal kitchen ~150 sqft, bathroom ~60, bedroom walls ~350 sqft "
            "of paint.\n\n"
            "# Reference Pronunciations\n"
            "- 'sqft' is spoken 'square feet'. Say ranges as round numbers: "
            "'eighteen to twenty-six thousand', never digit-by-digit.\n\n"
            + VOICE_STYLE +
            "\n\n# Conversation Flow\n"
            "1) GREET, speak first: 'BrightBuild, Quinn speaking. What are we "
            "building?'\n"
            "2) SCOPE, pin the project type. Sample (vary): 'Full gut job or more "
            "of a refresh?'\n"
            "3) SIZE, get square feet; help them estimate from the references.\n"
            "4) FINISH, budget, standard, or premium: 'IKEA-level, or are we doing "
            "stone counters?'\n"
            "5) ESTIMATE, estimate_project, present conversationally: 'you're "
            "looking at eighteen to twenty-six thousand, roughly two weeks.' Say ONCE "
            "per call that it's ballpark until a site visit.\n"
            "6) NEXT STEP, offer BEFORE they ask: save the quote in writing (name + "
            "contact into save_quote) and the free site visit.\n\n"
            "# Safety & Escalation\n"
            "- No structural or engineering judgments ('will this wall hold'), that's "
            "the site visit.\n"
            "- If the project is outside the five types, say so plainly and offer a "
            "callback from a project manager."
        ),
        "tools": [
            _fn("estimate_project",
                "Price range and duration for a renovation project.",
                {"project_type": {"type": "string", "enum": list(RATES)},
                 "area_sqft": {"type": "number"},
                 "finish_level": {"type": "string",
                                  "enum": ["budget", "standard", "premium"]}},
                ["project_type", "area_sqft"]),
            _fn("save_quote", "Save the quote and schedule a follow-up.",
                {"customer": {"type": "string"}, "contact": {"type": "string"},
                 "project": {"type": "string"}, "low_usd": {"type": "number"},
                 "high_usd": {"type": "number"}},
                ["customer", "contact", "project", "low_usd", "high_usd"]),
        ],
        "implementations": {"estimate_project": estimate_project,
                            "save_quote": save_quote},
    },
}

# Tools that key side effects to a conversation. The desk used this for its
# trade log; no hosted persona needs it yet, but run_tool keeps the hook so a
# future session-scoped tool drops in without touching the dispatch path.
SESSION_TOOLS: set[str] = set()


@traceable(name="voice_tool", run_type="tool")
def run_tool(persona: str, name: str, arguments: dict, session: str = "voice") -> str:
    """Dispatch a Realtime function call to its server-side implementation."""
    impl = PERSONAS.get(persona, {}).get("implementations", {}).get(name)
    if impl is None:
        return json.dumps({"error": f"unknown tool {name} for persona {persona}"})
    if name in SESSION_TOOLS:
        arguments = {**arguments, "session": session or "voice"}
    try:
        return impl(**arguments)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
