"""
LLM service — GPT-4o-mini for intent classification and slot extraction.

The LLM handles:
  - Intent classification
  - Entity / slot extraction
  - Generating the voice response text (1–2 sentences max)

Booking tool calls (create/reschedule/cancel) are made deterministically by
the orchestrator based on state machine state — not by LLM discretion.
The orchestrator controls all state transitions; the LLM cannot override them.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from openai import AsyncOpenAI

from packages.core.models import CallSession, LLMTurnResult
from apps.orchestrator.services.state_machine import get_missing_slots

logger = logging.getLogger(__name__)

_PRICING_RESOURCE = Path(__file__).parents[3] / "apps" / "resources" / "pricing" / "after_hours_fees.json"

# ─── System prompt ────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    today_str = date.today().isoformat()
    max_date_str = (date.today() + timedelta(days=30)).isoformat()

    return f"""You are a data extraction engine for Toronto HVAC Services.
Your ONLY job is to extract structured data from caller utterances. Python code handles all spoken responses — do NOT generate conversational text.

EMERGENCY DETECTION (critical):
- Set is_emergency=true ONLY for explicit mentions of: gas smell, gas leak, carbon monoxide, CO detector, fire, smoke from furnace, explosion, cannot breathe.
- A broken furnace, no heat, or stopped heating is NOT an emergency.

ESCALATION DETECTION:
- Set intent=escalate ONLY when the caller explicitly asks to speak to a person, agent, or human.
- Words like "wrong", "no", "cancel", "that's not right" are corrections — NOT escalation.

DATE EXTRACTION:
- Today is {today_str}. Maximum booking date: {max_date_str}.
- When a caller gives a month+day without a year, always resolve to the nearest upcoming date (e.g. "sixth March" when today is {today_str} → {today_str[:4]}-03-06).
- Do NOT reject or validate dates — just extract what the caller said in YYYY-MM-DD format.
- preferred_time must exactly match the slot format shown in [Available slots] (e.g. "10:00-12:00").

SLOT ACCEPTANCE (collecting_booking_details only):
- Extract preferred_date and preferred_time ONLY when the caller explicitly accepts or names a specific slot.
- If the caller says "no", "doesn't work", "different day", or similar — do NOT extract preferred_date or preferred_time.

OUTPUT FORMAT (JSON only, no prose):
{{
  "intent": "new_booking|reschedule|cancellation|pricing|emergency|escalate|general|unknown|null",
  "is_emergency": false,
  "extracted_slots": {{
    "city": null,
    "customer_name": null,
    "issue_description": null,
    "preferred_date": null,
    "preferred_time": null,
    "booking_id": null,
    "confirmed": null,
    "after_hours_accepted": null,
    "more_help": null
  }},
  "response_text": ""
}}

Only include fields you are confident about. Set response_text to empty string."""


# Extraction-only hints injected per state.
# These guide WHAT TO EXTRACT — not what to say. Python generates all spoken responses.
_STATE_GUIDANCE: dict[str, str] = {
    "collecting_customer_info": "Extract the caller's name into customer_name.",
    "collecting_city": "Extract the caller's city into city.",
    "intent_detection": (
        "Classify the caller's intent from their utterance.\n"
        "- new_booking: any mention of booking, appointment, service call, schedule, new service, fix, repair, maintenance.\n"
        "- reschedule: change, move, reschedule an existing appointment.\n"
        "- cancellation: cancel an existing appointment.\n"
        "- pricing: asking about cost, price, rates, fees, how much.\n"
        "- unknown: ONLY if the utterance is truly ambiguous and matches none of the above.\n"
        "Do not extract any other slots in this state."
    ),
    "pricing": "Set intent=pricing. No slots to extract.",
    "pricing_followup": (
        "If caller says yes/sure/want to book: set intent=new_booking. "
        "If caller says no/that's all/not right now: set intent=unknown."
    ),
    "collecting_booking_ref": (
        "Extract the booking reference number into booking_id. "
        "Format: bk_ followed by alphanumeric characters."
    ),
    "collecting_booking_details": (
        "Extract issue_description from any problem description.\n"
        "Extract preferred_date (YYYY-MM-DD) and preferred_time (e.g. '10:00-12:00') "
        "ONLY when the caller explicitly accepts or names a specific slot from [Available slots]. "
        "Do NOT extract date/time if caller says no, doesn't work, different day, or similar."
    ),
    "collecting_booking_details_reschedule": (
        "Extract preferred_date and preferred_time ONLY when caller accepts a slot. "
        "Do not re-extract issue_description."
    ),
    "after_hours_disclosure": (
        "Set after_hours_accepted=true if caller agrees to proceed. "
        "Set after_hours_accepted=false if caller declines."
    ),
    "confirming_booking": (
        "Set confirmed=true ONLY for explicit yes/correct/go ahead/sure. "
        "Do NOT set confirmed=true for 'thanks', 'okay', 'cool', or filler words. "
        "Set confirmed=false if caller says no/cancel/different."
    ),
    "wrap_up": (
        "Set more_help=true if caller wants more help. "
        "Set more_help=false if caller is done (no/that's all/goodbye). "
        "You MUST set one or the other — never leave more_help null when the caller has responded."
    ),
}


def _fmt_date(date_str: str) -> str:
    """'2026-03-05' → 'March 5th'"""
    from datetime import datetime as _dt
    d = _dt.strptime(date_str, "%Y-%m-%d")
    day = d.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{d.strftime('%B')} {day}{suffix}"


def _fmt_time(time_str: str) -> str:
    """'10:00-12:00' → '10:00 AM to 12:00 PM'"""
    try:
        start, end = time_str.split("-")
        def _t(s: str) -> str:
            h, m = map(int, s.split(":"))
            ampm = "AM" if h < 12 else "PM"
            return f"{h % 12 or 12}:{m:02d} {ampm}"
        return f"{_t(start)} to {_t(end)}"
    except Exception:
        return time_str


def _build_context_messages(session: CallSession, utterance: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _build_system_prompt()}]

    # Inject only the guidance for the current state so the LLM cannot apply logic
    # from other states (e.g. generating an out_of_area message while in collecting_city).
    state_key = session.state.value
    if (session.state.value == "collecting_booking_details"
            and session.intent and session.intent.value == "reschedule"):
        state_key = "collecting_booking_details_reschedule"
    guidance = _STATE_GUIDANCE.get(state_key, "")

    state_ctx = (
        f"[State: {session.state.value} | Intent: {session.intent} | "
        f"Emergency: {session.is_emergency} | "
        f"Slots collected: {session.slots.model_dump(exclude_none=True)} | "
        f"Missing slots: {get_missing_slots(session)}]\n"
        + (f"[Your instructions for this turn: {guidance}]" if guidance else "")
    )
    messages.append({"role": "system", "content": state_ctx})

    # Inject available slots if fetched (when collecting booking details).
    # Rate labels are intentionally omitted — the backend discloses pricing at confirmation.
    if session.available_slots:
        lines = [
            f"  {_fmt_date(s['date'])}, {_fmt_time(s['time_slot'])}"
            for s in session.available_slots[:12]
        ]
        messages.append({
            "role": "system",
            "content": "[Available slots — offer ONE at a time, starting with the first:\n" + "\n".join(lines) + "]",
        })

    # Rolling last-5 turns
    for turn in session.turns[-5:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": utterance})
    return messages


# ─── Main call ────────────────────────────────────────────────────────────────

async def run_turn(
    session: CallSession,
    utterance: str,
    openai_api_key: str,
) -> LLMTurnResult:
    """
    Run one conversational turn through GPT-4o-mini.

    The LLM classifies intent, extracts slots, and generates a response.
    Booking tool calls (create/reschedule/cancel) are handled deterministically
    by the orchestrator based on state machine state — not by the LLM.
    """
    client = AsyncOpenAI(api_key=openai_api_key)
    messages = _build_context_messages(session, utterance)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = response.choices[0].message.content or "{}"
    logger.info("LLM raw [%s]: %s", session.state.value, raw)
    try:
        data: dict = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON: %s", raw)
        data = {"response_text": raw}

    return LLMTurnResult(
        intent=data.get("intent"),
        is_emergency=bool(data.get("is_emergency", False)),
        extracted_slots={
            k: v for k, v in (data.get("extracted_slots") or {}).items() if v is not None
        },
        response_text=data.get("response_text", "I'm sorry, can you repeat that?"),
    )
