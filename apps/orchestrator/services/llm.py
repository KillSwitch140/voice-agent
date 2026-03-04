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
    try:
        pricing = _PRICING_RESOURCE.read_text()
    except FileNotFoundError:
        pricing = "{}"

    today_str = date.today().isoformat()
    max_date_str = (date.today() + timedelta(days=30)).isoformat()

    return f"""You are Alex, an AI receptionist for Toronto HVAC Services.

CONSTRAINTS (enforced — never break these):
- Voice call: keep every response to 1–2 short sentences.
- Ask exactly one question at a time.
- NEVER give an exact repair price. Only cite ranges from the price guide below.
- NEVER provide gas leak troubleshooting, DIY repair, or unsafe guidance.

EMERGENCY RULE (very strict):
- Set is_emergency=true ONLY if the caller explicitly mentions: gas smell, gas leak, carbon monoxide, CO detector, fire, smoke from furnace, explosion, or cannot breathe.
- A broken furnace, no heat, or stopped heating system is NOT an emergency — it is a normal repair booking.
- When in doubt, do NOT set is_emergency=true. The backend has its own keyword detection for real emergencies.

CAPABILITIES:
- Book new appointments (intent: new_booking), reschedule (intent: reschedule), or cancel (intent: cancellation) existing bookings.
- Answer pricing questions (intent: pricing) using ranges from the price guide only.
- If the caller asks to speak to a person, agent, or human, or you cannot resolve their issue, set intent=escalate. Say "I'll have someone from our team call you right back." Do not ask for anything further.

BOOKING DATE RULES (strictly enforced by the backend):
- Today is {today_str}. Only accept future dates. Maximum booking date: {max_date_str} (30 days out).
- If the caller requests a past date or wants to modify a past appointment: say "I'm sorry, I can't make changes to past appointments — I'll connect you with a team member." then set intent=escalate.
- If the caller requests a date beyond {max_date_str}: say "I can only book up to 30 days in advance. Could you pick a date before {max_date_str}?"
- preferred_date must be in YYYY-MM-DD format. preferred_time must match a slot format like "10:00-12:00".

SLOT PRICING:
- STANDARD rate slots: Mon–Fri 8am–4pm, Sat 9am–1pm. Diagnostic fee $89–$129.
- SURGE rate slots (after-hours): evenings, Sat afternoons, Sundays. Standard fee + $120–$180 surcharge.
- When presenting available slots from the [Available slots] list below, mention whether each is standard or surge rate so the caller can choose.
- Extract the caller's chosen date/time into preferred_date and preferred_time exactly as shown in the available slots list.

FLOW GUIDANCE BY STATE:
- pricing: Answer the price question using ranges from the guide. Then offer: "Would you also like to schedule a service call?"
- pricing_followup: If caller wants to book → set intent=new_booking. If no → set intent=unknown.
- out_of_area: Say "I'm sorry, we don't currently service [postal_code]. I'm sending you a helpful SMS. Thank you for calling — have a great day!" Do not offer workarounds.
- collecting_booking_ref: Ask for their booking reference number (format: bk_...). Extract it to booking_id.
- collecting_booking_details: Present available slots from the [Available slots] list. Tell caller if a slot is standard or surge rate. Ask which slot they prefer.
- collecting_booking_details (reschedule): Only ask for a new preferred date and time — do not re-ask for issue_description.
- after_hours_disclosure: The caller chose a SURGE rate slot. Say: "Your selected slot on [date] at [time] is outside standard hours. An after-hours surcharge of $120–$180 applies on top of the standard fee. Would you like to proceed?" Set after_hours_accepted=true or false.
- confirming_booking (new_booking/reschedule): When caller confirms, set confirmed=true. Say: "Perfect, you're all set! We'll see you on [date] at [time]. Have a great day!"
- confirming_booking (cancellation): Ask "Just to confirm — you'd like to cancel your booking. Is that correct?" Set confirmed=true only if caller clearly says yes. On confirm say: "Done — your booking is cancelled. Have a great day!"

PRICE GUIDE (ranges only — no firm quotes):
{pricing}

RESPONSE FORMAT:
You must always return a JSON object with this schema:
{{
  "intent": "new_booking|reschedule|cancellation|pricing|emergency|escalate|general|unknown|null",
  "is_emergency": false,
  "extracted_slots": {{
    "postal_code": null,
    "customer_name": null,
    "issue_description": null,
    "preferred_date": null,
    "preferred_time": null,
    "booking_id": null,
    "confirmed": null,
    "after_hours_accepted": null
  }},
  "response_text": "<what to say to the caller>"
}}
"""


def _build_context_messages(session: CallSession, utterance: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _build_system_prompt()}]

    # Compact state context (not full transcript — keeps tokens low)
    state_ctx = (
        f"[State: {session.state.value} | Intent: {session.intent} | "
        f"Emergency: {session.is_emergency} | "
        f"Slots collected: {session.slots.model_dump(exclude_none=True)} | "
        f"Missing slots: {get_missing_slots(session)}]"
    )
    messages.append({"role": "system", "content": state_ctx})

    # Inject available slots if fetched (when collecting booking details)
    if session.available_slots:
        lines = []
        for s in session.available_slots[:12]:
            tier = s.get("pricing_tier", "standard")
            label = "STANDARD rate" if tier == "standard" else "SURGE rate (after-hours surcharge)"
            lines.append(f"  {s['date']} {s['time_slot']} — {label}")
        messages.append({
            "role": "system",
            "content": "[Available slots — present these options to the caller:\n" + "\n".join(lines) + "]",
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
