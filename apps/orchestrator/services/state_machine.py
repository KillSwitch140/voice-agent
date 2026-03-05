"""
Deterministic state machine for the HVAC call flow.

State transitions are driven by collected slots and LLM signals.
The LLM can influence but never override safety-critical paths.
"""
from __future__ import annotations

from packages.core.models import CallSession, CallState, Intent, LLMTurnResult, Priority
from packages.core.utils import is_gta_city
from apps.orchestrator.services.tools import _is_slot_surge


# Slots required before leaving each collection state.
# AFTER_HOURS_DISCLOSURE is handled directly in _next_state (Optional[bool] gate).
_REQUIRED_SLOTS: dict[CallState, list[str]] = {
    CallState.COLLECTING_CITY: ["city"],
    CallState.COLLECTING_BOOKING_REF: ["booking_id"],
    CallState.COLLECTING_CUSTOMER_INFO: ["customer_name"],
    CallState.COLLECTING_BOOKING_DETAILS: ["issue_description", "preferred_date", "preferred_time"],
    CallState.CONFIRMING_BOOKING: ["confirmed"],
}


def _required_for(session: CallSession) -> list[str]:
    """Required slot names for the session's current state.
    Reschedule only needs new date/time — not a fresh issue_description.
    """
    if session.state == CallState.COLLECTING_BOOKING_DETAILS and session.intent == Intent.RESCHEDULE:
        return ["preferred_date", "preferred_time"]
    return _REQUIRED_SLOTS.get(session.state, [])


def _slots_complete(session: CallSession) -> bool:
    slot_dict = session.slots.model_dump()
    return all(slot_dict.get(s) for s in _required_for(session))


def get_missing_slots(session: CallSession) -> list[str]:
    """Return slot names still needed for the current state (used as LLM context)."""
    slot_dict = session.slots.model_dump()
    return [s for s in _required_for(session) if not slot_dict.get(s)]


def apply_llm_result(session: CallSession, result: LLMTurnResult) -> CallSession:
    """
    Merge LLM output into the session:
      1. Emergency flag — irreversible, always wins.
      2. Escalate intent — overrides state from anywhere.
      3. Intent — set once, except PRICING_FOLLOWUP and WRAP_UP allow re-detection.
      4. Merge extracted slots.
      5. Advance state deterministically.
      6. Slot reset when WRAP_UP → INTENT_DETECTION (city+name kept).
    """
    # 1. Emergency — always wins, irreversible
    if result.is_emergency:
        session.is_emergency = True
        session.priority = Priority.EMERGENCY
        session.state = CallState.EMERGENCY_TRIAGE
        return session

    # 2. Escalate — caller requested a human; jump immediately from any state
    if result.intent in ("escalate", Intent.ESCALATE.value):
        session.intent = Intent.ESCALATE
        session.state = CallState.ESCALATING
        return session

    # 3. Intent — set once (first confident detection wins).
    #    Exception: INTENT_DETECTION, PRICING_FOLLOWUP, and WRAP_UP allow re-detection
    #    so a misclassified "unknown" doesn't permanently lock the caller out.
    if result.intent:
        try:
            new_intent = Intent(result.intent)
            if session.intent is None or session.state in (
                CallState.INTENT_DETECTION, CallState.PRICING_FOLLOWUP, CallState.WRAP_UP
            ):
                session.intent = new_intent
        except ValueError:
            pass

    # 4. Merge slots
    slots_data = session.slots.model_dump()
    for key, value in result.extracted_slots.items():
        if value is not None and key in slots_data:
            slots_data[key] = value
    session.slots = session.slots.model_copy(update=slots_data)

    # 5. Advance state
    old_state = session.state
    session.state = _next_state(session)

    # 6. Reset booking slots when looping back from WRAP_UP → INTENT_DETECTION
    if old_state == CallState.WRAP_UP and session.state == CallState.INTENT_DETECTION:
        session.intent = None
        session.slots = session.slots.model_copy(update={
            "booking_id": None,
            "issue_description": None,
            "preferred_date": None,
            "preferred_time": None,
            "confirmed": False,
            "after_hours_accepted": None,
            "more_help": None,
        })

    return session


def _next_state(session: CallSession) -> CallState:  # noqa: C901
    current = session.state
    intent = session.intent
    slots = session.slots

    if current == CallState.GREETING:
        return CallState.COLLECTING_CUSTOMER_INFO

    if current == CallState.COLLECTING_CUSTOMER_INFO:
        if slots.customer_name:
            return CallState.COLLECTING_CITY
        return CallState.COLLECTING_CUSTOMER_INFO

    if current == CallState.COLLECTING_CITY:
        if slots.city:
            if not is_gta_city(slots.city):
                return CallState.OUT_OF_AREA
            return CallState.INTENT_DETECTION
        return CallState.COLLECTING_CITY

    if current == CallState.OUT_OF_AREA:
        return CallState.ENDED  # voice.py handles SMS + hangup immediately

    if current == CallState.INTENT_DETECTION:
        if intent == Intent.PRICING:
            return CallState.PRICING
        if intent == Intent.EMERGENCY:
            return CallState.EMERGENCY_TRIAGE
        if intent == Intent.NEW_BOOKING:
            return CallState.COLLECTING_BOOKING_DETAILS
        if intent in (Intent.RESCHEDULE, Intent.CANCELLATION):
            return CallState.COLLECTING_BOOKING_REF
        return CallState.INTENT_DETECTION

    if current == CallState.PRICING:
        return CallState.PRICING_FOLLOWUP

    if current == CallState.PRICING_FOLLOWUP:
        if intent == Intent.NEW_BOOKING:
            return CallState.COLLECTING_BOOKING_DETAILS
        if intent in (Intent.RESCHEDULE, Intent.CANCELLATION):
            return CallState.COLLECTING_BOOKING_REF
        if intent in (Intent.GENERAL, Intent.UNKNOWN):
            return CallState.WRAP_UP
        return CallState.PRICING_FOLLOWUP

    if current == CallState.COLLECTING_BOOKING_REF:
        if slots.booking_id:
            if intent == Intent.CANCELLATION:
                return CallState.CONFIRMING_BOOKING
            return CallState.COLLECTING_BOOKING_DETAILS
        return CallState.COLLECTING_BOOKING_REF

    if current == CallState.COLLECTING_BOOKING_DETAILS:
        if _slots_complete(session):
            if _is_slot_surge(slots.preferred_date or "", slots.preferred_time or "") and slots.after_hours_accepted is None:
                return CallState.AFTER_HOURS_DISCLOSURE
            return CallState.CONFIRMING_BOOKING
        return CallState.COLLECTING_BOOKING_DETAILS

    if current == CallState.AFTER_HOURS_DISCLOSURE:
        if slots.after_hours_accepted is True:
            return CallState.CONFIRMING_BOOKING
        if slots.after_hours_accepted is False:
            return CallState.WRAP_UP
        return CallState.AFTER_HOURS_DISCLOSURE

    if current == CallState.CONFIRMING_BOOKING:
        if slots.confirmed:
            return CallState.CLOSING
        return CallState.CONFIRMING_BOOKING

    if current == CallState.EMERGENCY_TRIAGE:
        return CallState.ESCALATING

    if current == CallState.ESCALATING:
        return CallState.ENDED

    if current == CallState.CLOSING:
        return CallState.WRAP_UP

    if current == CallState.WRAP_UP:
        if slots.more_help is False:
            return CallState.ENDED
        if slots.more_help is True:
            return CallState.INTENT_DETECTION
        return CallState.WRAP_UP

    return current
