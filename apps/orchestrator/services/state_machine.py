"""
Deterministic state machine for the HVAC call flow.

State transitions are driven by collected slots and LLM signals.
The LLM can influence but never override safety-critical paths.
"""
from __future__ import annotations

from packages.core.models import CallSession, CallState, Intent, LLMTurnResult, Priority
from packages.core.utils import is_toronto_service_area


# Slots required before leaving each collection state.
# AFTER_HOURS_DISCLOSURE is handled directly in _next_state (Optional[bool] gate).
_REQUIRED_SLOTS: dict[CallState, list[str]] = {
    CallState.COLLECTING_POSTAL: ["postal_code"],
    CallState.COLLECTING_BOOKING_REF: ["booking_id"],
    CallState.COLLECTING_CUSTOMER_INFO: ["customer_name"],
    CallState.COLLECTING_BOOKING_DETAILS: ["issue_description", "preferred_date", "preferred_time"],
    CallState.CONFIRMING_BOOKING: ["confirmed"],
}


def _is_slot_after_hours(preferred_date: str | None, preferred_time: str | None) -> bool:
    """Return True if the booked slot falls outside standard business hours.

    Mirrors the tier logic in tools.py:
      Mon–Fri: slots starting 08:00–14:00 are standard (surge starts at 16:00)
      Sat: slots starting 09:00–11:00 are standard (surge starts at 13:00)
      Sun: always surge
    """
    if not preferred_date or not preferred_time:
        return False
    try:
        from datetime import datetime as _dt
        d = _dt.strptime(preferred_date, "%Y-%m-%d").date()
        hour = int(preferred_time.split(":")[0])
        wd = d.weekday()
        if wd < 5:   return not (8 <= hour < 16)   # Mon–Fri surge starts at 16:00
        if wd == 5:  return not (9 <= hour < 13)   # Sat surge starts at 13:00
        return True                                  # Sun always after-hours
    except Exception:
        return False


def _slots_complete(state: CallState, session: CallSession) -> bool:
    """Check if all required slots for the state are filled.

    Reschedule only needs new date/time — not a fresh issue_description.
    """
    if state == CallState.COLLECTING_BOOKING_DETAILS and session.intent == Intent.RESCHEDULE:
        required = ["preferred_date", "preferred_time"]
    else:
        required = _REQUIRED_SLOTS.get(state, [])
    slot_dict = session.slots.model_dump()
    return all(slot_dict.get(s) for s in required)


def apply_llm_result(session: CallSession, result: LLMTurnResult) -> CallSession:
    """
    Merge LLM output into the session:
      1. Emergency flag — irreversible, always wins.
      2. Escalate intent — overrides state from anywhere.
      3. Intent — set once, except PRICING_FOLLOWUP allows re-detection.
      4. Merge extracted slots.
      5. Advance state deterministically.
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
    #    Exception: PRICING_FOLLOWUP allows re-detection so caller can switch to booking.
    if result.intent:
        try:
            new_intent = Intent(result.intent)
            if session.intent is None or session.state == CallState.PRICING_FOLLOWUP:
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
    session.state = _next_state(session)
    return session


def _next_state(session: CallSession) -> CallState:  # noqa: C901
    current = session.state
    intent = session.intent
    slots = session.slots

    if current == CallState.GREETING:
        return CallState.INTENT_DETECTION

    if current == CallState.INTENT_DETECTION:
        if intent == Intent.PRICING:
            return CallState.PRICING
        if intent == Intent.EMERGENCY:
            return CallState.EMERGENCY_TRIAGE
        if intent == Intent.NEW_BOOKING:
            return CallState.COLLECTING_POSTAL
        if intent in (Intent.RESCHEDULE, Intent.CANCELLATION):
            return CallState.COLLECTING_BOOKING_REF   # ask for booking_id first
        return CallState.INTENT_DETECTION  # keep asking

    if current == CallState.PRICING:
        # Answer given; now offer to book — re-enter intent detection
        return CallState.PRICING_FOLLOWUP

    if current == CallState.PRICING_FOLLOWUP:
        # Caller either wants to book or is done — intent was re-detected here
        if intent == Intent.NEW_BOOKING:
            return CallState.COLLECTING_POSTAL
        if intent in (Intent.RESCHEDULE, Intent.CANCELLATION):
            return CallState.COLLECTING_BOOKING_REF
        if intent in (Intent.GENERAL, Intent.UNKNOWN):
            return CallState.ENDED      # politely declined booking offer
        return CallState.PRICING_FOLLOWUP   # still deciding; wait for explicit answer

    if current == CallState.COLLECTING_BOOKING_REF:
        if slots.booking_id:
            if intent == Intent.CANCELLATION:
                return CallState.CONFIRMING_BOOKING   # cancel only needs confirmation
            return CallState.COLLECTING_BOOKING_DETAILS  # reschedule needs new date/time
        return CallState.COLLECTING_BOOKING_REF

    if current == CallState.COLLECTING_POSTAL:
        if slots.postal_code:
            if not is_toronto_service_area(slots.postal_code):
                return CallState.OUT_OF_AREA
            return CallState.COLLECTING_CUSTOMER_INFO
        return CallState.COLLECTING_POSTAL

    if current == CallState.OUT_OF_AREA:
        return CallState.ENDED

    if current == CallState.COLLECTING_CUSTOMER_INFO:
        if slots.customer_name:
            return CallState.COLLECTING_BOOKING_DETAILS
        return CallState.COLLECTING_CUSTOMER_INFO

    if current == CallState.COLLECTING_BOOKING_DETAILS:
        if _slots_complete(CallState.COLLECTING_BOOKING_DETAILS, session):
            # Disclose surcharge if the chosen SLOT is after-hours (not call time)
            if _is_slot_after_hours(slots.preferred_date, slots.preferred_time) and slots.after_hours_accepted is None:
                return CallState.AFTER_HOURS_DISCLOSURE
            return CallState.CONFIRMING_BOOKING
        return CallState.COLLECTING_BOOKING_DETAILS

    if current == CallState.AFTER_HOURS_DISCLOSURE:
        if slots.after_hours_accepted is True:
            return CallState.CONFIRMING_BOOKING
        if slots.after_hours_accepted is False:
            return CallState.ENDED          # caller declined surcharge
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
        return CallState.ENDED

    return current


def get_missing_slots(session: CallSession) -> list[str]:
    """Return slot names still needed for the current state (used as LLM context)."""
    if session.state == CallState.COLLECTING_BOOKING_DETAILS and session.intent == Intent.RESCHEDULE:
        required = ["preferred_date", "preferred_time"]
    else:
        required = _REQUIRED_SLOTS.get(session.state, [])
    slot_dict = session.slots.model_dump()
    return [s for s in required if not slot_dict.get(s)]
