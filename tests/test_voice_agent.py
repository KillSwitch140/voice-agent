"""
Comprehensive test suite for the HVAC Voice Agent.

Tests the state machine, emergency detection, service area validation,
slot pricing, and end-to-end call flows — all independent of any server.
Simulates LLM outputs deterministically to test every flow path.
"""
from __future__ import annotations

import pytest
from packages.core.models import (
    CallSession, CallSlots, CallState, Intent, LLMTurnResult, Priority,
)
from packages.core.utils import detect_emergency, is_gta_city
from apps.orchestrator.services.state_machine import (
    apply_llm_result, get_missing_slots, _next_state,
)
from apps.orchestrator.services.tools import _is_slot_surge


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _session(**overrides) -> CallSession:
    """Create a CallSession with sensible defaults, easy to override."""
    defaults = {"call_sid": "TEST", "state": CallState.GREETING}
    defaults.update(overrides)
    return CallSession(**defaults)


def _result(**overrides) -> LLMTurnResult:
    """Create an LLMTurnResult with defaults, easy to override."""
    defaults = {
        "intent": None,
        "is_emergency": False,
        "extracted_slots": {},
        "response_text": "",
    }
    defaults.update(overrides)
    return LLMTurnResult(**defaults)


def _advance(session: CallSession, result: LLMTurnResult) -> CallSession:
    """Apply an LLM result and return the updated session."""
    return apply_llm_result(session, result)


# ═════════════════════════════════════════════════════════════════════════════
# 1. EMERGENCY DETECTION  (packages/core/utils.py)
# ═════════════════════════════════════════════════════════════════════════════

class TestEmergencyDetection:
    """Hard-coded keyword detection — must bypass LLM entirely."""

    @pytest.mark.parametrize("phrase", [
        "I smell gas in my basement",
        "There's a gas leak",
        "My carbon monoxide detector is going off",
        "There's smoke from furnace",
        "The furnace is on fire",
        "I heard an explosion from my furnace",
        "My pipes are frozen and bursting",
        "I can't breathe, something is wrong",
    ])
    def test_real_emergencies_detected(self, phrase):
        assert detect_emergency(phrase) is True

    @pytest.mark.parametrize("phrase", [
        "My furnace stopped working",
        "There's no heat in my house",
        "The heater is broken",
        "I need an emergency repair",      # "emergency" alone is NOT a trigger
        "My AC unit is making a loud noise",
        "I want to book an appointment",
        "Can you fix my thermostat?",
    ])
    def test_non_emergencies_not_triggered(self, phrase):
        assert detect_emergency(phrase) is False

    def test_case_insensitive(self):
        assert detect_emergency("I SMELL GAS") is True
        assert detect_emergency("Carbon Monoxide alarm") is True

    def test_emergency_overrides_state_machine(self):
        """Emergency flag in LLM result → EMERGENCY_TRIAGE regardless of current state."""
        for state in [
            CallState.COLLECTING_CUSTOMER_INFO,
            CallState.INTENT_DETECTION,
            CallState.COLLECTING_BOOKING_DETAILS,
            CallState.CONFIRMING_BOOKING,
        ]:
            s = _session(state=state)
            s = _advance(s, _result(is_emergency=True))
            assert s.state == CallState.EMERGENCY_TRIAGE
            assert s.is_emergency is True
            assert s.priority == Priority.EMERGENCY

    def test_emergency_is_irreversible(self):
        """Once is_emergency is set, it can never be unset."""
        s = _session(state=CallState.INTENT_DETECTION)
        s = _advance(s, _result(is_emergency=True))
        assert s.is_emergency is True
        # Even a non-emergency result cannot clear the flag
        s.state = CallState.INTENT_DETECTION  # force back for testing
        s = _advance(s, _result(is_emergency=False, intent="new_booking"))
        assert s.is_emergency is True  # still True


# ═════════════════════════════════════════════════════════════════════════════
# 2. SERVICE AREA  (packages/core/utils.py)
# ═════════════════════════════════════════════════════════════════════════════

class TestServiceArea:
    """City-based GTA service area validation."""

    @pytest.mark.parametrize("city", [
        "Toronto", "toronto", "TORONTO",
        "Scarborough", "North York", "Etobicoke",
        "Mississauga", "Brampton", "Vaughan", "Markham",
        "Ajax", "Whitby", "Oshawa", "Oakville", "Burlington",
    ])
    def test_gta_cities_accepted(self, city):
        assert is_gta_city(city) is True

    @pytest.mark.parametrize("city", [
        "Ottawa", "Montreal", "Vancouver", "London", "Kingston",
        "Barrie", "Hamilton", "Guelph", "Kitchener", "Windsor",
    ])
    def test_non_gta_cities_rejected(self, city):
        assert is_gta_city(city) is False

    def test_whitespace_handling(self):
        assert is_gta_city("  Toronto  ") is True
        assert is_gta_city(" North York ") is True

    def test_empty_city_rejected(self):
        assert is_gta_city("") is False

    def test_out_of_area_state_transition(self):
        """Non-GTA city → COLLECTING_CITY → OUT_OF_AREA → ENDED."""
        s = _session(state=CallState.COLLECTING_CITY)
        s = _advance(s, _result(extracted_slots={"city": "Ottawa"}))
        assert s.state == CallState.OUT_OF_AREA

        # OUT_OF_AREA always transitions to ENDED
        s = _advance(s, _result())
        assert s.state == CallState.ENDED


# ═════════════════════════════════════════════════════════════════════════════
# 3. SLOT PRICING  (apps/orchestrator/services/tools.py)
# ═════════════════════════════════════════════════════════════════════════════

class TestSlotPricing:
    """Pricing tier determination based on booked slot time."""

    # 2026-03-09 is a Monday, 2026-03-14 is a Saturday, 2026-03-15 is a Sunday
    @pytest.mark.parametrize("date_str,time_slot,expected_surge", [
        # Weekday standard hours
        ("2026-03-09", "08:00-10:00", False),
        ("2026-03-09", "10:00-12:00", False),
        ("2026-03-09", "14:00-16:00", False),
        # Weekday surge hours
        ("2026-03-09", "16:00-18:00", True),
        ("2026-03-09", "18:00-20:00", True),
        # Saturday standard
        ("2026-03-14", "09:00-11:00", False),
        ("2026-03-14", "11:00-13:00", False),
        # Saturday surge
        ("2026-03-14", "13:00-15:00", True),
        ("2026-03-14", "15:00-17:00", True),
        # Sunday — always surge
        ("2026-03-15", "10:00-12:00", True),
        ("2026-03-15", "14:00-16:00", True),
    ], ids=[
        "Mon-8am-std", "Mon-10am-std", "Mon-2pm-std",
        "Mon-4pm-surge", "Mon-6pm-surge",
        "Sat-9am-std", "Sat-11am-std",
        "Sat-1pm-surge", "Sat-3pm-surge",
        "Sun-10am-surge", "Sun-2pm-surge",
    ])
    def test_pricing_tiers(self, date_str, time_slot, expected_surge):
        assert _is_slot_surge(date_str, time_slot) is expected_surge

    def test_unknown_slot_defaults_to_surge(self):
        """Safety: unrecognized time slots default to surge pricing."""
        assert _is_slot_surge("2026-03-09", "07:00-09:00") is True
        assert _is_slot_surge("2026-03-09", "22:00-24:00") is True

    def test_invalid_date_defaults_to_surge(self):
        assert _is_slot_surge("not-a-date", "10:00-12:00") is True
        assert _is_slot_surge("", "") is True


# ═════════════════════════════════════════════════════════════════════════════
# 4. STATE MACHINE — CORE TRANSITIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestStateMachineCore:
    """Fundamental state machine transitions."""

    def test_greeting_to_collecting_customer_info(self):
        s = _session(state=CallState.GREETING)
        s = _advance(s, _result())
        assert s.state == CallState.COLLECTING_CUSTOMER_INFO

    def test_collecting_name(self):
        s = _session(state=CallState.COLLECTING_CUSTOMER_INFO)
        # No name → stays
        s = _advance(s, _result())
        assert s.state == CallState.COLLECTING_CUSTOMER_INFO
        # Name given → moves to city
        s = _advance(s, _result(extracted_slots={"customer_name": "Alice"}))
        assert s.state == CallState.COLLECTING_CITY
        assert s.slots.customer_name == "Alice"

    def test_collecting_city_gta(self):
        s = _session(state=CallState.COLLECTING_CITY)
        s.slots = CallSlots(customer_name="Alice")
        s = _advance(s, _result(extracted_slots={"city": "Toronto"}))
        assert s.state == CallState.INTENT_DETECTION

    def test_collecting_city_non_gta(self):
        s = _session(state=CallState.COLLECTING_CITY)
        s.slots = CallSlots(customer_name="Alice")
        s = _advance(s, _result(extracted_slots={"city": "Ottawa"}))
        assert s.state == CallState.OUT_OF_AREA

    def test_intent_detection_all_intents(self):
        """Each valid intent routes to the correct next state."""
        cases = [
            (Intent.NEW_BOOKING, CallState.COLLECTING_BOOKING_DETAILS),
            (Intent.RESCHEDULE, CallState.COLLECTING_BOOKING_REF),
            (Intent.CANCELLATION, CallState.COLLECTING_BOOKING_REF),
            (Intent.PRICING, CallState.PRICING),
            (Intent.EMERGENCY, CallState.EMERGENCY_TRIAGE),
        ]
        for intent, expected_state in cases:
            s = _session(state=CallState.INTENT_DETECTION)
            s = _advance(s, _result(intent=intent.value))
            assert s.state == expected_state, (
                f"Intent {intent.value} → expected {expected_state}, got {s.state}"
            )

    def test_intent_detection_unknown_stays(self):
        s = _session(state=CallState.INTENT_DETECTION)
        s = _advance(s, _result(intent="unknown"))
        assert s.state == CallState.INTENT_DETECTION

    def test_escalate_from_any_state(self):
        """Escalation must work from any state."""
        for state in [
            CallState.COLLECTING_CUSTOMER_INFO,
            CallState.INTENT_DETECTION,
            CallState.COLLECTING_BOOKING_DETAILS,
            CallState.CONFIRMING_BOOKING,
            CallState.PRICING_FOLLOWUP,
        ]:
            s = _session(state=state)
            s = _advance(s, _result(intent="escalate"))
            assert s.state == CallState.ESCALATING, (
                f"Escalate from {state} should go to ESCALATING, got {s.state}"
            )
            assert s.intent == Intent.ESCALATE


# ═════════════════════════════════════════════════════════════════════════════
# 5. INTENT RE-DETECTION  (the bug we fixed)
# ═════════════════════════════════════════════════════════════════════════════

class TestIntentReDetection:
    """Verify that misclassified intents don't permanently lock the caller out."""

    def test_unknown_to_new_booking(self):
        """Key bug fix: unknown → new_booking must work in INTENT_DETECTION."""
        s = _session(state=CallState.INTENT_DETECTION, intent=Intent.UNKNOWN)
        s = _advance(s, _result(intent="new_booking"))
        assert s.intent == Intent.NEW_BOOKING
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS

    def test_unknown_to_reschedule(self):
        s = _session(state=CallState.INTENT_DETECTION, intent=Intent.UNKNOWN)
        s = _advance(s, _result(intent="reschedule"))
        assert s.intent == Intent.RESCHEDULE
        assert s.state == CallState.COLLECTING_BOOKING_REF

    def test_unknown_to_pricing(self):
        s = _session(state=CallState.INTENT_DETECTION, intent=Intent.UNKNOWN)
        s = _advance(s, _result(intent="pricing"))
        assert s.intent == Intent.PRICING
        assert s.state == CallState.PRICING

    def test_intent_locked_in_collecting_details(self):
        """Outside re-detection states, intent must not change."""
        s = _session(
            state=CallState.COLLECTING_BOOKING_DETAILS,
            intent=Intent.NEW_BOOKING,
        )
        s = _advance(s, _result(
            intent="cancellation",
            extracted_slots={"issue_description": "heater broken"},
        ))
        assert s.intent == Intent.NEW_BOOKING  # locked

    def test_pricing_followup_allows_redetection(self):
        s = _session(state=CallState.PRICING_FOLLOWUP, intent=Intent.PRICING)
        s = _advance(s, _result(intent="new_booking"))
        assert s.intent == Intent.NEW_BOOKING
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS

    def test_wrap_up_allows_redetection(self):
        s = _session(state=CallState.WRAP_UP, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(customer_name="Bob", city="Toronto", more_help=True)
        s = _advance(s, _result(
            intent="reschedule",
            extracted_slots={"more_help": True},
        ))
        # WRAP_UP + more_help=True → INTENT_DETECTION (with slot reset)
        assert s.state == CallState.INTENT_DETECTION
        assert s.intent is None  # reset on WRAP_UP → INTENT_DETECTION


# ═════════════════════════════════════════════════════════════════════════════
# 6. NEW BOOKING FLOW
# ═════════════════════════════════════════════════════════════════════════════

class TestNewBookingFlow:
    """Full new booking flow: intent → details → confirm → close → wrap_up."""

    def test_happy_path_standard_slot(self):
        """Complete new booking with a standard-rate weekday slot."""
        s = _session(state=CallState.INTENT_DETECTION)
        s.slots = CallSlots(customer_name="Bob", city="Toronto")

        # Intent detected
        s = _advance(s, _result(intent="new_booking"))
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS

        # Issue described
        s = _advance(s, _result(extracted_slots={"issue_description": "furnace not heating"}))
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS  # still needs date/time

        # Date and time selected (standard slot: Mon 10am)
        s = _advance(s, _result(extracted_slots={
            "preferred_date": "2026-03-09",
            "preferred_time": "10:00-12:00",
        }))
        assert s.state == CallState.CONFIRMING_BOOKING  # all slots filled, standard → no disclosure

        # Confirmed
        s = _advance(s, _result(extracted_slots={"confirmed": True}))
        assert s.state == CallState.CLOSING

        # CLOSING → WRAP_UP
        s = _advance(s, _result())
        assert s.state == CallState.WRAP_UP

    def test_surge_slot_triggers_disclosure(self):
        """After-hours slot triggers AFTER_HOURS_DISCLOSURE before confirmation."""
        s = _session(state=CallState.COLLECTING_BOOKING_DETAILS, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(
            customer_name="Alice", city="Toronto",
            issue_description="AC broken",
        )

        # Sunday slot = always surge
        s = _advance(s, _result(extracted_slots={
            "preferred_date": "2026-03-15",
            "preferred_time": "10:00-12:00",
        }))
        assert s.state == CallState.AFTER_HOURS_DISCLOSURE

    def test_surge_accepted(self):
        """Caller accepts after-hours surcharge → proceeds to confirmation."""
        s = _session(state=CallState.AFTER_HOURS_DISCLOSURE, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(
            customer_name="Alice", city="Toronto",
            issue_description="AC broken",
            preferred_date="2026-03-15", preferred_time="10:00-12:00",
        )
        s = _advance(s, _result(extracted_slots={"after_hours_accepted": True}))
        assert s.state == CallState.CONFIRMING_BOOKING

    def test_surge_declined(self):
        """Caller declines after-hours surcharge → wrap up."""
        s = _session(state=CallState.AFTER_HOURS_DISCLOSURE, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(
            customer_name="Alice", city="Toronto",
            issue_description="AC broken",
            preferred_date="2026-03-15", preferred_time="10:00-12:00",
        )
        s = _advance(s, _result(extracted_slots={"after_hours_accepted": False}))
        assert s.state == CallState.WRAP_UP

    def test_missing_slots_stays(self):
        """Stays in COLLECTING_BOOKING_DETAILS until all slots filled."""
        s = _session(state=CallState.COLLECTING_BOOKING_DETAILS, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(customer_name="Bob", city="Toronto")

        missing = get_missing_slots(s)
        assert "issue_description" in missing
        assert "preferred_date" in missing
        assert "preferred_time" in missing

        # Only issue given
        s = _advance(s, _result(extracted_slots={"issue_description": "no heat"}))
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS

        # Only date given
        s = _advance(s, _result(extracted_slots={"preferred_date": "2026-03-09"}))
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS

        # Time given → all complete → confirming
        s = _advance(s, _result(extracted_slots={"preferred_time": "10:00-12:00"}))
        assert s.state == CallState.CONFIRMING_BOOKING


# ═════════════════════════════════════════════════════════════════════════════
# 7. RESCHEDULE FLOW
# ═════════════════════════════════════════════════════════════════════════════

class TestRescheduleFlow:
    """Reschedule: intent → booking ref → new date/time → confirm → close."""

    def test_reschedule_happy_path(self):
        s = _session(state=CallState.INTENT_DETECTION)
        s.slots = CallSlots(customer_name="Bob", city="Toronto")

        s = _advance(s, _result(intent="reschedule"))
        assert s.state == CallState.COLLECTING_BOOKING_REF

        s = _advance(s, _result(extracted_slots={"booking_id": "bk_test001"}))
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS

        # Reschedule only needs date + time (not issue_description)
        missing = get_missing_slots(s)
        assert "issue_description" not in missing
        assert "preferred_date" in missing
        assert "preferred_time" in missing

        s = _advance(s, _result(extracted_slots={
            "preferred_date": "2026-03-10",
            "preferred_time": "10:00-12:00",
        }))
        assert s.state == CallState.CONFIRMING_BOOKING

        s = _advance(s, _result(extracted_slots={"confirmed": True}))
        assert s.state == CallState.CLOSING

    def test_reschedule_no_booking_ref_stays(self):
        s = _session(state=CallState.COLLECTING_BOOKING_REF, intent=Intent.RESCHEDULE)
        s = _advance(s, _result())  # no booking_id extracted
        assert s.state == CallState.COLLECTING_BOOKING_REF


# ═════════════════════════════════════════════════════════════════════════════
# 8. CANCELLATION FLOW
# ═════════════════════════════════════════════════════════════════════════════

class TestCancellationFlow:
    """Cancel: intent → booking ref → confirm → close (no date/time needed)."""

    def test_cancellation_happy_path(self):
        s = _session(state=CallState.INTENT_DETECTION)
        s.slots = CallSlots(customer_name="Bob", city="Toronto")

        s = _advance(s, _result(intent="cancellation"))
        assert s.state == CallState.COLLECTING_BOOKING_REF

        # Cancellation skips COLLECTING_BOOKING_DETAILS — goes straight to confirm
        s = _advance(s, _result(extracted_slots={"booking_id": "bk_test002"}))
        assert s.state == CallState.CONFIRMING_BOOKING

        s = _advance(s, _result(extracted_slots={"confirmed": True}))
        assert s.state == CallState.CLOSING

    def test_cancellation_denied(self):
        """Caller says no at confirmation → stays in CONFIRMING_BOOKING."""
        s = _session(state=CallState.CONFIRMING_BOOKING, intent=Intent.CANCELLATION)
        s.slots = CallSlots(
            customer_name="Bob", city="Toronto",
            booking_id="bk_test002", confirmed=False,
        )
        s = _advance(s, _result(extracted_slots={"confirmed": False}))
        assert s.state == CallState.CONFIRMING_BOOKING  # stays — confirmed is False


# ═════════════════════════════════════════════════════════════════════════════
# 9. PRICING FLOW
# ═════════════════════════════════════════════════════════════════════════════

class TestPricingFlow:
    """Pricing inquiry → follow-up → optional booking or wrap up."""

    def test_pricing_then_book(self):
        s = _session(state=CallState.INTENT_DETECTION)
        s.slots = CallSlots(customer_name="Bob", city="Toronto")

        s = _advance(s, _result(intent="pricing"))
        assert s.state == CallState.PRICING

        # PRICING always advances to PRICING_FOLLOWUP
        s = _advance(s, _result())
        assert s.state == CallState.PRICING_FOLLOWUP

        # Caller wants to book after hearing prices
        s = _advance(s, _result(intent="new_booking"))
        assert s.intent == Intent.NEW_BOOKING
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS

    def test_pricing_then_decline(self):
        s = _session(state=CallState.PRICING_FOLLOWUP, intent=Intent.PRICING)
        s.slots = CallSlots(customer_name="Bob", city="Toronto")

        s = _advance(s, _result(intent="unknown"))
        assert s.state == CallState.WRAP_UP

    def test_pricing_followup_to_reschedule(self):
        s = _session(state=CallState.PRICING_FOLLOWUP, intent=Intent.PRICING)
        s.slots = CallSlots(customer_name="Bob", city="Toronto")

        s = _advance(s, _result(intent="reschedule"))
        assert s.intent == Intent.RESCHEDULE
        assert s.state == CallState.COLLECTING_BOOKING_REF


# ═════════════════════════════════════════════════════════════════════════════
# 10. WRAP-UP / MULTI-TASK FLOW
# ═════════════════════════════════════════════════════════════════════════════

class TestWrapUpFlow:
    """WRAP_UP gate: caller can do another task or end the call."""

    def test_wrap_up_more_help(self):
        """Caller wants more help → back to INTENT_DETECTION with reset slots."""
        s = _session(state=CallState.WRAP_UP, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(
            customer_name="Bob", city="Toronto",
            issue_description="furnace broken",
            preferred_date="2026-03-09", preferred_time="10:00-12:00",
            confirmed=True, booking_id="bk_abc123",
        )
        s = _advance(s, _result(extracted_slots={"more_help": True}))

        assert s.state == CallState.INTENT_DETECTION
        # Intent and booking-specific slots are reset
        assert s.intent is None
        assert s.slots.booking_id is None
        assert s.slots.issue_description is None
        assert s.slots.preferred_date is None
        assert s.slots.preferred_time is None
        assert s.slots.confirmed is False
        # Name and city are preserved
        assert s.slots.customer_name == "Bob"
        assert s.slots.city == "Toronto"

    def test_wrap_up_done(self):
        """Caller is done → ENDED."""
        s = _session(state=CallState.WRAP_UP, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(customer_name="Bob", city="Toronto")
        s = _advance(s, _result(extracted_slots={"more_help": False}))
        assert s.state == CallState.ENDED

    def test_wrap_up_no_answer_stays(self):
        """No more_help extracted → stays in WRAP_UP."""
        s = _session(state=CallState.WRAP_UP, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(customer_name="Bob", city="Toronto")
        s = _advance(s, _result())
        assert s.state == CallState.WRAP_UP


# ═════════════════════════════════════════════════════════════════════════════
# 11. SLOT MERGE / EXTRACTION RULES
# ═════════════════════════════════════════════════════════════════════════════

class TestSlotMerge:
    """Verify how extracted slots are merged into the session."""

    def test_none_values_not_merged(self):
        """Slots with None value should not overwrite existing data."""
        s = _session(state=CallState.COLLECTING_BOOKING_DETAILS, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(
            customer_name="Bob", city="Toronto",
            issue_description="heater broken",
        )
        s = _advance(s, _result(extracted_slots={
            "issue_description": None,  # should NOT clear existing
            "preferred_date": "2026-03-09",
        }))
        assert s.slots.issue_description == "heater broken"
        assert s.slots.preferred_date == "2026-03-09"

    def test_unknown_slot_keys_ignored(self):
        """Slots not in CallSlots schema are silently ignored."""
        s = _session(state=CallState.COLLECTING_CUSTOMER_INFO)
        s = _advance(s, _result(extracted_slots={
            "customer_name": "Alice",
            "favorite_color": "blue",  # not a real slot
        }))
        assert s.slots.customer_name == "Alice"
        assert not hasattr(s.slots, "favorite_color")

    def test_invalid_intent_string_ignored(self):
        """An invalid intent string should be silently ignored, not crash."""
        s = _session(state=CallState.INTENT_DETECTION)
        s = _advance(s, _result(intent="buy_pizza"))  # not a valid Intent enum
        assert s.intent is None
        assert s.state == CallState.INTENT_DETECTION


# ═════════════════════════════════════════════════════════════════════════════
# 12. MISSING SLOTS TRACKING
# ═════════════════════════════════════════════════════════════════════════════

class TestMissingSlots:
    """Verify get_missing_slots returns the right set for each state."""

    def test_collecting_city_missing(self):
        s = _session(state=CallState.COLLECTING_CITY)
        assert get_missing_slots(s) == ["city"]

    def test_collecting_customer_info_missing(self):
        s = _session(state=CallState.COLLECTING_CUSTOMER_INFO)
        assert get_missing_slots(s) == ["customer_name"]

    def test_new_booking_details_all_missing(self):
        s = _session(state=CallState.COLLECTING_BOOKING_DETAILS, intent=Intent.NEW_BOOKING)
        missing = get_missing_slots(s)
        assert "issue_description" in missing
        assert "preferred_date" in missing
        assert "preferred_time" in missing

    def test_reschedule_details_no_issue_needed(self):
        """Reschedule only needs date + time, not issue_description."""
        s = _session(state=CallState.COLLECTING_BOOKING_DETAILS, intent=Intent.RESCHEDULE)
        missing = get_missing_slots(s)
        assert "issue_description" not in missing
        assert "preferred_date" in missing
        assert "preferred_time" in missing

    def test_no_missing_when_state_has_no_required(self):
        s = _session(state=CallState.PRICING)
        assert get_missing_slots(s) == []


# ═════════════════════════════════════════════════════════════════════════════
# 13. END-TO-END MULTI-TURN SCENARIOS
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndScenarios:
    """Full multi-turn call simulations testing the complete flow.

    Note: In the live system, voice.py pre-advances GREETING → COLLECTING_CUSTOMER_INFO
    before the LLM runs. These tests start from COLLECTING_CUSTOMER_INFO to match that.
    """

    def test_full_new_booking_call(self):
        """Simulate a complete new booking call from name collection to end."""
        s = _session(state=CallState.COLLECTING_CUSTOMER_INFO)

        # Turn 1: name → city
        s = _advance(s, _result(extracted_slots={"customer_name": "Sarah"}))
        assert s.state == CallState.COLLECTING_CITY

        # Turn 2: city
        s = _advance(s, _result(extracted_slots={"city": "Scarborough"}))
        assert s.state == CallState.INTENT_DETECTION

        # Turn 3: intent
        s = _advance(s, _result(intent="new_booking"))
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS

        # Turn 4: issue
        s = _advance(s, _result(extracted_slots={"issue_description": "no hot water"}))
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS  # still needs date/time

        # Turn 5: date + time (standard slot)
        s = _advance(s, _result(extracted_slots={
            "preferred_date": "2026-03-09",
            "preferred_time": "10:00-12:00",
        }))
        assert s.state == CallState.CONFIRMING_BOOKING

        # Turn 6: confirm
        s = _advance(s, _result(extracted_slots={"confirmed": True}))
        assert s.state == CallState.CLOSING

        # Turn 7: close → wrap up
        s = _advance(s, _result())
        assert s.state == CallState.WRAP_UP

        # Turn 8: done
        s = _advance(s, _result(extracted_slots={"more_help": False}))
        assert s.state == CallState.ENDED

    def test_booking_then_cancel_same_call(self):
        """Caller books, then cancels another booking in the same call."""
        s = _session(state=CallState.COLLECTING_CUSTOMER_INFO)

        # Get through name/city
        s = _advance(s, _result(extracted_slots={"customer_name": "Dave"}))
        s = _advance(s, _result(extracted_slots={"city": "Toronto"}))
        s = _advance(s, _result(intent="new_booking"))
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS

        # Complete booking (standard slot)
        s = _advance(s, _result(extracted_slots={
            "issue_description": "thermostat broken",
            "preferred_date": "2026-03-09",
            "preferred_time": "08:00-10:00",
        }))
        assert s.state == CallState.CONFIRMING_BOOKING
        s = _advance(s, _result(extracted_slots={"confirmed": True}))
        assert s.state == CallState.CLOSING
        s = _advance(s, _result())
        assert s.state == CallState.WRAP_UP

        # Want more help → back to intent detection with reset
        s = _advance(s, _result(extracted_slots={"more_help": True}))
        assert s.state == CallState.INTENT_DETECTION
        assert s.intent is None
        assert s.slots.customer_name == "Dave"  # preserved

        # Now cancel a different booking
        s = _advance(s, _result(intent="cancellation"))
        assert s.state == CallState.COLLECTING_BOOKING_REF

        s = _advance(s, _result(extracted_slots={"booking_id": "bk_test002"}))
        assert s.state == CallState.CONFIRMING_BOOKING

        s = _advance(s, _result(extracted_slots={"confirmed": True}))
        assert s.state == CallState.CLOSING
        s = _advance(s, _result())
        assert s.state == CallState.WRAP_UP

        s = _advance(s, _result(extracted_slots={"more_help": False}))
        assert s.state == CallState.ENDED

    def test_out_of_area_call(self):
        """Caller from outside GTA — rejected early."""
        s = _session(state=CallState.COLLECTING_CUSTOMER_INFO)
        s = _advance(s, _result(extracted_slots={"customer_name": "Marc"}))
        s = _advance(s, _result(extracted_slots={"city": "Montreal"}))
        assert s.state == CallState.OUT_OF_AREA

        s = _advance(s, _result())
        assert s.state == CallState.ENDED

    def test_emergency_mid_booking(self):
        """Emergency mentioned mid-booking overrides everything."""
        s = _session(state=CallState.COLLECTING_BOOKING_DETAILS, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(
            customer_name="Kim", city="Toronto",
            issue_description="furnace issue",
        )

        s = _advance(s, _result(is_emergency=True))
        assert s.state == CallState.EMERGENCY_TRIAGE
        assert s.is_emergency is True

    def test_escalation_mid_booking(self):
        """Caller asks for human during booking → ESCALATING."""
        s = _session(state=CallState.COLLECTING_BOOKING_DETAILS, intent=Intent.NEW_BOOKING)
        s.slots = CallSlots(customer_name="Pat", city="Toronto")

        s = _advance(s, _result(intent="escalate"))
        assert s.state == CallState.ESCALATING
        assert s.intent == Intent.ESCALATE

    def test_pricing_to_booking_flow(self):
        """Caller asks about pricing, then decides to book."""
        s = _session(state=CallState.COLLECTING_CUSTOMER_INFO)
        s = _advance(s, _result(extracted_slots={"customer_name": "Liz"}))
        s = _advance(s, _result(extracted_slots={"city": "Markham"}))
        s = _advance(s, _result(intent="pricing"))
        assert s.state == CallState.PRICING

        s = _advance(s, _result())
        assert s.state == CallState.PRICING_FOLLOWUP

        s = _advance(s, _result(intent="new_booking"))
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS
        assert s.intent == Intent.NEW_BOOKING

    def test_surge_slot_accepted_full_flow(self):
        """Full flow with after-hours slot accepted."""
        s = _session(state=CallState.INTENT_DETECTION)
        s.slots = CallSlots(customer_name="Tom", city="Ajax")

        s = _advance(s, _result(intent="new_booking"))
        s = _advance(s, _result(extracted_slots={
            "issue_description": "AC not cooling",
            "preferred_date": "2026-03-15",    # Sunday = surge
            "preferred_time": "10:00-12:00",
        }))
        assert s.state == CallState.AFTER_HOURS_DISCLOSURE

        s = _advance(s, _result(extracted_slots={"after_hours_accepted": True}))
        assert s.state == CallState.CONFIRMING_BOOKING

        s = _advance(s, _result(extracted_slots={"confirmed": True}))
        assert s.state == CallState.CLOSING

    def test_three_unknown_then_correct_intent(self):
        """Caller gives ambiguous responses 3 times, then says 'new booking'."""
        s = _session(state=CallState.INTENT_DETECTION)
        s.slots = CallSlots(customer_name="Jo", city="Toronto")

        # Three rounds of unknown
        for _ in range(3):
            s = _advance(s, _result(intent="unknown"))
            assert s.state == CallState.INTENT_DETECTION

        # Fourth attempt — correct intent
        s = _advance(s, _result(intent="new_booking"))
        assert s.intent == Intent.NEW_BOOKING
        assert s.state == CallState.COLLECTING_BOOKING_DETAILS
