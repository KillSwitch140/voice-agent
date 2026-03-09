"""
Voice router — Twilio webhooks + Media Streams WebSocket + simulation endpoint.

Flow:
  POST /voice/inbound  →  TwiML (<Connect><Stream>)
  WS   /voice/stream   →  Deepgram STT → LLM → Deepgram TTS → Twilio
  POST /voice/simulate →  Inject a text transcript for local testing
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytz
from twilio.rest import Client as TwilioClient

TORONTO_TZ = pytz.timezone("America/Toronto")

from fastapi import APIRouter, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

from apps.orchestrator.config import get_settings
from apps.orchestrator.services import session_store
from apps.orchestrator.services.deepgram_stt import DeepgramSTT
from apps.orchestrator.services.llm import run_turn, try_regex_extraction
from apps.orchestrator.services.state_machine import apply_llm_result
from apps.orchestrator.services.supabase_logger import SupabaseLogger
from apps.orchestrator.services.tools import call_tool, _is_slot_surge
from apps.orchestrator.services.llm import _fmt_date, _fmt_time
from apps.orchestrator.services.tts import TTSService
from packages.core.models import CallState, Intent
from packages.core.utils import detect_emergency

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

settings = get_settings()

_RESOURCES = Path(__file__).parents[3] / "apps" / "resources"
_SAFETY_SCRIPT = (_RESOURCES / "scripts" / "safety_gas_smell.txt").read_text().strip()
_OPENING_SCRIPT = (_RESOURCES / "scripts" / "call_opening.txt").read_text().strip()


_tts_instance: TTSService | None = None
_db_instance: SupabaseLogger | None = None

# Fire-and-forget task set (prevents GC of background tasks)
_bg_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:
    """Schedule a coroutine as a fire-and-forget background task."""
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def _tts() -> TTSService:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSService(settings.deepgram_api_key)
    return _tts_instance


def _db() -> SupabaseLogger:
    global _db_instance
    if _db_instance is None:
        _db_instance = SupabaseLogger(settings.supabase_url, settings.supabase_service_key)
    return _db_instance


def _twilio() -> TwilioClient:
    return TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)


async def cleanup() -> None:
    """Close persistent clients (called on app shutdown)."""
    if _tts_instance:
        await _tts_instance.aclose()
    if _db_instance:
        await _db_instance.aclose()


async def _hangup_call(call_sid: str) -> None:
    """Terminate the live Twilio call via REST API."""
    def _end() -> None:
        try:
            _twilio().calls(call_sid).update(status="completed")
        except Exception as exc:
            logger.warning("Hangup failed for %s: %s", call_sid, exc)
    await asyncio.to_thread(_end)



# ─── Helpers ──────────────────────────────────────────────────────────────────

def _outcome_for_state(session) -> str:
    """Map terminal session intent/state to a human-readable DB outcome."""
    if session.slots.city and session.intent is None:
        return "out_of_area"
    if session.intent == Intent.CANCELLATION:
        return "cancelled"
    if session.intent == Intent.RESCHEDULE:
        return "rescheduled"
    if session.intent == Intent.PRICING:
        return "pricing_answered"
    return "completed"


async def _send_audio(ws: WebSocket, stream_sid: str, audio: bytes) -> None:
    """Send mulaw audio back to Twilio over the Media Stream WebSocket."""
    if not audio:
        return
    await ws.send_text(json.dumps({
        "event": "media",
        "streamSid": stream_sid,
        "media": {"payload": base64.b64encode(audio).decode()},
    }))


def _matches_time_of_day(time_slot: str, tod: str) -> bool:
    """Return True if the slot's start hour falls in the requested time-of-day band."""
    try:
        start_hour = int(time_slot.split(":")[0])
    except (ValueError, IndexError):
        return True  # unknown format — don't filter
    if tod == "morning":
        return start_hour < 12
    if tod == "afternoon":
        return 12 <= start_hour < 16
    if tod == "evening":
        return start_hour >= 16
    return True


async def _reload_slots(session, job_type: str | None = None) -> None:
    """Fetch available booking slots and store them on the session."""
    today = datetime.now(TORONTO_TZ).date()
    try:
        avail = await call_tool("get_availability", {
            "date_range": f"{today + timedelta(days=1)}/{today + timedelta(days=30)}",
            "job_type": job_type or session.slots.issue_description or "hvac_service",
            "city": session.slots.city or "",
        })
        session.available_slots = avail.get("available_slots", [])
        session.slot_offer_index = 0  # reset to first slot whenever slots are refreshed
        logger.info("Availability fetched: %d open slots", len(session.available_slots))
    except Exception as exc:
        logger.warning("get_availability failed: %s", exc)
        session.available_slots = []


def _clear_slot_preference(session) -> None:
    """Reset date/time/confirmation slots so the caller can pick again (used after a conflict)."""
    session.slots = session.slots.model_copy(
        update={"preferred_date": None, "preferred_time": None, "after_hours_accepted": None, "confirmed": False}
    )


async def _fetch_availability(session, prev_state: CallState):
    """Fetch slot availability whenever we need it in COLLECTING_BOOKING_DETAILS.

    Triggers on:
    - First entry into COLLECTING_BOOKING_DETAILS (reschedule — issue already known)
    - Any turn in COLLECTING_BOOKING_DETAILS once issue_description is collected
      but available_slots are not yet loaded (covers the common new-booking path where
      issue is given one turn after intent detection).
    """
    if session.state != CallState.COLLECTING_BOOKING_DETAILS:
        return session
    # Don't fetch until we have the issue description (new bookings).
    if session.intent != Intent.RESCHEDULE and not session.slots.issue_description:
        return session
    # Only fetch if we don't have slots yet.
    if not session.available_slots:
        await _reload_slots(session)
    return session


async def _run_booking_tools(session, prev_state: CallState) -> tuple:
    """Step 4b: call booking/cancel tools when state transitions to CLOSING.

    Returns (session, response_override) where response_override is a string
    to speak instead of the LLM response, or None to use the LLM response.
    """
    if prev_state == CallState.CLOSING or session.state != CallState.CLOSING:
        return session, None

    response_override = None

    if session.intent == Intent.NEW_BOOKING:
        try:
            result = await call_tool("create_booking", {
                "customer_name": session.slots.customer_name,
                "phone": session.from_number,
                "city": session.slots.city,
                "issue_description": session.slots.issue_description,
                "preferred_date": session.slots.preferred_date,
                "preferred_time": session.slots.preferred_time,
                "call_id": session.call_id or "",
                "is_emergency": session.is_emergency,
            })
            if result.get("success"):
                session.slots = session.slots.model_copy(update={"booking_id": result["booking_id"]})
                logger.info("Booking created: %s (tier=%s)", result["booking_id"], result.get("pricing_tier"))
            elif result.get("error") == "slot_taken":
                _clear_slot_preference(session)
                session.state = CallState.COLLECTING_BOOKING_DETAILS
                response_override = (
                    "I'm sorry, that slot was just taken by another caller. "
                    "Let me show you what's still available."
                )
                await _reload_slots(session)
            elif result.get("error") in ("past_date", "too_far"):
                session.state = CallState.ESCALATING
                response_override = (
                    "I'm sorry, I can only book within the next 30 days. "
                    "I'll connect you with a team member right away."
                )
            else:
                logger.error("create_booking failed: %s", result)
                session.state = CallState.ESCALATING
                response_override = (
                    "I'm sorry, I'm having trouble saving your booking right now. "
                    "Let me connect you with a team member who can help."
                )
        except Exception as exc:
            logger.error("create_booking exception: %s", exc)
            session.state = CallState.ESCALATING
            response_override = (
                "I'm sorry, I'm having trouble saving your booking right now. "
                "Let me connect you with a team member who can help."
            )

    elif session.intent == Intent.RESCHEDULE and session.slots.booking_id:
        try:
            result = await call_tool("reschedule_booking", {
                "booking_id": session.slots.booking_id,
                "new_date": session.slots.preferred_date,
                "new_time_slot": session.slots.preferred_time,
            })
            if result.get("error") == "slot_taken":
                _clear_slot_preference(session)
                session.state = CallState.COLLECTING_BOOKING_DETAILS
                response_override = (
                    "I'm sorry, that slot is already booked. "
                    "Let me show you what's available."
                )
                await _reload_slots(session, "reschedule")
            elif result.get("needs_human"):
                session.state = CallState.ESCALATING
                response_override = (
                    "Changes to past appointments need a team member. "
                    "I'll have someone call you right back."
                )
            elif result.get("error") == "not_found":
                session.slots = session.slots.model_copy(update={"booking_id": None})
                session.state = CallState.COLLECTING_BOOKING_REF
                response_override = (
                    "I couldn't find that booking reference. "
                    "Could you double-check the number and try again?"
                )
        except Exception as exc:
            logger.warning("reschedule_booking failed: %s", exc)

    elif session.intent == Intent.CANCELLATION and session.slots.booking_id:
        try:
            result = await call_tool("cancel_booking", {
                "booking_id": session.slots.booking_id,
                "reason": "Caller requested cancellation",
            })
            if result.get("needs_human"):
                session.state = CallState.ESCALATING
                response_override = (
                    "That appointment is in the past — a team member will need to help. "
                    "I'll have someone call you right back."
                )
            elif result.get("error") == "not_found":
                session.slots = session.slots.model_copy(update={"booking_id": None})
                session.state = CallState.COLLECTING_BOOKING_REF
                response_override = (
                    "I couldn't find that booking reference. "
                    "Could you double-check the number and try again?"
                )
        except Exception as exc:
            logger.warning("cancel_booking failed: %s", exc)

    return session, response_override


_PRICING_SCRIPT = (
    "Our standard diagnostic fee is $89 to $129. "
    "After-hours slots carry an additional $120 to $180 surcharge. "
    "Would you like to schedule a service call?"
)


def _transition_overrides(session, prev_state: CallState, response_override: str | None) -> str | None:
    """Apply deterministic response overrides for state transitions.

    Shared between _process_transcript and /simulate. May modify session.slots
    (e.g. resetting confirmed=False when entering CONFIRMING_BOOKING).
    Returns the response override text, or the incoming override if a booking tool
    already set one.
    """
    if response_override is not None:
        return response_override

    # 1. After city: capabilities menu (GTA city) or out-of-area rejection
    if prev_state == CallState.COLLECTING_CITY:
        if session.state == CallState.INTENT_DETECTION:
            name = session.slots.customer_name or ""
            return (
                f"Great{', ' + name if name else ''}! "
                "I can help you book a new service appointment, reschedule or cancel an existing "
                "booking, or answer pricing questions. What can I help you with today?"
            )
        if session.state == CallState.OUT_OF_AREA:
            city = session.slots.city or "your area"
            return (
                "I'm sorry, we only service the Greater Toronto Area. "
                f"Unfortunately, {city} is outside our coverage area. Have a great day!"
            )

    # 2. Entering CONFIRMING_BOOKING: always ask the confirmation question.
    #    Reset confirmed=False to prevent premature skip to CLOSING.
    if (prev_state != CallState.CONFIRMING_BOOKING
            and session.state == CallState.CONFIRMING_BOOKING):
        session.slots = session.slots.model_copy(update={"confirmed": False})
        if session.intent == Intent.CANCELLATION:
            return (
                f"Just to confirm, you'd like to cancel booking "
                f"{session.slots.booking_id}. Is that correct?"
            )
        if session.intent == Intent.RESCHEDULE and session.slots.preferred_date and session.slots.preferred_time:
            return (
                f"Just to confirm, we'll reschedule your appointment to "
                f"{_fmt_date(session.slots.preferred_date)} "
                f"at {_fmt_time(session.slots.preferred_time)}. Is that correct?"
            )
        if session.slots.preferred_date and session.slots.preferred_time:
            return (
                f"Just to confirm, we have you booked for "
                f"{_fmt_date(session.slots.preferred_date)} "
                f"at {_fmt_time(session.slots.preferred_time)}. Shall I go ahead?"
            )

    # 3. CONFIRMING_BOOKING → CLOSING: booking is done.
    if (prev_state == CallState.CONFIRMING_BOOKING
            and session.state == CallState.CLOSING):
        if session.intent == Intent.CANCELLATION:
            return "Your booking has been cancelled."
        if session.intent == Intent.RESCHEDULE:
            return (
                f"Your appointment has been rescheduled to "
                f"{_fmt_date(session.slots.preferred_date)} "
                f"at {_fmt_time(session.slots.preferred_time)}."
            )
        return (
            f"You're all set! We have you booked for "
            f"{_fmt_date(session.slots.preferred_date)} "
            f"at {_fmt_time(session.slots.preferred_time)}."
        )

    # 4. Entering WRAP_UP from any state: always ask "anything else?"
    if prev_state != CallState.WRAP_UP and session.state == CallState.WRAP_UP:
        return "Is there anything else I can help you with today?"

    # 5. WRAP_UP → INTENT_DETECTION (caller wants more help).
    if prev_state == CallState.WRAP_UP and session.state == CallState.INTENT_DETECTION:
        name = session.slots.customer_name or ""
        return (
            f"Of course{', ' + name if name else ''}! "
            "I can help you book a new service appointment, reschedule or cancel an existing "
            "booking, or answer pricing questions. What can I help you with today?"
        )

    # 6. WRAP_UP → ENDED (caller is done).
    if prev_state == CallState.WRAP_UP and session.state == CallState.ENDED:
        return "Thank you for calling Toronto HVAC Services. Have a great day!"

    return None


def _scripted_response(session, prev_state: CallState) -> str | None:
    """Return the deterministic spoken response for the current state.

    Covers every state that is NOT already handled by the inline override blocks
    in _process_transcript. Returns None only if an inline override already exists
    for this transition (e.g. capabilities menu, confirmation question).
    """
    state = session.state
    slots = session.slots

    if state == CallState.COLLECTING_CUSTOMER_INFO:
        return "Could I get your name please?"

    if state == CallState.COLLECTING_CITY:
        return "And what city are you calling from?"

    # INTENT_DETECTION staying (no intent detected yet).
    # Transitions INTO INTENT_DETECTION already have inline overrides (capabilities menu).
    if state == CallState.INTENT_DETECTION and prev_state == CallState.INTENT_DETECTION:
        return (
            "I can help with a new service appointment, rescheduling, cancellations, or pricing. "
            "What would you like to do?"
        )

    if state == CallState.COLLECTING_BOOKING_REF:
        if session.intent == Intent.RESCHEDULE:
            return (
                "Sure, I can help you reschedule. "
                "Could you please provide your booking reference number? It usually starts with bk_."
            )
        return (
            "Sure, I can help you with that. "
            "Could you please provide your booking reference number? It usually starts with bk_."
        )

    if state == CallState.COLLECTING_BOOKING_DETAILS:
        if not slots.issue_description:
            return "What's the issue with your HVAC system?"
        # Issue collected — offer a slot from the available list with pricing tier.
        if session.available_slots:
            idx = min(session.slot_offer_index, len(session.available_slots) - 1)
            slot = session.available_slots[idx]
            tier = slot.get("pricing_tier", "standard")
            rate_info = "at our standard rate" if tier == "standard" else "at our after-hours rate"
            return (
                f"We have {_fmt_date(slot['date'])} at {_fmt_time(slot['time_slot'])}, "
                f"{rate_info}. Does that work for you?"
            )
        # Slots not loaded yet (still fetching or none available).
        return "What day and time works best for you?"

    if state == CallState.AFTER_HOURS_DISCLOSURE:
        date_str = _fmt_date(slots.preferred_date) if slots.preferred_date else "that slot"
        time_str = f" at {_fmt_time(slots.preferred_time)}" if slots.preferred_time else ""
        return (
            f"Just so you know, {date_str}{time_str} is outside our standard hours. "
            "An after-hours surcharge of $120 to $180 applies on top of the standard fee. "
            "Would you like to proceed?"
        )

    # PRICING — play the static pricing script (state advances to PRICING_FOLLOWUP next turn).
    if state == CallState.PRICING:
        return _PRICING_SCRIPT

    if state == CallState.PRICING_FOLLOWUP:
        if session.intent == Intent.NEW_BOOKING:
            return "Sure, let me help you schedule that!"
        return "Would you like to go ahead and book a service call?"

    # CONFIRMING_BOOKING staying — entry is already handled by inline override.
    if state == CallState.CONFIRMING_BOOKING and prev_state == CallState.CONFIRMING_BOOKING:
        if slots.preferred_date and slots.preferred_time:
            return (
                f"To confirm — {_fmt_date(slots.preferred_date)} "
                f"at {_fmt_time(slots.preferred_time)}. Shall I go ahead?"
            )
        return "Shall I go ahead and book that for you?"

    # ESCALATING from any non-escalating state.
    if state == CallState.ESCALATING and prev_state != CallState.ESCALATING:
        return "I'll have someone from our team call you right back. Thank you for your patience."

    # All remaining states (CLOSING, WRAP_UP entry, WRAP_UP→ENDED, OUT_OF_AREA)
    # are covered by the existing inline overrides — return None to let those run.
    return None


async def _process_transcript(transcript: str, call_sid: str, ws: WebSocket) -> None:
    """Core pipeline: transcript → state machine → LLM → TTS → Twilio."""
    session = session_store.get_session(call_sid)
    if session is None or session.state == CallState.ENDED:
        return

    db = _db()
    tts = _tts()

    # 1. Hard-coded emergency detection — bypasses LLM
    if detect_emergency(transcript) or session.is_emergency:
        session.is_emergency = True
        session.state = CallState.EMERGENCY_TRIAGE
        safety_audio = await tts.synthesize(_SAFETY_SCRIPT)
        await _send_audio(ws, session.stream_sid, safety_audio)
        await asyncio.sleep(len(safety_audio) / 8000 + 0.5)
        await call_tool("escalate_call", {
            "call_id": session.call_id or session.call_sid,
            "reason": "Emergency keyword detected",
            "transcript_summary": transcript,
            "is_emergency": True,
        })
        session.state = CallState.ENDED
        session_store.save_session(session)
        if session.call_id:
            await db.close_call(session.call_id, "emergency_escalated")
        # Hang up after script plays — caller must leave the building
        try:
            await _hangup_call(session.call_sid)
        except Exception as exc:
            logger.warning("Emergency hangup failed: %s", exc)
        return

    # 2. Pre-advance GREETING → COLLECTING_CUSTOMER_INFO so the LLM always runs
    #    with the correct initial state and knows to ask for the caller's name first.
    if session.state == CallState.GREETING:
        session.state = CallState.COLLECTING_CUSTOMER_INFO

    # Add user turn to rolling context
    session.turns.append({"role": "user", "content": transcript})
    if session.call_id:
        _fire_and_forget(db.log_turn(session.call_id, "user", transcript, session.state.value))

    # 3. LLM turn — extraction only; Python generates all spoken responses.
    # Track whether we were already in slot-offering mode before this turn so we
    # know whether to advance the slot index if the user declines the offered slot.
    was_offering_slot = (
        session.state == CallState.COLLECTING_BOOKING_DETAILS
        and bool(session.slots.issue_description)
        and bool(session.available_slots)
    )

    # Fast path: regex extraction for simple yes/no states (skips LLM entirely)
    llm_result = try_regex_extraction(session, transcript)
    if llm_result is None:
        llm_result = await run_turn(
            session=session,
            utterance=transcript,
            openai_api_key=settings.openai_api_key,
        )

    # 4. Apply result to state machine
    prev_state = session.state
    session = apply_llm_result(session, llm_result)

    # 4a+4b. Fetch availability and run booking tools (deterministic)
    session = await _fetch_availability(session, prev_state)
    session, response_override = await _run_booking_tools(session, prev_state)

    # Navigate slot_offer_index based on caller's date/time-of-day preference or decline.
    if session.state == CallState.COLLECTING_BOOKING_DETAILS and session.available_slots:
        req_date = session.slots.requested_date
        tod = session.slots.requested_time_of_day
        if req_date or tod:
            # Determine start index: jump to requested date, or stay at current index.
            if req_date:
                start_idx = next(
                    (i for i, s in enumerate(session.available_slots) if s["date"] >= req_date),
                    len(session.available_slots) - 1,
                )
            else:
                start_idx = session.slot_offer_index
            # Apply time-of-day filter from start_idx onward.
            if tod:
                session.slot_offer_index = next(
                    (i for i in range(start_idx, len(session.available_slots))
                     if _matches_time_of_day(session.available_slots[i]["time_slot"], tod)),
                    min(start_idx, len(session.available_slots) - 1),
                )
            else:
                session.slot_offer_index = start_idx
            session.slots = session.slots.model_copy(
                update={"requested_date": None, "requested_time_of_day": None}
            )
        elif was_offering_slot and not session.slots.preferred_date:
            # Caller declined the offered slot — advance to the next one.
            session.slot_offer_index = min(
                session.slot_offer_index + 1,
                len(session.available_slots) - 1,
            )

    # ── Deterministic response overrides (shared with /simulate) ─────────────
    response_override = _transition_overrides(session, prev_state, response_override)

    # Speak the response — scripted responses cover every state; LLM text is last resort.
    if response_override is None:
        response_override = _scripted_response(session, prev_state)
    response_text = response_override or llm_result.response_text or "I'm sorry, could you repeat that?"
    session.turns.append({"role": "assistant", "content": response_text})
    if session.call_id:
        _fire_and_forget(db.log_turn(session.call_id, "assistant", response_text, session.state.value))
    response_audio = await tts.synthesize(response_text)
    await _send_audio(ws, session.stream_sid, response_audio)
    # mulaw 8kHz = 8000 bytes/sec — used below to wait for playback before hangup
    _response_play_duration = len(response_audio) / 8000

    # Defensive: if LLM said goodbye in WRAP_UP but didn't extract more_help=false,
    # infer it so the ENDED block below fires and the call terminates.
    if session.state == CallState.WRAP_UP and session.slots.more_help is None:
        _goodbye = ("great day", "good day", "goodbye", "take care", "bye", "thank you for calling")
        if any(p in response_text.lower() for p in _goodbye):
            session.slots = session.slots.model_copy(update={"more_help": False})
            session.state = CallState.ENDED

    # 6. Handle terminal states — escalation logging, booking SMS, and call close
    if session.state == CallState.ESCALATING:
        await call_tool("escalate_call", {
            "call_id": session.call_id or session.call_sid,
            "reason": "Caller requested human agent",
            "transcript_summary": transcript,
            "is_emergency": session.is_emergency,
        })
        session.state = CallState.ENDED
        if session.call_id:
            await db.close_call(session.call_id, "escalated")
        await asyncio.sleep(_response_play_duration + 0.5)
        try:
            await ws.close()
        except Exception:
            pass
        try:
            await _hangup_call(session.call_sid)
        except Exception as exc:
            logger.warning("Escalation hangup failed: %s", exc)

    elif session.state == CallState.EMERGENCY_TRIAGE:
        # LLM detected an emergency (backend keyword filter didn't catch the phrasing).
        await call_tool("escalate_call", {
            "call_id": session.call_id or session.call_sid,
            "reason": "Emergency detected by LLM",
            "transcript_summary": transcript,
            "is_emergency": True,
        })
        session.state = CallState.ENDED
        if session.call_id:
            await db.close_call(session.call_id, "emergency_escalated")
        await asyncio.sleep(_response_play_duration + 0.5)
        try:
            await ws.close()
        except Exception:
            pass
        try:
            await _hangup_call(session.call_sid)
        except Exception as exc:
            logger.warning("Emergency (LLM) hangup failed: %s", exc)

    elif prev_state != CallState.OUT_OF_AREA and session.state == CallState.OUT_OF_AREA:
        # Out-of-area: send referral SMS then hang up immediately (no "anything else?" wait)
        city = session.slots.city or "your area"
        if session.from_number:
            await call_tool("send_sms", {
                "phone": session.from_number,
                "message": (
                    f"Toronto HVAC Services: Unfortunately we don't service {city} at this time. "
                    "For local HVAC help, contact a licensed contractor in your area. "
                    "Thank you for calling — we hope to serve your area soon!"
                ),
            })
        if session.call_id:
            await db.close_call(session.call_id, "out_of_area")
        session.state = CallState.ENDED
        await asyncio.sleep(_response_play_duration + 0.5)
        try:
            await ws.close()
        except Exception:
            pass
        try:
            await _hangup_call(session.call_sid)
        except Exception as exc:
            logger.warning("Out-of-area hangup failed: %s", exc)

    elif prev_state != CallState.CLOSING and session.state == CallState.CLOSING:
        # Send booking confirmation SMS — state advances to WRAP_UP next turn
        phone = session.from_number
        name = session.slots.customer_name or "Valued Customer"
        sms_msg: str | None = None
        if phone:
            if session.intent == Intent.RESCHEDULE and session.slots.booking_id:
                sms_msg = (
                    f"Hi {name}, your HVAC appointment (ref: {session.slots.booking_id}) "
                    f"has been rescheduled to {session.slots.preferred_date} "
                    f"({session.slots.preferred_time or ''}). "
                    "Toronto HVAC Services — reply STOP to opt out."
                )
            elif session.intent == Intent.CANCELLATION and session.slots.booking_id:
                sms_msg = (
                    f"Hi {name}, your HVAC appointment (ref: {session.slots.booking_id}) "
                    "has been cancelled as requested. "
                    "Toronto HVAC Services — reply STOP to opt out."
                )
            elif session.slots.preferred_date:
                sms_msg = (
                    f"Hi {name}, your HVAC appointment is confirmed for "
                    f"{session.slots.preferred_date} ({session.slots.preferred_time or ''}). "
                    "Toronto HVAC Services — reply STOP to opt out."
                )
        if sms_msg:
            await call_tool("send_sms", {"phone": phone, "message": sms_msg})

    elif session.state == CallState.ENDED:
        await asyncio.sleep(_response_play_duration + 0.5)
        if session.call_id:
            await db.close_call(session.call_id, _outcome_for_state(session))
        try:
            await ws.close()
        except Exception:
            pass
        try:
            await _hangup_call(session.call_sid)
        except Exception as exc:
            logger.warning("Hangup failed: %s", exc)

    session_store.save_session(session)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/inbound")
async def inbound(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
):
    """
    Twilio calls this webhook when a call arrives.
    Returns TwiML that opens a bidirectional Media Stream WebSocket.
    """
    session = session_store.create_session(call_sid=CallSid, from_number=From)

    db = _db()
    call_id = await db.log_call(CallSid, From)
    session.call_id = call_id
    session_store.save_session(session)

    ws_url = settings.orchestrator_base_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{ws_url}/voice/stream/{CallSid}"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@router.websocket("/stream/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str):
    """
    Twilio Media Streams WebSocket.
    Receives mulaw audio → Deepgram STT → LLM → Deepgram TTS → Twilio.
    """
    await websocket.accept()
    session = session_store.get_session(call_sid)
    if session is None:
        await websocket.close(code=1008)
        return

    tts = _tts()
    stt = DeepgramSTT(api_key=settings.deepgram_api_key)

    async def on_transcript(text: str) -> None:
        await _process_transcript(text, call_sid, websocket)

    await stt.start(on_transcript)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                session.stream_sid = msg["streamSid"]
                session.turns.append({"role": "assistant", "content": _OPENING_SCRIPT})
                session_store.save_session(session)
                logger.info("Stream started: %s", session.stream_sid)
                await _send_audio(websocket, session.stream_sid, await tts.synthesize(_OPENING_SCRIPT))

            elif event == "media":
                await stt.send(base64.b64decode(msg["media"]["payload"]))

            elif event == "stop":
                logger.info("Stream stopped: %s", call_sid)
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", call_sid)
    finally:
        await stt.close()
        # Ensure the call is always closed in the DB, even if the caller
        # hung up mid-conversation without reaching a terminal state.
        final_session = session_store.get_session(call_sid)
        if final_session and final_session.call_id:
            if final_session.state != CallState.ENDED:
                await _db().close_call(final_session.call_id, "abandoned")
        session_store.delete_session(call_sid)


@router.get("/callback-twiml")
async def callback_twiml():
    """
    TwiML served to outbound callback calls made during human escalation.
    Plays a hold message — a team member answers on the business line.
    """
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Please hold. A Toronto HVAC Services team member will be with you shortly.</Say>
  <Pause length="3"/>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@router.post("/simulate")
async def simulate(request: Request):
    """
    Simulate a single transcript turn without a real phone call.
    Body: {"call_sid": "...", "transcript": "...", "from_number": "+1..."}
    """
    body = await request.json()
    call_sid: str = body.get("call_sid", "SIM_001")
    transcript: str = body.get("transcript", "")
    from_number: str = body.get("from_number", "+14161234567")

    if not transcript:
        return {"error": "transcript required"}

    session = session_store.get_session(call_sid)
    if session is None:
        session = session_store.create_session(call_sid, from_number)
        db = _db()
        call_id = await db.log_call(call_sid, from_number)
        session.call_id = call_id
        session_store.save_session(session)

    if detect_emergency(transcript):
        session.is_emergency = True
        session.state = CallState.EMERGENCY_TRIAGE
        session_store.save_session(session)
        return {
            "call_sid": call_sid,
            "response": _SAFETY_SCRIPT,
            "state": "emergency_triage",
            "is_emergency": True,
        }

    # Pre-advance GREETING → COLLECTING_CUSTOMER_INFO (same as live path)
    if session.state == CallState.GREETING:
        session.state = CallState.COLLECTING_CUSTOMER_INFO

    session.turns.append({"role": "user", "content": transcript})

    was_offering_slot = (
        session.state == CallState.COLLECTING_BOOKING_DETAILS
        and bool(session.slots.issue_description)
        and bool(session.available_slots)
    )

    # Fast path: regex extraction for simple yes/no states (skips LLM entirely)
    llm_result = try_regex_extraction(session, transcript)
    if llm_result is None:
        llm_result = await run_turn(
            session=session,
            utterance=transcript,
            openai_api_key=settings.openai_api_key,
        )
    prev_state = session.state
    session = apply_llm_result(session, llm_result)

    # Same availability fetch + booking tool logic as the live call path
    session = await _fetch_availability(session, prev_state)
    session, response_override = await _run_booking_tools(session, prev_state)

    if session.state == CallState.COLLECTING_BOOKING_DETAILS and session.available_slots:
        req_date = session.slots.requested_date
        tod = session.slots.requested_time_of_day
        if req_date or tod:
            if req_date:
                start_idx = next(
                    (i for i, s in enumerate(session.available_slots) if s["date"] >= req_date),
                    len(session.available_slots) - 1,
                )
            else:
                start_idx = session.slot_offer_index
            if tod:
                session.slot_offer_index = next(
                    (i for i in range(start_idx, len(session.available_slots))
                     if _matches_time_of_day(session.available_slots[i]["time_slot"], tod)),
                    min(start_idx, len(session.available_slots) - 1),
                )
            else:
                session.slot_offer_index = start_idx
            session.slots = session.slots.model_copy(
                update={"requested_date": None, "requested_time_of_day": None}
            )
        elif was_offering_slot and not session.slots.preferred_date:
            session.slot_offer_index = min(
                session.slot_offer_index + 1,
                len(session.available_slots) - 1,
            )

    # Apply the same transition overrides as the live call path
    response_override = _transition_overrides(session, prev_state, response_override)
    if response_override is None:
        response_override = _scripted_response(session, prev_state)
    response_text = response_override or llm_result.response_text or "I'm sorry, could you repeat that?"
    session.turns.append({"role": "assistant", "content": response_text})
    session_store.save_session(session)

    return {
        "call_sid": call_sid,
        "response": response_text,
        "state": session.state.value,
        "intent": session.intent,
        "slots": session.slots.model_dump(exclude_none=True),
        "is_emergency": session.is_emergency,
    }


# ─── WebRTC test client ────────────────────────────────────────────────────────

@router.get("/token")
async def get_access_token():
    """Generate a Twilio Access Token for the browser WebRTC test client."""
    from twilio.jwt.access_token import AccessToken
    from twilio.jwt.access_token.grants import VoiceGrant

    if not all([settings.twilio_api_key_sid, settings.twilio_api_key_secret, settings.twilio_twiml_app_sid]):
        return JSONResponse(
            {"error": "Set TWILIO_API_KEY_SID, TWILIO_API_KEY_SECRET, and TWILIO_TWIML_APP_SID in .env"},
            status_code=503,
        )

    token = AccessToken(
        settings.twilio_account_sid,
        settings.twilio_api_key_sid,
        settings.twilio_api_key_secret,
        identity="test_caller",
        ttl=3600,
    )
    token.add_grant(VoiceGrant(
        outgoing_application_sid=settings.twilio_twiml_app_sid,
        incoming_allow=False,
    ))
    jwt = token.to_jwt()
    # to_jwt() returns bytes in some SDK versions
    return {"token": jwt.decode() if isinstance(jwt, bytes) else jwt}


@router.post("/client-inbound")
async def client_inbound(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(default="client:test_caller"),
    PhoneNumber: str = Form(default=""),
):
    """
    TwiML App Voice URL — Twilio calls this when the browser WebRTC client dials.
    Creates a session and returns the same <Connect><Stream> TwiML as /voice/inbound.
    PhoneNumber is an optional custom param forwarded by the test page for SMS delivery.
    """
    # Use the provided real phone number (for SMS) or fall back to a placeholder
    from_number = PhoneNumber.strip() if PhoneNumber.strip().startswith("+") else (
        From if From.startswith("+") else "+10000000000"
    )
    session = session_store.create_session(call_sid=CallSid, from_number=from_number)

    db = _db()
    call_id = await db.log_call(CallSid, from_number)
    session.call_id = call_id
    session_store.save_session(session)

    ws_url = settings.orchestrator_base_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{ws_url}/voice/stream/{CallSid}"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@router.get("/session-info/{call_sid}")
async def session_info(call_sid: str):
    """Return live session state for the test client debug panel."""
    session = session_store.get_session(call_sid)
    if session is None:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return {
        "state": session.state.value,
        "intent": session.intent.value if session.intent else None,
        "is_emergency": session.is_emergency,
        "slots": session.slots.model_dump(exclude_none=True),
        "turns": [{"role": t["role"], "content": t["content"]} for t in session.turns],
    }


@router.get("/test-client", response_class=HTMLResponse)
async def test_client():
    """Serve the browser WebRTC test UI."""
    return HTMLResponse((_RESOURCES / "test_client.html").read_text(encoding="utf-8"))
