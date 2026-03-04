"""
Voice router — Twilio webhooks + Media Streams WebSocket + simulation endpoint.

Flow:
  POST /voice/inbound  →  TwiML (<Connect><Stream>)
  WS   /voice/stream   →  Deepgram STT → LLM → Deepgram TTS → Twilio
  POST /voice/simulate →  Inject a text transcript for local testing
"""
from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytz

TORONTO_TZ = pytz.timezone("America/Toronto")

from fastapi import APIRouter, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from apps.orchestrator.config import get_settings
from apps.orchestrator.services import session_store
from apps.orchestrator.services.deepgram_stt import DeepgramSTT
from apps.orchestrator.services.llm import run_turn
from apps.orchestrator.services.state_machine import apply_llm_result
from apps.orchestrator.services.supabase_logger import SupabaseLogger
from apps.orchestrator.services.tools import call_tool
from apps.orchestrator.services.tts import TTSService
from packages.core.models import CallState, Intent
from packages.core.utils import detect_emergency

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

settings = get_settings()

_RESOURCES = Path(__file__).parents[3] / "apps" / "resources"
_SAFETY_SCRIPT = (_RESOURCES / "scripts" / "safety_gas_smell.txt").read_text().strip()
_OPENING_SCRIPT = (_RESOURCES / "scripts" / "call_opening.txt").read_text().strip()


def _tts() -> TTSService:
    return TTSService(settings.deepgram_api_key)


def _db() -> SupabaseLogger:
    return SupabaseLogger(settings.supabase_url, settings.supabase_service_key)


async def _hangup_call(call_sid: str) -> None:
    """Terminate the live Twilio call via REST API.

    Used after safety-critical scripts (emergency, out-of-area) where we must
    end the call without waiting for the caller to hang up.
    """
    import asyncio
    from twilio.rest import Client as TwilioClient

    def _end() -> None:
        try:
            client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
            client.calls(call_sid).update(status="completed")
        except Exception as exc:
            logger.warning("Hangup failed for %s: %s", call_sid, exc)

    await asyncio.to_thread(_end)


async def _initiate_callback(from_number: str) -> None:
    """Place an outbound callback call to the caller via Twilio REST API.

    Used for human escalation: instead of a warm transfer (not possible mid-stream),
    we redial the caller so a team member can pick up on their end.
    """
    import asyncio
    from twilio.rest import Client as TwilioClient

    def _call() -> None:
        client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
        callback_url = f"{settings.orchestrator_base_url}/voice/callback-twiml"
        client.calls.create(
            to=from_number,
            from_=settings.twilio_phone_number,
            url=callback_url,
        )

    await asyncio.to_thread(_call)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _outcome_for_state(session) -> str:
    """Map terminal session intent/state to a human-readable DB outcome."""
    if session.state == CallState.OUT_OF_AREA:
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


async def _fetch_availability(session, prev_state: CallState):
    """Step 4a: fetch real slot availability on first entry into COLLECTING_BOOKING_DETAILS."""
    if prev_state == CallState.COLLECTING_BOOKING_DETAILS or session.state != CallState.COLLECTING_BOOKING_DETAILS:
        return session
    today = datetime.now(TORONTO_TZ).date()
    try:
        avail = await call_tool("get_availability", {
            "date_range": f"{today + timedelta(days=1)}/{today + timedelta(days=14)}",
            "job_type": session.slots.issue_description or "hvac_service",
            "postal_code": session.slots.postal_code or "",
        })
        session.available_slots = avail.get("available_slots", [])
        logger.info("Availability fetched: %d open slots", len(session.available_slots or []))
    except Exception as exc:
        logger.warning("get_availability failed: %s", exc)
        session.available_slots = []
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
                "postal_code": session.slots.postal_code,
                "issue_description": session.slots.issue_description,
                "preferred_date": session.slots.preferred_date,
                "preferred_time": session.slots.preferred_time,
                "call_id": session.call_id or session.call_sid,
                "is_emergency": session.is_emergency,
            })
            if result.get("success"):
                session.slots = session.slots.model_copy(update={"booking_id": result["booking_id"]})
                logger.info("Booking created: %s (tier=%s)", result["booking_id"], result.get("pricing_tier"))
            elif result.get("error") == "slot_taken":
                session.slots = session.slots.model_copy(
                    update={"preferred_date": None, "preferred_time": None, "after_hours_accepted": None}
                )
                session.state = CallState.COLLECTING_BOOKING_DETAILS
                response_override = (
                    "I'm sorry, that slot was just taken by another caller. "
                    "Let me show you what's still available."
                )
                today = datetime.now(TORONTO_TZ).date()
                avail = await call_tool("get_availability", {
                    "date_range": f"{today + timedelta(days=1)}/{today + timedelta(days=14)}",
                    "job_type": session.slots.issue_description or "hvac_service",
                    "postal_code": session.slots.postal_code or "",
                })
                session.available_slots = avail.get("available_slots", [])
            elif result.get("error") in ("past_date", "too_far"):
                session.state = CallState.ESCALATING
                response_override = (
                    "I'm sorry, I can only book within the next 30 days. "
                    "I'll connect you with a team member right away."
                )
            else:
                logger.warning("create_booking error: %s", result)
        except Exception as exc:
            logger.warning("create_booking failed: %s", exc)

    elif session.intent == Intent.RESCHEDULE and session.slots.booking_id:
        try:
            result = await call_tool("reschedule_booking", {
                "booking_id": session.slots.booking_id,
                "new_date": session.slots.preferred_date,
                "new_time_slot": session.slots.preferred_time,
            })
            if result.get("error") == "slot_taken":
                session.slots = session.slots.model_copy(
                    update={"preferred_date": None, "preferred_time": None, "after_hours_accepted": None}
                )
                session.state = CallState.COLLECTING_BOOKING_DETAILS
                response_override = (
                    "I'm sorry, that slot is already booked. "
                    "Let me show you what's available."
                )
                today = datetime.now(TORONTO_TZ).date()
                avail = await call_tool("get_availability", {
                    "date_range": f"{today + timedelta(days=1)}/{today + timedelta(days=14)}",
                    "job_type": "reschedule",
                    "postal_code": session.slots.postal_code or "",
                })
                session.available_slots = avail.get("available_slots", [])
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
        await _send_audio(ws, session.stream_sid, await tts.synthesize(_SAFETY_SCRIPT))
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
        # Hang up proactively — caller must leave the building, not stay on the line
        try:
            await _hangup_call(session.call_sid)
        except Exception as exc:
            logger.warning("Emergency hangup failed: %s", exc)
        return

    # 2. Add user turn to rolling context
    session.turns.append({"role": "user", "content": transcript})
    if session.call_id:
        await db.log_turn(session.call_id, "user", transcript, session.state.value)

    # 3. LLM turn
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

    # 5. Speak the response
    response_text = response_override if response_override else llm_result.response_text
    session.turns.append({"role": "assistant", "content": response_text})
    if session.call_id:
        await db.log_turn(session.call_id, "assistant", response_text, session.state.value)
    await _send_audio(ws, session.stream_sid, await tts.synthesize(response_text))

    # 6. Handle terminal states — escalation callback, booking SMS, and call close
    if session.state == CallState.ESCALATING:
        # Log the escalation record
        await call_tool("escalate_call", {
            "call_id": session.call_id or session.call_sid,
            "reason": "Caller requested human agent",
            "transcript_summary": transcript,
            "is_emergency": session.is_emergency,
        })
        # Redial the caller so a team member can pick up
        try:
            await _initiate_callback(session.from_number)
            logger.info("Callback initiated to %s", session.from_number)
        except Exception as exc:
            logger.warning("Callback initiation failed: %s", exc)
        session.state = CallState.ENDED
        if session.call_id:
            await db.close_call(session.call_id, "escalated")

    elif session.state == CallState.OUT_OF_AREA:
        # Send referral SMS then hang up — no point waiting for caller input
        postal = session.slots.postal_code or "your area"
        if session.from_number:
            await call_tool("send_sms", {
                "phone": session.from_number,
                "message": (
                    f"Toronto HVAC Services: Unfortunately we don't service {postal} at this time. "
                    "For local HVAC help, contact a licensed contractor in your area. "
                    "Thank you for calling — we hope to serve your area soon!"
                ),
            })
        try:
            await _hangup_call(session.call_sid)
        except Exception as exc:
            logger.warning("Hangup after out_of_area failed: %s", exc)
        session.state = CallState.ENDED
        if session.call_id:
            await db.close_call(session.call_id, "out_of_area")

    elif session.state in (CallState.CLOSING, CallState.ENDED):
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
        session.state = CallState.ENDED
        if session.call_id:
            outcome = _outcome_for_state(session)
            await db.close_call(session.call_id, outcome)

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
        return {
            "call_sid": call_sid,
            "response": _SAFETY_SCRIPT,
            "state": "emergency_triage",
            "is_emergency": True,
        }

    session.turns.append({"role": "user", "content": transcript})
    llm_result = await run_turn(
        session=session,
        utterance=transcript,
        openai_api_key=settings.openai_api_key,
    )
    prev_state = session.state
    session = apply_llm_result(session, llm_result)

    # 4a+4b. Same availability fetch + booking tool logic as the live call path
    session = await _fetch_availability(session, prev_state)
    session, response_override = await _run_booking_tools(session, prev_state)

    response_text = response_override if response_override else llm_result.response_text
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
