"""
Business logic tools — inlined from the former MCP server.

All tools are plain async functions called directly from the orchestrator.
call_tool(name, args) is the single dispatch entry point.

Booking rules enforced here (not just in prompts):
  - Future-only: preferred_date must be > today (Toronto time)
  - Max 30 days in advance
  - No double-booking: slot conflict check against existing confirmed bookings
  - Pricing tier: STANDARD during business hours, SURGE otherwise
  - Past bookings (modify/cancel): requires human escalation
"""
from __future__ import annotations

import logging
import os
import asyncio
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytz

from apps.orchestrator.config import get_settings

logger = logging.getLogger(__name__)

_RESOURCES_DIR = Path(__file__).parents[3] / "apps" / "resources"
TORONTO_TZ = pytz.timezone("America/Toronto")


# ─── Slot definitions ─────────────────────────────────────────────────────────
# Each tuple: (time_slot_string, pricing_tier)

_WEEKDAY_SLOTS = [          # Mon–Fri
    ("08:00-10:00", "standard"),
    ("10:00-12:00", "standard"),
    ("12:00-14:00", "standard"),
    ("14:00-16:00", "standard"),
    ("16:00-18:00", "surge"),
    ("18:00-20:00", "surge"),
]
_SATURDAY_SLOTS = [
    ("09:00-11:00", "standard"),
    ("11:00-13:00", "standard"),
    ("13:00-15:00", "surge"),
    ("15:00-17:00", "surge"),
]
_SUNDAY_SLOTS = [
    ("10:00-12:00", "surge"),
    ("12:00-14:00", "surge"),
    ("14:00-16:00", "surge"),
    ("16:00-18:00", "surge"),
]


def _slots_for_date(d: date) -> list[tuple[str, str]]:
    wd = d.weekday()  # 0=Mon, 6=Sun
    if wd < 5:
        return _WEEKDAY_SLOTS
    if wd == 5:
        return _SATURDAY_SLOTS
    return _SUNDAY_SLOTS


def _is_slot_surge(date_str: str, time_slot: str) -> bool:
    """Return True if the given slot falls outside standard business hours."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        for slot, tier in _slots_for_date(d):
            if slot == time_slot:
                return tier == "surge"
    except Exception:
        pass
    return True  # unknown slot → default to surge (safe)


# ─── Shared HTTP client (persistent, connection-pooled) ─────────────────────

_http: httpx.AsyncClient | None = None
_base_url: str = ""


def _ensure_client() -> None:
    """Lazily initialize the shared httpx client using app settings."""
    global _http, _base_url
    if _http is not None:
        return
    s = get_settings()
    _base_url = f"{s.supabase_url.rstrip('/')}/rest/v1"
    _http = httpx.AsyncClient(
        timeout=10,
        headers={
            "apikey": s.supabase_service_key,
            "Authorization": f"Bearer {s.supabase_service_key}",
            "Content-Type": "application/json",
        },
    )
    logger.info("Supabase HTTP client initialized: %s", _base_url)


async def close_http_client() -> None:
    """Close the shared HTTP client (called on app shutdown)."""
    global _http
    if _http:
        await _http.aclose()
        _http = None


# ─── Supabase helpers ─────────────────────────────────────────────────────────

async def _sb_get(table: str, params) -> list:
    """params can be a dict or a list of (key, val) tuples (use tuples for duplicate keys)."""
    _ensure_client()
    pairs = list(params.items()) if isinstance(params, dict) else list(params)
    pairs.append(("select", "*"))
    r = await _http.get(f"{_base_url}/{table}", params=pairs)
    if r.status_code != 200:
        logger.warning("Supabase GET %s failed (%d): %s", table, r.status_code, r.text[:200])
    return r.json() if r.status_code == 200 else []


async def _sb_post(table: str, payload: dict) -> dict:
    _ensure_client()
    r = await _http.post(
        f"{_base_url}/{table}",
        json=payload,
        headers={"Prefer": "return=representation"},
    )
    if r.status_code not in (200, 201):
        logger.warning("Supabase POST %s failed (%d): %s", table, r.status_code, r.text[:200])
    return {"ok": r.status_code in (200, 201), "body": r.text}


async def _sb_patch(table: str, filter_param: str, filter_val: str, payload: dict) -> dict:
    _ensure_client()
    r = await _http.patch(
        f"{_base_url}/{table}",
        params={filter_param: f"eq.{filter_val}"},
        json=payload,
    )
    if r.status_code not in (200, 204):
        logger.warning("Supabase PATCH %s failed (%d): %s", table, r.status_code, r.text[:200])
    return {"ok": r.status_code in (200, 204)}


async def _upsert_customer(phone: str, name: str, city: str) -> None:
    """Insert or update a customer record keyed by phone number."""
    _ensure_client()
    try:
        r = await _http.post(
            f"{_base_url}/customers",
            json={
                "phone": phone,
                "name": name,
                "postal_code": city,     # live DB column (stores city name)
                "updated_at": datetime.utcnow().isoformat(),
            },
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )
        if r.status_code not in (200, 201, 204):
            logger.warning("Customer upsert failed (%d): %s", r.status_code, r.text[:200])
    except Exception as exc:
        logger.warning("Customer upsert failed for %s: %s", phone, exc)


# ─── Tools ────────────────────────────────────────────────────────────────────


async def get_availability(date_range: str, job_type: str, city: str = "") -> dict:
    """
    Return available booking slots for the given date range.

    Queries the bookings table for already-confirmed slots and returns
    only the open ones, annotated with their pricing tier (standard/surge).
    Date range format: "YYYY-MM-DD/YYYY-MM-DD"
    """
    today = datetime.now(TORONTO_TZ).date()
    max_date = today + timedelta(days=30)

    try:
        start_str, end_str = date_range.split("/")
        start = max(datetime.strptime(start_str, "%Y-%m-%d").date(), today + timedelta(days=1))
        end = min(datetime.strptime(end_str, "%Y-%m-%d").date(), max_date)
    except Exception:
        start = today + timedelta(days=1)
        end = today + timedelta(days=7)

    if start > end:
        return {"available_slots": [], "note": "No dates in range."}

    # Fetch already-booked (non-cancelled) slots in this window
    # Use list of tuples so both preferred_date filters are sent (dict would deduplicate)
    rows = await _sb_get("bookings", [
        ("preferred_date", f"gte.{start}"),
        ("preferred_date", f"lte.{end}"),
        ("status", "not.in.(cancelled)"),
    ])
    booked: set[tuple] = {(r["preferred_date"], r["preferred_time_slot"]) for r in rows}

    available = []
    d = start
    while d <= end:
        for time_slot, tier in _slots_for_date(d):
            if (str(d), time_slot) not in booked:
                available.append({
                    "date": str(d),
                    "time_slot": time_slot,
                    "pricing_tier": tier,
                    "rate_label": "Standard rate" if tier == "standard" else "Surge rate (after-hours surcharge applies)",
                })
        d += timedelta(days=1)

    return {"available_slots": available[:20], "job_type": job_type}


async def create_booking(
    customer_name: str,
    phone: str,
    city: str,
    issue_description: str,
    preferred_date: str,
    preferred_time: str,
    call_id: str = "",
    is_emergency: bool = False,
) -> dict:
    """
    Create a confirmed booking with full calendar validation:
      - Date must be in the future (Toronto time)
      - Date must be within 30 days
      - Slot must not already be taken
      - Pricing tier set based on slot time
      - Customer record upserted
    """
    today = datetime.now(TORONTO_TZ).date()
    max_date = today + timedelta(days=30)

    # 1. Validate date
    try:
        slot_date = datetime.strptime(preferred_date, "%Y-%m-%d").date()
    except ValueError:
        return {"success": False, "error": "invalid_date"}

    if slot_date <= today:
        return {"success": False, "error": "past_date",
                "message": "Cannot book appointments in the past."}
    if slot_date > max_date:
        return {"success": False, "error": "too_far",
                "message": f"Cannot book more than 30 days out (max: {max_date})."}

    # 2. Check slot conflict
    rows = await _sb_get("bookings", {
        "preferred_date": f"eq.{preferred_date}",
        "preferred_time_slot": f"eq.{preferred_time}",
        "status": "not.in.(cancelled)",
    })
    if rows:
        return {"success": False, "error": "slot_taken",
                "message": "That time slot is already booked."}

    # 3. Determine pricing tier
    pricing_tier = "surge" if _is_slot_surge(preferred_date, preferred_time) else "standard"

    # 4. Upsert customer
    await _upsert_customer(phone, customer_name, city)

    # 5. Verify call_id FK exists (if provided) — skip FK if call record is missing
    actual_call_id = None
    if call_id:
        existing_call = await _sb_get("calls", {"id": f"eq.{call_id}"})
        actual_call_id = call_id if existing_call else None
        if not existing_call:
            logger.warning("call_id %s not found in calls table — booking without FK", call_id)

    # 6. Create booking row
    booking_id = f"bk_{uuid.uuid4().hex[:12]}"
    payload = {
        "id": booking_id,
        "call_id": actual_call_id,
        "customer_name": customer_name,
        "phone": phone,
        "postal_code": city,     # live DB column (stores city name)
        "issue_description": issue_description,
        "preferred_date": preferred_date,
        "preferred_time_slot": preferred_time,
        "priority": "emergency" if is_emergency else "normal",
        "status": "confirmed",
        "pricing_tier": pricing_tier,
        "created_at": datetime.utcnow().isoformat(),
    }
    result = await _sb_post("bookings", payload)
    if result["ok"]:
        logger.info("Booking created: %s (%s, %s %s, tier=%s)",
                    booking_id, customer_name, preferred_date, preferred_time, pricing_tier)
        return {"success": True, "booking_id": booking_id, "pricing_tier": pricing_tier}
    logger.error("Booking DB insert failed: %s", result["body"][:300])
    return {"success": False, "error": "db_error", "detail": result["body"]}


async def reschedule_booking(booking_id: str, new_date: str, new_time_slot: str) -> dict:
    """
    Reschedule a booking to a new slot.
    Validates: new date is future + within 30 days, slot is free, original booking exists.
    If the original booking is in the past, returns needs_human=True.
    """
    today = datetime.now(TORONTO_TZ).date()
    max_date = today + timedelta(days=30)

    # Validate new date
    try:
        slot_date = datetime.strptime(new_date, "%Y-%m-%d").date()
    except ValueError:
        return {"success": False, "error": "invalid_date"}

    if slot_date <= today:
        return {"success": False, "error": "past_date"}
    if slot_date > max_date:
        return {"success": False, "error": "too_far"}

    # Verify existing booking exists
    existing = await _sb_get("bookings", {"id": f"eq.{booking_id}"})
    if not existing:
        return {"success": False, "error": "not_found",
                "message": "Booking not found. Please check the reference number."}

    # Check if original booking is in the past (needs human)
    try:
        orig_date = datetime.strptime(existing[0]["preferred_date"], "%Y-%m-%d").date()
        if orig_date < today:
            return {"success": False, "error": "past_booking", "needs_human": True,
                    "message": "This appointment is in the past and requires a team member."}
    except Exception:
        pass

    # Check conflict on new slot (excluding the booking being rescheduled)
    rows = await _sb_get("bookings", {
        "preferred_date": f"eq.{new_date}",
        "preferred_time_slot": f"eq.{new_time_slot}",
        "status": "not.in.(cancelled)",
        "id": f"neq.{booking_id}",
    })
    if rows:
        return {"success": False, "error": "slot_taken",
                "message": "That slot is already taken."}

    pricing_tier = "surge" if _is_slot_surge(new_date, new_time_slot) else "standard"
    result = await _sb_patch("bookings", "id", booking_id, {
        "preferred_date": new_date,
        "preferred_time_slot": new_time_slot,
        "pricing_tier": pricing_tier,
        "status": "rescheduled",
        "updated_at": datetime.utcnow().isoformat(),
    })
    if result["ok"]:
        return {"success": True, "booking_id": booking_id, "pricing_tier": pricing_tier}
    return {"success": False, "error": "db_error"}


async def cancel_booking(booking_id: str, reason: str) -> dict:
    """
    Cancel a booking. Returns needs_human=True if the appointment is in the past.
    """
    today = datetime.now(TORONTO_TZ).date()

    existing = await _sb_get("bookings", {"id": f"eq.{booking_id}"})
    if not existing:
        return {"success": False, "error": "not_found",
                "message": "Booking not found. Please check the reference number."}

    try:
        orig_date = datetime.strptime(existing[0]["preferred_date"], "%Y-%m-%d").date()
        if orig_date < today:
            return {"success": False, "error": "past_booking", "needs_human": True,
                    "message": "This appointment is in the past and requires a team member."}
    except Exception:
        pass

    result = await _sb_patch("bookings", "id", booking_id, {
        "status": "cancelled",
        "cancellation_reason": reason,
        "updated_at": datetime.utcnow().isoformat(),
    })
    if result["ok"]:
        return {"success": True, "booking_id": booking_id}
    return {"success": False, "error": "db_error"}


async def send_sms(phone: str, message: str) -> dict:
    """Send an SMS via Twilio (runs in a thread — Twilio client is synchronous)."""
    from twilio.rest import Client

    def _send() -> dict:
        try:
            s = get_settings()
            client = Client(s.twilio_account_sid, s.twilio_auth_token)
            msg = client.messages.create(
                body=message,
                from_=s.twilio_phone_number,
                to=phone,
            )
            return {"success": True, "sid": msg.sid}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    return await asyncio.to_thread(_send)


async def escalate_call(
    call_id: str,
    reason: str,
    transcript_summary: str,
    is_emergency: bool = False,
) -> dict:
    """Log a call escalation for human follow-up."""
    payload = {
        "call_id": call_id,
        "reason": reason,
        "transcript_summary": transcript_summary,
        "is_emergency": is_emergency,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
    }
    return await _sb_post("escalations", payload)


# ─── Dispatcher ───────────────────────────────────────────────────────────────

_TOOL_MAP: dict[str, Any] = {
    "get_availability": get_availability,
    "create_booking": create_booking,
    "reschedule_booking": reschedule_booking,
    "cancel_booking": cancel_booking,
    "send_sms": send_sms,
    "escalate_call": escalate_call,
}


async def call_tool(name: str, args: dict) -> Any:
    """Dispatch a tool call by name."""
    fn = _TOOL_MAP.get(name)
    if fn is None:
        logger.warning("Unknown tool: %s", name)
        return {"error": f"Unknown tool: {name}"}
    logger.info("Tool: %s %s", name, args)
    return await fn(**args)
