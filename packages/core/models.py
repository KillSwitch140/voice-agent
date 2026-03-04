from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CallState(str, Enum):
    GREETING = "greeting"
    INTENT_DETECTION = "intent_detection"
    COLLECTING_POSTAL = "collecting_postal"
    COLLECTING_BOOKING_REF = "collecting_booking_ref"   # reschedule / cancel
    OUT_OF_AREA = "out_of_area"
    COLLECTING_CUSTOMER_INFO = "collecting_customer_info"
    COLLECTING_BOOKING_DETAILS = "collecting_booking_details"
    AFTER_HOURS_DISCLOSURE = "after_hours_disclosure"   # after-hours surcharge warning
    CONFIRMING_BOOKING = "confirming_booking"
    EMERGENCY_TRIAGE = "emergency_triage"
    PRICING = "pricing"
    PRICING_FOLLOWUP = "pricing_followup"    # offer booking after answering price question
    ESCALATING = "escalating"
    CLOSING = "closing"
    ENDED = "ended"


class Intent(str, Enum):
    NEW_BOOKING = "new_booking"
    RESCHEDULE = "reschedule"
    CANCELLATION = "cancellation"
    PRICING = "pricing"
    EMERGENCY = "emergency"
    ESCALATE = "escalate"           # caller requests a human agent
    GENERAL = "general"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    NORMAL = "normal"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class CallSlots(BaseModel):
    """Slot-filling container for a single call."""

    postal_code: Optional[str] = None
    customer_name: Optional[str] = None
    issue_description: Optional[str] = None
    preferred_date: Optional[str] = None        # YYYY-MM-DD
    preferred_time: Optional[str] = None        # e.g. "10:00-12:00"
    booking_id: Optional[str] = None            # for reschedule / cancel
    confirmed: bool = False
    after_hours_accepted: Optional[bool] = None  # gate for AFTER_HOURS_DISCLOSURE


class CallSession(BaseModel):
    call_sid: str
    call_id: Optional[str] = None          # Supabase row id
    stream_sid: Optional[str] = None       # Twilio Media Stream SID
    state: CallState = CallState.GREETING
    intent: Optional[Intent] = None
    priority: Priority = Priority.NORMAL
    slots: CallSlots = Field(default_factory=CallSlots)
    # Compact rolling context sent to the LLM (last N turns only)
    turns: List[Dict[str, str]] = Field(default_factory=list)
    from_number: str = ""
    started_at: datetime = Field(default_factory=datetime.utcnow)
    is_emergency: bool = False
    available_slots: Optional[List[Dict]] = None  # fetched on entering COLLECTING_BOOKING_DETAILS


class LLMTurnResult(BaseModel):
    """Structured output expected from GPT-4o-mini each turn."""

    intent: Optional[str] = None
    is_emergency: bool = False
    extracted_slots: Dict[str, Any] = Field(default_factory=dict)
    response_text: str
