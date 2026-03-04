"""
In-memory session store keyed by Twilio call_sid.
For production, replace with Redis or a persistent store.
"""
from __future__ import annotations

from typing import Dict, Optional

from packages.core.models import CallSession

_sessions: Dict[str, CallSession] = {}


def get_session(call_sid: str) -> Optional[CallSession]:
    return _sessions.get(call_sid)


def create_session(call_sid: str, from_number: str) -> CallSession:
    session = CallSession(call_sid=call_sid, from_number=from_number)
    _sessions[call_sid] = session
    return session


def save_session(session: CallSession) -> None:
    _sessions[session.call_sid] = session


def delete_session(call_sid: str) -> None:
    _sessions.pop(call_sid, None)


def all_sessions() -> Dict[str, CallSession]:
    return dict(_sessions)
