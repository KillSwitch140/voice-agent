"""
Async Supabase logging helpers.

All writes are best-effort — a logging failure must never crash a live call.
Uses a persistent httpx.AsyncClient for connection reuse.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class SupabaseLogger:
    def __init__(self, supabase_url: str, service_key: str) -> None:
        self._url = supabase_url.rstrip("/")
        self._headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        self._client = httpx.AsyncClient(timeout=5)

    async def _post(self, table: str, payload: dict) -> bool:
        try:
            r = await self._client.post(
                f"{self._url}/rest/v1/{table}",
                json=payload,
                headers=self._headers,
            )
            if r.status_code not in (200, 201, 204):
                logger.warning("Supabase %s insert failed (%d): %s", table, r.status_code, r.text[:200])
                return False
            return True
        except Exception as exc:
            logger.warning("Supabase logger error (%s): %s", table, exc)
            return False

    async def log_call(
        self,
        call_sid: str,
        from_number: str,
        call_id: Optional[str] = None,
    ) -> Optional[str]:
        """Log a new call. Returns the call row ID, or None if the insert failed."""
        row_id = call_id or f"call_{call_sid[:16]}"
        ok = await self._post("calls", {
            "id": row_id,
            "twilio_call_sid": call_sid,
            "from_number": from_number,
            "status": "in_progress",
            "started_at": datetime.utcnow().isoformat(),
        })
        return row_id if ok else None

    async def log_turn(
        self,
        call_id: str,
        role: str,
        content: str,
        state: str,
    ) -> None:
        await self._post("call_turns", {
            "call_id": call_id,
            "role": role,
            "content": content,
            "state": state,
            "created_at": datetime.utcnow().isoformat(),
        })

    async def close_call(self, call_id: str, outcome: str) -> None:
        try:
            await self._client.patch(
                f"{self._url}/rest/v1/calls",
                params={"id": f"eq.{call_id}"},
                json={
                    "status": outcome,
                    "ended_at": datetime.utcnow().isoformat(),
                },
                headers=self._headers,
            )
        except Exception as exc:
            logger.warning("Supabase close_call error: %s", exc)

    async def aclose(self) -> None:
        await self._client.aclose()
