"""
TTS service using Deepgram Aura.

Returns mulaw 8kHz audio directly — Twilio's native format, no conversion needed.
Typical latency: ~300ms for a short sentence.

Voice options: aura-asteria-en, aura-luna-en, aura-stella-en, aura-athena-en
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_AURA_URL = "https://api.deepgram.com/v1/speak"
_VOICE_MODEL = "aura-asteria-en"


class TTSService:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        logger.debug("TTS synthesizing: %s", text[:80])

        async with httpx.AsyncClient() as client:
            response = await client.post(
                _AURA_URL,
                params={
                    "model": _VOICE_MODEL,
                    "encoding": "mulaw",
                    "sample_rate": "8000",
                    "container": "none",
                },
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": text},
                timeout=10.0,
            )
            response.raise_for_status()

        logger.debug("TTS done: %d bytes", len(response.content))
        return response.content
