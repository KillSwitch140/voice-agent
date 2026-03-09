"""
TTS service using Deepgram Aura.

Returns mulaw 8kHz audio directly — Twilio's native format, no conversion needed.
Uses a persistent httpx.AsyncClient for connection reuse (avoids TCP+TLS handshake per request).
Caches synthesized audio by text key — static/scripted responses are synthesized once.
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
        self._client = httpx.AsyncClient(timeout=10.0)
        self._cache: dict[str, bytes] = {}

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        cached = self._cache.get(text)
        if cached is not None:
            return cached

        logger.debug("TTS synthesizing: %s", text[:80])

        response = await self._client.post(
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
        )
        response.raise_for_status()

        audio = response.content
        self._cache[text] = audio
        logger.debug("TTS done: %d bytes (cached)", len(audio))
        return audio

    async def aclose(self) -> None:
        await self._client.aclose()
