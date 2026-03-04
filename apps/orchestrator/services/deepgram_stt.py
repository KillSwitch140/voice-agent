"""
Deepgram live transcription handler.

Wraps the Deepgram Python SDK v3 async live client.
Receives raw mulaw 8kHz audio from Twilio and emits final transcripts.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from deepgram import (
    DeepgramClient,
    LiveOptions,
    LiveTranscriptionEvents,
)

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str], Awaitable[None]]


class DeepgramSTT:
    """
    Thin wrapper around Deepgram's async live transcription.

    Usage:
        stt = DeepgramSTT(api_key)
        await stt.start(on_final_transcript)
        await stt.send(mulaw_audio_bytes)
        ...
        await stt.close()
    """

    def __init__(self, api_key: str) -> None:
        self._client = DeepgramClient(api_key)
        self._connection = None
        self._on_transcript: TranscriptCallback | None = None

    async def start(self, on_transcript: TranscriptCallback) -> None:
        self._on_transcript = on_transcript
        self._connection = self._client.listen.asynclive.v("1")

        async def _on_message(_, result, **__):
            transcript = result.channel.alternatives[0].transcript
            if result.is_final and transcript.strip():
                logger.debug("Deepgram final: %s", transcript)
                if self._on_transcript:
                    await self._on_transcript(transcript)

        async def _on_error(_, error, **__):
            logger.error("Deepgram error: %s", error)

        self._connection.on(LiveTranscriptionEvents.Transcript, _on_message)
        self._connection.on(LiveTranscriptionEvents.Error, _on_error)

        options = LiveOptions(
            model="nova-2",
            language="en",
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            punctuate=True,
            interim_results=False,
            endpointing=400,   # declare end-of-utterance after 400ms silence
            utterance_end_ms="1000",
        )

        started = await self._connection.start(options)
        if not started:
            raise RuntimeError("Failed to start Deepgram live transcription")

        logger.info("Deepgram STT started")

    async def send(self, audio_bytes: bytes) -> None:
        if self._connection:
            await self._connection.send(audio_bytes)

    async def close(self) -> None:
        if self._connection:
            await self._connection.finish()
            self._connection = None
            logger.info("Deepgram STT closed")
