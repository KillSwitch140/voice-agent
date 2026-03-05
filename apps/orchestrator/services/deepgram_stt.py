"""
Deepgram live transcription handler.

Wraps the Deepgram Python SDK v3 async live client.
Receives raw mulaw 8kHz audio from Twilio and emits complete utterances.

Uses interim_results=True + utterance_end_ms so that short mid-sentence
pauses don't prematurely fire the pipeline.  Final transcript segments are
accumulated until Deepgram fires UtteranceEnd (1.5 s of silence), then the
full accumulated text is emitted as a single callback.
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
        self._segments: list[str] = []   # accumulated final segments for current utterance

    async def start(self, on_transcript: TranscriptCallback) -> None:
        self._on_transcript = on_transcript
        self._connection = self._client.listen.asynclive.v("1")

        async def _on_message(_, result, **__):
            transcript = result.channel.alternatives[0].transcript
            if result.is_final and transcript.strip():
                self._segments.append(transcript.strip())
                logger.debug("Deepgram segment: %s", transcript)

        async def _on_utterance_end(_, utterance_end, **__):
            """Fire the pipeline once Deepgram detects 1.5 s of silence."""
            if self._segments:
                full_text = " ".join(self._segments)
                self._segments.clear()
                logger.debug("Deepgram utterance complete: %s", full_text)
                if self._on_transcript:
                    await self._on_transcript(full_text)

        async def _on_error(_, error, **__):
            logger.error("Deepgram error: %s", error)

        self._connection.on(LiveTranscriptionEvents.Transcript, _on_message)
        self._connection.on(LiveTranscriptionEvents.UtteranceEnd, _on_utterance_end)
        self._connection.on(LiveTranscriptionEvents.Error, _on_error)

        options = LiveOptions(
            model="nova-2",
            language="en",
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            punctuate=True,
            interim_results=True,       # required for utterance_end_ms
            utterance_end_ms="1500",    # fire UtteranceEnd after 1.5 s silence
            endpointing=300,            # still break long speech into segments
        )

        started = await self._connection.start(options)
        if not started:
            raise RuntimeError("Failed to start Deepgram live transcription")

        logger.info("Deepgram STT started (utterance_end_ms=1500)")

    async def send(self, audio_bytes: bytes) -> None:
        if self._connection:
            await self._connection.send(audio_bytes)

    async def close(self) -> None:
        if self._connection:
            self._segments.clear()
            await self._connection.finish()
            self._connection = None
            logger.info("Deepgram STT closed")
