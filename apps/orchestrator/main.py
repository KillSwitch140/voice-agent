"""
HVAC Voice Agent — FastAPI Orchestrator

Run with:
    python -m uvicorn apps.orchestrator.main:app --reload --port 8080
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps.orchestrator.config import get_settings
from apps.orchestrator.routers import health, voice

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup persistent HTTP clients on shutdown
    from apps.orchestrator.routers.voice import cleanup as voice_cleanup
    from apps.orchestrator.services.tools import close_http_client
    from apps.orchestrator.services.llm import close_llm_client
    await voice_cleanup()
    await close_http_client()
    await close_llm_client()


app = FastAPI(title="HVAC Voice Agent Orchestrator", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(voice.router, prefix="/voice")
