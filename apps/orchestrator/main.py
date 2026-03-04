"""
HVAC Voice Agent — FastAPI Orchestrator

Run with:
    python -m uvicorn apps.orchestrator.main:app --reload --port 8080
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from apps.orchestrator.config import get_settings
from apps.orchestrator.routers import health, voice

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())

app = FastAPI(title="HVAC Voice Agent Orchestrator", version="0.1.0")
app.include_router(health.router)
app.include_router(voice.router, prefix="/voice")
