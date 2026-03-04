# CLAUDE.md

## Project: HVAC Vertical Voice Agent (Toronto, Canada)

This document defines how Claude should behave when assisting with this codebase.

The goal of this project is to build a production-style MVP of an AI-powered HVAC voice receptionist using:

- Twilio (telephony + SMS)
- Deepgram (speech-to-text + text-to-speech via Aura)
- GPT-4o-mini (LLM reasoning)
- FastAPI (backend orchestration)
- Supabase (Postgres)
- Render (deployment)

This is NOT a research project.
This is a revenue-focused automation system.

---

# 1. System Purpose

The system replaces or augments an HVAC receptionist in Toronto.

It must:

- Answer inbound calls
- Identify intent
- Book appointments
- Reschedule bookings
- Cancel bookings
- Answer pricing questions
- Triage emergencies
- Escalate when required
- Send SMS confirmations

Primary KPI:
- % of calls fully handled
- Booking conversion rate
- Emergency prioritization accuracy
- Escalation rate

---

# 2. Architectural Principles

Claude must follow these constraints:

1. Deterministic > autonomous
2. Safety overrides always win
3. Business rules are enforced in backend code, not only prompts
4. Keep latency low (voice system)
5. Keep responses concise (voice UX)
6. Avoid overengineering
7. Prefer simple explicit state machines over complex agent loops

Do NOT introduce unnecessary frameworks or abstraction layers.

---

# 3. High-Level Architecture

Call Flow:

1. Twilio receives inbound call
2. Twilio webhook → FastAPI Orchestrator (`POST /voice/inbound`)
3. TwiML opens bidirectional Media Stream WebSocket
4. Audio streamed to Deepgram STT (mulaw 8kHz)
5. Final transcript → GPT-4o-mini (intent + slot extraction)
6. State machine advances deterministically
7. Orchestrator calls booking tools directly (not via LLM)
8. Response text → Deepgram Aura TTS → mulaw audio → Twilio
9. All events logged to Supabase

Key design decision: **the LLM only classifies intent and extracts slots. All booking tool calls (create/reschedule/cancel) are made deterministically by the orchestrator when the state machine reaches CLOSING.**

---

# 4. Directory Structure

```
apps/
  orchestrator/        — FastAPI app, state machine, LLM, tools
    routers/
      voice.py         — Twilio webhooks + WebSocket + /simulate
    services/
      llm.py           — GPT-4o-mini: intent + slot extraction
      state_machine.py — Deterministic call flow state machine
      tools.py         — Booking tools (Supabase direct calls)
      tts.py           — Deepgram Aura TTS
      deepgram_stt.py  — Deepgram live STT
      session_store.py — In-memory session store (keyed by call_sid)
      supabase_logger.py — Async call/turn logging
      config.py        — Pydantic settings from .env
  resources/           — Static config files read at runtime
    pricing/           — after_hours_fees.json
    policies/          — emergency_triage.md
    scripts/           — call_opening.txt, safety_gas_smell.txt
    service_area/      — toronto_postal_prefixes.json

packages/
  core/
    models.py          — CallSession, CallState, Intent, LLMTurnResult
    utils.py           — detect_emergency(), is_toronto_service_area()

sql/
  schema.sql           — Full Supabase schema + seed data

infra/
  render.yaml          — Render deployment config
```

---

# 5. Tools (Direct Function Calls)

All tools are async functions in `apps/orchestrator/services/tools.py` dispatched via `call_tool(name, args)`.

Active tools:
- `get_availability` — fetch open slots from Supabase, annotated with pricing tier
- `create_booking` — validate + insert booking, upsert customer record
- `reschedule_booking` — validate + update booking date/time
- `cancel_booking` — cancel booking with past-date guard
- `send_sms` — Twilio SMS via asyncio.to_thread
- `escalate_call` — log escalation record to Supabase

Business rules enforced in tool code (not prompts):
- Future-only dates (Toronto time)
- Max 30 days in advance
- Slot conflict check against existing confirmed bookings
- Past booking modifications require human escalation
- Pricing tier (standard/surge) determined by slot time

---

# 6. Business Logic Rules (Non-Negotiable)

These rules must be implemented in backend code, not only prompts.

## Emergency Overrides

If caller mentions: gas smell, gas leak, carbon monoxide, CO detector, smoke from furnace, fire, explosion, cannot breathe, pipes frozen

Then:
- Hard-coded detection in `detect_emergency()` — bypasses LLM
- Play safety script immediately
- Log escalation record
- Hang up proactively (caller must leave building)

## After-Hours Slot Pricing

Pricing is based on the **booked appointment slot time**, not the time of the call.

- **Standard rate**: Mon–Fri slots starting before 16:00, Sat slots starting before 13:00
- **Surge rate** (after-hours surcharge +$120–$180): Mon–Fri 16:00+, Sat 13:00+, all Sunday slots
- Caller is informed via `AFTER_HOURS_DISCLOSURE` state before confirming

## Service Area

If postal code prefix not in M1–M9 (Toronto FSA):
- Politely decline, send referral SMS, hang up

---

# 7. Voice UX Rules

- Keep every response to 1–2 short sentences
- Ask exactly one question at a time
- Confirm critical info (date, time, booking ref)
- Never provide exact repair costs — ranges only

Tone: professional, polite, efficient.

Never:
- Give gas leak troubleshooting or DIY repair guidance
- Overpromise on pricing
- Let the LLM directly trigger booking operations

---

# 8. Database (Supabase)

Tables: `calls`, `call_turns`, `customers`, `bookings`, `escalations`

`bookings` key columns:
- `preferred_date` (date), `preferred_time_slot` (text)
- `pricing_tier` (text): `standard` | `surge`
- `status`: `confirmed` | `rescheduled` | `cancelled` | `completed`
- Unique partial index on `(preferred_date, preferred_time_slot) WHERE status NOT IN ('cancelled')`

---

# 9. LLM Usage Rules

GPT-4o-mini is used **only** for:
- Intent classification
- Slot/entity extraction
- Generating voice response text (1–2 sentences)

The LLM does **NOT**:
- Make tool calls or trigger bookings directly
- See the full transcript (rolling last-5 turns only)
- Generate pricing numbers (reads from `after_hours_fees.json`)

State context and available slots are injected as system messages each turn.

---

# 10. Safety and Liability

Never provide:
- Gas leak troubleshooting instructions
- Unsafe electrical guidance
- DIY repair instructions

Always redirect to licensed technician or emergency services.

---

# 11. What This Project Is NOT

- Not a generic chatbot
- Not a multi-agent experiment
- Not a LangChain or MCP playground

It is a revenue-focused, operational voice automation system.

---

# 12. Claude Development Behavior

When assisting with code:

- Prefer explicit FastAPI endpoints
- Prefer simple state machine logic over agent loops
- Avoid unnecessary abstractions
- Use Pydantic for all schemas
- Optimize for clarity and latency over cleverness
- Always confirm before making functional changes

When unsure: ask for clarification instead of inventing architecture.

---

# 13. MVP Scope (Implemented)

- New booking (full calendar: conflict check, future-only, 30-day limit)
- Reschedule (validates new slot, excludes own slot from conflict check)
- Cancellation
- Emergency triage (hard-coded bypass)
- Pricing explanation (ranges from JSON file)
- SMS confirmation (booking / reschedule / cancellation / out-of-area)
- Escalation logging + outbound callback via Twilio
- After-hours surcharge disclosure
- Out-of-area detection + referral SMS

---

# 14. Security

- All secrets stored in `.env` (never committed)
- `.env` is listed in `.gitignore`
- Supabase service key used server-side only
- No secrets in source code or logs

---

End of CLAUDE.md
