# HVAC Voice Agent

An AI-powered HVAC receptionist for the Greater Toronto Area. Handles inbound calls end-to-end — booking appointments, rescheduling, cancellations, pricing questions, and emergency triage — without human involvement.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Telephony | Twilio (inbound calls + SMS) |
| Speech-to-Text | Deepgram Live Streaming (mulaw 8kHz) |
| Text-to-Speech | Deepgram Aura |
| LLM | OpenAI GPT-4o-mini (intent + slot extraction only) |
| Backend | FastAPI + Python |
| Database | Supabase (PostgreSQL) |
| Hosting | Render |

---

## Getting Started

### 1 — Install dependencies

```bash
pip install -r apps/orchestrator/requirements.txt
```

### 2 — Configure environment

Copy `.env.example` to `.env` and fill in all values:

| Variable | Where to get it |
|---|---|
| `TWILIO_ACCOUNT_SID` | Twilio Console → Account Info |
| `TWILIO_AUTH_TOKEN` | Twilio Console → Account Info |
| `TWILIO_PHONE_NUMBER` | Twilio Console → Phone Numbers (E.164 format) |
| `DEEPGRAM_API_KEY` | console.deepgram.com → API Keys |
| `OPENAI_API_KEY` | platform.openai.com → API Keys |
| `SUPABASE_URL` | Supabase project → Settings → API → Project URL |
| `SUPABASE_SERVICE_KEY` | Supabase project → Settings → API → service_role key |
| `ORCHESTRATOR_BASE_URL` | Your ngrok or Render public URL |

### 3 — Set up the database

Run `sql/schema.sql` in the **Supabase SQL Editor** once. This creates all tables, indexes, and seeds 5 test bookings for conflict detection testing.

Verify tables exist: `calls`, `call_turns`, `customers`, `bookings`, `escalations`

### 4 — Start the server

```bash
PYTHONPATH=$(pwd) uvicorn apps.orchestrator.main:app --reload --port 8080
```

```bash
curl http://localhost:8080/health
# → {"status": "ok"}
```

---

## Architecture

```
Inbound call
    │
    ▼
Twilio ──POST /voice/inbound──► FastAPI Orchestrator (:8080)
                                      │  TwiML: <Connect><Stream>
                                      │
                                WebSocket /voice/stream/{call_sid}
                                      │
                   ┌──────────────────┴──────────────────┐
                   │                                     │
            Deepgram STT                        Deepgram Aura TTS
           (mulaw 8kHz in)                     (mulaw 8kHz out)
                   │                                     ▲
             Final transcript                      response_text
                   │                                     │
                   └─────► GPT-4o-mini ─────► State Machine
                           (intent +           (deterministic)
                           slot extract)              │
                                              Booking Tools
                                              (Supabase direct)
```

**Key design decision:** the LLM only classifies intent and extracts slots. All booking operations are executed deterministically by the state machine when it reaches `CLOSING`. The LLM never triggers tool calls directly.

---

## Call State Machine

```
GREETING
  └─► INTENT_DETECTION
        ├─► COLLECTING_POSTAL ─► OUT_OF_AREA ─► ENDED
        │     └─► COLLECTING_CUSTOMER_INFO
        │           └─► COLLECTING_BOOKING_DETAILS
        │                 └─► AFTER_HOURS_DISCLOSURE (surge slot only)
        │                       └─► CONFIRMING_BOOKING ─► CLOSING ─► ENDED
        ├─► COLLECTING_BOOKING_REF (reschedule / cancel)
        │     ├─► COLLECTING_BOOKING_DETAILS ─► (see above)
        │     └─► CONFIRMING_BOOKING ─► CLOSING ─► ENDED (cancel)
        ├─► PRICING ─► PRICING_FOLLOWUP ─► (book or ENDED)
        └─► EMERGENCY_TRIAGE ─► ESCALATING ─► ENDED

Any state ─► EMERGENCY_TRIAGE   (detect_emergency() hard bypass)
Any state ─► ESCALATING          (intent=escalate or booking error)
```

---

## Directory Structure

```
apps/
  orchestrator/
    routers/
      voice.py         — Twilio webhooks, WebSocket handler, /simulate endpoint
    services/
      llm.py           — GPT-4o-mini intent + slot extraction, regex fast path
      state_machine.py — Deterministic call flow
      tools.py         — Booking tools (Supabase direct calls)
      tts.py           — Deepgram Aura TTS
      deepgram_stt.py  — Deepgram live STT
      session_store.py — In-memory session store (keyed by call_sid)
      supabase_logger.py — Async call/turn logging
      config.py        — Pydantic settings from .env

  resources/
    pricing/           — after_hours_fees.json
    policies/          — emergency_triage.md
    scripts/           — call_opening.txt, safety_gas_smell.txt
    service_area/      — gta_cities.json
    test_client.html   — Browser WebRTC test UI (served at /voice/test-client)

packages/
  core/
    models.py          — CallSession, CallState, Intent, LLMTurnResult
    utils.py           — detect_emergency(), is_gta_city()

sql/
  schema.sql           — Full schema + seed data

infra/
  render.yaml          — Render deployment config
```

---

## The LLM's Role Is Intentionally Narrow

The LLM does one thing: take a caller's raw utterance and return structured JSON — which intent was expressed and what slot values were mentioned. It does not generate the voice responses the caller hears (those are scripted Python strings), it does not decide what to do next (the state machine does), and it does not call any functions (the orchestrator does, only when the state machine reaches `CLOSING`).

This constraint was deliberate. In a voice system operating on a live phone call, the cost of an LLM making a wrong decision is high — a misfired booking, an incorrectly cancelled appointment, or a missed emergency. Keeping the LLM as a dumb parser means all of that logic sits in auditable Python code that behaves identically every time.

The LLM also only sees the last 5 turns of the conversation, not the full history. This keeps token usage low and forces the prompt to stay focused on what the caller just said.

---

## Features

### Appointment Booking

The full booking flow collects: caller name, city, issue description, preferred date, and preferred time. Each maps to a required slot that must be filled before the state machine advances — the system will keep asking until it has everything it needs.

**Calendar validation is enforced in backend code, not prompts:**

- Dates must be in the future (evaluated against Toronto local time — `America/Toronto` — not UTC)
- Bookings are capped at 30 days out
- Every slot is checked for conflicts against existing confirmed bookings before a new one is created
- If a requested slot is already taken, the system backs up and offers the next available option

Available time slots are fixed 2-hour blocks:

| Day | Slots |
|---|---|
| Mon–Fri | 08:00–10:00, 10:00–12:00, 12:00–14:00, 14:00–16:00, 16:00–18:00, 18:00–20:00 |
| Saturday | 09:00–11:00, 11:00–13:00, 13:00–15:00, 15:00–17:00 |
| Sunday | 10:00–12:00, 12:00–14:00, 14:00–16:00, 16:00–18:00 |

Slots are offered one at a time. The system does not dump a full list of options at the caller — it presents the first available slot and waits for acceptance or an alternative request.

### Slot Navigation

Callers rarely say "I want the 10:00 to 12:00 slot on March 15th." They say things like "do you have anything on the 15th?" or "something in the morning."

The system handles this with two auxiliary slots — `requested_date` and `requested_time_of_day` — separate from `preferred_date` and `preferred_time`. When a caller asks about a date without committing to a specific time, the orchestrator filters the available slot list to that day only. When they express a time-of-day preference (morning / afternoon / evening), the list filters further.

`preferred_date` and `preferred_time` are only set when the caller explicitly accepts a specific slot. This prevents the system from prematurely locking in a date the caller was just asking about.

### Reschedule and Cancellation

Both flows start by collecting the caller's booking reference number (`bk_` followed by alphanumeric characters).

**For reschedule:** collects a new date and time, runs the same future-date and 30-day validation as a new booking, then checks the new slot for conflicts — explicitly excluding the caller's own existing slot from that check (otherwise rescheduling to the same time window would always fail). The record is updated and a confirmation SMS is sent.

**For cancellation:** confirms intent, marks the booking cancelled with the caller's stated reason, and sends a confirmation SMS.

**Past booking guard:** if the original appointment date has already passed, the system does not attempt to modify it. It escalates to a human — past-date modifications require a team member to handle manually. This is enforced in tool code, not prompts.

### Emergency Triage

Emergency detection runs **before the LLM** on every transcript, as a hard-coded keyword check. There is no configuration option to disable it.

Trigger phrases include: gas smell, gas leak, carbon monoxide, CO detector, smoke from furnace, fire, explosion, cannot breathe, pipes frozen.

When triggered:
1. A pre-scripted safety response plays immediately (leave the building, call 911)
2. An escalation record is written to Supabase
3. The call is terminated proactively — the agent does not wait for the caller to respond

This runs before the LLM rather than being handled as an LLM intent because a keyword match is deterministic and instantaneous. An LLM call adds 300–800ms of round-trip time and introduces the theoretical risk of misclassification on the one scenario where a wrong answer is unacceptable.

> "Broken furnace", "no heat", and "heating stopped working" are **not** treated as emergencies — they are standard service calls. Only explicit life-safety language triggers the bypass.

### After-Hours Pricing Disclosure

Pricing is determined by **when the appointment is scheduled** (the slot time), not when the caller phones in. A caller ringing at 9pm asking for a next-morning slot gets standard pricing. A caller ringing at 9am asking for an evening slot gets the surge rate.

- **Standard rate**: Mon–Fri slots starting before 16:00, Sat slots starting before 13:00
- **Surge rate** (after-hours surcharge +$120–$180): Mon–Fri 16:00+, Sat 13:00+, all Sunday slots

When a caller picks a surge-tier slot, the state machine routes through `AFTER_HOURS_DISCLOSURE` before reaching `CONFIRMING_BOOKING`. The agent explicitly discloses the surcharge and asks the caller to confirm. If they decline, the session returns to `WRAP_UP` so they can choose a different slot or end the call — the system does not silently continue to confirmation.

No caller can be confirmed into a surcharge slot without having explicitly heard and accepted the pricing disclosure first.

### Out-of-Area Detection

The service area covers the GTA: City of Toronto, Peel Region (Mississauga, Brampton, Caledon), York Region, Durham Region, and Halton Region.

Detection is city-name based. The caller's spoken city is extracted by the LLM and checked against `gta_cities.json`. If not in the list, the call routes to `OUT_OF_AREA` — a polite decline plays, a referral SMS is sent, and the call ends.

City-name matching was chosen over postal codes because callers are far more likely to say "I'm in Barrie" than to spell out a postal code. Postal codes also require geocoding lookups; city names are a simple set membership check.

### SMS Confirmations

Outbound SMS is sent via Twilio for every significant outcome:

- **New booking**: date, time, booking reference number
- **Reschedule**: updated date, time, same booking reference
- **Cancellation**: confirmation the booking was cancelled
- **Out-of-area**: referral message pointing to alternative services

SMS runs in a background thread (`asyncio.to_thread`) because the Twilio Python client is synchronous. This keeps it off the async event loop so it cannot block the voice response pipeline.

### Regex Fast Path

For states where the only meaningful response is yes or no — `CONFIRMING_BOOKING`, `AFTER_HOURS_DISCLOSURE`, `WRAP_UP`, `PRICING_FOLLOWUP` — the LLM is skipped entirely. A phrase-match check runs first. If the utterance matches a known yes or no pattern, the slot is set directly and the state machine advances without a GPT-4o-mini round trip.

This covers the majority of caller utterances in the back half of a call. Phrases like "yes", "yeah", "sounds good", "that works", "no thanks", "that's all", "I'm done" all resolve instantly. The LLM fallback only kicks in if the utterance is ambiguous.

### Multi-Intent Calls

After any completed task, the state machine enters `WRAP_UP` and asks the caller if there's anything else. If yes, the session loops back to `INTENT_DETECTION` with a clean slate — booking-specific slots (date, time, issue, booking reference) are cleared, but the caller's name and city are retained so they don't have to repeat themselves.

A single call can handle: ask about pricing → book an appointment → reschedule a different booking. Each task runs through the full state machine independently.

### Intent Lock

Once an intent is detected (`new_booking`, `reschedule`, `cancellation`), it is locked for the duration of that task. A mid-booking utterance like "actually, never mind" is treated as a correction, not a cancellation intent, and the state machine does not restart.

The intent can only be re-detected in three states: `INTENT_DETECTION` (initial detection), `PRICING_FOLLOWUP` (caller may want to book after asking about pricing), and `WRAP_UP` (starting a new task). This prevents a single misclassified "unknown" from permanently locking the caller out of completing their request.

### Escalation

The agent escalates to a human in three scenarios:

1. **Caller explicitly asks** — "can I speak to someone", "I want a human", "transfer me"
2. **Emergency detected** — gas/CO/fire keywords trigger immediate escalation
3. **Past-date booking modification** — the system cannot alter history; a team member must handle it

All escalations write a record to the `escalations` table with the reason, call ID, and transcript summary, and trigger an outbound callback via Twilio.

---

## Security

- All secrets stored in `.env` (never committed — listed in `.gitignore`)
- Supabase service key used server-side only
- No secrets in source code or logs
