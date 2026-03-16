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

## Core Features

### Appointment Booking

Full booking flow with validation enforced in backend code:

- Future-only dates (Toronto timezone)
- Maximum 30 days in advance
- Slot conflict check against confirmed bookings
- Caller-provided city validated against GTA service area

### Reschedule and Cancellation

- Reschedule validates the new slot and excludes the caller's existing booking from the conflict check
- Past bookings require human escalation — the agent cannot modify them
- Both flows send SMS confirmations on completion

### Emergency Triage

Hard-coded keyword detection runs before the LLM on every transcript. If a caller mentions gas smell, carbon monoxide, smoke from furnace, or similar emergencies:

- Pre-scripted safety response plays immediately
- Escalation record logged to Supabase
- Call terminated proactively (caller must leave the building)

### After-Hours Pricing Disclosure

Pricing is based on the **booked appointment slot time**, not when the caller phones in.

- **Standard rate**: Mon–Fri slots before 16:00, Sat slots before 13:00
- **Surge rate** (+$120–$180): Mon–Fri 16:00+, Sat 13:00+, all Sunday slots

Callers are informed and must confirm before the booking is created.

### Out-of-Area Detection

City name extracted from caller speech and checked against `gta_cities.json` (covers City of Toronto, Peel, York, Durham, and Halton regions). Out-of-area callers receive a referral SMS and the call ends.

### SMS Confirmations

Twilio SMS sent on: new booking, reschedule, cancellation, and out-of-area referral.

### Regex Fast Path

For states where only a yes/no is needed (`CONFIRMING_BOOKING`, `AFTER_HOURS_DISCLOSURE`, `WRAP_UP`, `PRICING_FOLLOWUP`), the LLM is bypassed entirely. This eliminates LLM latency on the most common caller responses.

### Multi-Intent Calls (WRAP_UP)

After completing a task, the agent checks whether the caller needs anything else. A single call can book an appointment, ask about pricing, and reschedule — without restarting.

---

## Testing Without a Phone

A browser WebRTC test client is available at `http://localhost:8080/voice/test-client`.

You can also hit the `/voice/simulate` endpoint directly to test the full pipeline (LLM → state machine → booking tools) over HTTP:

```bash
# Start a new booking
curl -X POST http://localhost:8080/voice/simulate \
  -H "Content-Type: application/json" \
  -d '{"call_sid": "TEST001", "transcript": "Hi, I need to book a furnace repair", "from_number": "+14161234567"}'

# Continue the session
curl -X POST http://localhost:8080/voice/simulate \
  -H "Content-Type: application/json" \
  -d '{"call_sid": "TEST001", "transcript": "My name is John Smith"}'

# Test emergency bypass
curl -X POST http://localhost:8080/voice/simulate \
  -H "Content-Type: application/json" \
  -d '{"call_sid": "EMRG001", "transcript": "I can smell gas in my basement", "from_number": "+14161234567"}'
```

Response includes: `state`, `intent`, `slots`, `response` (what the agent would say).

---

## Connecting Twilio (Local Dev)

```bash
# Start ngrok (binary included in repo root)
./ngrok http 8080
# Copy the https:// URL, e.g. https://abc123.ngrok-free.app
```

Set `ORCHESTRATOR_BASE_URL=https://abc123.ngrok-free.app` in `.env`.

In **Twilio Console → Phone Numbers → Voice Configuration**:
- A call comes in → Webhook → `https://abc123.ngrok-free.app/voice/inbound`
- HTTP method: `POST`

---

## Deploying to Render

1. Push repo to GitHub (`.env` is gitignored — never committed)
2. Render Dashboard → **New** → **Blueprint** → select repo
3. Render reads `infra/render.yaml` and creates the service
4. Set all secret env vars in the Render dashboard
5. After first deploy, update `ORCHESTRATOR_BASE_URL` and the Twilio webhook URL

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Booking not saved to DB | Check `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` |
| STT produces no transcripts | Check `DEEPGRAM_API_KEY`; verify mulaw audio is reaching the WebSocket |
| LLM returns garbage | Check `OPENAI_API_KEY`; inspect logs for JSON parse errors |
| SMS not delivered | Check `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` |
| Slot conflict not detected | Confirm `sql/schema.sql` was run and the unique index exists on bookings |
| Emergency not escalating | Check that the phrase matches `EMERGENCY_KEYWORDS` in `packages/core/utils.py` |

---

## Security

- All secrets stored in `.env` (never committed — listed in `.gitignore`)
- Supabase service key used server-side only
- No secrets in source code or logs
