# HVAC Voice Agent — Local Dev Runbook

## Required Environment Variables

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

---

## Database Setup (Supabase) — run once

1. Open your Supabase project → **SQL Editor**
2. Paste and run the full contents of `sql/schema.sql`
3. This creates all tables, indexes, and seeds 5 test bookings
4. Verify tables exist: `calls`, `call_turns`, `customers`, `bookings`, `escalations`

---

## Running Locally

### 1 — Install dependencies

```bash
# From repo root
pip install -r apps/orchestrator/requirements.txt
```

### 2 — Start the Orchestrator

```bash
PYTHONPATH=$(pwd) uvicorn apps.orchestrator.main:app --reload --port 8080
```

Health check:
```bash
curl http://localhost:8080/health
# → {"status": "ok"}
```

---

## Connecting Twilio Webhook via ngrok

```bash
# Use the included ngrok binary or install separately
./ngrok http 8080
# Copy the https:// forwarding URL, e.g. https://abc123.ngrok-free.app
```

In `.env`:
```
ORCHESTRATOR_BASE_URL=https://abc123.ngrok-free.app
```

In Twilio Console → **Phone Numbers** → your number → **Voice Configuration**:
- **A call comes in** → Webhook → `https://abc123.ngrok-free.app/voice/inbound`
- HTTP method: **POST**

---

## Testing Without a Real Phone

Use the `/voice/simulate` endpoint to test the full call pipeline locally.
It runs the same LLM → state machine → booking tools logic as a live call.

```bash
# 1. Start a new booking conversation
curl -X POST http://localhost:8080/voice/simulate \
  -H "Content-Type: application/json" \
  -d '{"call_sid": "TEST001", "transcript": "Hi, I need to book a furnace repair", "from_number": "+14161234567"}'

# 2. Continue — provide postal code
curl -X POST http://localhost:8080/voice/simulate \
  -H "Content-Type: application/json" \
  -d '{"call_sid": "TEST001", "transcript": "My postal code is M5V 2K3"}'

# 3. Continue — provide name
curl -X POST http://localhost:8080/voice/simulate \
  -H "Content-Type: application/json" \
  -d '{"call_sid": "TEST001", "transcript": "My name is John Smith"}'

# Test emergency detection (hard-coded bypass)
curl -X POST http://localhost:8080/voice/simulate \
  -H "Content-Type: application/json" \
  -d '{"call_sid": "EMRG001", "transcript": "I can smell gas in my basement", "from_number": "+14161234567"}'

# Test out-of-area
curl -X POST http://localhost:8080/voice/simulate \
  -H "Content-Type: application/json" \
  -d '{"call_sid": "OOA001", "transcript": "My postal code is L4C 2H1", "from_number": "+14161234567"}'
```

The response includes: `state`, `intent`, `slots`, `response` (what Alex would say).

---

## Slot Availability (Test Data)

`sql/schema.sql` seeds the following pre-booked slots so you can test conflict detection:

| Date | Time | Customer | Tier |
|---|---|---|---|
| 2026-03-05 | 10:00–12:00 | Michael Chen | standard |
| 2026-03-05 | 14:00–16:00 | Sarah Williams | standard |
| 2026-03-06 | 09:00–11:00 | David Patel | standard |
| 2026-03-07 | 10:00–12:00 | Jennifer Kim | standard |
| 2026-03-05 | 18:00–20:00 | Robert Singh | surge |

Attempting to book any of these slots will trigger the slot-conflict backtrack and show the next available options.

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
                                                      │
                                                  Supabase DB
```

---

## Deploying to Render

1. Push repo to GitHub (`.env` is gitignored — never committed)
2. Render Dashboard → **New** → **Blueprint** → select repo
3. Render reads `infra/render.yaml` and creates the service
4. Set all secret env vars in the Render dashboard (marked `sync: false` in the yaml)
5. After first deploy, copy the public URL into:
   - `.env` `ORCHESTRATOR_BASE_URL`
   - Twilio Console webhook config: `https://<your-url>/voice/inbound`

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Booking not saved to DB | Check `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in `.env` |
| STT produces no transcripts | Check `DEEPGRAM_API_KEY`; verify mulaw audio is reaching the WebSocket |
| LLM returns garbage | Check `OPENAI_API_KEY`; inspect logs for JSON parse errors |
| SMS not delivered | Check `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` |
| Slot conflict not detected | Confirm `sql/schema.sql` was run and unique index exists on bookings |
| Emergency not escalating | Confirm the phrase matches `EMERGENCY_KEYWORDS` in `packages/core/utils.py` |
