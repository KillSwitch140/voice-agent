# Emergency Triage Policy — HVAC Voice Agent

## Trigger Phrases (auto-detected by backend — not LLM-only)

Any of the following utterances must immediately activate the emergency flow:

- "gas smell" / "smell gas" / "gas leak" / "leaking gas"
- "carbon monoxide" / "CO detector" / "CO alarm"
- "no heat" when caller indicates elderly resident, infant, or medical dependency
- "pipes frozen" / "pipe burst"
- "smoke from furnace" / "furnace on fire"
- "explosion" / "loud bang from furnace"
- "can't breathe" / "chest pain" associated with HVAC issue

## Emergency Response Protocol

1. **Do not attempt to troubleshoot.** Do not give DIY instructions.
2. Play the safety script (`scripts://safety_gas_smell` for gas/CO emergencies).
3. Advise caller to:
   - Leave the building immediately if gas/CO is suspected.
   - Call **911** and **Enbridge Gas (1-866-763-5427)** for gas emergencies.
   - Call **Toronto Hydro** for electrical emergencies.
4. Flag the call as `priority = emergency` in the system.
5. Log an escalation record immediately via `escalate_call` tool.
6. Bypass standard scheduling flow — book earliest available slot OR page on-call technician.
7. End normal conversational flow after safety message is delivered.

## After-Hours Emergency

If call is received outside Mon–Fri 8am–5pm or Sat 9am–2pm (Toronto time):
- Inform caller that emergency after-hours rates apply.
- Confirm they accept before dispatching.
- Priority flag: `emergency`, not `urgent`.

## What the Agent Must NEVER Do

- Provide gas leak detection instructions.
- Advise the caller to check the furnace or pilot light themselves.
- Delay the safety message to collect booking information first.
- Promise a specific technician arrival time during an emergency.
