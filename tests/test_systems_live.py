"""
Full systems test — exercises all voice agent flows via /voice/simulate.
Hits real APIs (OpenAI, Supabase). Measures per-turn and per-flow latency.

Uses httpx.AsyncClient + ASGITransport so the event loop stays alive across
all requests (required for persistent httpx clients in the app).

Usage: python tests/test_systems_live.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field

# Ensure the project root is importable
sys.path.insert(0, ".")

import httpx
from apps.orchestrator.main import app


# ─── Helpers ──────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn: int
    transcript: str
    state: str
    intent: str | None
    response: str
    latency_ms: float
    slots: dict = field(default_factory=dict)
    is_emergency: bool = False


@dataclass
class FlowResult:
    name: str
    turns: list[TurnResult] = field(default_factory=list)
    success: bool = True
    error: str = ""

    @property
    def total_ms(self) -> float:
        return sum(t.latency_ms for t in self.turns)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / len(self.turns) if self.turns else 0

    @property
    def max_ms(self) -> float:
        return max(t.latency_ms for t in self.turns) if self.turns else 0


def _sid() -> str:
    return f"SYS_{uuid.uuid4().hex[:8]}"


# Global async client — set in main()
_client: httpx.AsyncClient | None = None


async def simulate(call_sid: str, transcript: str, from_number: str = "+14161234567") -> tuple[dict, float]:
    start = time.perf_counter()
    r = await _client.post("/voice/simulate", json={
        "call_sid": call_sid,
        "transcript": transcript,
        "from_number": from_number,
    })
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json(), elapsed_ms


async def run_flow(name: str, turns: list[str], call_sid: str | None = None) -> FlowResult:
    sid = call_sid or _sid()
    result = FlowResult(name=name)
    for i, transcript in enumerate(turns):
        try:
            data, ms = await simulate(sid, transcript)
            tr = TurnResult(
                turn=i + 1,
                transcript=transcript,
                state=data.get("state", "?"),
                intent=data.get("intent"),
                response=data.get("response", ""),
                latency_ms=ms,
                slots=data.get("slots", {}),
                is_emergency=data.get("is_emergency", False),
            )
            result.turns.append(tr)
            # Stop if call ended
            if data.get("state") == "ended":
                break
        except Exception as e:
            result.success = False
            result.error = f"Turn {i+1} failed: {e}"
            break
    return result


def print_flow(flow: FlowResult) -> None:
    status = "PASS" if flow.success else "FAIL"
    print(f"\n{'='*80}")
    print(f"  {status}  {flow.name}")
    print(f"  Total: {flow.total_ms:.0f}ms | Avg: {flow.avg_ms:.0f}ms | Max: {flow.max_ms:.0f}ms | Turns: {len(flow.turns)}")
    print(f"{'='*80}")
    for t in flow.turns:
        bypass = " [REGEX]" if t.latency_ms < 100 else ""
        resp_preview = t.response[:72]
        print(f"  T{t.turn:2d} [{t.latency_ms:6.0f}ms]{bypass} {t.state:30s} | {t.transcript[:40]}")
        print(f"       {'':30s} > {resp_preview}")
    if flow.error:
        print(f"  ERROR: {flow.error}")
    final = flow.turns[-1] if flow.turns else None
    if final:
        print(f"  Final: state={final.state} intent={final.intent}")


# ─── Flow definitions ────────────────────────────────────────────────────────

async def test_emergency() -> FlowResult:
    """Emergency detection (hard-coded bypass, no LLM)"""
    return await run_flow("Emergency Triage", [
        "I smell gas in my basement please help",
    ])


async def test_out_of_area() -> FlowResult:
    """Out-of-area rejection with proper scripted message"""
    return await run_flow("Out of Area", [
        "Hi there",
        "Alex Brown",
        "Ottawa",
    ])


async def test_escalation() -> FlowResult:
    """Caller requests human agent"""
    return await run_flow("Escalation (human request)", [
        "Hello",
        "Lisa Wang",
        "Vaughan",
        "I'd like to speak to a real person please",
    ])


async def test_pricing_only() -> FlowResult:
    """Pricing question, decline booking"""
    return await run_flow("Pricing Only (no booking)", [
        "Hi",
        "Bob Lee",
        "Mississauga",
        "What are your rates?",
        "No not right now",                     # regex bypass → intent=unknown
        "No that's all",                        # regex bypass → more_help=false
    ])


async def test_new_booking() -> FlowResult:
    """Full new booking flow"""
    sid = _sid()
    result = FlowResult(name="New Booking (happy path)")

    # Turns 1-4: name, city, intent
    for text in ["Hi there", "John Smith", "Toronto", "I'd like to book a new appointment"]:
        data, ms = await simulate(sid, text)
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript=text,
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Turn 5: issue description → should trigger slot offer
    data, ms = await simulate(sid, "My furnace isn't heating properly")
    result.turns.append(TurnResult(
        turn=len(result.turns) + 1, transcript="My furnace isn't heating properly",
        state=data["state"], intent=data.get("intent"),
        response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
    ))

    # Turn 6: accept offered slot
    data, ms = await simulate(sid, "Yes that works")
    result.turns.append(TurnResult(
        turn=len(result.turns) + 1, transcript="Yes that works",
        state=data["state"], intent=data.get("intent"),
        response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
    ))

    # If we hit AFTER_HOURS_DISCLOSURE, accept it
    if data.get("state") == "after_hours_disclosure":
        data, ms = await simulate(sid, "Yes proceed")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="Yes proceed (after-hours)",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Confirm booking
    if data.get("state") == "confirming_booking":
        data, ms = await simulate(sid, "Yes")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="Yes (confirm)",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Wrap up
    if data.get("state") in ("closing", "wrap_up"):
        data, ms = await simulate(sid, "No that's all thanks")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="No that's all thanks",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    return result


async def test_cancellation() -> FlowResult:
    """Cancel an existing booking (uses seed data bk_test003)"""
    return await run_flow("Cancellation", [
        "Hey",
        "David Patel",
        "Scarborough",
        "I need to cancel my appointment",
        "bk_test003",
        "Yes that's correct",                   # regex bypass
        "No that's all",                        # regex bypass
    ])


async def test_reschedule() -> FlowResult:
    """Reschedule an existing booking (uses seed data bk_test004)"""
    sid = _sid()
    result = FlowResult(name="Reschedule")

    for text in ["Hello", "Jennifer Kim", "North York", "I'd like to reschedule my appointment", "bk_test004"]:
        data, ms = await simulate(sid, text)
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript=text,
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Accept offered slot
    data, ms = await simulate(sid, "Yes that works")
    result.turns.append(TurnResult(
        turn=len(result.turns) + 1, transcript="Yes that works",
        state=data["state"], intent=data.get("intent"),
        response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
    ))

    # Handle after-hours disclosure if triggered
    if data.get("state") == "after_hours_disclosure":
        data, ms = await simulate(sid, "Yes")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="Yes (after-hours)",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Confirm
    if data.get("state") == "confirming_booking":
        data, ms = await simulate(sid, "Yes")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="Yes (confirm)",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Wrap up
    if data.get("state") in ("closing", "wrap_up"):
        data, ms = await simulate(sid, "No I'm done")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="No I'm done",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    return result


async def test_pricing_then_book() -> FlowResult:
    """Pricing inquiry followed by booking"""
    sid = _sid()
    result = FlowResult(name="Pricing -> Booking")

    for text in ["Hi", "Mike Chen", "Markham", "How much does a service call cost?"]:
        data, ms = await simulate(sid, text)
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript=text,
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Accept pricing followup → new_booking (regex bypass)
    data, ms = await simulate(sid, "Sure")
    result.turns.append(TurnResult(
        turn=len(result.turns) + 1, transcript="Sure (regex)",
        state=data["state"], intent=data.get("intent"),
        response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
    ))

    # Issue + slot
    data, ms = await simulate(sid, "My thermostat stopped working")
    result.turns.append(TurnResult(
        turn=len(result.turns) + 1, transcript="My thermostat stopped working",
        state=data["state"], intent=data.get("intent"),
        response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
    ))

    # Accept slot
    data, ms = await simulate(sid, "Yes that works")
    result.turns.append(TurnResult(
        turn=len(result.turns) + 1, transcript="Yes that works",
        state=data["state"], intent=data.get("intent"),
        response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
    ))

    # After-hours if needed
    if data.get("state") == "after_hours_disclosure":
        data, ms = await simulate(sid, "Yes")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="Yes (after-hours)",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Confirm
    if data.get("state") == "confirming_booking":
        data, ms = await simulate(sid, "Yes")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="Yes (confirm)",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Wrap up
    if data.get("state") in ("closing", "wrap_up"):
        data, ms = await simulate(sid, "No thanks")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="No thanks",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    return result


async def test_slot_presentation() -> FlowResult:
    """Verify slots are offered with pricing tier info"""
    sid = _sid()
    result = FlowResult(name="Slot Presentation (pricing tier)")

    for text in ["Hi", "Test User", "Toronto", "I'd like to book", "My AC is broken"]:
        data, ms = await simulate(sid, text)
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript=text,
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Check if the slot offer mentions rate info
    last_resp = result.turns[-1].response.lower()
    if "standard rate" in last_resp or "after-hours rate" in last_resp:
        pass  # Good — pricing tier is shown
    else:
        result.success = False
        result.error = f"Missing pricing tier in slot offer: '{result.turns[-1].response[:100]}'"

    return result


async def test_regex_bypass_comparison() -> FlowResult:
    """Compare latency of regex-bypassed turns vs LLM turns"""
    sid = _sid()
    result = FlowResult(name="Regex Bypass Latency Comparison")

    # LLM turns
    for text in ["Hi there", "John", "Toronto", "I want to book", "Furnace broken"]:
        data, ms = await simulate(sid, text)
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript=text,
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Accept slot (LLM)
    data, ms = await simulate(sid, "Yes that works")
    result.turns.append(TurnResult(
        turn=len(result.turns) + 1, transcript="Yes that works (LLM)",
        state=data["state"], intent=data.get("intent"),
        response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
    ))

    # After-hours if needed (regex)
    if data.get("state") == "after_hours_disclosure":
        data, ms = await simulate(sid, "Yes")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="Yes (after-hours, regex)",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Confirm (regex)
    if data.get("state") == "confirming_booking":
        data, ms = await simulate(sid, "Yes")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="Yes (confirm, regex)",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    # Wrap up (regex)
    if data.get("state") in ("closing", "wrap_up"):
        data, ms = await simulate(sid, "No")
        result.turns.append(TurnResult(
            turn=len(result.turns) + 1, transcript="No (wrap-up, regex)",
            state=data["state"], intent=data.get("intent"),
            response=data["response"], latency_ms=ms, slots=data.get("slots", {}),
        ))

    return result


# ─── Main ────────────────────────────────────────────────────────────────────

async def async_main() -> int:
    global _client

    print("\n" + "=" * 80)
    print("  HVAC VOICE AGENT - FULL SYSTEMS TEST")
    print("  Real APIs: OpenAI, Supabase | Regex bypass for yes/no states")
    print("=" * 80)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        _client = client

        flows: list[FlowResult] = []
        tests = [
            test_emergency,
            test_out_of_area,
            test_escalation,
            test_pricing_only,
            test_new_booking,
            test_cancellation,
            test_reschedule,
            test_pricing_then_book,
            test_slot_presentation,
            test_regex_bypass_comparison,
        ]

        for test_fn in tests:
            try:
                flow = await test_fn()
                flows.append(flow)
                print_flow(flow)
            except Exception as e:
                print(f"\n  CRASH  {test_fn.__name__}: {e}")
                import traceback; traceback.print_exc()
                flows.append(FlowResult(name=test_fn.__name__, success=False, error=str(e)))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    all_turns = [t for f in flows for t in f.turns]
    passed = sum(1 for f in flows if f.success)
    failed = sum(1 for f in flows if not f.success)

    # Heuristic: regex-bypassed turns are < 100ms
    regex_turns = [t for t in all_turns if t.latency_ms < 100]
    llm_turns = [t for t in all_turns if t.latency_ms >= 100]

    print(f"\n  Flows: {passed} passed, {failed} failed, {len(flows)} total")
    print(f"  Turns: {len(all_turns)} total ({len(llm_turns)} LLM, {len(regex_turns)} regex)")

    if all_turns:
        print(f"\n  Overall latency:")
        print(f"    Avg: {sum(t.latency_ms for t in all_turns)/len(all_turns):.0f}ms")
        print(f"    Min: {min(t.latency_ms for t in all_turns):.0f}ms")
        print(f"    Max: {max(t.latency_ms for t in all_turns):.0f}ms")
        print(f"    Total: {sum(t.latency_ms for t in all_turns):.0f}ms")

    if llm_turns:
        print(f"\n  LLM turns ({len(llm_turns)}):")
        print(f"    Avg: {sum(t.latency_ms for t in llm_turns)/len(llm_turns):.0f}ms")
        print(f"    Min: {min(t.latency_ms for t in llm_turns):.0f}ms")
        print(f"    Max: {max(t.latency_ms for t in llm_turns):.0f}ms")

    if regex_turns:
        print(f"\n  Regex bypass turns ({len(regex_turns)}):")
        print(f"    Avg: {sum(t.latency_ms for t in regex_turns)/len(regex_turns):.0f}ms")
        print(f"    Min: {min(t.latency_ms for t in regex_turns):.0f}ms")
        print(f"    Max: {max(t.latency_ms for t in regex_turns):.0f}ms")

    if llm_turns and regex_turns:
        avg_llm = sum(t.latency_ms for t in llm_turns) / len(llm_turns)
        avg_regex = sum(t.latency_ms for t in regex_turns) / len(regex_turns)
        if avg_regex > 0:
            print(f"\n  Regex bypass speedup: {avg_llm/avg_regex:.0f}x faster than LLM")

    # Per-flow table
    print(f"\n  {'Flow':<42} {'Status':>6} {'Turns':>5} {'Avg ms':>7} {'Total ms':>8}")
    print(f"  {'-'*42} {'-'*6} {'-'*5} {'-'*7} {'-'*8}")
    for f in flows:
        status = "PASS" if f.success else "FAIL"
        print(f"  {f.name:<42} {status:>6} {len(f.turns):>5} {f.avg_ms:>7.0f} {f.total_ms:>8.0f}")

    if failed:
        print(f"\n  FAILED FLOWS:")
        for f in flows:
            if not f.success:
                print(f"    - {f.name}: {f.error}")

    print()
    return 1 if failed else 0


def main():
    # Suppress noisy HTTP logs — we show latency in our own output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("apps.orchestrator.services.llm").setLevel(logging.WARNING)
    logging.getLogger("apps.orchestrator.services.tools").setLevel(logging.WARNING)
    logging.getLogger("apps.orchestrator.services.supabase_logger").setLevel(logging.WARNING)

    exit_code = asyncio.run(async_main())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
