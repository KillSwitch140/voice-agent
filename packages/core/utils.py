from __future__ import annotations

# ─── Emergency detection ─────────────────────────────────────────────────────

EMERGENCY_KEYWORDS: list[str] = [
    # Gas
    "gas smell", "smell gas", "gas leak", "leaking gas", "smell of gas",
    # Carbon monoxide
    "carbon monoxide", "co detector", "co alarm", "co leak",
    # Fire / smoke
    "fire", "smoke from furnace", "furnace on fire", "smoke coming from",
    # Explosion / loud bang
    "explosion", "exploded", "loud bang from furnace",
    # Frozen / burst pipes (water damage + flooding risk)
    "pipes frozen", "frozen pipes", "pipe burst", "burst pipe",
    # Breathing emergency
    "cannot breathe", "can't breathe",
]


def detect_emergency(text: str) -> bool:
    """Return True if the utterance contains any emergency trigger phrase."""
    lower = text.lower()
    return any(kw in lower for kw in EMERGENCY_KEYWORDS)


# ─── Service area ─────────────────────────────────────────────────────────────

TORONTO_POSTAL_PREFIXES: set[str] = {
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
}


def is_toronto_service_area(postal_code: str) -> bool:
    prefix = postal_code.upper().replace(" ", "")[:2]
    return prefix in TORONTO_POSTAL_PREFIXES

