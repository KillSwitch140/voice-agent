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

GTA_CITIES: set[str] = {
    # City of Toronto + former municipalities
    "toronto", "north york", "scarborough", "etobicoke", "east york", "york",
    # Peel Region
    "mississauga", "brampton", "caledon",
    # York Region
    "vaughan", "markham", "richmond hill", "newmarket", "aurora",
    "king", "king city", "east gwillimbury", "georgina", "stouffville",
    "whitchurch-stouffville",
    # Durham Region
    "ajax", "whitby", "oshawa", "pickering", "uxbridge", "clarington", "bowmanville",
    # Halton Region
    "oakville", "burlington", "halton hills", "georgetown", "milton",
}


def is_gta_city(city: str) -> bool:
    """Return True if the caller's city is within the Greater Toronto Area service area."""
    return city.strip().lower() in GTA_CITIES

