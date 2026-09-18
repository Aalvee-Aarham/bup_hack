"""Section 08 guardrails: LLM output is untrusted until it passes here.

validate() either returns an entry in the exact section 04 shape or None. It normalises only
what is unambiguous (hour order, duplicates, numeric strings, key synonyms) and rejects
anything it would have to guess, such as a factor of 5 or a reserve above capacity.
A rejected answer is treated as a failed LLM attempt, so another model gets a turn; it is
never clamped into a directive the note did not state.
"""
from __future__ import annotations

import math
import re

TYPES = ("solar_reduction", "minimum_battery_reserve", "no_charge_window",
         "no_discharge_window", "max_grid_window", "no_op")

# the exact value key per type, plus names models reach for instead
_VALUE_KEYS = {
    "solar_reduction": ("factor", ("reduction_factor", "solar_factor", "fraction",
                                   "remaining_fraction", "remaining_factor")),
    "minimum_battery_reserve": ("minimum_energy_kwh", ("reserve_kwh", "min_kwh", "minimum_kwh",
                                                       "energy_kwh", "min_energy_kwh")),
    "max_grid_window": ("max_grid_kwh", ("grid_kwh", "max_kwh", "limit_kwh", "cap_kwh",
                                         "max_grid_import_kwh")),
}

NO_OP_EXPLANATION = "This note does not affect today's energy schedule."


def _finite(x):
    if isinstance(x, bool):
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _hours(adj):
    """Unique ascending hours 0-23, or None if the model gave something unusable."""
    raw = adj.get("hours")
    if raw is None:
        # tolerate {"start_hour": 13, "end_hour": 15}: end exclusive, may wrap midnight
        a = _finite(adj.get("start_hour", adj.get("start")))
        b = _finite(adj.get("end_hour", adj.get("end")))
        if a is None or b is None or a != int(a) or b != int(b):
            return None
        a, b = int(a) % 24, int(b) % 24
        raw = [(a + k) % 24 for k in range((b - a) % 24 or 24)]
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    out = set()
    for h in raw:
        v = _finite(h)
        if v is None or v != int(v):
            return None  # "13.5" or "afternoon": not an hour
        if 0 <= v <= 23:
            out.add(int(v))
    return sorted(out) or None


def validate(entry, capacity):
    """A guardrail-clean copy of one interpretation entry, or None if it cannot be trusted."""
    if not isinstance(entry, dict):
        return None
    t = entry.get("directive_type")
    if t not in TYPES:
        return None
    explanation = entry.get("explanation")
    explanation = str(explanation)[:300] if explanation else None
    if t == "no_op":
        return {"applies": False, "directive_type": "no_op", "structured_adjustment": None,
                "explanation": explanation or NO_OP_EXPLANATION}

    adj = entry.get("structured_adjustment")
    if not isinstance(adj, dict):
        return None
    hours = _hours(adj)
    if hours is None:
        return None
    out = {"hours": hours}
    if t in _VALUE_KEYS:
        key, synonyms = _VALUE_KEYS[t]
        v = next((_finite(adj[k]) for k in (key, *synonyms) if k in adj), None)
        if v is None or v < 0:
            return None
        if t == "solar_reduction" and v > 1:
            return None  # 5, 20 or 1.5 would all be guesses about what the model meant
        if t == "minimum_battery_reserve" and capacity is not None and v > capacity:
            return None
        out[key] = v
    return {"applies": True, "directive_type": t, "structured_adjustment": out,
            "explanation": explanation or f"{t.replace('_', ' ')} for hours {hours}."}


# A directive must be about energy. The list is deliberately broad: it exists to veto a model
# that hallucinates a directive onto "The cafeteria menu changes tomorrow.", not to judge
# paraphrases. One hit anywhere in the note is enough.
_ENERGY = re.compile(
    r"solar|\bpv\b|panel|photovolt|\bsun|array|generat|output|yield|produc|irradian|cloud|rooftop|"
    r"module|inverter|batter|reserve|storage|\bstor|charg|\bsoc\b|state of charge|\bbess\b|\bess\b|"
    r"kwh|mwh|\bkw\b|energy|power|electric|backup|\bpack\b|cells?\b|grid|import|utility|mains|"
    r"feeder|supply|draw|purchas|\bbuy|demand|\bload|substation|transformer|tariff|top.?up|"
    r"drain|discharg|capacity|full|empty|export",
    re.I)


def grounded(note, entry):
    """False when a non-no_op directive comes from a note with no energy vocabulary at all."""
    return entry["directive_type"] == "no_op" or bool(_ENERGY.search(note or ""))


def no_op(index, explanation=NO_OP_EXPLANATION):
    return {"note_index": index, "applies": False, "directive_type": "no_op",
            "structured_adjustment": None, "explanation": explanation}


def normalize(entry, index, capacity):
    """validate(), tagged with note_index, falling back to no_op: never a guessed directive."""
    v = validate(entry, capacity)
    return {"note_index": index, **v} if v else no_op(index)
