"""Independent judge for /optimize-energy responses.

Written from the Problem Statement alone (sections 04, 05, 08, 09, 10, 11) and imports nothing
from app, solver or llm: it must be able to catch their bugs, not share them. Every check
replays the response against GROUND-TRUTH directives supplied by the caller, never against
the service's own interpretation.

    truth = [{"directive_type": "solar_reduction", "structured_adjustment": {"hours": [13, 14], "factor": 0.2}},
             {"directive_type": "no_op", "structured_adjustment": None}]
    problems = validate(request, response, truth)      # [] means the case is valid
    scores = interpretation(response, truth)           # per-note rubric points
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog

TOL = 0.01
H = 24
SHAPES = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
}
TYPES = set(SHAPES) | {"no_op"}
VALUE_KEY = {"solar_reduction": "factor", "minimum_battery_reserve": "minimum_energy_kwh",
             "max_grid_window": "max_grid_kwh"}
PLAN_FIELDS = {"hour", "grid_kwh", "solar_used_kwh", "battery_action", "battery_kwh",
               "battery_energy_after_kwh"}


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def schema(request, response):
    """Section 10 response shape and section 08 guardrails on the reported interpretation."""
    bad = []
    if not isinstance(response, dict):
        return ["response is not a JSON object"]
    for k in ("scenario_id", "directive_interpretation", "hourly_plan", "total_grid_kwh",
              "total_cost_bdt", "peak_grid_kwh", "plan_summary"):
        if k not in response:
            bad.append(f"missing top-level field {k}")
    if bad:
        return bad
    if response["scenario_id"] != request["scenario_id"]:
        bad.append("scenario_id not echoed")
    if not isinstance(response["plan_summary"], str):
        bad.append("plan_summary must be a string")
    for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"):
        if not _num(response[k]):
            bad.append(f"{k} must be a finite number")

    di, notes = response["directive_interpretation"], request["operator_notes"]
    if not isinstance(di, list) or len(di) != len(notes):
        return bad + [f"directive_interpretation must have exactly {len(notes)} entries"]
    capacity = request["battery"]["capacity_kwh"]
    for i, d in enumerate(di):
        where = f"directive_interpretation[{i}]"
        if not isinstance(d, dict):
            bad.append(f"{where} is not an object")
            continue
        if d.get("note_index") != i or isinstance(d.get("note_index"), bool):
            bad.append(f"{where}: note_index must be {i}")
        t = d.get("directive_type")
        if t not in TYPES:
            bad.append(f"{where}: unsupported directive_type {t!r}")
            continue
        if not isinstance(d.get("explanation"), str):
            bad.append(f"{where}: explanation must be a string")
        adj = d.get("structured_adjustment")
        if t == "no_op":
            if d.get("applies") is not False or adj is not None:
                bad.append(f"{where}: no_op needs applies=false and structured_adjustment=null")
            continue
        if d.get("applies") is not True:
            bad.append(f"{where}: {t} needs applies=true")
        if not isinstance(adj, dict) or set(adj) != SHAPES[t]:
            bad.append(f"{where}: structured_adjustment must have exactly {sorted(SHAPES[t])}")
            continue
        hrs = adj["hours"]
        if (not isinstance(hrs, list) or not hrs
                or not all(isinstance(h, int) and not isinstance(h, bool) and 0 <= h < H for h in hrs)
                or hrs != sorted(set(hrs))):
            bad.append(f"{where}: hours must be unique ascending integers 0-23, got {hrs}")
        v = adj.get(VALUE_KEY.get(t, ""), 0)
        if t in VALUE_KEY and not _num(v):
            bad.append(f"{where}: {VALUE_KEY[t]} must be a finite number")
        elif t == "solar_reduction" and not 0 <= v <= 1:
            bad.append(f"{where}: factor {v} outside [0, 1]")
        elif t == "minimum_battery_reserve" and not 0 <= v <= capacity:
            bad.append(f"{where}: reserve {v} outside [0, capacity]")
        elif t == "max_grid_window" and v < 0:
            bad.append(f"{where}: max_grid_kwh {v} is negative")

    plan = response["hourly_plan"]
    if not isinstance(plan, list) or len(plan) != H:
        return bad + ["hourly_plan must have 24 entries"]
    for p in plan:
        if not isinstance(p, dict) or set(p) < PLAN_FIELDS:
            bad.append(f"hourly_plan entry missing fields: {p}")
            return bad
    if sorted(p["hour"] for p in plan) != list(range(H)):
        bad.append("hourly_plan hours must be 0..23 exactly once")
    return bad


def _truth_model(request, truth):
    hours = {h["hour"]: h for h in request["hours"]}
    b = request["battery"]
    m = {
        "demand": [float(hours[h]["demand_kwh"]) for h in range(H)],
        "tariff": [float(hours[h]["tariff_bdt_per_kwh"]) for h in range(H)],
        "solar": [float(hours[h]["solar_kwh"]) for h in range(H)],
        "min_e": [float(b["minimum_energy_kwh"])] * H,
        "cap": [math.inf] * H,
        "no_charge": [False] * H,
        "no_discharge": [False] * H,
    }
    for d in truth:
        adj = d.get("structured_adjustment") or {}
        for h in adj.get("hours", []):
            t = d["directive_type"]
            if t == "solar_reduction":  # section 5.3, applied per directive
                m["solar"][h] *= adj["factor"]
            elif t == "minimum_battery_reserve":
                m["min_e"][h] = max(m["min_e"][h], adj["minimum_energy_kwh"])
            elif t == "no_charge_window":
                m["no_charge"][h] = True
            elif t == "no_discharge_window":
                m["no_discharge"][h] = True
            elif t == "max_grid_window":
                m["cap"][h] = min(m["cap"][h], adj["max_grid_kwh"])
    return m


def replay(request, response, truth):
    """Sections 9 and 11.2-11.3: replay the plan hour by hour under the ground truth."""
    bad = []
    b = request["battery"]
    cap, e0 = float(b["capacity_kwh"]), float(b["initial_energy_kwh"])
    m = _truth_model(request, truth)
    soc = e0
    grid_sum = cost = 0.0
    peak = -math.inf
    for p in sorted(response["hourly_plan"], key=lambda p: p["hour"]):
        h = p["hour"]
        g, s, kwh, after, act = (p["grid_kwh"], p["solar_used_kwh"], p["battery_kwh"],
                                 p["battery_energy_after_kwh"], p["battery_action"])
        if not all(_num(v) for v in (g, s, kwh, after)):
            bad.append(f"h{h}: non-finite or non-numeric value")
            continue
        for name, v in (("grid_kwh", g), ("solar_used_kwh", s), ("battery_kwh", kwh),
                        ("battery_energy_after_kwh", after)):
            if v < -TOL:
                bad.append(f"h{h}: {name}={v} is negative")
        if act not in ("charge", "discharge", "idle"):
            bad.append(f"h{h}: battery_action {act!r} not charge/discharge/idle")
            continue
        chg = kwh if act == "charge" else 0.0
        dis = kwh if act == "discharge" else 0.0
        if act == "idle" and abs(kwh) > TOL:
            bad.append(f"h{h}: idle with battery_kwh={kwh}")
        if chg > float(b["max_charge_kwh_per_hour"]) + TOL:
            bad.append(f"h{h}: charge {chg} over hourly limit")
        if dis > float(b["max_discharge_kwh_per_hour"]) + TOL:
            bad.append(f"h{h}: discharge {dis} over hourly limit")
        if abs((soc + chg - dis) - after) > TOL:
            bad.append(f"h{h}: transition {soc}+{chg}-{dis} != reported {after}")
        soc = after
        if after < m["min_e"][h] - TOL:
            bad.append(f"h{h}: battery {after} below required minimum {m['min_e'][h]}")
        if after > cap + TOL:
            bad.append(f"h{h}: battery {after} above capacity {cap}")
        if s > m["solar"][h] + TOL:
            bad.append(f"h{h}: solar_used {s} exceeds effective solar {m['solar'][h]:.4f}")
        if abs(g + s + dis - m["demand"][h] - chg) > TOL:
            bad.append(f"h{h}: energy balance off by {g + s + dis - m['demand'][h] - chg:.4f}")
        if m["no_charge"][h] and (act == "charge" or chg > TOL):
            bad.append(f"h{h}: charging inside a no_charge_window")
        if m["no_discharge"][h] and (act == "discharge" or dis > TOL):
            bad.append(f"h{h}: discharging inside a no_discharge_window")
        if g > m["cap"][h] + TOL:
            bad.append(f"h{h}: grid {g} over max_grid_window cap {m['cap'][h]}")
        grid_sum += g
        cost += g * m["tariff"][h]
        peak = max(peak, g)
    if abs(soc - e0) > TOL:
        bad.append(f"final battery {soc} != initial {e0}")
    for k, v in (("total_grid_kwh", grid_sum), ("total_cost_bdt", cost), ("peak_grid_kwh", peak)):
        if _num(response.get(k)) and abs(response[k] - v) > TOL:
            bad.append(f"{k}={response[k]} but hourly_plan gives {v:.4f}")
    return bad


def validate(request, response, truth):
    """Everything the judge checks for validity. [] means the case is valid."""
    return schema(request, response) or replay(request, response, truth)


def plan_cost(request, response):
    tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in request["hours"]}
    return sum(p["grid_kwh"] * tariff[p["hour"]] for p in response["hourly_plan"])


def interpretation(response, truth):
    """Rubric category 1, per note: relevance, type, hours, value (each True/False)."""
    out = []
    for i, want in enumerate(truth):
        got = response["directive_interpretation"][i]
        wt, gt = want["directive_type"], got.get("directive_type")
        wa, ga = want.get("structured_adjustment") or {}, got.get("structured_adjustment") or {}
        row = {"relevance": (wt == "no_op") == (gt == "no_op"), "type": wt == gt}
        row["hours"] = wt == "no_op" and gt == "no_op" or (row["type"] and wa.get("hours") == ga.get("hours"))
        k = VALUE_KEY.get(wt)
        row["value"] = row["hours"] and (k is None or (
            _num(ga.get(k)) and abs(ga[k] - wa[k]) <= TOL))
        out.append(row)
    return out


# --- independent optimum: a different LP formulation, solved with interior point ---------

def lp_optimum(req, truth):
    """Organizer-style optimal cost under the ground truth, or None if infeasible."""
    m = _truth_model(req, truth)
    b = req["battery"]
    e0 = float(b["initial_energy_kwh"])
    n = 24
    # variables per hour: g, s, c, d, E  (explicit state of charge, unlike the service)
    G, S, C, Dd, E = (np.arange(n) + k * n for k in range(5))
    nv = 5 * n
    cost = np.zeros(nv)
    cost[G] = m["tariff"]
    A, rhs = [], []
    for h in range(n):
        row = np.zeros(nv)
        row[G[h]] = row[S[h]] = row[Dd[h]] = 1
        row[C[h]] = -1
        A.append(row)
        rhs.append(m["demand"][h])
        row = np.zeros(nv)
        row[E[h]], row[C[h]], row[Dd[h]] = 1, -1, 1
        if h:
            row[E[h - 1]] = -1
        A.append(row)
        rhs.append(0 if h else e0)
    bounds = ([(0, None if math.isinf(c) else c) for c in m["cap"]]
              + [(0, s) for s in m["solar"]]
              + [(0, 0 if m["no_charge"][h] else b["max_charge_kwh_per_hour"]) for h in range(n)]
              + [(0, 0 if m["no_discharge"][h] else b["max_discharge_kwh_per_hour"]) for h in range(n)]
              + [(m["min_e"][h], b["capacity_kwh"]) for h in range(n)])
    lo, hi = bounds[E[-1]]
    if not lo - 1e-9 <= e0 <= hi + 1e-9:
        return None
    bounds[E[-1]] = (e0, e0)
    r = linprog(cost, A_eq=np.array(A), b_eq=rhs, bounds=bounds, method="highs-ipm")
    return r.fun if r.status == 0 else None
