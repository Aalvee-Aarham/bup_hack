"""Self-check: python test_app.py          (deterministic, no network, no keys)
               python test_app.py --live   (also runs paraphrases through your real keys)

Every scenario goes through the real HTTP route and is then replayed against the judge's
section 11.3 rules.
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from fastapi.testclient import TestClient

import app as api
import llm
import solver

KEY_VARS = ("GROQ_API_KEYS", "GEMINI_API_KEYS", "OPENROUTER_API_KEYS",
            "GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")

DEMAND = [180, 170, 165, 160, 160, 170, 200, 240, 280, 300, 310, 320,
          330, 325, 320, 310, 300, 320, 360, 380, 340, 290, 240, 200]
SOLAR = [0, 0, 0, 0, 0, 5, 30, 80, 140, 200, 250, 280,
         290, 280, 250, 200, 140, 70, 20, 0, 0, 0, 0, 0]
TARIFF = [7, 7, 6, 6, 6, 7, 8, 10, 12, 12, 11, 11,
          10, 10, 11, 12, 14, 16, 18, 18, 15, 12, 9, 9]

PARAPHRASES = [
    ("PV production will drop to about 20% between 13:00 and 15:00.", "solar_reduction", [13, 14], 0.2),
    ("Panel washing from one until three will leave roughly one-fifth of normal solar output.", "solar_reduction", [13, 14], 0.2),
    ("Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.", "solar_reduction", [13, 14], 0.2),
    ("Do not charge the battery between 2 PM and 4 PM.", "no_charge_window", [14, 15], None),
    ("Keep at least 120 kWh in reserve from 6 PM until 9 PM.", "minimum_battery_reserve", [18, 19, 20], 120),
    ("The cafeteria menu changes tomorrow.", "no_op", None, None),
]

client = TestClient(api.app)  # no context manager: skips the lifespan warmup, stays offline


def scenario(notes, sid="TEST-1", **battery):
    s = {
        "scenario_id": sid,
        "operator_notes": notes,
        "hours": [{"hour": h, "demand_kwh": float(DEMAND[h]), "solar_kwh": float(SOLAR[h]),
                   "tariff_bdt_per_kwh": float(TARIFF[h])} for h in range(24)],
        "battery": {"capacity_kwh": 500.0, "initial_energy_kwh": 200.0, "minimum_energy_kwh": 50.0,
                    "max_charge_kwh_per_hour": 100.0, "max_discharge_kwh_per_hour": 100.0},
    }
    s["battery"].update(battery)
    return s


def run(s, strict=True):
    r = client.post("/optimize-energy", json=s)
    assert r.status_code == 200, (r.status_code, r.text[:300])
    resp = r.json()
    assert resp["scenario_id"] == s["scenario_id"]
    assert [d["note_index"] for d in resp["directive_interpretation"]] == \
        list(range(len(s["operator_notes"]))), "one entry per note, in note_index order"
    for d in resp["directive_interpretation"]:
        assert d["applies"] == (d["directive_type"] != "no_op"), "applies must track no_op"
        assert (d["structured_adjustment"] is None) == (d["directive_type"] == "no_op")
        assert d["directive_type"] in solver.DIRECTIVE_TYPES
        hrs = (d["structured_adjustment"] or {}).get("hours")
        if hrs is not None:
            assert hrs == sorted(set(hrs)) and all(0 <= h < 24 for h in hrs), hrs
    # replay: strict against the directives, or against base rules for an infeasible case
    model = solver.build_model(s, resp["directive_interpretation"] if strict else [])
    tot = {k: resp[k] for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh")}
    problems = solver.validate_plan(model, resp["hourly_plan"], tot)
    assert not problems, f"{s['scenario_id']}: {problems}"
    return resp


def kinds(resp):
    return [d["directive_type"] for d in resp["directive_interpretation"]]


def offline():
    for k in KEY_VARS:
        os.environ.pop(k, None)
    llm.reset()


# --- fake transport for scheduler tests ---------------------------------------

def http_error(code, retry_after=None):
    req = httpx.Request("POST", "http://fake")
    headers = {"retry-after": str(retry_after)} if retry_after else {}
    return httpx.HTTPStatusError("fake", request=req, response=httpx.Response(code, headers=headers, request=req))


def fake_answer(prompt):
    notes = json.loads(prompt)["notes"]
    return [{**llm.rule_parse(n["note_index"], n["text"]), "explanation": "fake llm"} for n in notes]


def with_fake_keys(groq=("gA", "gB"), gemini=("mA",), behaviour=None):
    """Install fake keys and a fake _call(slot, prompt, timeout) -> list."""
    offline()
    os.environ["GROQ_API_KEYS"] = ",".join(groq)
    os.environ["GEMINI_API_KEYS"] = ",".join(gemini)
    os.environ["GROQ_MODELS"] = "fast-model"
    os.environ["GEMINI_MODELS"] = "gem-model"
    llm.reset()
    llm._call = behaviour


def restore(real_call):
    llm._call = real_call
    for k in ("GROQ_MODELS", "GEMINI_MODELS"):
        os.environ.pop(k, None)
    offline()


def test_scheduler():
    real_call = llm._call

    # hedging: first slot hangs, the hedge on the other key answers
    async def hang_on_a(s, prompt, timeout):
        if s.key == "gA":
            await asyncio.sleep(10)
        await asyncio.sleep(0.05)
        return fake_answer(prompt)
    with_fake_keys(behaviour=hang_on_a)
    llm.HEDGE_AFTER_S = 0.3
    t = time.perf_counter()
    entries, source = asyncio.run(llm.interpret(["Do not charge the battery between 2 PM and 4 PM."]))
    took = time.perf_counter() - t
    assert source.startswith("groq") and entries[0]["directive_type"] == "no_charge_window", (source, entries)
    assert took < 1.5, f"hedge should answer in ~0.35s, took {took:.2f}s"

    # 429 on one key: fail over to the other key, cool the limited slot for Retry-After
    async def limit_a(s, prompt, timeout):
        if s.key == "gA":
            raise http_error(429, retry_after=7)
        return fake_answer(prompt)
    with_fake_keys(behaviour=limit_a)
    entries, source = asyncio.run(llm.interpret(["Keep at least 120 kWh in reserve from 6 PM until 9 PM."]))
    assert source.startswith("groq") and entries[0]["directive_type"] == "minimum_battery_reserve"
    a = next(s for s in llm.slots() if s.key == "gA")
    assert 5 < a.cool_until - time.monotonic() <= 7.1, "429 must honour Retry-After"

    # 401 on every groq key: kill them, fall to gemini (priority order holds)
    async def groq_dead(s, prompt, timeout):
        if s.provider == "groq":
            raise http_error(401)
        return fake_answer(prompt)
    with_fake_keys(behaviour=groq_dead)
    _, source = asyncio.run(llm.interpret(["Do not discharge the battery from 6 PM to 9 PM."]))
    assert source.startswith("gemini"), source
    assert all(s.cool_until == float("inf") for s in llm.slots() if s.provider == "groq")

    # everything broken: regex net, no crash
    async def all_down(s, prompt, timeout):
        raise http_error(503)
    with_fake_keys(behaviour=all_down)
    entries, source = asyncio.run(llm.interpret(["anything at all"]))
    assert source == "rules" and entries == []

    # coalescing + cache: 30 concurrent identical requests make exactly one LLM call
    calls = []
    async def counted(s, prompt, timeout):
        calls.append(s.key)
        await asyncio.sleep(0.1)
        return fake_answer(prompt)
    with_fake_keys(behaviour=counted)
    note = ["Solar output will drop to about 20% from 1 PM to 3 PM."]
    async def burst():
        return await asyncio.gather(*(llm.interpret(note) for _ in range(30)))
    results = asyncio.run(burst())
    assert len(calls) == 1, f"coalescing failed: {len(calls)} calls"
    assert all(r[0][0]["directive_type"] == "solar_reduction" for r in results)
    assert asyncio.run(llm.interpret(note))[1] == "cache"
    # the cache is per note: a new combination of seen notes costs nothing
    assert asyncio.run(llm.interpret(note + note))[1] == "cache"

    # load spreading: distinct concurrent requests fan out across both groq keys
    calls.clear()
    async def spread():
        return await asyncio.gather(*(llm.interpret([f"Do not charge the battery from {h} AM to {h + 1} AM."])
                                      for h in range(1, 11)))
    asyncio.run(spread())
    assert {"gA", "gB"} <= set(calls), f"load did not spread across keys: {calls}"

    # budgets refill continuously: a spent slot is usable again in seconds, not a minute
    assert llm._duration("11m31.2s") == 691.2 and llm._duration("250ms") == 0.25
    b = llm.Bucket(2000)
    now = time.monotonic()
    b.take(900, now)
    b.take(900, now)
    assert 20.5 < b.wait(900, now) < 21.5, "700 tokens short at 2000/min is a 21s wait"
    assert b.wait(900, now + 22) == 0, "and affordable once refilled"

    # provider headers are the truth: a low remaining count delays the slot proportionally
    s = llm.slots()[0]
    s.cool_until, s.inflight, s.need = 0.0, 0, 1200
    s.tok = llm.Bucket(8000)
    llm._observe(s, httpx.Response(200, headers={"x-ratelimit-limit-tokens": "8000",
                                                 "x-ratelimit-remaining-tokens": "900"}))
    assert 2.0 < s.wait(time.monotonic()) < 2.5, "300 tokens short at 8000/min is ~2.25s"
    s.tok = llm.Bucket(8000)
    llm._observe(s, httpx.Response(200, headers={"x-ratelimit-remaining-tokens": "7000"}))
    assert s.ready(time.monotonic()), "plenty of budget must not delay the slot"

    # a 429 naming a hidden cap teaches it to every key of that model
    body = ("Rate limit reached for model `fast-model` on output tokens per minute "
            "(OTPM): Limit 1000, Used 1000, Requested 236.")
    llm._penalize(s, httpx.Response(429, text=body))
    assert all(x.out.cap == 1000 for x in llm.slots() if x.model == "fast-model"), "OTPM learned for all keys"
    assert all(x.out.cap == float("inf") for x in llm.slots() if x.model != "fast-model")

    # gemini's 429 carries its quota and retry delay in the body (verbatim shape from the API)
    g = next(x for x in llm.slots() if x.provider == "gemini")
    gemini_429 = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
            "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "15"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "13s"}]}}
    g.cool_until = 0.0
    llm._penalize(g, httpx.Response(429, json=gemini_429))
    assert g.req.cap == 15, "gemini RPM learned from quotaValue"
    assert 12 < g.cool_until - time.monotonic() <= 13.1, "gemini retryDelay honoured"

    # routing waits briefly for a fast slot instead of taking a much slower provider now
    for x in llm.slots():
        x.cool_until, x.inflight = 0.0, 0
        x.tok, x.out, x.req = llm.Bucket(8000), llm.Bucket(float("inf")), llm.Bucket(30)
    groq = [x for x in llm.slots() if x.provider == "groq"]
    gem = next(x for x in llm.slots() if x.provider == "gemini")
    for x in groq:
        x.lat = 0.6
        x.tok.level = x.need - 8000 / 60 * 0.5  # affordable again in ~0.5s
    gem.lat = 4.0
    pick, wait = llm._pick(set(), left=10)
    assert pick.provider == "groq" and 0.3 < wait < 0.7, (pick, wait)
    gem.lat = 1.5
    for x in groq:
        x.tok.level = x.need - 8000 / 60 * 3  # groq busy for ~3s: gemini now answers sooner
    pick, wait = llm._pick(set(), left=10)
    assert pick is gem and wait == 0, (pick, wait)

    llm.HEDGE_AFTER_S = 1.5
    restore(real_call)


def main():
    offline()

    # --- regex net -------------------------------------------------------------
    for text, kind, hours, value in PARAPHRASES:
        d = llm.rule_parse(0, text)
        assert d["directive_type"] == kind, (text, d)
        if hours:
            assert d["structured_adjustment"]["hours"] == hours, (text, d)
        if value is not None:
            adj = d["structured_adjustment"]
            got = adj.get("factor", adj.get("minimum_energy_kwh"))
            assert abs(got - value) < 1e-6, (text, d)

    # --- guardrails --------------------------------------------------------------
    junk = [
        {"note_index": 0, "directive_type": "make_it_cheaper", "structured_adjustment": {"hours": [1]}},
        {"note_index": 0, "directive_type": "solar_reduction", "structured_adjustment": {"hours": [99, 99], "factor": 0.5}},
        {"note_index": 0, "directive_type": "solar_reduction", "structured_adjustment": None},
        {"note_index": 0, "directive_type": "solar_reduction", "structured_adjustment": {"hours": [3], "factor": -1}},
        {"note_index": 0, "directive_type": "max_grid_window", "structured_adjustment": {"hours": [2], "max_grid_kwh": float("nan")}},
        {"note_index": 0, "directive_type": "no_charge_window", "structured_adjustment": {"hours": [13.5]}},
        "not even an object",
    ]
    for e in junk:
        assert api.normalize(e, 0, 500.0)["directive_type"] == "no_op", e

    def adj(entry):
        return api.normalize(entry, 0, 500.0)["structured_adjustment"]

    assert adj({"directive_type": "solar_reduction",
                "structured_adjustment": {"hours": [14, 13, 13, "14"], "factor": 1.0}}) == {"hours": [13, 14], "factor": 1.0}
    # out-of-range values are rejected, never clamped or rescaled into a directive
    assert api.normalize({"directive_type": "solar_reduction",
                          "structured_adjustment": {"hours": [13], "factor": 20}}, 0, 500.0)["directive_type"] == "no_op"
    assert api.normalize({"directive_type": "minimum_battery_reserve",
                          "structured_adjustment": {"hours": [3], "minimum_energy_kwh": 9999}}, 0, 500.0)["directive_type"] == "no_op"
    assert adj({"directive_type": "no_charge_window",
                "structured_adjustment": {"start_hour": 22, "end_hour": 2}})["hours"] == [0, 1, 22, 23], "wraps midnight"

    filled = api.interpret_notes([], ["Do not charge the battery between 2 PM and 4 PM.", "Hi."], 500.0)
    assert [d["directive_type"] for d in filled] == ["no_charge_window", "no_op"], "missing notes filled by rules"
    assert len(api.interpret_notes([{"note_index": 0, "directive_type": "no_op"}] * 2, ["n"], 500.0)) == 1

    # --- HTTP contract ------------------------------------------------------------
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200
    assert "slots" in client.get("/stats").json()
    bad = client.post("/optimize-energy", content=b'{"broken', headers={"content-type": "application/json"})
    assert bad.status_code == 400, bad.status_code
    short = scenario(["x"])
    short["hours"] = short["hours"][:23]
    assert client.post("/optimize-energy", json=short).status_code == 400, "23 hours is structural"
    assert client.post("/optimize-energy", json={"scenario_id": "x"}).status_code == 400
    nan = json.dumps(scenario(["x"])).replace('"demand_kwh": 180.0', '"demand_kwh": NaN')
    assert client.post("/optimize-energy", content=nan, headers={"content-type": "application/json"}).status_code == 400
    assert client.post("/optimize-energy", json=scenario(["x"], initial_energy_kwh=900.0)).status_code == 400
    assert client.post("/optimize-energy", json=scenario(["x"], minimum_energy_kwh=600.0)).status_code == 400

    # --- end-to-end, each replayed against the judge -------------------------------
    base = run(scenario(["Nothing unusual is planned for tomorrow."]))
    assert kinds(base) == ["no_op"]
    m0 = solver.build_model(scenario(["x"]), [])
    assert base["total_cost_bdt"] < solver.totals(m0, solver.idle_plan(m0))["total_cost_bdt"], \
        "the optimiser must beat holding the battery idle"

    solar = run(scenario(["Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window."]))
    assert kinds(solar) == ["solar_reduction"] and solar["total_cost_bdt"] > base["total_cost_bdt"]

    nocharge = run(scenario(["Do not charge the battery between 2 AM and 5 AM."]))
    assert kinds(nocharge) == ["no_charge_window"]
    assert all(p["battery_action"] != "charge" for p in nocharge["hourly_plan"] if p["hour"] in (2, 3, 4))

    nodis = run(scenario(["Battery discharging is unavailable from 6 PM to 9 PM."]))
    assert kinds(nodis) == ["no_discharge_window"]
    assert all(p["battery_action"] != "discharge" for p in nodis["hourly_plan"] if p["hour"] in (18, 19, 20))

    reserve = run(scenario(["Keep at least 300 kWh in reserve from 6 PM until 9 PM."]))
    assert kinds(reserve) == ["minimum_battery_reserve"]
    assert all(p["battery_energy_after_kwh"] >= 300 - solver.TOL
               for p in reserve["hourly_plan"] if p["hour"] in (18, 19, 20))

    cap = run(scenario(["Grid import must not exceed 300 kWh during the 6 PM to 9 PM peak."]))
    assert kinds(cap) == ["max_grid_window"]
    assert all(p["grid_kwh"] <= 300 + solver.TOL for p in cap["hourly_plan"] if p["hour"] in (18, 19, 20))

    stacked = run(scenario([
        "Solar output will drop to about 20% from 1 PM to 3 PM.",
        "Do not charge the battery between 2 PM and 4 PM.",
        "The cafeteria menu changes tomorrow.",
    ]))
    assert kinds(stacked) == ["solar_reduction", "no_charge_window", "no_op"]

    night = scenario(["Routine night shift, nothing to report."], initial_energy_kwh=50.0)
    for h in night["hours"]:
        h["solar_kwh"] = 0.0
    run(night)

    frozen = scenario(["Nothing to report."], max_charge_kwh_per_hour=0.0, max_discharge_kwh_per_hour=0.0)
    assert all(p["battery_action"] == "idle" for p in run(frozen)["hourly_plan"])

    assert client.post("/optimize-energy", json=scenario(["   "])).status_code == 400  # blank note
    run(scenario(["x" * 50000]))                                                    # huge note, no blow-up

    # infeasible as interpreted: the ladder degrades instead of breaking
    hard = run(scenario(["Grid import must not exceed 1 kWh from 6 PM to 9 PM.",
                         "Do not discharge the battery from 6 PM to 9 PM."]), strict=False)
    assert "relaxed" in hard["plan_summary"], hard["plan_summary"]

    # --- scheduler (fake transport) ------------------------------------------------
    test_scheduler()

    # --- organiser sample cases, if present --------------------------------------
    sample_file = next((f for f in ("samples.json", "sample.json") if os.path.isfile(f)), None)
    if sample_file:
        raw = json.load(open(sample_file))
        cases = (raw if isinstance(raw, list) else [raw] if "hours" in raw
                 else raw.get("cases", raw.get("scenarios", [])))
        for c in cases:
            run(c if "hours" in c else c["request"])
        print(f"{sample_file}: {len(cases)} sample cases pass")

    print("all offline checks pass")

    if "--live" in sys.argv:
        live()


def live():
    """Real keys from .env: paraphrases must come back right from an actual model."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
    llm.reset()
    cases = PARAPHRASES + [  # needs battery_capacity_kwh from the prompt
        ("Keep the battery at least half full from 6 PM to 9 PM.", "minimum_battery_reserve", [18, 19, 20], 250),
    ]
    notes = [p[0] for p in cases]
    t = time.perf_counter()
    entries, source = asyncio.run(llm.interpret(notes, 500.0))
    took = time.perf_counter() - t
    got = {e["note_index"]: api.normalize(e, e["note_index"], 500.0) for e in entries}
    wrong = []
    for i, (text, kind, hours, value) in enumerate(cases):
        d = got.get(i)
        adj = (d or {}).get("structured_adjustment") or {}
        v = adj.get("factor", adj.get("minimum_energy_kwh", adj.get("max_grid_kwh")))
        if (not d or d["directive_type"] != kind or (hours and adj.get("hours") != hours)
                or (value is not None and (v is None or abs(v - value) > 1e-6))):
            wrong.append((text, d))
    print(f"live: {len(cases) - len(wrong)}/{len(cases)} correct via {source} in {took:.2f}s")
    for w in wrong:
        print("   wrong:", w)
    assert source != "rules", "no provider answered — check keys"


if __name__ == "__main__":
    main()
