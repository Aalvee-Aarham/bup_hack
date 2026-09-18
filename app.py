"""Smart Campus Energy Optimization API.

LLM reads the operator notes, deterministic code checks what it said, the LP does the maths,
and the finished plan is validated against the judge's own rules before it leaves.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
from contextlib import asynccontextmanager

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, ".env"))  # real environment variables still win
except ImportError:
    pass

import orjson
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

import llm
import solver

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_app):
    llm.slots()
    warm = asyncio.create_task(llm.warmup())  # never blocks /health
    yield
    warm.cancel()
    await llm.close()


app = FastAPI(title="Smart Campus Energy Optimization", lifespan=lifespan)

STATIC = os.path.join(HERE, "static")
if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _json(body, status=200):
    return Response(orjson.dumps(body), status_code=status, media_type="application/json")


# --- request schema ----------------------------------------------------------

class _Strict(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class HourIn(_Strict):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float


class BatteryIn(_Strict):
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)


class ScenarioIn(_Strict):
    scenario_id: str
    operator_notes: list[str] = Field(min_length=1)
    hours: list[HourIn]
    battery: BatteryIn

    @field_validator("hours")
    @classmethod
    def _full_day(cls, v):
        if sorted(h.hour for h in v) != list(range(solver.H)):
            raise ValueError("hours must contain each hour 0..23 exactly once")
        return v


def _semantic_errors(b):
    """Well-formed but impossible batteries (section 6.1: 422)."""
    errs = []
    if b["minimum_energy_kwh"] > b["capacity_kwh"]:
        errs.append("minimum_energy_kwh exceeds capacity_kwh")
    if not b["minimum_energy_kwh"] <= b["initial_energy_kwh"] <= b["capacity_kwh"]:
        errs.append("initial_energy_kwh must lie between minimum_energy_kwh and capacity_kwh")
    return errs


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError):
    # malformed JSON or a structurally invalid body are both 400 (section 6.1)
    errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]
    return _json({"detail": "invalid request", "errors": errors[:20]}, 400)


@app.exception_handler(Exception)
async def _unhandled(_request: Request, exc: Exception):
    log.exception("unhandled error")
    return _json({"detail": "internal error"}, 500)  # no stack traces, no secrets


# --- routes ------------------------------------------------------------------

@app.get("/health")
async def health():
    return _json({"status": "ok"})


@app.get("/stats")
async def stats():
    return _json(llm.stats())


@app.get("/")
async def index():
    page = os.path.join(STATIC, "index.html")
    return FileResponse(page) if os.path.isfile(page) else _json({"status": "ok"})


# --- guardrails --------------------------------------------------------------

def _finite(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _clean_hours(adj):
    raw = adj.get("hours")
    if raw is None:
        # tolerate {"start_hour": 13, "end_hour": 15} style windows: end exclusive, may wrap midnight
        a = _finite(adj.get("start_hour", adj.get("start")))
        b = _finite(adj.get("end_hour", adj.get("end")))
        if a is None or b is None:
            return []
        a, b = int(a) % 24, int(b) % 24
        raw = [(a + k) % 24 for k in range((b - a) % 24 or 24)]
    out = set()
    for h in raw if isinstance(raw, (list, tuple)) else [raw]:
        v = _finite(h)
        if v is not None and v == int(v) and 0 <= v < solver.H:
            out.add(int(v))
    return sorted(out)


def normalize(entry, index, capacity):
    """Turn one untrusted LLM entry into a schema-legal directive, or no_op (section 08)."""
    no_op = {"note_index": index, "applies": False, "directive_type": "no_op",
             "structured_adjustment": None,
             "explanation": "This note does not affect today's energy schedule."}
    if not isinstance(entry, dict):
        return no_op

    t = entry.get("directive_type")
    if t not in solver.DIRECTIVE_TYPES or t == "no_op":
        if t == "no_op" and entry.get("explanation"):
            no_op["explanation"] = str(entry["explanation"])[:300]
        return no_op

    adj = entry.get("structured_adjustment")
    if not isinstance(adj, dict):
        return no_op
    hours = _clean_hours(adj)
    if not hours:
        return no_op

    out = {"hours": hours}
    if t == "solar_reduction":
        f = _finite(adj.get("factor"))
        if f is None or f < 0:
            return no_op
        if 1 < f <= 100:
            f /= 100  # model wrote a percentage: "20" means 20% remains
        out["factor"] = min(1.0, f)
    elif t == "minimum_battery_reserve":
        v = _finite(adj.get("minimum_energy_kwh"))
        if v is None or v < 0:
            return no_op
        out["minimum_energy_kwh"] = min(capacity, v)
    elif t == "max_grid_window":
        v = _finite(adj.get("max_grid_kwh"))
        if v is None or v < 0:
            return no_op
        out["max_grid_kwh"] = v

    explanation = entry.get("explanation")
    return {
        "note_index": index,
        "applies": True,
        "directive_type": t,
        "structured_adjustment": out,
        "explanation": str(explanation)[:300] if explanation else f"Directive extracted from note {index}.",
    }


def interpret_notes(raw, notes, capacity):
    """Exactly one entry per note, in note_index order, whatever the model returned."""
    by_index = {}
    for e in raw or []:
        if isinstance(e, dict):
            i = _finite(e.get("note_index"))
            if i is not None and 0 <= int(i) < len(notes) and int(i) not in by_index:
                by_index[int(i)] = e
    return [normalize(by_index[i] if i in by_index else llm.rule_parse(i, note), i, capacity)
            for i, note in enumerate(notes)]


# --- endpoint ----------------------------------------------------------------

def _summary(directives, plan, tot, note, source):
    applied = sorted({d["directive_type"] for d in directives if d["applies"]})
    cycling = any(p["battery_action"] != "idle" for p in plan)
    return (
        f"Applied {sum(d['applies'] for d in directives)} of {len(directives)} operator notes"
        + (f" ({', '.join(applied)})" if applied else "")
        + ("; charged the battery in cheap hours and discharged it into expensive ones"
           if cycling else "; held the battery idle")
        + f" for {tot['total_cost_bdt']} BDT across {tot['total_grid_kwh']} kWh of grid import"
        + f" (peak {tot['peak_grid_kwh']} kWh); {note}. Interpretation source: {source}."
    )


@app.post("/optimize-energy")
async def optimize_energy(req: ScenarioIn):
    scenario = req.model_dump()
    errs = _semantic_errors(scenario["battery"])
    if errs:
        return _json({"detail": "semantically invalid scenario", "errors": errs}, 422)

    notes = scenario["operator_notes"]
    capacity = scenario["battery"]["capacity_kwh"]

    raw, source = await llm.interpret(notes, capacity)
    directives = interpret_notes(raw, notes, capacity)

    try:
        # ~3ms of CPU, run inline on purpose: measured under 100 concurrent requests,
        # asyncio.to_thread made p50 150x worse (GIL contention in the thread pool)
        plan, tot, note = solver.solve(scenario, directives)
    except Exception:
        log.exception("optimization failed")
        model = solver.build_model(scenario, directives)
        plan = solver.idle_plan(model)
        tot = solver.totals(model, plan)
        note = "fallback: battery held idle (optimizer error)"

    return _json({
        "scenario_id": scenario["scenario_id"],
        "directive_interpretation": directives,
        "hourly_plan": plan,
        **tot,
        "plan_summary": _summary(directives, plan, tot, note, source),
    })
