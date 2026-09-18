"""Smart Campus Energy Optimization API.

LLM reads the operator notes, guard.py checks what it said, the LP does the maths, and the
finished plan is validated against the judge's own rules before it leaves.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import guard
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


# --- request schema (section 07) ----------------------------------------------
# Strict numbers: true/false or "180" are not kWh. Any violation is a 400 (section 6.1).

Num = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNeg = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]
Note = Annotated[str, Field(strict=True)]


class HourIn(BaseModel):
    hour: Annotated[int, Field(strict=True, ge=0, le=23)]
    demand_kwh: NonNeg
    solar_kwh: NonNeg
    tariff_bdt_per_kwh: Num


class BatteryIn(BaseModel):
    capacity_kwh: Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0)]
    initial_energy_kwh: NonNeg
    minimum_energy_kwh: NonNeg
    max_charge_kwh_per_hour: NonNeg
    max_discharge_kwh_per_hour: NonNeg

    @model_validator(mode="after")
    def _consistent(self):
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh exceeds capacity_kwh")
        if not self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh:
            raise ValueError("initial_energy_kwh must lie between minimum_energy_kwh and capacity_kwh")
        return self


class ScenarioIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=False)

    scenario_id: Annotated[str, Field(strict=True)]
    operator_notes: Annotated[list[Note], Field(min_length=1, max_length=3)]
    hours: list[HourIn]
    battery: BatteryIn

    @field_validator("operator_notes")
    @classmethod
    def _non_empty(cls, v):
        if any(not n.strip() for n in v):
            raise ValueError("operator_notes must be non-empty strings")
        return v

    @field_validator("hours")
    @classmethod
    def _full_day(cls, v):
        if sorted(h.hour for h in v) != list(range(solver.H)):
            raise ValueError("hours must contain each hour 0..23 exactly once")
        return v


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError):
    errors = [{"loc": [str(x) for x in e.get("loc", ())], "msg": str(e.get("msg", ""))}
              for e in exc.errors()]
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


# --- interpretation ------------------------------------------------------------

normalize = guard.normalize


def interpret_notes(raw, notes, capacity):
    """Exactly one guardrail-clean entry per note, in note_index order.

    A note the LLM did not answer usably goes to the regex fallback, and failing that to
    no_op. Nothing is ever clamped or guessed into a directive.
    """
    by_index = {}
    for e in raw or []:
        if isinstance(e, dict):
            i = guard._finite(e.get("note_index"))
            if i is not None and i == int(i) and 0 <= int(i) < len(notes) and int(i) not in by_index:
                by_index[int(i)] = e
    out = []
    for i, note in enumerate(notes):
        v = guard.validate(by_index.get(i), capacity)
        if v is None:
            v = guard.validate(llm.rule_parse(i, note, capacity), capacity)
        if v is not None and not guard.grounded(note, v):
            log.warning("vetoed %s on a note with no energy content", v["directive_type"])
            v = None
        out.append({"note_index": i, **v} if v else guard.no_op(i))
    return out


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
        # still honour the directives we can (reduced solar, blocked windows) in the idle plan
        try:
            model = solver.build_model(scenario, directives)
        except Exception:
            model = solver.build_model(scenario, [])
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
