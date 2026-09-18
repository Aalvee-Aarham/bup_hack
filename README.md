# Smart Campus Energy Optimization — BUP CSE Fest 2026 Preliminary

An HTTP API that reads campus operators' natural-language notes with an LLM, turns them into
validated directives, and returns the cost-optimal 24-hour grid / solar / battery schedule
that obeys them.

```
operator notes ──► LLM (Groq → Gemini → OpenRouter) ──► guard.py (section 08 guardrails)
                                                             │ invalid → next model → regex fallback → no_op
                                                             ▼
scenario ─────────────────────────────────────────► solver.py: exact LP (scipy HiGHS)
                                                             │ replayed against section 11.3 before return
                                                             ▼
                                        directive_interpretation + hourly_plan + totals
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"status":"ok"}`, ready ~2 s after start (no LLM call) |
| `POST /optimize-energy` | the contract in the Problem Statement, sections 07 and 10 |
| `GET /` | small browser UI (optional) |
| `GET /stats` | key/model budgets and health, keys masked (optional) |

---

## 1. Local quickstart (clean machine)

Requires Python 3.11+ (tested on 3.12 and 3.14) and git.

```bash
git clone https://github.com/Aalvee-Aarham/bup_hack.git
cd bup_hack
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # then paste your keys into .env (see section 3)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

In a second terminal:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/optimize-energy \
     -H "content-type: application/json" --data-binary @sample.json
```

`sample.json` is the Problem Statement's section 7.4 example. **Expected result:**

| Field | Expected |
|---|---|
| `directive_interpretation` | `solar_reduction` hours `[13,14]` factor `0.2`; `no_charge_window` hours `[14,15]`; `no_op` |
| `hourly_plan` | 24 entries, hours 0-23 |
| `total_cost_bdt` | `47925.0` (the LP optimum; any equally cheap valid plan is acceptable) |
| `total_grid_kwh` / `peak_grid_kwh` | `4559.0` / `369.0` |

The service also starts with **no keys at all**: it then answers from the regex fallback, which
is enough to verify startup and the contract, but the LLM is the intended interpreter.

## 2. Docker fallback image

Built and published by `.github/workflows/docker.yml` on every push to `main`, then smoke
tested (the published image must answer `/health` and `/optimize-energy`).

```bash
docker pull ghcr.io/aalvee-aarham/bup_hack:latest
docker run --rm -p 8000:8000 \
  -e GROQ_API_KEYS=... -e GEMINI_API_KEYS=... -e OPENROUTER_API_KEYS=... \
  ghcr.io/aalvee-aarham/bup_hack:latest
curl http://localhost:8000/health
```

- Exact tags: `:latest` and `:<commit-sha>` (use the SHA of the submitted commit).
- Port **8000**, bound to `0.0.0.0`; `PORT` overrides it.
- The image contains **no credentials** (`.dockerignore` excludes `.env`); pass keys with `-e`
  or `--env-file .env`.
- Build locally instead: `docker build -t campus-energy . && docker run --rm -p 8000:8000 --env-file .env campus-energy`

## 3. Configuration

| Variable | Required | Meaning |
|---|---|---|
| `GROQ_API_KEYS` | recommended | comma-separated Groq keys (primary provider) |
| `GEMINI_API_KEYS` | optional | comma-separated Google AI Studio keys (second tier) |
| `OPENROUTER_API_KEYS` | optional | comma-separated OpenRouter keys (third tier) |
| `GROQ_MODELS` / `GEMINI_MODELS` / `OPENROUTER_MODELS` | optional | override model lists, most accurate first |
| `LLM_TOTAL_BUDGET_S` | optional | whole LLM cascade per request, default 15 |
| `LLM_ATTEMPT_TIMEOUT_S` | optional | one model call, default 8; a slow call is retried elsewhere |
| `PORT` | optional | listen port, default 8000 |
| `WEB_CONCURRENCY` | optional | keep at 1 (see Limitations) |

Singular names (`GROQ_API_KEY`, …) are also read. `.env.example` lists every knob.

## 4. Models and the LLM's role

The LLM is the **primary interpreter** of every operator note: all notes of a request go to
one model call (temperature 0, JSON mode, ~480-token prompt), and its structured answer is
what the optimizer applies.

| Provider (priority) | Models, most accurate first |
|---|---|
| Groq | `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b` |
| Google Gemini | `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-flash-lite-latest` |
| OpenRouter | `deepseek/deepseek-v4-flash-0731:free`, `qwen/qwen3.8-27b:free`, `google/gemma-4-31b-it:free` |

Model order comes from a measured run of the live paraphrase suite (`test_live.py`):
`qwen3.8-27b` answered 65/65 notes correctly. Each (provider, model, key) is a separate rate
budget; requests go to the slot expected to answer soonest, with a bias that keeps the
priority above. Failed or slow calls are retried on another slot; identical notes are cached
and concurrent identical requests share one call.

A deterministic regex parser (`llm.rule_parse`) runs **only** for notes no model answered
usably. It never replaces the LLM path, and anything it is unsure about becomes `no_op`.

## 5. Guardrails (`guard.py`, Problem Statement section 08)

LLM output is untrusted. Each entry must pass `guard.validate`, or it is discarded and that
note is retried on another model, then the regex fallback, then `no_op`:

- `directive_type` must be one of the six supported types; `note_index` must map to an
  existing note exactly once.
- `hours`: unique ascending integers 0-23 (unsorted or duplicated lists are normalised;
  fractional or non-numeric hours are rejected).
- `factor` in [0, 1]; reserve in [0, capacity]; grid cap finite and ≥ 0. Out-of-range values
  are **rejected, never clamped** — a factor of 5 is not silently turned into anything.
- `applies` is derived from the type; `structured_adjustment` is rebuilt in the exact shape.
- A directive on a note with no energy vocabulary at all (e.g. a hallucination on
  "The cafeteria menu changes tomorrow.") is vetoed to `no_op`.
- Notes are treated as data; the prompt instructs the model to ignore instructions inside them.

## 6. Optimizer (`solver.py`)

An exact linear program over 96 variables (grid, solar used, charge, discharge per hour),
solved with scipy's HiGHS: energy balance, battery bounds and rate limits, effective solar,
end-of-day neutrality, and all five directive types as constraints or bounds. The LP result is
rounded onto an exactly consistent schedule and **replayed against the section 11.3 rules**
before it is returned; totals are recomputed from the returned plan. If interpreted directives
are infeasible together, they are relaxed one rung at a time; the last resort is a
battery-idle plan. Identical scenarios are memoised.

## 7. Tests

```bash
python test_app.py            # contract, guardrails, scheduler, sample.json (offline, no keys)
python test_judge.py          # independent judge suite (offline, no keys)
python test_live.py           # real keys: paraphrases, distractors, injection, load
python test_live.py https://api-production-c4f7.up.railway.app   # same, against a deployment
```

`judge.py` is an independent validator written from the Problem Statement only (it imports
nothing from the service). It replays a response against **ground-truth** directives and
computes the organizer-style optimal cost with a differently formulated LP.

Latest results:

| Suite | Result |
|---|---|
| Bad requests (45 cases) | all 400/404/405, JSON bodies, no stack traces |
| LLM failures (32 cases: timeouts, bad JSON, unknown types, factor > 1, bad indices, 401/429/5xx, injections) | 200, never an invented or wrong directive |
| Optimizer edge cases (26) | valid against ground truth, optimal |
| Optimality (60 random scenarios) | cost ratio 1.000000 vs independent LP; LP confirmed by an exact DP |
| Live interpretation (65 notes, `qwen3.8-27b`) | 65/65; 29/29 requests valid downstream |
| Live, default routing under 4-way concurrency | 63-64/65 |
| Sequential requests with new notes | p50 1.2 s, p95 3.0 s, 0 failures |

## 8. Known limitations

- Throughput with fresh notes is bound by LLM quota: two free Groq keys sustain about 50
  new-note requests per minute (Gemini absorbs overflow). Bursts beyond that queue for up to
  15 s, then fall back to the regex parser. More keys scale linearly.
- `qwen3.8-27b` has a 1,000 output-tokens/minute cap per key on Groq's free tier; overflow goes
  to `gpt-oss-120b`, which is slightly less accurate on unusual time phrasings.
- Rate budgets, cache and coalescing live in-process: run one worker. With `WEB_CONCURRENCY=N`
  each worker gets 1/N of every budget.
- Genuinely ambiguous notes ("through 3 PM", "overnight") follow the model's reading of the
  start-inclusive / end-exclusive rule.

## 9. Secret handling

- Keys are read from environment variables only; `.env` is git- and docker-ignored.
- Logs and `/stats` show at most the last 4 characters of a key; error responses never include
  stack traces, prompts or keys.
- The Docker image contains no credentials.

## 10. Files and credits

| File | Job |
|---|---|
| `app.py` | FastAPI app, request validation, pipeline |
| `guard.py` | section 08 guardrails |
| `llm.py` | provider cascade, rate budgets, cache, regex fallback |
| `solver.py` | LP, repair, replay, relaxation ladder |
| `judge.py`, `test_*.py` | independent judge and test suites |
| `static/index.html` | browser UI |

Built with [FastAPI](https://fastapi.tiangolo.com), [Uvicorn](https://www.uvicorn.org),
[Pydantic](https://docs.pydantic.dev), [HTTPX](https://www.python-httpx.org),
[SciPy / HiGHS](https://scipy.org), [NumPy](https://numpy.org), [orjson](https://github.com/ijl/orjson)
and [python-dotenv](https://github.com/theskumar/python-dotenv); LLMs via Groq, Google Gemini
and OpenRouter. Developed with AI coding assistance (Claude Code).
