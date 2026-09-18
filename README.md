# Smart Campus Energy Optimization

BUP CSE Fest 2026 preliminary round. One HTTP service that reads natural-language operator notes
with an LLM, turns them into validated directives, and returns a cost-optimal 24-hour
grid / solar / battery schedule.

```
POST /optimize-energy   scenario + operator notes  ->  interpretation + 24h plan
GET  /health            {"status": "ok"}
GET  /stats             per-key/model budgets, latency, health (keys masked)
GET  /                  browser UI
```

## Run

```bash
pip install -r requirements.txt
cp .env.example .env              # paste key pools
uvicorn app:app --port 8000 --workers 1
python test_app.py                # offline self-check, no keys needed
python test_app.py --live         # also checks your real keys end to end
```

## Keys and models

Comma-separated pools per provider, priority **groq → gemini → openrouter**:

```
GROQ_API_KEYS=gsk_a,gsk_b
GEMINI_API_KEYS=AIza_a
OPENROUTER_API_KEYS=sk-or-a
```

Every key × model pair is its own rate budget, so each added key adds throughput.
Default models were chosen by measurement (September 2026, 12 paraphrased notes, all 12/12 correct):

| Provider | Models, fastest first | Typical latency (3 notes) |
|---|---|---|
| groq | `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | 0.6–1.0 s |
| gemini | `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-flash-lite-latest` | 1.2–1.6 s |
| openrouter | `deepseek/deepseek-v4-flash-0731:free`, `qwen/qwen3.8-27b:free`, `google/gemma-4-31b-it:free` | 7–17 s |

`llama-3.3-70b-versatile` is no longer served by Groq (404). `gemini-3.8-flash` is listed first
as requested but was returning Google's "high demand" 503 during testing; the scheduler routes
around it. Override any list with `GROQ_MODELS` / `GEMINI_MODELS` / `OPENROUTER_MODELS`.

## How requests are routed

Each (provider, model, key) slot keeps continuously refilling budgets for requests, tokens and
output tokens, plus a measured latency. A request goes to the slot with the **earliest expected
answer**: time until it can afford the call + its latency + a 1 s bias per provider tier, so
priority holds unless a lower tier is clearly faster right now.

| Mechanism | Effect |
|---|---|
| Per-note LRU cache (keyed with battery capacity) | a note seen before costs nothing |
| Request coalescing | 50 simultaneous identical requests make one LLM call |
| Budget booking at dispatch | a burst spreads across every key and model instead of piling onto one |
| Provider headers + 429 bodies | real limits learned at runtime: Groq's 8K tokens/min, qwen's hidden 1K output tokens/min, Gemini's 15 req/min and `retryDelay` |
| Hedging | if an attempt is slow, a second free slot races it; first valid answer wins |
| Pooled HTTP/2 client, warmed at startup | no TLS handshake per call; rejected keys are dropped before traffic arrives |
| Health routing | 401/403 kills a key, 429 cools one slot, 404/5xx/timeouts cool the model |
| Regex parser | last resort if every provider is down; the service always answers |

## Correctness

- The LLM output is untrusted: `app.normalize` enforces the schema, clamps values, fixes
  percentage-vs-fraction factors, and degrades anything malformed to `no_op`.
- The schedule is an exact LP (scipy HiGHS): the global cost optimum, not a heuristic.
- Every plan is replayed against the judge's section 11.3 rules (`solver.validate_plan`) before
  it is returned. Infeasible directive combinations relax one rung at a time; the floor is a
  battery-idle plan that is always valid.
- Malformed JSON or structurally invalid body → 400. Impossible battery → 422. NaN/Infinity → 400.

## Measured (single worker, Windows dev box, real keys)

| Scenario | Result |
|---|---|
| Sequential requests, every note new | p50 680 ms, p90 1.4 s |
| 30 simultaneous requests, all notes new | all 200, p50 0.95 s, p95 1.5 s |
| 50 simultaneous identical new requests | one LLM call, p50 0.59 s |
| Repeated scenario (cache + solve memo) | ~1,100 req/s in-process, p50 0.8 ms |
| Unique scenarios, LP-bound | ~170 req/s in-process, p50 5 ms |

The ceiling under sustained load is LLM quota, not CPU: two Groq keys give roughly 50 fresh
calls/minute before overflow goes to Gemini. Add keys to raise it.

## Deploy

**Render:** push the repo, use `render.yaml`, set the three key variables in the dashboard.
**Docker:** `docker build -t campus-energy . && docker run -p 8000:8000 --env-file .env campus-energy`
(`.dockerignore` keeps `.env` out of the image).

Keep **one worker**: rate budgets, the cache and coalescing live in-process, and LLM quota is
the bottleneck. If you raise `WEB_CONCURRENCY`, each worker gets 1/N of every budget.
The Render free tier sleeps when idle; ping `/health` every few minutes during judging.

## Files

| File | Job |
|---|---|
| `app.py` | endpoints, request schema, LLM guardrails |
| `llm.py` | slot scheduler, provider adapters, cache, coalescing, regex fallback |
| `solver.py` | LP, drift repair, validator, relaxation ladder, solve memo |
| `test_app.py` | offline + `--live` checks |
| `static/index.html` | browser UI |
