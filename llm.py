"""Operator-note interpretation over a pool of LLM "slots".

A slot is one (provider, model, key). Each slot carries continuously refilling budgets for
requests, tokens and output tokens (synced to provider headers, caps learned from 429 bodies)
and a measured latency. A request goes to the slot with the earliest expected answer:
time until it can afford the call + its latency + a per-tier bias that keeps the priority
groq -> gemini -> openrouter. So a burst fans out over every key and model, briefly waits
for a refilling groq slot rather than taking a 7s provider, and never spends into a 429 it
can predict.

Latency, cheapest first:
  per-note LRU cache    a note seen before costs nothing
  request coalescing    concurrent requests with the same new notes share one LLM call
  pooled HTTP/2 client  no TLS handshake per call; warmed at startup
  hedged attempts       a slow first slot gets raced by a free one; first valid answer wins
  health-aware routing  401 kills a key, 429 cools a slot, 404/5xx/timeouts cool a model

Under all of it sits a deterministic regex parser, so the service always answers.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import math
import os
import re
import time

import httpx

from solver import DIRECTIVE_TYPES

log = logging.getLogger("llm")


def _num(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


TOTAL_BUDGET_S = _num("LLM_TOTAL_BUDGET_S", 15)     # whole cascade, then the regex net
HEDGE_AFTER_S = _num("LLM_HEDGE_AFTER_S", 1.5)      # race a second slot after this long
MAX_PARALLEL = int(_num("LLM_MAX_PARALLEL", 2))     # attempts racing for one request
SLOT_MAX_INFLIGHT = int(_num("LLM_SLOT_MAX_INFLIGHT", 8))
PRIORITY_BIAS_S = _num("LLM_PRIORITY_BIAS_S", 1.0)  # seconds a lower provider tier must beat
# Rate state lives in-process. With N workers each one may only spend 1/N of a key's budget.
# One worker is the recommended setup: LLM quota, not CPU, is the bottleneck, and a single
# process shares the note cache and request coalescing across all traffic.
WORKERS = max(1, int(_num("WEB_CONCURRENCY", 1)))
CACHE_SIZE = int(_num("LLM_CACHE_SIZE", 20000))
MAX_NOTE_CHARS = 1000

PROVIDERS = [
    {
        "name": "groq",
        "kind": "openai",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "warm": "https://api.groq.com/openai/v1/models",
        "keys": ("GROQ_API_KEYS", "GROQ_API_KEY"),
        "models": ("GROQ_MODELS", ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]),
        "rpm": ("GROQ_RPM", 30),
        "tpm": ("GROQ_TPM", 8000),
        "lat": 0.9,  # seed for the latency estimate; replaced by measurements
    },
    {
        "name": "gemini",
        "kind": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "warm": "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
        "keys": ("GEMINI_API_KEYS", "GEMINI_API_KEY"),
        "models": ("GEMINI_MODELS",
                   ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]),
        "rpm": ("GEMINI_RPM", 10),
        "tpm": ("GEMINI_TPM", "inf"),
        "lat": 1.8,
    },
    {
        "name": "openrouter",
        "kind": "openai",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "warm": "https://openrouter.ai/api/v1/key",
        "keys": ("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY"),
        "models": ("OPENROUTER_MODELS", ["deepseek/deepseek-v4-flash-0731:free",
                                         "qwen/qwen3.8-27b:free", "google/gemma-4-31b-it:free"]),
        "rpm": ("OPENROUTER_RPM", 20),
        "tpm": ("OPENROUTER_TPM", "inf"),
        "lat": 8.0,
    },
]
_BY_NAME = {p["name"]: p for p in PROVIDERS}

# Measured against the old prompt with multi-turn few-shot: 45% fewer input tokens and the
# same 12/12 accuracy on every model. Tokens matter: groq's binding limit is tokens/minute.
SYSTEM = """Convert campus operator notes into energy directives. Reply with JSON only:
{"directives":[{"note_index":0,"applies":true,"directive_type":"...","structured_adjustment":{...},"explanation":"<=12 words"}]}
One entry per note, in note order.

directive_type -> structured_adjustment:
solar_reduction -> {"hours":[...],"factor":f}   f = fraction of solar that REMAINS, 0..1
minimum_battery_reserve -> {"hours":[...],"minimum_energy_kwh":x}
no_charge_window -> {"hours":[...]}
no_discharge_window -> {"hours":[...]}
max_grid_window -> {"hours":[...],"max_grid_kwh":x}
no_op -> null   (the note does not change today's electricity schedule)

Rules:
- hours: integers 0-23, ascending, start INCLUSIVE, end EXCLUSIVE.
  "1 PM to 3 PM"->[13,14]  "6 PM until 9 PM"->[18,19,20]  "13:00-15:00"->[13,14]
  "10 PM to 2 AM"->[0,1,22,23]  "at 5 PM"->[17]  "from one until three" (daytime)->[13,14]
- factor: "drops to 20%"->0.2  "80% reduction"->0.2  "one-fifth of normal"->0.2  "offline"->0.0
- Convert a share of the battery ("40% of capacity", "half full") to kWh using battery_capacity_kwh.
- applies=false with structured_adjustment=null ONLY for no_op; all other types applies=true.
- Never invent values or directive types."""


# --- slots -------------------------------------------------------------------

def _env_list(*names):
    out = []
    for name in names:
        for v in os.environ.get(name, "").split(","):
            v = v.strip()
            if v and v not in out:
                out.append(v)
    return out


class Bucket:
    """A per-minute budget that refills continuously, the way groq enforces its limits.

    level may go negative: that is debt from calls booked on estimates, repaid by refill.
    """

    __slots__ = ("cap", "level", "t")

    def __init__(self, cap):
        self.cap = self.level = float(cap) / WORKERS
        self.t = time.monotonic()

    def _refill(self, now):
        if now > self.t:
            self.level = min(self.cap, self.level + (now - self.t) * self.cap / 60)
            self.t = now

    def wait(self, n, now):
        """Seconds until n units are affordable: 0 if now."""
        if math.isinf(self.cap):
            return 0.0
        self._refill(now)
        return max(0.0, (min(n, self.cap) - self.level) * 60 / self.cap)

    def take(self, n, now):
        self._refill(now)
        self.level -= n

    def sync(self, remaining, now):
        """The provider's own count wins when lower: other clients may share the key."""
        self._refill(now)
        self.level = min(self.level, remaining)

    def set_cap(self, cap):
        self.cap = float(cap) / WORKERS
        self.level = min(self.level, self.cap)


class Slot:
    """One (provider, model, key): its budgets, health and observed latency."""

    __slots__ = ("provider", "kind", "url", "model", "key", "tier", "req", "tok", "out",
                 "need", "need_out", "lat", "inflight", "cool_until", "ok", "fail")

    def __init__(self, p, tier, model, key, rpm, tpm):
        self.provider, self.kind, self.url = p["name"], p["kind"], p["url"]
        self.model, self.key, self.tier = model, key, tier
        self.req, self.tok = Bucket(rpm), Bucket(tpm)
        self.out = Bucket(math.inf)       # output tokens/min: learned from the first 429 naming it
        self.need, self.need_out = 900, 250  # what one call costs; tracks real usage
        self.lat = p["lat"]               # seconds per successful call; tracks real latency
        self.inflight = 0
        self.cool_until = 0.0
        self.ok = self.fail = 0

    def wait(self, now):
        """Seconds until this slot can take another call; inf if it never will."""
        if self.inflight >= SLOT_MAX_INFLIGHT:
            return max(self.lat, self.cool_until - now)
        return max(self.cool_until - now, self.req.wait(1, now),
                   self.tok.wait(self.need, now), self.out.wait(self.need_out, now), 0.0)

    def ready(self, now):
        return self.wait(now) == 0

    def book(self, now):
        """Reserve budget synchronously, so a burst picking in the same tick sees it."""
        self.req.take(1, now)
        self.tok.take(self.need, now)
        self.out.take(self.need_out, now)
        self.inflight += 1

    def __repr__(self):
        return f"{self.provider}:{self.model}:...{self.key[-4:]}"


_slots: list[Slot] | None = None


def slots():
    """All slots in priority order, built from the environment on first use."""
    global _slots
    if _slots is None:
        _slots = []
        for pi, p in enumerate(PROVIDERS):
            keys = _env_list(*p["keys"])
            models = _env_list(p["models"][0]) or p["models"][1]
            rpm = max(1.0, _num(*p["rpm"]))
            tpm = _num(*p["tpm"])
            for mi, model in enumerate(models):
                for key in keys:
                    _slots.append(Slot(p, (pi, mi), model, key, rpm, tpm))
        log.info("llm slots: %s", ", ".join(map(repr, _slots)) or "none (regex net only)")
    return _slots


def _cool(pred, seconds, why):
    until = time.monotonic() + seconds
    hit = [s for s in slots() if pred(s)]
    for s in hit:
        s.cool_until = max(s.cool_until, until)
    if hit:
        log.warning("cooling %s for %s: %s", hit[0] if len(hit) == 1 else f"{len(hit)} slots",
                    "ever" if math.isinf(seconds) else f"{seconds:.0f}s", why)


def _gemini_details(resp):
    try:
        return resp.json()["error"].get("details") or []
    except Exception:
        return []


def _retry_after(resp):
    """Seconds to back off: the retry-after header (groq, openrouter) or gemini's RetryInfo."""
    try:
        return min(float(resp.headers.get("retry-after", "")), 3600.0)
    except ValueError:
        pass
    for d in _gemini_details(resp):
        if wait := _duration(d.get("retryDelay")):
            return min(wait, 3600.0)
    return None


def _duration(v):
    """Parse groq/openai reset strings: "18.697s", "11m31.2s", "1h2m3s", "250ms"."""
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", v or ""):
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total or None


def _learn(s, total, out):
    if total:
        s.need = max(300, int(0.7 * s.need + 0.3 * total))
    if out:
        s.need_out = max(50, int(0.7 * s.need_out + 0.3 * out))


# groq names the exhausted budget in the 429 body, e.g. "(OTPM): Limit 1000". Some caps
# (qwen's output tokens/minute) appear nowhere else, so learn them here for every key.
_LIMIT = re.compile(r"\((RPM|TPM|OTPM)\): Limit (\d+)")
_BUCKET_FOR = {"RPM": "req", "TPM": "tok", "OTPM": "out"}


def _set_limit(s, kind, limit):
    attr = _BUCKET_FOR[kind]
    for x in slots():
        if x.provider == s.provider and x.model == s.model and getattr(x, attr).cap != limit / WORKERS:
            getattr(x, attr).set_cap(limit)
            if x is s:
                log.warning("learned %s limit %d for every %s key", kind, limit, s.model)
    getattr(s, attr).level = 0.0  # the provider just told us this one is spent


def _learn_limit(s, resp):
    """Read which budget ran out from a 429 body. groq: "(OTPM): Limit 1000".
    gemini: QuotaFailure {"quotaId": "GenerateRequestsPerMinute...", "quotaValue": "15"}."""
    if m := _LIMIT.search(resp.text):
        _set_limit(s, m.group(1), float(m.group(2)))
        return
    for d in _gemini_details(resp):
        for v in d.get("violations", []):
            qid, value = v.get("quotaId", ""), _num_or_none(v.get("quotaValue"))
            if value is None:
                continue
            if "PerDay" in qid:  # resets at Pacific midnight; look again in an hour
                _cool(lambda x: x.key == s.key and x.model == s.model, 3600.0, f"daily quota ({qid})")
            elif "RequestsPerMinute" in qid:
                _set_limit(s, "RPM", value)
            elif "TokensPerMinute" in qid:
                _set_limit(s, "TPM", value)


def _num_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _observe(s, resp):
    """Sync a slot's budgets with the provider's rate-limit headers (groq, openrouter send them),
    so the next request is routed before a 429 costs a round trip."""
    h = resp.headers
    now = time.monotonic()
    try:
        if limit := h.get("x-ratelimit-limit-tokens"):
            s.tok.set_cap(float(limit))
        if (remaining := h.get("x-ratelimit-remaining-tokens")) is not None:
            s.tok.sync(float(remaining) / WORKERS, now)
        # groq's request counter is a daily quota, not a rate: only act when it is gone
        if (reqs := h.get("x-ratelimit-remaining-requests")) is not None and float(reqs) < 1:
            if wait := _duration(h.get("x-ratelimit-reset-requests")):
                s.cool_until = max(s.cool_until, now + wait)
    except ValueError:
        pass


def _penalize(s, resp):
    code, text = resp.status_code, resp.text[:300]
    same_key = lambda x: x.key == s.key                                   # noqa: E731
    same_model = lambda x: x.provider == s.provider and x.model == s.model  # noqa: E731
    if code in (401, 403) or (code == 400 and "API_KEY_INVALID" in text):
        _cool(same_key, math.inf, f"key rejected ({code})")
    elif code == 429:
        _learn_limit(s, resp)
        _cool(lambda x: x is s, _retry_after(resp) or 20.0, "rate limited")
    elif code == 404:
        _cool(same_model, 300.0, "model not found")
    elif code in (400, 422):
        _cool(same_model, 60.0, f"request rejected ({code}): {text[:120]}")
    elif code >= 500:
        _cool(same_model, 15.0, f"provider error {code}")
    else:
        _cool(lambda x: x is s, 5.0, f"http {code}")


# --- transport ---------------------------------------------------------------

_client: httpx.AsyncClient | None = None
_client_loop = None


def _http():
    """One pooled client per event loop: keep-alive and HTTP/2 multiplexing across requests."""
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        limits = httpx.Limits(max_connections=256, max_keepalive_connections=64, keepalive_expiry=120)
        try:
            _client = httpx.AsyncClient(http2=True, limits=limits)
        except ImportError:  # h2 not installed
            _client = httpx.AsyncClient(limits=limits)
        _client_loop = loop
    return _client


async def close():
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def _extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.M).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON in response")
    end = max(text.rfind("}"), text.rfind("]"))
    data = json.loads(text[start:end + 1])
    if isinstance(data, dict):
        data = data.get("directives") or data.get("interpretations") or data.get("results")
    if not isinstance(data, list):
        raise ValueError("response did not contain a directive list")
    return data


async def _call(s, prompt, timeout):
    t = httpx.Timeout(timeout, connect=min(3.0, timeout))
    if s.kind == "gemini":
        r = await _http().post(
            s.url.format(model=s.model),
            headers={"x-goog-api-key": s.key},
            timeout=t,
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                # no temperature: Gemini 3 is documented to degrade below its default
                "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 4096},
            },
        )
        _observe(s, r)
        r.raise_for_status()
        j = r.json()
        u = j.get("usageMetadata", {})
        _learn(s, u.get("totalTokenCount"), u.get("candidatesTokenCount"))
        parts = j["candidates"][0]["content"]["parts"]
        return _extract_json("".join(p.get("text", "") for p in parts if not p.get("thought")))

    body = {
        "model": s.model,
        "temperature": 0,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    if "gpt-oss" in s.model:  # reasoning model: low effort is plenty for extraction, and faster
        if s.provider == "groq":
            body["reasoning_effort"] = "low"
        else:
            body["reasoning"] = {"effort": "low"}
    r = await _http().post(s.url, headers={"Authorization": f"Bearer {s.key}"}, json=body, timeout=t)
    _observe(s, r)
    r.raise_for_status()
    j = r.json()
    u = j.get("usage", {})
    _learn(s, u.get("total_tokens"), u.get("completion_tokens"))
    return _extract_json(j["choices"][0]["message"]["content"])


def _complete(data, n):
    """Index the answer by note; None unless every note got a known directive type."""
    got = {}
    for e in data:
        if not isinstance(e, dict) or e.get("directive_type") not in DIRECTIVE_TYPES:
            continue
        try:
            i = int(e.get("note_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < n:
            got.setdefault(i, e)
    return got if len(got) == n else None


async def _attempt(s, prompt, n, deadline):
    t0 = time.monotonic()
    try:
        got = _complete(await _call(s, prompt, max(deadline - t0, 0.5)), n)
        if got is None:
            log.warning("%s returned an incomplete answer", s)
            s.fail += 1
        else:
            s.ok += 1
            s.lat = 0.7 * s.lat + 0.3 * (time.monotonic() - t0)
        return got
    except asyncio.CancelledError:
        raise
    except httpx.HTTPStatusError as e:
        _penalize(s, e.response)
    except httpx.TimeoutException:
        _cool(lambda x: x.provider == s.provider and x.model == s.model, 20.0, "timed out")
    except Exception as e:
        _cool(lambda x: x is s, 3.0, f"{type(e).__name__}: {str(e)[:120]}")
    s.fail += 1
    return None


def _release(s):
    s.inflight -= 1


def _pick(tried, left, ready_only=False):
    """The slot with the earliest expected answer: time until it can take the call plus its
    observed latency, plus PRIORITY_BIAS_S per provider tier so groq -> gemini -> openrouter
    holds whenever the estimates are close. Returns (slot, seconds to wait) or (None, None)."""
    now = time.monotonic()
    tried_models = {(s.provider, s.model) for s in tried}
    best, best_key, best_wait = None, None, None
    for s in slots():
        if s in tried:
            continue
        w = s.wait(now)
        if w >= left or (ready_only and w > 0):
            continue
        eta = w + s.lat + PRIORITY_BIAS_S * s.tier[0]
        k = (eta, (s.provider, s.model) in tried_models, s.tier, s.inflight)
        if best_key is None or k < best_key:
            best, best_key, best_wait = s, k, w
    return best, best_wait


async def _cascade(notes, capacity):
    """Race slots until one returns a complete answer or the budget runs out."""
    n = len(notes)
    prompt = json.dumps({"battery_capacity_kwh": capacity,
                         "notes": [{"note_index": i, "text": t} for i, t in enumerate(notes)]})
    deadline = time.monotonic() + TOTAL_BUDGET_S
    tried, running = set(), {}

    def launch(s):
        tried.add(s)
        # book now, synchronously: a burst picking in the same tick must see each other's load
        s.book(time.monotonic())
        task = asyncio.ensure_future(_attempt(s, prompt, n, deadline))
        task.add_done_callback(lambda _t, s=s: _release(s))  # runs even if cancelled before start
        running[task] = s

    try:
        while (left := deadline - time.monotonic()) > 0:
            if not running:
                s, wait = _pick(tried, left)
                if s is None:
                    break  # every remaining slot is dead or cannot start before the deadline
                if wait > 0:
                    # the best slot frees up soon: waiting beats a slower provider right now
                    await asyncio.sleep(min(wait, 2.0))
                    continue
                launch(s)
            done, _ = await asyncio.wait(set(running), timeout=min(HEDGE_AFTER_S, left),
                                         return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                s = running.pop(t)
                got = None if t.cancelled() else t.result()
                if got is not None:
                    return got, f"{s.provider}:{s.model}"
            # slow (hedge) or failed (replace): race another slot, but only one that is free now
            if running and len(running) < MAX_PARALLEL:
                s, _ = _pick(tried, deadline - time.monotonic(), ready_only=True)
                if s is not None:
                    launch(s)
    finally:
        for t in running:
            t.cancel()
    return None, "rules"


# --- public ------------------------------------------------------------------

_cache: collections.OrderedDict[tuple, dict] = collections.OrderedDict()  # (capacity, note) -> entry
_flights: dict[tuple, asyncio.Future] = {}


def _norm(note):
    return " ".join(str(note).split())[:MAX_NOTE_CHARS]


def _remember(key, entry):
    _cache[key] = {k: v for k, v in entry.items() if k != "note_index"}
    _cache.move_to_end(key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)


async def _resolve(texts, capacity):
    got, source = await _cascade(texts, capacity)
    if got is None:
        return {}, source
    for i, t in enumerate(texts):
        _remember((capacity, t), got[i])
    return {t: got[i] for i, t in enumerate(texts)}, source


async def interpret(notes, capacity=None):
    """Return (entries, source). Never raises.

    entries holds one dict per note the LLM (or cache) answered, tagged with note_index.
    Notes it could not answer are simply absent; the caller fills them from rule_parse.
    """
    texts = [_norm(n) for n in notes]
    entries, missing = [], []
    for i, t in enumerate(texts):
        # keyed with capacity: "keep it half full" means different kWh on different batteries
        hit = _cache.get((capacity, t)) if t else None
        if hit is not None:
            _cache.move_to_end((capacity, t))
            entries.append({**hit, "note_index": i})
        elif t:
            missing.append(i)
    if not missing:
        return entries, "cache"
    if not slots():
        return entries, "rules"

    # coalesce: identical in-flight note sets share one cascade
    want = (capacity, *dict.fromkeys(texts[i] for i in missing))
    fut = _flights.get(want)
    if fut is None:
        fut = asyncio.ensure_future(_resolve(list(want[1:]), capacity))
        _flights[want] = fut
        fut.add_done_callback(lambda _f, k=want: _flights.pop(k, None))
    try:
        answers, source = await asyncio.shield(fut)  # a disconnecting client must not cancel it for others
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("interpretation cascade crashed")
        answers, source = {}, "rules"

    for i in missing:
        e = answers.get(texts[i])
        if e is not None:
            entries.append({**e, "note_index": i})
    return entries, source if answers else "rules"


async def warmup():
    """Open pooled connections and weed out rejected keys before the first real request."""
    seen, pings = set(), []
    for s in slots():
        if (s.provider, s.key) in seen:
            continue
        seen.add((s.provider, s.key))
        headers = {"x-goog-api-key": s.key} if s.kind == "gemini" else {"Authorization": f"Bearer {s.key}"}
        pings.append((s, _http().get(_BY_NAME[s.provider]["warm"], headers=headers, timeout=8)))
    for (s, _), r in zip(pings, await asyncio.gather(*(c for _, c in pings), return_exceptions=True)):
        if isinstance(r, httpx.Response) and (
                r.status_code in (401, 403) or (r.status_code == 400 and "API_KEY_INVALID" in r.text)):
            _cool(lambda x, k=s.key: x.key == k, math.inf, f"key rejected at startup ({r.status_code})")
    log.info("llm warmup done: %d provider keys pinged", len(pings))


def stats():
    now = time.monotonic()

    def cap(b):
        return None if math.isinf(b.cap) else int(b.cap)

    def level(b):
        return None if math.isinf(b.cap) else int(b.level)

    return {
        "cache_entries": len(_cache),
        "coalesced_in_flight": len(_flights),
        "slots": [{
            "slot": repr(s),
            "wait_s": None if math.isinf(w := s.wait(now)) else round(w, 2),
            "latency_s": round(s.lat, 2),
            "inflight": s.inflight,
            "tokens_left": level(s.tok), "tpm": cap(s.tok),
            "output_tokens_left": level(s.out), "otpm": cap(s.out),
            "dead": math.isinf(s.cool_until),
            "ok": s.ok,
            "fail": s.fail,
        } for s in slots()],
    }


def reset():
    """Forget slots, cache and in-flight state (tests, or after changing keys)."""
    global _slots
    _slots = None
    _cache.clear()
    _flights.clear()


# --- deterministic net -------------------------------------------------------
# ponytail: handles the common phrasings only; the LLM is the real path. This exists so a
# total provider outage degrades instead of failing. Upgrade only if that actually happens.

_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "midnight": 0, "noon": 12}
_FRACTIONS = {"half": 0.5, "a third": 1 / 3, "one-third": 1 / 3, "a quarter": 0.25,
              "one-quarter": 0.25, "one-fifth": 0.2, "a fifth": 0.2, "one fifth": 0.2}
_TIME = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b|\b(\d{1,2}):(\d{2})\b|"
    r"\b(" + "|".join(_WORDS) + r")\b", re.I)


# "1-3 PM" / "1 to 3 PM" / "between 1 and 3 PM": the meridiem belongs to both endpoints
_SHARED_MERIDIEM = re.compile(
    r"\b(\d{1,2})\s*(?:[-\u2013\u2014]|to|and|until|through|till)\s*(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)\b", re.I)


def _hours(text):
    """Pull a half-open hour window out of free text. Returns [] when unsure."""
    text = _SHARED_MERIDIEM.sub(r"\1 \3 to \2 \3", text)
    found = []
    for m in _TIME.finditer(text):
        if m.group(1):
            h = int(m.group(1)) % 12
            if m.group(3).lower().startswith("p"):
                h += 12
            found.append((h, True))
        elif m.group(4):
            found.append((int(m.group(4)) % 24, True))
        else:
            found.append((_WORDS[m.group(6).lower()], False))
    if not found:
        return []
    if len(found) == 1:
        return [found[0][0] % 24]
    a, b = found[0], found[1]
    start, end = a[0], b[0]
    # bare word times in an operator note mean the afternoon far more often than the small hours
    if not a[1] and not b[1] and start < 12 and end < 12 and start != 0:
        start, end = start + 12, end + 12
    if end <= start:
        end += 12 if end + 12 > start else 24
    return [h % 24 for h in range(start, min(end, start + 24))]


def _number(text, pattern):
    m = re.search(pattern, text, re.I)
    return float(m.group(1)) if m else None


def rule_parse(index, note):
    """Last-resort interpretation of one note."""
    t = note.lower()
    hours = _hours(note)
    out = {"note_index": index, "applies": False, "directive_type": "no_op",
           "structured_adjustment": None, "explanation": "No energy-schedule impact detected."}
    if not hours:
        return out

    def hit(*words):
        return any(w in t for w in words)

    blocked = hit("do not", "don't", "dont", "cannot", "can't", "no ", "avoid", "unavailable",
                  "disabled", "suspended", "prohibited", "not allowed", "refrain", "offline")

    if hit("solar", "pv", "panel", "photovoltaic"):
        pct = _number(note, r"(\d+(?:\.\d+)?)\s*(?:%|percent)")
        factor = None
        if pct is not None:
            reduced = hit("reduction", "reduced by", "drop by", "decrease", "less", "lower by", "down by")
            factor = (100 - pct) / 100 if reduced else pct / 100
        else:
            for word, val in _FRACTIONS.items():
                if word in t:
                    factor = 1 - val if hit("reduction", "reduced by", "drop by") else val
                    break
        if factor is None and hit("offline", "no output", "zero", "shut down", "shutdown"):
            factor = 0.0
        if factor is not None:
            return {"note_index": index, "applies": True, "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": hours, "factor": max(0.0, min(1.0, factor))},
                    "explanation": "Reduced solar availability during the stated window."}

    kwh = _number(note, r"(\d+(?:\.\d+)?)\s*kwh")
    if hit("reserve", "at least", "minimum", "no lower than", "maintain") and kwh is not None:
        return {"note_index": index, "applies": True, "directive_type": "minimum_battery_reserve",
                "structured_adjustment": {"hours": hours, "minimum_energy_kwh": kwh},
                "explanation": "Battery must stay above the stated reserve."}

    if hit("grid", "import", "draw") and kwh is not None and hit(
            "cap", "limit", "exceed", "no more than", "at most", "max"):
        return {"note_index": index, "applies": True, "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": hours, "max_grid_kwh": kwh},
                "explanation": "Grid import is capped during the stated window."}

    if blocked and "discharg" in t:
        return {"note_index": index, "applies": True, "directive_type": "no_discharge_window",
                "structured_adjustment": {"hours": hours},
                "explanation": "Battery discharging is unavailable during the stated window."}
    if blocked and "charg" in t:
        return {"note_index": index, "applies": True, "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": hours},
                "explanation": "Battery charging is unavailable during the stated window."}
    return out
