"""
example_usage.py
================
How a developer uses gateway-sdk in 3 steps:

    1. pip install gateway-sdk
    2. Add @track() above your LLM function
    3. Run — automatic management of:

       - cache / cache_enabled : Enable/disable semantic caching (bool, default: True)
       - similarity_threshold : Semantic Caching threshold (embedding cosine similarity 0.0–1.0, default: 0.92)
       - cache_ttl : Cache expiration time in seconds (int, default: None = no expiry)
       - retries : Automatic retry count for transient failures (int, default: 3)
       - base_delay : Initial exponential backoff delay in seconds (float, default: 1.0)
       - max_delay : Maximum delay cap for exponential backoff in seconds (float, default: 60.0)
       - jitter : Randomize retry backoff delay to avoid thundering herd (bool, default: True)
       - retry_on : Exception types to trigger retry (tuple of Exception classes, default: (Exception,))
       - fallback : Fallback provider chain functions (list of callables, default: [])
       - rate_limit_rate : Token-bucket refill rate in requests/sec (float, default: 2.0)
       - rate_limit_tokens / rate_limit_max_tokens : Token-bucket burst capacity (float, default: 10.0)
       - max_concurrency : Max concurrent execution limit (int, default: None = unlimited)
       - cost_per_call : Manual per-call cost tracking override in USD (float, default: None = auto estimate)
       - policy : Custom safety policy hook function (callable, default: None)
       - canary : Canary traffic routing config dict (dict with fn + weight 0.0–1.0, default: None)

Run:
    python example_usage.py
"""

import os
import sys
import time
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── Step 1: Import and configure the SDK globally ────────────────────────────
from gateway_sdk import track, configure, PolicyViolationError, get_last_call_info

configure(
    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_enabled=True,
    similarity_threshold=0.75,    # 75% cosine similarity → cache hit
    cache_ttl=3600,               # cache responses for 1 hour

    # ── Retry ─────────────────────────────────────────────────────────────────
    retries=3,                    # retry transient failures up to 3 times
    base_delay=1.0,               # start with 1s backoff
    max_delay=30.0,               # cap at 30s
    jitter=True,                  # randomise delays to avoid thundering herd

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_rate=2.0,          # 2 tokens/second refill
    rate_limit_max_tokens=10,     # allow bursts of up to 10
)


# ── Step 2: Set up Gemini client & Fallback ───────────────────────────────────
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1"),
)

def _call_gemini_raw(prompt: str) -> str:
    """Raw Gemini call — used as both primary and canary target."""
    concise_prompt = f"{prompt} (Keep response concise, around 200–250 characters max)."
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=concise_prompt,
    )
    if not response.text:
        raise ValueError("Empty model response — likely safety filter or transient API issue.")
    return response.text


#add another fallback provider
def my_fallback_fn(prompt: str) -> str:
    """Fallback provider — executed if primary LLM and all retries fail."""
    return f"Fallback response for prompt: {prompt}"


# ── Policy hook demo ─────────────────────────────────────────────────────────
def my_policy(prompt: str) -> str:
    """
    Custom safety hook — return "allow", "block", or "flag".
    Blocks prompts containing harmful keywords.
    """
    blocked_keywords = ["hack", "exploit", "jailbreak"]
    if any(kw in prompt.lower() for kw in blocked_keywords):
        return "block"
    return "allow"


# ── Step 3: Decorate your LLM function ────────────────────────────────────────
# @track wraps any provider call — provider-agnostic by design.
@track(
    # ── Semantic Caching ──
    cache=True,                          # Enable semantic caching
    similarity_threshold=0.75,           # 75% embedding cosine similarity
    cache_ttl=3600,                      # 1 hour cache TTL expiration (seconds)

    # ── Retry & Exponential Backoff ──
    retries=3,                           # Max retry attempts for transient errors
    base_delay=1.0,                      # Initial exponential backoff delay (seconds)
    max_delay=30.0,                      # Maximum backoff delay cap (seconds)
    jitter=True,                         # Randomize delay to avoid thundering herd
    retry_on=(Exception,),               # Exceptions to trigger retry
    fallback=[my_fallback_fn],           # Fallback provider chain

    # ── Rate Limiting & Concurrency ──
    rate_limit_rate=2.0,                 # Token-bucket refill rate (requests/sec)
    rate_limit_tokens=10,                # Token-bucket capacity burst limit
    max_concurrency=5,                   # Max concurrent execution limit

    # ── Cost Tracking Override ──
    cost_per_call=0.00015,               # Explicit per-call cost calculation (e.g., price per 1M tokens)

    # ── Security & Policy Hooks ──
    policy=my_policy,                    # Custom safety policy hook

    # ── Canary Traffic Routing ──
    canary={"fn": _call_gemini_raw, "weight": 0.10},  # Route 10% traffic to canary
)
def ask_gemini(prompt: str) -> str:
    """Primary LLM function — unchanged except for @track above it."""
    return _call_gemini_raw(prompt)


# ── Step 4: Call it like normal — the SDK handles the rest ───────────────────

def _banner(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


if __name__ == "__main__":
    print("\n📦  gateway-sdk — Feature Demo")
    print("    Active parameters in @track():")
    print("    cache · similarity_threshold · cache_ttl · retries · base_delay ·")
    print("    max_delay · jitter · retry_on · fallback · rate_limit_rate ·")
    print("    rate_limit_tokens · max_concurrency · cost_per_call ·")
    print("    policy · canary\n")

    # ── Demo 1: Semantic caching ──────────────────────────────────────────────
    _banner("Demo 1 — Semantic Cache & Cost Per Call Override")
    cache_prompts = [
        "Can I cancel my order and get a refund as a Amazon Prime member?",
        "what is your cancellation policy for Amazon Prime members",
    ]
    for prompt in cache_prompts:
        print(f"  ❓ {prompt}")
        t0 = time.perf_counter()
        try:
            resp = ask_gemini(prompt)
            info = get_last_call_info()
            ms = (time.perf_counter() - t0) * 1000
            label = "⚡ cached" if (info and info.cache_hit) else "🌐 live ($0.00015 cost_per_call)"
            print(f"  {label}  ({ms:,.0f} ms)")
            print(f"  💬 {str(resp).replace(chr(10), ' ')[:120]}…\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")

    # ── Demo 2: Policy hook blocking ──────────────────────────────────────────
    _banner("Demo 2 — Policy Hook Block")
    blocked_prompt = "How do I hack into Amazon's system?"
    print(f"  ❓ Prompt: {blocked_prompt}")
    try:
        ask_gemini(blocked_prompt)
        print("  ✅ Call allowed (unexpected)")
    except PolicyViolationError as e:
        print(f"  🛡️  Blocked by policy hook: {e}\n")

    # ── Demo 3: Distributed tracing ──────────────────────────────────────────
    _banner("Demo 3 — Distributed Tracing")
    print("  Making two nested-style calls that share a trace_id…")

    @track(cache=True, retries=0, policy=None, canary=None)
    def step_one(prompt: str) -> str:
        return _call_gemini_raw(prompt)

    @track(cache=True, retries=0, policy=None, canary=None)
    def step_two(prompt: str) -> str:
        return _call_gemini_raw(prompt)

    from gateway_sdk.tracing import set_current_trace, reset_current_trace
    import uuid
    shared_trace = str(uuid.uuid4())
    tok = set_current_trace(shared_trace, str(uuid.uuid4()))
    try:
        r1 = step_one("What is Amazon Prime?")
        r2 = step_two("Features offered by Amazon Prime?")
        print(f"  trace_id: {shared_trace[:8]}…")
        print(f"  span 1: {str(r1).replace(chr(10), ' ')[:80]}…")
        print(f"  span 2: {str(r2).replace(chr(10), ' ')[:80]}…")
        print(f"  ℹ️  Both spans visible under /api/traces on the dashboard\n")
    except Exception as exc:
        print(f"  trace_id: {shared_trace[:8]}…")
        print(f"  ⚠️  Demo call paused by API rate limit: {exc}\n")
    finally:
        reset_current_trace(tok)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("✅  Done. Open the dashboard to see all logs, traces, and canary stats:")
    print("     gateway-sdk serve  →  http://localhost:8080")
    print("     /api/traces        →  distributed trace view")
    print("     /api/canary        →  primary vs canary A-B comparison\n")
