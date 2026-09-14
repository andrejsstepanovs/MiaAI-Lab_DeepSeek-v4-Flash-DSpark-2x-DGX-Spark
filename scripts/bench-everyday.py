#!/usr/bin/env python3
"""bench-everyday.py — baseline benchmark using the real everyday Hermes system prompt.

Sends /var/tmp/random_message_prompt_all.md (~7,560 tokens) as the system message plus
a realistic user turn, with the server's DEFAULT_THINKING=max behavior (unless
--thinking is given). Every request gets a unique nonce at the very start of the
system content, so the prefix cache is never hit (cold prefill baseline), across
consecutive trials and between the parallel requests of one trial.

Usage:
    python3 scripts/bench-everyday.py --concurrency 2,4 --repeat 3
    python3 scripts/bench-everyday.py --multiplier 5 --concurrency 2 --repeat 1

Per request: TTFT (first output delta), answer-start (first content delta), per-stream
decode tok/s after first token, reasoning/content token split when the server reports
it, prompt tokens. Per case: aggregate tok/s, median per-stream decode, median TTFT,
failures. A JSON report is written under results/ for cross-run comparison.
"""
import argparse
import asyncio
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

DEFAULT_PROMPT_FILE = "/var/tmp/random_message_prompt_all.md"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "deepseek-v4-flash-vision-exp"
DEFAULT_USER_MESSAGE = ("Give me a brief overview of what's going on right now and "
                        "what's coming up.")


def request_json(url, body, timeout=120):
    for attempt in range(4):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.URLError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def tokenize_count(base_url, model, text):
    return request_json(base_url.removesuffix("/v1") + "/tokenize",
                        {"model": model, "prompt": text})["count"]


def load_system_text(prompt_file, multiplier):
    text = open(prompt_file, encoding="utf-8").read().strip()
    if multiplier <= 1:
        return text
    return "\n\n---\n\n".join([text] * multiplier)


def build_system(nonce, prompt_file, multiplier):
    body = load_system_text(prompt_file, multiplier)
    return "<!-- bench everyday: " + nonce + " -->\n\n" + body


def stream_one(base_url, model, system, user_message, max_tokens, min_tokens,
               thinking, temperature, top_p):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if min_tokens:
        body["min_tokens"] = min_tokens
    if thinking == "off":
        body["chat_template_kwargs"] = {"thinking": False}
    elif thinking in ("low", "high", "max"):
        body["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": thinking}

    req = urllib.request.Request(f"{base_url}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    first_content = None
    reasoning_chars = 0
    content_chars = 0
    usage = None
    error = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except ValueError:
                    continue
                choices = event.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                content = delta.get("content") or ""
                now = time.perf_counter()
                if reasoning or content:
                    if first is None:
                        first = now
                    if content and content.strip() and first_content is None:
                        first_content = now
                    reasoning_chars += len(reasoning)
                    content_chars += len(content)
                if event.get("usage"):
                    usage = event["usage"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finished = time.perf_counter()
    if error:
        return {"ok": False, "error": error, "ttft_s": None,
                "first_content_s": None, "elapsed_s": finished - started,
                "completion_tokens": 0, "prompt_tokens": 0, "reasoning_tokens": None,
                "cached_tokens": None, "decode_tok_s": None,
                "reasoning_chars": 0, "content_chars": 0}
    completion = (usage or {}).get("completion_tokens", 0)
    prompt = (usage or {}).get("prompt_tokens", 0)
    details = (usage or {}).get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    cached_tokens = (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens")
    ttft = (first or finished) - started
    decode_window = max(0.001, finished - (first or finished))
    return {
        "ok": True, "error": None,
        "ttft_s": ttft,
        "first_content_s": (first_content - started) if first_content is not None else None,
        "elapsed_s": finished - started,
        "completion_tokens": completion,
        "prompt_tokens": prompt,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "decode_tok_s": completion / decode_window,
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
    }


async def run_case(base_url, model, prompt_file, multiplier, concurrency, user_message,
                   max_tokens, min_tokens, thinking, temperature, top_p, trial, cached):
    async def one(nonce):
        system = build_system(nonce, prompt_file, multiplier)
        return await asyncio.to_thread(stream_one, base_url, model, system,
                                       user_message, max_tokens, min_tokens,
                                       thinking, temperature, top_p)

    if cached:
        warm = await asyncio.to_thread(stream_one, base_url, model,
                                       build_system("warmup", prompt_file, multiplier),
                                       user_message, max_tokens, min_tokens,
                                       thinking, temperature, top_p)
        if not warm["ok"]:
            return {"concurrency": concurrency, "trial": trial, "elapsed_s": None,
                    "aggregate_tok_s": None, "median_decode_tok_s": None,
                    "median_ttft_s": None, "n_ok": 0, "n_fail": concurrency,
                    "requests": [warm]}
        nonces = ["cached"] * concurrency
    else:
        nonces = [f"t{trial}-r{index}" for index in range(concurrency)]
    started = time.perf_counter()
    results = await asyncio.gather(*[one(nonce) for nonce in nonces])
    elapsed = time.perf_counter() - started
    ok = [r for r in results if r["ok"]]
    if not ok:
        return {"concurrency": concurrency, "trial": trial, "elapsed_s": elapsed,
                "aggregate_tok_s": None, "median_decode_tok_s": None,
                "median_ttft_s": None, "n_ok": 0, "n_fail": concurrency,
                "requests": results}
    total = sum(r["completion_tokens"] for r in ok)
    return {"concurrency": concurrency, "trial": trial, "elapsed_s": elapsed,
            "aggregate_tok_s": total / max(0.001, elapsed),
            "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in ok),
            "median_ttft_s": statistics.median(r["ttft_s"] for r in ok),
            "n_ok": len(ok), "n_fail": concurrency - len(ok), "requests": results}


def summarize(cases, concurrency):
    ok = [r for c in cases for r in c["requests"] if r["ok"]]
    n_fail = sum(c["n_fail"] for c in cases)
    if not ok:
        return {"concurrency": concurrency, "n_ok": 0, "n_fail": n_fail}
    decs = [r["decode_tok_s"] for r in ok]
    aggs = [c["aggregate_tok_s"] for c in cases if c["aggregate_tok_s"]]
    ttfts = [r["ttft_s"] for r in ok]
    proms = [r["prompt_tokens"] for r in ok]
    comps = [r["completion_tokens"] for r in ok]
    reas = [r["reasoning_tokens"] for r in ok if r["reasoning_tokens"] is not None]
    cached = [r["cached_tokens"] for r in ok if r.get("cached_tokens") is not None]
    share = (sum(reas) / sum(comps)) if reas and sum(comps) else None
    return {"concurrency": concurrency, "n_ok": len(ok), "n_fail": n_fail,
            "median_decode_tok_s": statistics.median(decs),
            "decode_min": min(decs), "decode_max": max(decs),
            "median_aggregate_tok_s": statistics.median(aggs) if aggs else None,
            "median_ttft_s": statistics.median(ttfts),
            "median_prompt_tokens": statistics.median(proms),
            "median_cached_tokens": (statistics.median(cached) if cached else None),
            "median_completion_tokens": statistics.median(comps),
            "reasoning_token_share": share}


async def main():
    ap = argparse.ArgumentParser(
        description="Baseline benchmark using the real everyday Hermes system prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    ap.add_argument("--multiplier", type=int, default=1,
                    help="repeat the prompt N times to simulate deeper conversation")
    ap.add_argument("--concurrency", default="2,4",
                    help="comma-separated concurrency levels (default 2,4)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="trials per concurrency; unique nonces each (default 3)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--min-tokens", type=int, default=0, help="0 = server default")
    ap.add_argument("--user-message", default=DEFAULT_USER_MESSAGE)
    ap.add_argument("--thinking", default="server",
                    choices=["server", "off", "low", "high", "max"],
                    help="server = server's DEFAULT_THINKING; else override")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--output", default=None,
                    help="JSON output path (default results/bench-everyday-<ts>.json)")
    ap.add_argument("--cached", action="store_true",
                    help="reuse one fixed system prefix so requests hit the vLLM "
                         "prefix cache (warms up once per concurrency level; reports "
                         "cached_tokens from the usage block)")
    args = ap.parse_args()

    if args.multiplier < 1:
        ap.error("--multiplier must be >= 1")
    if not os.path.isfile(args.prompt_file):
        ap.error(f"prompt file not found: {args.prompt_file}")

    probe = build_system("probe", args.prompt_file, args.multiplier)
    prompt_tokens = tokenize_count(args.base_url, args.model, probe)
    concurrencies = sorted({int(x) for x in args.concurrency.split(",") if x.strip()})

    print(f"prompt {args.prompt_file} x{args.multiplier} -> {prompt_tokens} tokens "
          f"({'cached prefix, warmup + identical system' if args.cached else 'unique nonce per request, prefix cache bypassed'})",
          flush=True)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url, "model": args.model,
            "prompt_file": args.prompt_file, "multiplier": args.multiplier,
            "prompt_tokens": prompt_tokens, "concurrencies": concurrencies,
            "repeat": args.repeat, "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens, "user_message": args.user_message,
            "thinking": args.thinking, "temperature": args.temperature,
            "top_p": args.top_p, "cached": args.cached,
        },
        "cases": [],
        "summary": {},
    }

    for concurrency in concurrencies:
        print(f"\n=== concurrency {concurrency} x {args.repeat} trials ===", flush=True)
        cases = []
        for trial in range(args.repeat):
            case = await run_case(args.base_url, args.model, args.prompt_file,
                                  args.multiplier, concurrency, args.user_message,
                                  args.max_tokens, args.min_tokens, args.thinking,
                                  args.temperature, args.top_p, trial, args.cached)
            cases.append(case)
            output["cases"].append(case)
            if case["n_ok"]:
                dec = case["median_decode_tok_s"]
                ag = f"{case['aggregate_tok_s']:.1f}" if case["aggregate_tok_s"] else "-"
                tt = f"{case['median_ttft_s']:.1f}s" if case["median_ttft_s"] else "-"
                print(f"  trial {trial}: ok={case['n_ok']}/{case['n_ok'] + case['n_fail']} "
                      f"median_decode={dec:.1f} tok/s agg={ag} ttft={tt} "
                      f"completion={[r['completion_tokens'] for r in case['requests']]}",
                      flush=True)
            else:
                errs = [r.get("error", "?") for r in case["requests"]]
                print(f"  trial {trial}: ALL FAILED {errs[:2]}", flush=True)
        s = summarize(cases, concurrency)
        output["summary"][str(concurrency)] = s
        print(f"  => c={concurrency}: median decode {s.get('median_decode_tok_s')} tok/s "
              f"(min {s.get('decode_min')} max {s.get('decode_max')}), "
              f"agg {s.get('median_aggregate_tok_s')} tok/s, "
              f"TTFT {s.get('median_ttft_s')}s, prompt {s.get('median_prompt_tokens')} tok, "
              f"cached {s.get('median_cached_tokens')} tok, "
              f"completion {s.get('median_completion_tokens')} tok, "
              f"reasoning share {s.get('reasoning_token_share')}, "
              f"ok {s.get('n_ok')} fail {s.get('n_fail')}", flush=True)

    out_path = args.output
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "results",
            f"bench-everyday-{ts}.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
