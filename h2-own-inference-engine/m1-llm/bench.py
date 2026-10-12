"""Benchmark the H2 LLM engine in real Chrome: correctness against reference.json, then speed.

Usage (repo root served by `python serve.py`, which sets the headers threads need):
    python h2-own-inference-engine/m1-llm/bench.py                     # GPU + CPU (default threads)
    python h2-own-inference-engine/m1-llm/bench.py cpu:1 cpu:4:4 cpu:8:4 cpu:1:simd128 webgpu baseline:webgpu baseline:wasm
    MODEL=http://127.0.0.1:8765/models/qwen2.5-0.5b-instruct-q4_0.gguf python ...   # a local copy instead of Hugging Face
    (use the full URL: Git Bash on Windows rewrites values that look like /paths)

Each configuration gets a fresh page. "cpu:8:4" = 8 threads for prompts, 4 for generation; "cpu:4:simd128" forces the plain-SIMD build (no relaxed SIMD).
"baseline:webgpu" / "baseline:wasm" measure the same model through Transformers.js (the H1 libraries).
"""
import json, os, sys
from playwright.sync_api import sync_playwright

BASE = os.environ.get("URL", "http://127.0.0.1:8765/h2-own-inference-engine/m1-llm/")
MODEL = os.environ.get("MODEL", "")
configs = sys.argv[1:] or ["webgpu", "cpu"]

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    summary = []
    for c in configs:
        parts = c.split(":")
        opts = {"backend": parts[0]}
        if len(parts) > 1 and parts[0] != "baseline": opts["threads"] = int(parts[1])
        if len(parts) > 2 and parts[2] == "simd128": opts["relaxed"] = False
        elif len(parts) > 2: opts["decodeThreads"] = int(parts[2])
        if MODEL: opts["url"] = MODEL
        page = browser.new_page()
        page.on("pageerror", lambda e: print("  [pageerror]", e))
        if parts[0] == "baseline":                 # the H1 libraries on the same model, for comparison
            page.goto(BASE + "baseline.html")
            page.wait_for_function("window.ready", timeout=60000)
            print(f"\n== {c}")
            info = page.evaluate("(o) => base.load(o)", {"device": parts[1], "threads": int(parts[2]) if len(parts) > 2 else 4})
            print(f"   {info['info']}; load {info['loadMs'] / 1000:.1f} s")
            b = page.evaluate("base.bench()")
            pre = "  ".join(f"{x['tokens']} tok {x['ms']:.0f} ms ({x['tokPerS']:.0f}/s)" for x in b["prefill"])
            print(f"   prefill: {pre}")
            print(f"   decode: {b['decode']['tokPerS']:.1f} tok/s (median {b['decode']['msPerToken']:.1f} ms, p90 {b['decode']['p90']:.1f} ms)")
            b["turn"] = None
            summary.append((c, info["info"], None, b))
            page.close()
            continue
        page.goto(BASE)
        page.wait_for_function("window.ready", timeout=60000)
        print(f"\n== {c}")
        info = page.evaluate("(o) => lab.load(o)", opts)
        print(f"   {info['info']}; load {info['loadMs'] / 1000:.1f} s")
        chk = page.evaluate("lab.check()")
        ok = all(x["agree"] == x["of"] for x in chk)
        print("   check:", "identical to the reference" if ok else "DIFFERS", [f"{x['agree']}/{x['of']}" for x in chk])
        b = page.evaluate("lab.bench()")
        pre = "  ".join(f"{x['tokens']} tok {x['ms']:.0f} ms ({x['tokPerS']:.0f}/s)" for x in b["prefill"])
        print(f"   prefill: {pre}")
        print(f"   decode: {b['decode']['tokPerS']:.1f} tok/s (median {b['decode']['msPerToken']:.1f} ms, p90 {b['decode']['p90']:.1f} ms)")
        t = b["turn"]
        print(f"   voice turn: first token {t['firstTokenMs']:.0f} ms ({t['prefilled']} new tokens, {t['reused']} cached), "
              f"{t['tokens']} tokens, {t['decodeTokPerS']:.1f} tok/s: {t['text'].strip()[:80]!r}")
        summary.append((c, info["info"], ok, b))
        page.close()
    print("\n| config | engine | correct | prefill 64 tok | prefill 256 tok | decode tok/s | voice-turn first token |")
    print("|---|---|---|---|---|---|---|")
    for c, inf, ok, b in summary:
        p64 = next(x for x in b["prefill"] if x["tokens"] == 64)["ms"]
        p256 = next(x for x in b["prefill"] if x["tokens"] == 256)["ms"]
        turn = f"{b['turn']['firstTokenMs']:.0f} ms" if b.get("turn") else "–"
        okc = "–" if ok is None else "yes" if ok else "NO"
        print(f"| {c} | {inf} | {okc} | {p64:.0f} ms | {p256:.0f} ms | {b['decode']['tokPerS']:.1f} | {turn} |")
    browser.close()
