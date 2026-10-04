"""Benchmark single H1 components in Chrome, each in a fresh browser.

Usage (repo root served on 127.0.0.1:8765):
    python h1-browser-libs/bench/component_bench.py encoder w2a8:wasm:4 w2a8:webgpu w4a8:wasm:1
    python h1-browser-libs/bench/component_bench.py tts wasm:q8 wasm:fp32 webgpu:fp32

A fresh browser per run matters: ONNX Runtime fixes its thread count the first time it starts.
"""
import json, os, sys, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8765/h1-browser-libs/bench/components.html"
PROFILE = os.path.join(HERE, ".chrome-profile")
what, configs = sys.argv[1], sys.argv[2:]

with sync_playwright() as p:
    for cfg in configs:
        parts = cfg.split(":")
        ctx = p.chromium.launch_persistent_context(PROFILE, channel="chrome", headless=True)
        page = ctx.new_page()
        page.goto(URL); time.sleep(2)
        page.wait_for_function("window.ready && crossOriginIsolated", timeout=60000)
        if what == "encoder":
            call = "([v, ep, th]) => benchEncoder(v, ep, Number(th || 4))"
            args = [parts[0], parts[1], parts[2] if len(parts) > 2 else "4"]
        else:
            call = "([d, dt]) => benchTTS(d, dt)"
            args = parts
        try:
            print(cfg, json.dumps(page.evaluate(call, args)))
        except Exception as e:
            print(cfg, "FAILED", str(e).splitlines()[0][:300])
        ctx.close()
