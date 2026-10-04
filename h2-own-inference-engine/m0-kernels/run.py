"""Run the M0 kernel lab in Chrome and print the results table.

Usage (repo root served on 127.0.0.1:8765):  python h2-own-inference-engine/m0-kernels/run.py [url]
"""
import json, sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/h2-own-inference-engine/m0-kernels/"
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page()
    page.on("pageerror", lambda e: print("[pageerror]", e))
    page.on("console", lambda m: print("[console]", m.text[:300]) if m.type == "error" else None)
    page.goto(URL)
    page.wait_for_function("window.ready", timeout=60000)
    out = page.evaluate("runAll()")
    print("GPU:", out["gpu"])
    print(f"{'kernel':32} {'shape':22} {'weights':>9} {'ms/call':>9} {'throughput':>14} {'rel. error':>10}")
    for r in out["results"]:
        print(f"{r['kernel']:32} {r['shape']:22} {r['weights']:>9} {r['ms']:9.3f} {r['throughput']:>14} {r['err']:10.1e}")
    browser.close()
