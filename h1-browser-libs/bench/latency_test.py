"""Drive the H1 voice agent in real Chrome and print per-stage latency.

Usage (from the repo root, with `python -m http.server 8765 --bind 127.0.0.1` running):
    python h1-browser-libs/bench/latency_test.py                 # test clips, full GPU path
    NOGPU=1 python h1-browser-libs/bench/latency_test.py         # hide WebGPU: CPU-only path
    MIC=h1-browser-libs/bench/mic.wav python h1-browser-libs/bench/latency_test.py   # fake microphone

The Chrome profile is kept in bench/.chrome-profile so models stay cached between runs.
"""
import json, os, sys, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/h1-browser-libs/"
PROFILE = os.path.join(HERE, ".chrome-profile")
MIC = os.environ.get("MIC")       # wav file fed to Chrome as a fake microphone
NOGPU = os.environ.get("NOGPU")   # hide navigator.gpu to simulate a browser without WebGPU
CLIPS = ["samples/test.wav", "samples/short.wav"] + ([] if NOGPU else ["samples/test.wav", "samples/short.wav"])
KEYS = ["audioMs", "endpointMs", "asrMs", "asrWaitMs", "asrEncMs", "asrDecMs",
        "llmTtftMs", "llmFirstSentenceMs", "llmTokS", "ttsFirstMs", "v2vMs"]

args = ["--autoplay-policy=no-user-gesture-required"]
if MIC:
    args += ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
             f"--use-file-for-fake-audio-capture={os.path.abspath(MIC)}"]

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(PROFILE, channel="chrome", headless=True, args=args)
    page = ctx.new_page()
    if NOGPU:
        page.add_init_script("Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined })")
    page.on("pageerror", lambda e: print("  [pageerror]", e))
    page.goto(URL)
    time.sleep(2)  # the service worker reloads the page once to enable cross-origin isolation
    page.wait_for_function("window.agent", timeout=60000)
    print("crossOriginIsolated:", page.evaluate("crossOriginIsolated"))

    t = time.time()
    print(f"load: {json.dumps(page.evaluate('agent.load()'))}  (wall {time.time() - t:.1f}s)")

    if MIC:
        page.evaluate("agent.startMic()")
        page.wait_for_function("agent.results.length >= 2", timeout=180000)
        rows = page.evaluate("agent.results")
    else:
        rows = []
        for clip in CLIPS:
            page.wait_for_function("document.querySelector('#status').dataset.s === 'idle'", timeout=120000)
            rows.append(page.evaluate(f"agent.runClip('{clip}')"))

    print("\n" + " | ".join(["turn"] + KEYS))
    for r in rows:
        print(" | ".join([str(r.get("turn"))] + [f"{r[k]:.0f}" if isinstance(r.get(k), (int, float)) else "-" for k in KEYS]))
        print("   user :", r.get("text"))
        print("   agent:", r.get("reply"))
    ctx.close()
