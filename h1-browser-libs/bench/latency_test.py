"""Drive the H1 voice agent in real Chrome and print per-stage latency.

Usage (from the repo root, with `python -m http.server 8765 --bind 127.0.0.1` running):
    python h1-browser-libs/bench/latency_test.py                 # test clips, full GPU path
    NOGPU=1 python h1-browser-libs/bench/latency_test.py         # hide WebGPU: CPU-only path
    MIC=h1-browser-libs/bench/mic.wav python h1-browser-libs/bench/latency_test.py   # fake microphone
    VERSION=lite python h1-browser-libs/bench/latency_test.py    # normal | lite | superlite (default: the page's choice)

The Chrome profile is kept in bench/.chrome-profile so models stay cached between runs.
It also prints the peak memory of the page's process and of the GPU process (needs psutil).
"""
import json, os, sys, threading, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/h1-browser-libs/"
PROFILE = os.path.join(HERE, ".chrome-profile")
MIC = os.environ.get("MIC")       # wav file fed to Chrome as a fake microphone
NOGPU = os.environ.get("NOGPU")   # hide navigator.gpu to simulate a browser without WebGPU
VERSION = os.environ.get("VERSION")
if VERSION:
    URL += ("&" if "?" in URL else "?") + "profile=" + VERSION
CLIPS = ["samples/test.wav", "samples/short.wav"] + ([] if NOGPU else ["samples/test.wav", "samples/short.wav"])
KEYS = ["audioMs", "endpointMs", "asrMs", "asrWaitMs", "asrEncMs", "asrDecMs",
        "llmTtftMs", "llmFirstSentenceMs", "llmTokS", "ttsFirstMs", "v2vMs"]

args = ["--autoplay-policy=no-user-gesture-required"]
if MIC:
    args += ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
             f"--use-file-for-fake-audio-capture={os.path.abspath(MIC)}"]


def watch_memory(peak, stop):
    """Samples Chrome's private memory per process type: on phones the browser kills a tab whose memory peaks."""
    import psutil
    mem = lambda pr: getattr(pr.memory_info(), "private", 0) or pr.memory_info().rss
    browser = None
    while not stop.is_set():
        try:
            if browser is None:
                browser = next(pr for pr in psutil.process_iter(["cmdline"]) if pr.info["cmdline"] and
                               any(a.startswith("--user-data-dir=") and PROFILE in a for a in pr.info["cmdline"]) and
                               not any(a.startswith("--type=") for a in pr.info["cmdline"]))
            for pr in browser.children(recursive=True):
                kind = next((a[7:] for a in pr.cmdline() if a.startswith("--type=")), "browser")
                if kind in ("renderer", "gpu-process"):
                    peak[kind] = max(peak.get(kind, 0), mem(pr))
        except (StopIteration, psutil.Error):
            pass
        time.sleep(0.2)


with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(PROFILE, channel="chrome", headless=True, args=args)
    peak, stop = {}, threading.Event()
    threading.Thread(target=watch_memory, args=(peak, stop), daemon=True).start()
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
    print("peak memory after load (MB):", {k: round(v / 1e6) for k, v in peak.items()})

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
    print("\npeak memory overall (MB):", {k: round(v / 1e6) for k, v in peak.items()})
    stop.set()
    ctx.close()
