# H1: Browser libraries + cache

> **Hypothesis:** a single HTML file, using existing browser ML libraries and the browser's cache, can run a
> voice agent (speech recognition → language model → speech synthesis) with low enough latency to hold a conversation.

**Verdict: confirmed when the browser has WebGPU, rejected when it doesn't.**
On a 2019 laptop with a GTX 1050, the agent starts answering **1.4–2.1 s** after you stop talking.
Without WebGPU the same page works but takes **12–18 s**, too slow for a conversation.

[Try the demo](https://pedro1-21gw.github.io/e2e-browser-voice-agent/h1-browser-libs/) (Chrome or Edge; ~830 MB download on first visit, cached afterwards)

---

## Why it matters

Running everything on the user's device means no server bill, no API keys to leak, and privacy by design:
the audio never leaves the tab. The open question was speed: browsers sandbox code, and voice conversations
feel broken once the reply takes more than about 2 seconds.

## Setup

| Stage | Model | Runs on | Library |
|---|---|---|---|
| Voice activity detection | Silero VAD v5 (2 MB) | CPU | ONNX Runtime Web 1.30 (WebAssembly) |
| Speech to text | Parakeet TDT 0.6B v3 **Redux**: encoder weights are all −1, 0 or +1, stored in 2 bits (208 MB) | encoder on **GPU**, the rest on CPU | ONNX Runtime Web 1.30 (WebGPU + WebAssembly), **decoding loop hand-written** |
| Language model | LFM2 350M, 4-bit (293 MB) | GPU | Transformers.js 4.3 |
| Text to speech | Kokoro 82M, fp32 (325 MB) | GPU | kokoro-js 1.2.1 |

Hardware: Intel i5-9300HF (4 cores), GTX 1050 3 GB (no fp16 in shaders), 8 GB RAM, Windows 11, Chrome 154.

## Method

1. **Validate the speech-recognition decoder outside the browser first.** No browser library supports Parakeet,
   so the decoding loop (the TDT "token + how far to jump" loop) was written by hand. It was first written in
   Python ([bench/asr_reference.py](bench/asr_reference.py)) and checked on test clips: both transcripts came
   out word-perfect. Then it was ported to JavaScript.
2. **Build the page** ([index.html](index.html)) with hooks (`window.agent`) so a script can drive it.
3. **Measure in real Chrome** with Playwright ([bench/latency_test.py](bench/latency_test.py)):
   - **test clips**: a recorded question is fed straight to the pipeline. No silence wait, so it isolates the processing time.
   - **fake microphone**: Chrome plays [bench/mic.wav](bench/mic.wav) as if it were a mic, which exercises the
     whole path including voice activity detection and the silence wait.
4. **Benchmark components alone** ([bench/component_bench.py](bench/component_bench.py)) whenever a stage looked slow,
   to decide where it should run.

Test audio was generated with Windows' built-in voices (clean, no background noise), so it doesn't say
anything about accuracy on real noisy microphones.

The key number is **voice-to-voice latency**: from the moment you stop talking to the moment the agent's first
sound plays.

## Results

### How the latency came down

| Round | Change | Speech to text | LLM speed | First audio after first sentence | **Voice-to-voice** |
|---|---|---|---|---|---|
| 0 | Everything in place; TTS on CPU (8-bit) | 1.7–3.0 s | 1–4 tok/s | 3.8–23.5 s | **8.3–27.2 s** (clips) |
| 1 | TTS moved to GPU (fp32); first clause spoken at the first comma | 1.7–3.5 s | 11–22 tok/s | 1.1–2.4 s | **3.1–6.0 s** (clips) |
| 2 | Speech-recognition encoder moved to GPU | **0.13–0.53 s** | 12–23 tok/s | 1.0–2.1 s | **1.6–3.0 s** (clips) |
| 3 | Speculative transcription during the silence wait, measured through the mic | 0 s wait (hidden) | 12–45 tok/s | 0.7–1.4 s | **1.4–2.1 s** (mic, includes the 500 ms silence wait) |

Why round 0 was so slow: the TTS ran on the CPU on the same thread as everything else. While it synthesized,
the LLM couldn't produce tokens (1–4 tok/s), and the TTS itself was slower than real time.

### Component benchmarks

**Kokoro TTS: time to synthesize a sentence** ("× real time" above 1 means faster than it takes to say it)

| Device / precision | "Hello!" | 31-char sentence | 95-char sentence | × real time |
|---|---|---|---|---|
| CPU, 8-bit | 3.3 s | 5.8 s | 15.9 s | 0.40–0.42 |
| CPU, fp32 | 2.5 s | 3.2 s | 9.6 s | 0.56–0.73 |
| **GPU, fp32** | **0.72 s** | **0.88 s** | **2.0 s** | **1.9–3.1** |

Surprise: on the CPU, 8-bit was *slower* than fp32. Smaller weights don't help if the 8-bit math has no fast
path in WebAssembly.

**Parakeet encoder: time for a 6.06 s clip** (median of 4 runs after warm-up)

| Where | Time | Notes |
|---|---|---|
| Native Python, 1 thread | 1.84 s | reference |
| Native Python, 4 threads | 0.95 s | reference |
| Browser CPU, 1 thread | 6.0 s | WebAssembly is ~3.3× slower than native here |
| Browser CPU, 4 threads | 2.6–3.1 s | threads help 2×; 6 or 8 threads don't help on 4 cores |
| Browser CPU, 4-bit weights | 2.9–3.2 s | no faster than 2-bit |
| **Browser GPU (WebGPU)** | **~180 ms** | **~15× faster**; the first run takes 2.7 s (shader compilation), which warm-up absorbs |

For the 2.5 s clip the GPU encoder takes ~100 ms.

### Without WebGPU (same laptop, WebGPU hidden from the page)

| | With WebGPU | CPU only |
|---|---|---|
| Speech to text | 0.13–0.53 s | 1.9–4.5 s |
| LLM | 12–23 tok/s (LFM2 350M, 4-bit) | 2–5 tok/s (Qwen2.5 0.5B, 8-bit) |
| Voice | 2–3× faster than real time | slower than real time: audible gaps between sentences |
| **Voice-to-voice** | **1.4–2.1 s** | **11.8–18.2 s** |

## Problems found along the way (and fixes)

| Problem | Cause | Fix |
|---|---|---|
| Speech-recognition encoder failed on WebGPU with "no implementation for Cast" | The *preprocessor* (not the encoder) converts audio to float64, which ONNX Runtime's WebGPU build doesn't include | Two copies of ONNX Runtime: the WebAssembly build runs the preprocessor, the WebGPU build runs the encoder |
| Multithreaded WebAssembly unavailable on GitHub Pages | Threads need `SharedArrayBuffer`, which needs two HTTP headers (COOP/COEP) that GitHub Pages can't set | A tiny service worker ([sw.js](sw.js)) adds the headers to the page's own responses |
| The agent answered "Hi there." and dropped the rest of the question | The test voice paused for over 500 ms mid-question, and the mic was ignored while the agent was busy | If you speak again *before the agent makes a sound*, the pending reply is cancelled and both parts are merged |
| Page crashed without WebGPU | Both LFM2 exports (4- and 8-bit) use `GatherBlockQuantized`, which the CPU runtime lacks | CPU path switches to Qwen2.5 0.5B 8-bit, which uses standard integer ops |
| First GPU call of each model took 2–3 s | WebGPU compiles shaders on first use | Warm-up pass after loading (two audio lengths, one LLM call, one TTS call) |

## Verdict

**Confirmed, with a condition: the browser must have WebGPU.** With it, a single static HTML file reaches
1.4–2.1 s voice-to-voice on a modest 2019 laptop, using only off-the-shelf libraries plus a hand-written
decoding loop. Without WebGPU, the same page is ~8× slower and not conversational.

What made the difference wasn't any single model choice. It was putting each stage on the right processor
and **overlapping the stages** so the user rarely waits for one to finish before the next starts.

## Limitations

- One test machine. Integrated GPUs (Intel Iris Xe, Apple M-series) support WebGPU but weren't measured.
- Only tested in Chrome on Windows. Phones not tested (download size and memory are likely problems).
- Clean synthetic test audio; no accuracy measurement on real, noisy speech.
- The 350M language model is fluent but unreliable on facts (it gave two different temperatures for the same question).
- While the agent speaks, the mic is ignored unless the user enables interruptions (meant for headphones),
  because the agent would otherwise hear itself through the speakers.
- When a mid-sentence pause is merged by re-transcribing the joined audio, punctuation between the two parts can be lost.

## What's next

[H2](../h2-own-inference-engine/README.md): replace the libraries with our own inference engine.
The remaining bottleneck in H1 is the TTS sharing a small GPU with the LLM (~1 s to first audio).

## Reproduce

```bash
python -m http.server 8765 --bind 127.0.0.1          # from the repo root
pip install -r requirements.txt
python h1-browser-libs/bench/asr_reference.py h1-browser-libs/samples/test.wav
python h1-browser-libs/bench/latency_test.py
MIC=h1-browser-libs/bench/mic.wav python h1-browser-libs/bench/latency_test.py
NOGPU=1 python h1-browser-libs/bench/latency_test.py
python h1-browser-libs/bench/component_bench.py encoder w2a8:wasm:4 w2a8:webgpu
python h1-browser-libs/bench/component_bench.py tts wasm:q8 wasm:fp32 webgpu:fp32
```

URL options on the demo page: `?endpoint=500` (silence in ms that ends a turn), `?spec=160` (pause that triggers
speculative transcription; 0 disables it), `?asr=wasm` and `?tts=wasm` (force a stage onto the CPU).
