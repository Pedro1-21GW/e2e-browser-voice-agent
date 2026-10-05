# H1: Browser libraries + cache

> **Hypothesis:** a single HTML file, using existing browser ML libraries and the browser's cache, can run a
> voice agent (speech recognition → language model → speech synthesis) with low enough latency to hold a conversation.

**Verdict: confirmed when the browser has WebGPU, rejected when it doesn't.**
On a 2019 laptop with a GTX 1050, the agent starts answering **1.4–2.1 s** after you stop talking.
Without WebGPU the same page works but takes **12–18 s**, too slow for a conversation.

[Try the demo](https://pedro1-21gw.github.io/e2e-browser-voice-agent/h1-browser-libs/) (Chrome or Edge on a computer: ~830 MB download on first visit, cached afterwards.
On a phone or without WebGPU the page picks a [lighter version](#phone-versions-lite-and-super-lite): ~290 MB or ~170 MB.)

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

This is the **Normal** version; the phone versions are [further down](#phone-versions-lite-and-super-lite).

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

### Phone versions: Lite and Super-lite

On an iPhone 12 (4 GB of RAM, iOS 18, so Safari has no WebGPU) the page kept reloading while it loaded the
models. iOS closes a tab whose memory peaks too high, and the Normal version peaked at **3.4 GB** in the page's
process plus **2.6 GB** in the GPU process. The page now has a version picker; it chooses Normal on a computer with
WebGPU, Lite on a phone with WebGPU, and Super-lite without WebGPU.

| | Normal | Lite | Super-lite |
|---|---|---|---|
| Speech to text | Parakeet 0.6B Redux (GPU) | Moonshine tiny, 8-bit, **English only** (CPU) | same as Lite |
| Language model | LFM2 350M, 4-bit (GPU) | LFM2.5 350M, 4-bit (GPU) | SmolLM2 135M, 8-bit (**CPU**) |
| Voice | Kokoro 82M, fp32 (GPU) | the device's own voice | same as Lite |
| Download | ~830 MB | ~290 MB | ~170 MB |
| Needs WebGPU | yes (without it: 12–18 s per reply) | yes | **no** |
| Voice-to-voice (clips) | 1.6–2.6 s | **0.5–1.2 s** | 2.3–4.0 s |
| LLM speed | 6–19 tok/s | 46–64 tok/s | 17–19 tok/s |
| Peak memory, page process | 3.2 GB | **0.9 GB** | **0.9 GB** (0.65 GB once loaded, on one thread) |
| Peak memory, GPU process | 2.0 GB | 0.8 GB | 0.14 GB (not used) |

Same laptop, test clips, Chrome; Super-lite with WebGPU hidden. Memory is the process's private memory sampled
every 200 ms by [bench/latency_test.py](bench/latency_test.py). The one-thread figure (how Safari runs it, see below)
comes from a fresh profile with the service worker blocked.

What decided each choice:

- **Memory is mostly not the files.** ONNX Runtime holds a model's bytes, its own copy and a re-laid-out copy of the
  weights, and its WebAssembly memory never shrinks after a peak. Each copy of the runtime also costs ~115 MB
  before any model. Measured alone: Moonshine (28 MB) → ~250 MB, SmolLM2 (137 MB) → 600–680 MB, Kokoro fp32 (325 MB)
  → ~1 GB plus 0.8–0.9 GB on the GPU. Turning off weight pre-packing or the memory arena didn't help (both together
  made it 2.6 GB and slower).
- **One copy of ONNX Runtime.** Transformers.js and kokoro-js import their dependencies by name, so an import map
  points them at the runtime the page already loads, instead of each bringing its own: −115 to −190 MB per copy, same
  speed and outputs. A side effect: one runtime can't create two WebGPU sessions at once, so every version now loads
  one model at a time (Normal still loads in 7.5 s from cache, as before).
- **Moonshine tiny instead of Parakeet:** 0.77 s for the 6 s clip on one CPU thread (Parakeet: 6.0 s), both clips
  word-perfect. The cost is English only, which matches the agent's English prompt and voices.
- **The device's voice instead of Kokoro:** 8-bit Kokoro, the small one, is unusable on a GPU (17 s for a 5 s
  sentence on the GTX 1050, against 1.9 s for fp32), and fp32 alone needs ~1 GB. The fp16 version (163 MB) needs fp16
  shaders, which this laptop lacks, so it wasn't measured. The built-in voice costs nothing to download and starts
  speaking right away. Its first-audio time is taken at the utterance's start event, which waits for the main thread,
  so on Super-lite it includes finishing the language model's current step.
- **SmolLM2 135M** is the only chat model small enough for the CPU that still answers in sentences: 17–19 tok/s on
  the laptop's CPU, but often wrong (it gave Campinas's summer in Fahrenheit and invented facts about octopuses).
  Both small versions **stop replies after two sentences**: small models ramble, and a phone pays for every token.
- **Safari will likely run on one thread.** As far as we know, Safari ignores the `credentialless` header
  [sw.js](sw.js) uses for multithreading.
  The one-thread run above (0.93 GB peak) is the closest measurement to an iPhone here.

## Problems found along the way (and fixes)

| Problem | Cause | Fix |
|---|---|---|
| Speech-recognition encoder failed on WebGPU with "no implementation for Cast" | The *preprocessor* (not the encoder) converts audio to float64, which ONNX Runtime's WebGPU build doesn't include | Two copies of ONNX Runtime: the WebAssembly build runs the preprocessor, the WebGPU build runs the encoder |
| Multithreaded WebAssembly unavailable on GitHub Pages | Threads need `SharedArrayBuffer`, which needs two HTTP headers (COOP/COEP) that GitHub Pages can't set | A tiny service worker ([sw.js](sw.js)) adds the headers to the page's own responses |
| The agent answered "Hi there." and dropped the rest of the question | The test voice paused for over 500 ms mid-question, and the mic was ignored while the agent was busy | If you speak again *before the agent makes a sound*, the pending reply is cancelled and both parts are merged |
| Page crashed without WebGPU | Both LFM2 exports (4- and 8-bit) use `GatherBlockQuantized`, which the CPU runtime lacks | CPU path switches to Qwen2.5 0.5B 8-bit, which uses standard integer ops |
| First GPU call of each model took 2–3 s | WebGPU compiles shaders on first use | Warm-up pass after loading (two audio lengths, one LLM call, one TTS call) |
| iPhone 12 reloaded the page while loading | Normal peaks at 3.4 GB (page) + 2.6 GB (GPU); the phone has 4 GB | Lite and Super-lite versions, one copy of ONNX Runtime, one model at a time, downloads written into a single buffer |
| "another WebGPU EP inference session is being created" | With one shared runtime, two models can't start their WebGPU sessions at the same time | Load the models one after another |
| Small models cut at two sentences left a stray word ("…institutions. It") | Text arrives a word at a time, so the next word was already out when the cap triggered | Trim the reply at the end of the last sentence |

## Verdict

**Confirmed, with a condition: the browser must have WebGPU.** With it, a single static HTML file reaches
1.4–2.1 s voice-to-voice on a modest 2019 laptop, using only off-the-shelf libraries plus a hand-written
decoding loop. Without WebGPU, the same page is ~8× slower and not conversational.

What made the difference wasn't any single model choice. It was putting each stage on the right processor
and **overlapping the stages** so the user rarely waits for one to finish before the next starts.

## Limitations

- One test machine. Integrated GPUs (Intel Iris Xe, Apple M-series) support WebGPU but weren't measured.
- Only tested in Chrome on Windows. The phone versions were measured on the laptop (WebGPU hidden, one thread) and
  checked in phone emulation, **not yet on a real phone**. How much memory iOS allows a tab isn't published, so
  whether 0.9 GB fits an iPhone 12 is still to be confirmed.
- Lite and Super-lite understand English only, and Super-lite's language model is often wrong on facts.
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
VERSION=lite python h1-browser-libs/bench/latency_test.py
NOGPU=1 VERSION=superlite python h1-browser-libs/bench/latency_test.py
python h1-browser-libs/bench/component_bench.py encoder w2a8:wasm:4 w2a8:webgpu
python h1-browser-libs/bench/component_bench.py tts wasm:q8 wasm:fp32 webgpu:fp32
```

URL options on the demo page: `?endpoint=500` (silence in ms that ends a turn), `?spec=160` (pause that triggers
speculative transcription; 0 disables it), `?asr=wasm` and `?tts=wasm` (force a stage onto the CPU),
`?profile=normal|lite|superlite` (pick the version), `?tts=system` (the device's own voice in any version).
