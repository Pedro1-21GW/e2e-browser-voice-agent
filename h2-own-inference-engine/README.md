# H2: Our own inference engine

> **Hypothesis:** we can replace the browser ML libraries (ONNX Runtime Web, Transformers.js, kokoro-js) with an
> inference engine we write ourselves (our own WebGPU kernels and model code), still ship everything as one
> HTML file, and stay at least as fast as H1.

> **Experimental.** This is a research experiment, not a product: tested on one laptop (GTX 1050, Chrome on Windows), with three test prompts and clean synthetic audio. Expect rough edges on other hardware and browsers.

**Status: 🔬 in progress. Milestone 1 (the language model) done, on the GPU and on the CPU.** Qwen2.5 0.5B runs on
our own engine, gives the same tokens as Hugging Face Transformers, writes **6.5× (GPU) and 6× (CPU) faster** than
the libraries on the same model, and already powers two versions of the H1 voice agent. On a laptop with no usable
GPU, the agent now answers in **0.6–1.9 s** instead of 12–18 s. Details in the log below and in the
[Hypo 2 wiki](wiki/Home.md).

| Try it (local: `python serve.py`, then open) | What it is |
|---|---|
| [m1-llm/](m1-llm/index.html) | LLM lab: load the model on GPU or CPU, check it against the reference, benchmark, chat |
| [../h1-browser-libs/?profile=h2](../h1-browser-libs/index.html?profile=h2) | The voice agent with our engine as its language model, on the GPU |
| [../h1-browser-libs/?profile=h2cpu](../h1-browser-libs/index.html?profile=h2cpu) | The same, CPU only (Moonshine + our engine + the device's voice) |
| [m0-kernels/](m0-kernels/index.html) | The first kernel experiments |

## Why it matters

H1 proved the *idea* works with off-the-shelf libraries. But those libraries are black boxes: the page ended up
with four copies of ONNX Runtime, the speech model needed a workaround because one 140 KB preprocessor used
64-bit math, and the language model could only run on the CPU after switching to a different model.
Writing the engine ourselves means:

- **Understanding every millisecond.** We can see exactly which operation is slow and why.
- **Kernels shaped for these exact models**, such as a matrix multiply that reads the ternary (−1/0/+1)
  weights directly.
- **One small runtime** instead of four large ones.

It's also exactly the work of a model-optimization engineer: kernels, quantization, memory layout, measured on
real hardware.

## What "ourselves" means (the rules)

| Allowed | Not allowed |
|---|---|
| Published model weights (safetensors / GGUF / ONNX files read as plain data) | ONNX Runtime, Transformers.js, kokoro-js, WebLLM, or any ML runtime |
| WebGPU and WebAssembly as the "hardware" | Server-side inference |
| Plain JavaScript for orchestration | |
| **Temporary exception, to be removed by the end:** the TTS phonemizer (espeak-ng). (The tokenizer was on this list; it's now ours.) | |

## Success criteria

1. **Correct:** each model's outputs match a reference implementation (numpy/PyTorch) within a stated
   tolerance, and produce the same tokens / transcripts on the test clips.
2. **Fast:** each stage at least matches H1 on the same laptop:

   | Stage | H1 number to beat | H2 so far |
   |---|---|---|
   | Speech encoder, 6 s clip | ~180 ms | not started (M2) |
   | LLM decode | 12–23 tokens/s | ✅ **98 tokens/s** on the GPU, **28–33** on the CPU (Qwen2.5 0.5B, a larger model than H1's LFM2 350M) |
   | TTS | ≥ 1.9× real time | not started (M4) |
   | Voice-to-voice through the mic | 1.4–2.1 s | GPU: same range so far (the voice is now the slow stage); CPU only: 0.6–1.9 s (H1: 12–18 s) |
3. **One HTML file** (+ the service worker), no ML libraries. (During development the engine is a handful of ES
   modules plus one WebAssembly file; inlining them is part of M5.)
4. **(Added in M1) Good on the CPU too.** Phones and many laptops have no WebGPU; the engine has a CPU backend
   (C compiled to WebAssembly with SIMD and threads) held to the same correctness check.

## Plan

Ordered from easiest to verify to hardest. Each milestone ends with a measured comparison against H1.

```
M0  GPU harness + kernels ──▶ M1  LLM ──▶ M2  Parakeet encoder ──▶ M3  VAD ──▶ M4  TTS ──▶ M5  all in one page
    (matmul, norms,             (small,      (ternary kernel,        (tiny,      (hardest:     (replace H1's
     attention, quantized)       standard)    FastConformer)          CPU ok)     vocoder)      stages one by one)
```

| # | Milestone | What gets built | Done when |
|---|---|---|---|
| **M0** | GPU harness + kernels | WebGPU setup, buffer helpers, timing; kernels for matrix-vector and matrix-matrix multiply (fp32, 4-bit, ternary), RMSNorm/LayerNorm, softmax, RoPE, SiLU, attention | each kernel matches a CPU reference; bandwidth measured against the GPU's limit |
| **M1** ✅ | Language model | Llama-style transformer forward pass with KV cache, on WebGPU and on the CPU (WebAssembly); Qwen2.5-0.5B (same model as H1's CPU fallback, for direct comparison) | greedy tokens match the reference; tok/s vs Transformers.js |
| **M2** | Parakeet encoder | Subsampling convolutions + 24 Conformer blocks (feed-forward, relative-position attention, depthwise conv); ternary matmul kernel; the H1 decoding loop with our own LSTM predictor + joint | same transcripts as H1; encoder ≤ 180 ms for 6 s |
| **M3** | Silero VAD | small conv + LSTM network, likely on the CPU in plain JS | same speech probabilities as H1 |
| **M4** | Kokoro TTS | text encoder, duration/prosody predictor, iSTFTNet vocoder | audio matches the reference closely; ≥ 1.9× real time |
| **M5** | Integration | the H1 page with each library swapped for our engine, one stage at a time | voice-to-voice ≤ H1 |

Prior work (papers and engines) is collected in the wiki: [Prior work](wiki/06-prior-work.md). Useful prior work in sibling projects: `heavybuffer-parakeet-cpp` already runs Parakeet TDT 0.6B v3 from a
GGUF file with parakeet.cpp, which documents the weight layout we'll need in M2.

## How each kernel is checked

```
same random inputs ──┬──▶ CPU reference (plain JS / numpy) ──┐
                     │                                       ├──▶ max |difference| < tolerance ?
                     └──▶ our WebGPU kernel ─────────────────┘
                                   │
                                   └──▶ time it (many runs) ──▶ GB/s or GFLOP/s vs the GPU's limit
```

The GPU's limit gives each number meaning. During the LLM's decode step, every weight is read once per token
and used for just one multiply-add, so speed is set by **memory bandwidth**, not arithmetic:

```
GTX 1050 laptop memory bandwidth ≈ 112 GB/s (spec sheet)
LFM2 350M at 4-bit ≈ 0.2 GB of weights read per token
→ best case ≈ 112 / 0.2 ≈ 560 tokens/s;  H1 gets 12–23
```

That gap is the opportunity H2 is betting on.

## Log

### M0, round 1: first kernels ([m0-kernels/](m0-kernels/index.html))

Six hand-written WebGPU kernels, each checked against a plain-JavaScript reference on random data, timed over
50–200 back-to-back calls. Run with `python h2-own-inference-engine/m0-kernels/run.py`.

**All kernels are correct**: relative error 1e-7 (matrix-vector) and 1e-6 (matrix-matrix), i.e. float rounding.

**Matrix × vector** (one token through one layer, the LLM decode case). Best thread count per row shown:

| Weights | Shape | Weight bytes | Time | Weight bandwidth |
|---|---|---|---|---|
| fp32 | 4096×1024 | 16.8 MB | 0.25 ms | 61–67 GB/s (55–60% of the 112 GB/s limit) |
| 4-bit | 4096×1024 | 2.6 MB | 0.11–0.12 ms | 21–24 GB/s |
| ternary (2-bit) | 4096×1024 | 1.2 MB | 0.13–0.15 ms | 8–9 GB/s |
| empty kernel | 4096 workgroups | – | **0.024 ms** | (overhead floor) |

**Matrix × matrix** (many positions at once: the speech encoder and LLM prefill), 80×1024 · 1024×4096:

| Kernel | Time | Speed | Share of the GTX 1050's ~1.9 TFLOP/s |
|---|---|---|---|
| naive (one thread per output, all reads from memory) | 21.7 ms | 31 GFLOP/s | 2% |
| tiled 16×16 (shared-memory tiles) | 5.2 ms | 128 GFLOP/s | 7% |
| tiled + 4×4 outputs per thread | 8.3 ms (5.2 ms at 256 rows: 135 GFLOP/s) | 81 GFLOP/s | 4% |

**What we learned**

1. **Matching threads to work matters.** With 256 threads per row, a 4-bit row of 1,024 values has only 128
   packed words, so half the threads sit idle; 32 threads per row cut the 4-bit time from 0.21 to 0.11 ms.
2. **A wrong guess, corrected by measurement.** The quantized kernels stopped improving at ~0.12 ms, which
   looked like fixed per-call overhead. An empty kernel costs only 0.008–0.024 ms, so that's not it.
   The actual cost: every row re-reads the whole input vector. For 4096 rows × 1024 inputs that's 16.8 MB of
   (cached) reads, as much as the entire fp32 weight matrix. Compressing the weights 6–14× can't help much while
   the input reads stay the same size.
3. **Tiling gives 4× on matrix-matrix, but we're at 7% of peak.** By a rough estimate, the speech encoder does
   ~90 GFLOP for a 6 s clip (≈2 × 0.6 B weights × 76 positions), and H1's ONNX Runtime does it in ~180 ms,
   i.e. ~500 GFLOP/s. Our best matmul would need ~0.7 s. There's a ~4× gap to close.
4. The 4×4-per-thread version loses at 80 rows: its 64-row tiles waste 48 of 128 computed rows, and there are
   too few workgroups to keep the GPU busy. Kernel shape has to fit the problem shape.

**Next (M0, round 2)**
- Matrix × vector: load the input once into shared memory and compute several rows per workgroup.
- Matrix × matrix: vector (vec4) loads, bigger K tiles, a tile shape that fits 76–80 rows, then fused
  dequantization so 4-bit and ternary weights are unpacked while loading the tile.
- Then RMSNorm / LayerNorm, softmax, RoPE and attention, which M1 (the LLM) needs.

### M1: the language model, on the GPU and on the CPU ([m1-llm/](m1-llm/index.html), [engine/](https://github.com/Pedro1-21GW/e2e-browser-voice-agent/tree/main/h2-own-inference-engine/engine))

**What was built** (no ML library; ~1,700 lines):

```
gguf.js       read the GGUF file (metadata, tensor table, weights as bytes), cached in Cache Storage
tokenizer.js  byte-level BPE from the file's own vocabulary and merges + Qwen's chat format
llm.js        load on "webgpu" or "cpu"; chat() streams text and keeps the KV cache between turns
webgpu.js     WGSL kernels: embedding, RMSNorm, quantize, Q4_0 / Q8_0 matrix × vector, RoPE + cache, attention,
              SwiGLU, sampling (argmax / Gumbel); one command buffer per pass; the next token never leaves the GPU
kernels.c     the same model in C → WebAssembly (SIMD128 or relaxed SIMD, 1 or n threads), built by build.py (Zig)
cpu*.js       leader worker + helper workers on one shared WebAssembly memory
```

Model: Qwen2.5 0.5B Instruct, `qwen2.5-0.5b-instruct-q4_0.gguf` (429 MB: layers in Q4_0, output table in Q8_0).

**Correct.** [reference.py](m1-llm/reference.py) runs the same GGUF file through Hugging Face Transformers. The
tokenizer matches on 13/13 strings and prompts. Top-5 first-step scores: within 0.0005 on the GPU; within 0.11–0.25
on the CPU (8-bit activations, as llama.cpp). Greedy tokens: identical up to the end of the reply on all three
prompts, both backends, every thread count.

**Fast.** [bench.py](m1-llm/bench.py), same laptop, same day; "baseline" = Transformers.js 4.3 + ONNX Runtime Web
1.30 on the same model ([baseline.html](m1-llm/baseline.html)):

| | Prompt, 64 tokens | Prompt, 256 tokens | Writing | Voice-turn first token* |
|---|---|---|---|---|
| baseline, GPU (q4) | 238 ms | 702 ms | 15.3 tokens/s | – |
| **ours, GPU** | 178 ms | 766 ms | **98 tokens/s** | **69 ms** |
| baseline, CPU (q8, 4 threads) | 1,028 ms | 3,951 ms | 5.0 tokens/s | – |
| ours, CPU, 1 thread, plain SIMD | 2,928 ms | 10,936 ms | 12.9 tokens/s | 745 ms |
| ours, CPU, 1 thread, relaxed SIMD | 1,728 ms | 7,126 ms | 16.6 tokens/s | 634 ms |
| ours, CPU, 4 threads | 703 ms | 3,010 ms | 31.3 tokens/s | 211 ms |
| **ours, CPU, 8 threads for prompts / 4 for writing** | **571 ms** | **2,597 ms** | **28–33 tokens/s** | **163 ms** |

\* system prompt and one earlier exchange already in the cache, 18 new tokens. The laptop's speed varies ±15% from
run to run (heat), so writing speeds are given as ranges where runs differed.

**GPU log** (per-kernel times from timestamp queries, `--enable-webgpu-developer-features`):

| Step | gate+up matrix, one token | Whole token | Note |
|---|---|---|---|
| v1: thread per weight word, 4 rows per workgroup | 134 µs | 10.2 ms | 79% of the time in matrix × vector |
| v1, 16 rows per workgroup (sweep of rows × workgroup size) | ~100 µs | 8.0 ms | more loads in flight |
| v2: input in shared memory, a whole block per lane, RMSNorm/SwiGLU fused | 242 µs | 13.7 ms | **slower**: 32-way shared-memory bank conflicts |
| v3: + int8 input, `dot4I8Packed` (DP4A), quantized in every workgroup | 219 µs | 12.6 ms | **slower**: 1,216 workgroups redo the same setup |
| v4: input quantized once by a small kernel, DP4A matrix × vector | 107 µs | 8.5–9.5 ms | ties v1; kept (llama.cpp's MMVQ approach, fewer kernels) |

- A plain read kernel on this GPU reaches **76 GB/s** (spec: 112). 345 MB per token → 4.5 ms floor; we're at ~9–10 ms.
- The output table (153 MB) runs at 61–68 GB/s (80–90% of the practical limit); the 4-bit layer matrices at
  40–46 GB/s. An empty dispatch costs 5–10 µs and a token needs ~240 dispatches: ~2 ms of launch overhead.
- Reading the token back costs nothing measurable (GPU time 10.16 ms vs wall time 10.13 ms).

**CPU log:**

- Toolchain: `pip install ziglang`, then [build.py](engine/build.py) compiles kernels.c into 4 variants
  (1 thread / threads × SIMD128 / relaxed SIMD); the page picks the best one the browser supports.
- `-ffast-math` made the relaxed build output NaN everywhere; removed.
- Threads were bit-identical with 2 but garbage with 3–4, and deadlocked with mixed thread counts: helpers read the
  job's "compute logits" flag after the last barrier, when the leader could already be setting up the next pass.
  Every thread now copies the job's parameters before the first barrier. Barrier self-test: 20,000 rounds, 0 errors,
  0.3–0.7 µs per barrier.
- Profile (one token, 34.8 ms): output table 12.3 ms, gate+up 12.2 ms, down 6.1 ms. Memory bandwidth from
  WebAssembly: **14–17 GB/s with 1–8 threads** (one DDR4-2400 stick, 19.2 GB/s on paper), so writing is
  memory-bound: ceiling ~43 tokens/s.
- 4 rows per pass (one token) and 2-row × 4-token tiles (prompts): 256-token prompt 5.6 → 4.0 s; then 8 threads for
  prompts and 4 for writing: 2.6–3.0 s, while 8 threads for writing is slower (hyperthreads fight over memory).
- Not done (measured first): dropping the 34% of the vocabulary that isn't Latin script from the output table would
  save ~10% of writing time; not worth the risk for now.

**In the voice agent** (H1 page, `?profile=h2` / `?profile=h2cpu`; [latency_test.py](../h1-browser-libs/bench/latency_test.py), test clips):

| | First token | First sentence | First sound after it | Voice-to-voice |
|---|---|---|---|---|
| GPU, H1 Normal (LFM2 350M, library) | 189–258 ms | 191–436 ms | 0.72–1.70 s | 1.31–2.45 s |
| GPU, Hypo 2 engine (Qwen2.5 0.5B) | 59–106 ms | 126–255 ms | 0.91–1.99 s | 1.24–2.67 s |
| CPU only, H1 Super-lite (SmolLM2 135M, library) | 0.43–0.90 s | 1.2–1.6 s | 0.8–1.8 s | 2.2–3.9 s |
| **CPU only, Hypo 2 engine (Qwen2.5 0.5B)** | **0.16–0.39 s** | **0.34–1.04 s** | 0.08–0.14 s | **0.56–1.95 s** |

- A faster model first made the GPU voice *slower* (2.0 s to the first sound): the engine kept the GPU busy while
  Kokoro waited. Fix: after the first sentence, pause until the voice makes a sound (and flush the voice's sentence
  splitter first, or both wait on each other for 4 s).
- The engine gets the last 40 messages instead of 6, so the cache stays valid (a sliding window recomputed the whole
  conversation: 501 ms instead of ~70).
- Page memory, CPU version: 3.2 GB → 1.66 GB after reading the model file once instead of cloning the response.

**Verdict for M1: confirmed.** Our own engine runs the language model correctly and faster than the libraries, on
the GPU and on the CPU, inside the voice agent.

**Next**
- GPU: a tiled matrix × matrix kernel for prompts (now on par with the library, 766 vs 702 ms for 256 tokens);
  fewer, fused dispatches for one token (~240 today).
- M4 (the voice) is now the slow stage on the GPU, and M2 (the speech encoder) is the slow stage on the CPU
  (Moonshine on 1 thread, ~0.4–0.8 s).
- Stream the weights straight into GPU / WebAssembly memory instead of holding the file in JavaScript.
