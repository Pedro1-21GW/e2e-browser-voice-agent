# H2: Our own inference engine

> **Hypothesis:** we can replace the browser ML libraries (ONNX Runtime Web, Transformers.js, kokoro-js) with an
> inference engine we write ourselves (our own WebGPU kernels and model code), still ship everything as one
> HTML file, and stay at least as fast as H1.

**Status: 🔬 started.** Milestone 0 (GPU harness + first kernels) in progress; see the log at the bottom.

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
| **Temporary exceptions, to be removed by the end:** the tokenizer and the TTS phonemizer (espeak-ng) | |

## Success criteria

1. **Correct:** each model's outputs match a reference implementation (numpy/PyTorch) within a stated
   tolerance, and produce the same tokens / transcripts on the test clips.
2. **Fast:** each stage at least matches H1 on the same laptop:

   | Stage | H1 number to beat |
   |---|---|
   | Speech encoder, 6 s clip | ~180 ms |
   | LLM decode | 12–23 tokens/s |
   | TTS | ≥ 1.9× real time |
   | Voice-to-voice through the mic | 1.4–2.1 s |
3. **One HTML file** (+ the service worker), no ML libraries.

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
| **M1** | Language model | Llama-style transformer forward pass with KV cache; start with SmolLM2-135M for fast iteration, then Qwen2.5-0.5B (same model as H1's CPU fallback, for direct comparison) | greedy tokens match the reference; tok/s vs Transformers.js |
| **M2** | Parakeet encoder | Subsampling convolutions + 24 Conformer blocks (feed-forward, relative-position attention, depthwise conv); ternary matmul kernel; the H1 decoding loop with our own LSTM predictor + joint | same transcripts as H1; encoder ≤ 180 ms for 6 s |
| **M3** | Silero VAD | small conv + LSTM network, likely on the CPU in plain JS | same speech probabilities as H1 |
| **M4** | Kokoro TTS | text encoder, duration/prosody predictor, iSTFTNet vocoder | audio matches the reference closely; ≥ 1.9× real time |
| **M5** | Integration | the H1 page with each library swapped for our engine, one stage at a time | voice-to-voice ≤ H1 |

Useful prior work in sibling projects: `heavybuffer-parakeet-cpp` already runs Parakeet TDT 0.6B v3 from a
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
