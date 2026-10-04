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

Results are added here as milestones are run.
