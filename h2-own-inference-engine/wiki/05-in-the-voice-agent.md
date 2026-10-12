# 5. Inside the voice agent

**Core idea:** the engine now replaces the language-model library in the H1 voice agent (two new versions in its
menu). On a laptop without a usable GPU, the agent goes from unusable to conversational. On the GPU, the language
model is no longer the slow part; the voice is.

## The two versions

| Version | Speech to text | Language model | Voice |
|---|---|---|---|
| Hypo 2 engine, GPU | Parakeet (GPU) | **Qwen2.5 0.5B on our WebGPU kernels** | Kokoro (GPU) |
| Hypo 2 engine, CPU only | Moonshine tiny (CPU) | **Qwen2.5 0.5B on our C/WebAssembly kernels** | the device's own voice |

Open [the agent with `?profile=h2`](../../h1-browser-libs/index.html?profile=h2) or [`?profile=h2cpu`](../../h1-browser-libs/index.html?profile=h2cpu) (or pick it in the Version menu).

## CPU only: from 12–18 s to about 1 s

Test clips through the whole agent, WebGPU hidden from the page:

| Version | Language model | First token | First sentence | Voice-to-voice |
|---|---|---|---|---|
| H1 Normal, no GPU | Qwen2.5 0.5B (library) | – | – | **12–18 s** (measured in H1) |
| H1 Super-lite | SmolLM2 135M (library) | 0.43–0.90 s | 1.2–1.6 s | 2.2–3.9 s |
| **Hypo 2 engine, CPU** | **Qwen2.5 0.5B (ours)** | **0.16–0.39 s** | **0.34–1.04 s** | **0.56–1.95 s** |

And it is the better model: Super-lite's SmolLM2 answered the Campinas question in Fahrenheit with invented numbers.

## The surprising part: a faster model made the voice slower

On the GPU, the first long answers took **2.0 s** from "first sentence ready" to "first sound", against 0.7–1.4 s in
H1. The engine now writes ~20× faster than anyone can speak, so it kept the GPU busy, and Kokoro (on the same GPU)
waited behind it. The fix is **backpressure**: once the first sentence has gone to the voice, the engine pauses until
the voice makes a sound, then continues. Pausing costs nothing audible: the rest of the reply still arrives long
before it is needed.

```
before:  LLM ████████████████████  (keeps the GPU busy)
         TTS     ░░░░░░░░██████        first sound late
after:   LLM ████ (first sentence) ··pause··  ████████
         TTS     ██████ first sound ▶
```

The first version of the pause deadlocked for 4 s on short answers: Kokoro's sentence splitter waits for more text
(or a flush) before speaking a sentence, and the paused engine sent no more text. The engine now flushes the voice
before pausing.

## GPU: the language model got fast, the voice didn't

Same harness, same session, test clips:

| | First token | First sentence | First sound after the first sentence | Voice-to-voice |
|---|---|---|---|---|
| H1 Normal (LFM2 350M, library) | 189–258 ms | 191–436 ms | 0.72–1.70 s | 1.31–2.45 s |
| Hypo 2 engine (Qwen2.5 0.5B, ours) | **59–106 ms** | **126–255 ms** | 0.91–1.99 s | 1.24–2.67 s |

End to end it's a draw, because Kokoro's time depends mostly on how long the first phrase is (Qwen opened with a
12-word clause; LFM2 with "Hello!"). On the GPU, the next stage to own is the voice (milestone M4).

## Two smaller fixes the engine needed

- **Keep the history long.** H1 sends the last 6 messages; once the window slides, the prompt's beginning changes and
  the whole conversation is recomputed (one turn took 501 ms instead of ~70). The engine gets the last 40 messages
  and keeps its cache.
- **Don't hold the model twice.** The first CPU version peaked at **3.2 GB** of page memory: the downloader cloned
  the response to cache it, keeping two copies of the 429 MB file. Reading it once into a growing buffer brought the
  peak to **1.66 GB** (Super-lite: 0.98 GB).

**Bottom line:** with our engine, a laptop CPU alone holds a spoken conversation with a 0.5 B model (about 1 s to the
first sound); on the GPU the language model is now 2–3× quicker to its first sentence, and the voice is the next
bottleneck.
