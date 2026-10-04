# 4. The language model

**Core idea:** a language model writes one word-piece ("token") at a time, each time asking "given
everything so far, what comes next?". Reading your question is fast and happens once; writing the answer is
a loop that runs once per token.

## The analogy: a typist who first skims the letter, then types the reply key by key

Before replying, the typist reads your whole letter in one go. That's quick, because they can take in many
words at once (**prefill**). Then they type the reply one key at a time, and before each key they glance back
over everything written so far (**decode**). How long you wait for the *first* key is set by the skim; how fast
the rest comes out is set by the typing speed.

```
system prompt + your question ──▶ [ prefill: read all tokens at once ] ──▶ first token   (~0.2 s)
                                                                              │
                     ┌────────────────────────────────────────────────────────┘
                     ▼
              [ decode: 1 token ] ──▶ [ decode: 1 token ] ──▶ ... ──▶ end    (12–23 tokens/s)
```

## The real numbers

| | Measured |
|---|---|
| Model | LFM2 350M (Liquid AI), 4-bit weights, 293 MB |
| Time to first token | 0.12–0.26 s |
| Typing speed | 12–23 tokens/s (GTX 1050) |
| Time until the first full sentence | 0.13–0.5 s |

Is 12–23 tokens/s fast enough? People speak about 150 words per minute:

```
150 words/min ÷ 60 ≈ 2.5 words/s  ≈ 3.3 tokens/s   (a word is ~1.3 tokens)
LLM writes 12–23 tokens/s  →  4–7× faster than the agent can say it
```

So once the first sentence exists, the language model is never the bottleneck. Only its *first* sentence
matters for latency.

## Why 4-bit, and why it's "q4" and not "q4f16" here

Weights stored in 4 bits instead of 32:

```
350 million weights × 4 bits ≈ 175 MB, plus the parts kept at higher precision → 293 MB download
```

On most GPUs the math itself would also run in 16-bit floats ("q4f16"), halving memory traffic again. The
GTX 1050 doesn't support 16-bit floats in WebGPU shaders, so the page checks for that feature and falls back to
"q4" (4-bit weights, 32-bit math).

The system prompt asks for "one or two short, natural sentences, no lists or markdown", because everything
it writes will be read aloud.

## The surprising part

Without WebGPU, the page originally **crashed on load**. Both LFM2 files (4-bit *and* 8-bit) use an operation
called `GatherBlockQuantized` to look up compressed word embeddings, and the CPU version of the runtime doesn't
have it. The CPU fallback now uses Qwen2.5 0.5B in 8-bit, which uses only standard integer operations. It
works, at 2–5 tokens/s.

Also: small models are fluent but not reliable. Asked the same question twice, LFM2 said Campinas summers
peak around 25 °C and then around 30 °C.

**Bottom line:** the language model reads your question in ~0.2 s and then writes 4–7× faster than speech, so
it only costs latency up to its first sentence.
