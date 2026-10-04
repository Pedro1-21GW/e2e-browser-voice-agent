# 5. Text to speech: Kokoro

**Core idea:** text is first converted to pronunciation symbols (phonemes), and a small network turns those
symbols plus a "voice fingerprint" into the actual sound wave, 24,000 numbers per second.

## The analogy: a voice actor who must finish each line before the previous one ends

The agent's reply is read sentence by sentence. Kokoro is a voice actor who gets one line at a time. While
they say line 1 out loud, they're preparing line 2. The show runs smoothly as long as **preparing a line is
faster than saying it**. If preparing is slower, there's dead air between lines.

That ratio is the **real-time factor**: seconds of audio produced per second of work.

```
× real time > 1  →  ready before it's needed, no gaps
× real time < 1  →  silence between sentences
```

## The pipeline

```
"The capital of France is Paris."
        │
        ▼
┌──────────────────────────────┐
│ Phonemizer (espeak-ng, CPU)  │  roughly: ðə kˈæpɪtəl ʌv fɹˈæns ɪz pˈæɹɪs
└──────┬───────────────────────┘
       │ + voice fingerprint (af_heart: 256 numbers that define the voice)
       ▼
┌──────────────────────────────┐
│ Kokoro 82M network (GPU)     │  phonemes → waveform
└──────┬───────────────────────┘
       │ 2.3 s of audio at 24 kHz = ~55,000 numbers
       ▼
     speakers
```

## The real numbers

| Runs on | "Hello!" | 31-char sentence | 95-char sentence | × real time |
|---|---|---|---|---|
| CPU, 8-bit | 3.3 s | 5.8 s | 15.9 s | **0.40** (dead air) |
| CPU, 32-bit | 2.5 s | 3.2 s | 9.6 s | **0.56–0.73** (dead air) |
| **GPU, 32-bit** | **0.72 s** | **0.88 s** | **2.0 s** | **1.9–3.1** (smooth) |

Notice the fixed cost: even "Hello!" (1.4 s of audio) takes 0.72 s on the GPU. Short first sentences don't
come out proportionally faster, which is why the time to the *first* audio is the slowest remaining stage
(~1 s, because the GPU is also busy with the language model at that moment).

## The surprising part

On the CPU, the **8-bit version was slower than the 32-bit one** (0.40× vs 0.56–0.73× real time). Smaller
numbers only help if there's a fast way to multiply them; WebAssembly's 8-bit path isn't faster than its 32-bit
path, and unpacking the 8-bit weights costs extra. "Compressed" doesn't automatically mean "fast".

The first version ran Kokoro on the CPU, and it also blocked the language model, since they shared one thread:
first audio took 3.8–23.5 s. Moving it to the GPU brought that to ~1 s.

**Bottom line:** Kokoro on the GPU produces speech 2–3× faster than it's spoken, so after the first sentence
there are no gaps; on the CPU it can't keep up with its own voice.
