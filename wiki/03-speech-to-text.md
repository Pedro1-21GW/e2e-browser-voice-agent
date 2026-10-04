# 3. Speech to text: Parakeet

**Core idea:** the recording is turned into a picture of which frequencies are loud at each moment, a big
network reads that picture and summarizes each 80 ms of it, and a small loop walks along those summaries
deciding at each step "write a word-piece here, or skip ahead".

## The analogy: a court stenographer reading a strip of film

Imagine your speech printed as a long strip of film, one frame every 80 ms. A stenographer slides along the
strip. At each frame they make two decisions at once:

1. **What to type**: a word-piece like "▁cap", or nothing ("blank").
2. **How far to jump**: stay on this frame (0), or skip 1, 2, 3 or 4 frames ahead.

That second decision is what **TDT** (Token-and-Duration Transducer) means. Older models (RNN-T) could only
move one frame at a time and had to say "blank" at every silent frame. TDT can leap over silence and over
the middle of long sounds, so it needs fewer steps.

## The three parts

```
16,000 samples/s of audio
        │
        ▼
┌──────────────────────────────────────────┐
│ Preprocessor (nemo128.onnx, CPU)         │  sound → "mel spectrogram":
│                                          │  128 frequency bands, one column every 10 ms
└──────┬───────────────────────────────────┘
       │ 6.06 s of audio → 606 columns × 128 bands
       ▼
┌──────────────────────────────────────────┐
│ Encoder (24 Conformer layers, GPU)       │  reads the whole picture at once,
│ 2-bit weights, 190 MB                    │  squeezes time 8× → one summary every 80 ms
└──────┬───────────────────────────────────┘
       │ 606 ÷ 8 ≈ 76 frames, each a list of 1,024 numbers
       ▼
┌──────────────────────────────────────────┐
│ Decoder loop (hand-written, CPU)         │  the stenographer:
│ decoder_joint.onnx, 18 MB                │  at frame t → (token, jump)
└──────┬───────────────────────────────────┘
       │ 40 steps for 76 frames (it skipped a lot)
       ▼
"Hi there. Can you tell me what the weather is usually like in Campinas during the summer?"
```

The loop itself is short enough to read:

```
t = 0
while t < number_of_frames:
    scores = decoder_joint(frame[t], last_token, memory)
    token  = best of the first 8,193 scores      (8,192 word-pieces + 1 "blank")
    jump   = best of the last 5 scores           (0, 1, 2, 3 or 4 frames)
    if token is not blank: write it down, update memory
    if jump > 0: t += jump
    elif token is blank: t += 1                  (nothing to say here, move on)
```

It was written in Python first ([asr_reference.py](../h1-browser-libs/bench/asr_reference.py)), checked
word-perfect on the test clips, and then ported to JavaScript. No browser library supports Parakeet, so this
loop is the first piece of "our own engine".

## Why "Redux": weights that are only −1, 0 or +1

A neural network is mostly multiplication tables of learned numbers ("weights"). Normally each weight is a
32-bit decimal. Moondream retrained Parakeet's encoder so that **every weight is −1, 0 or +1** (scaled by one
shared number per block). Three possible values need only log₂(3) ≈ 1.58 bits; the file stores them in 2 bits.

```
~600 million weights × 32 bits = 2.4 GB    (normal fp32)
~600 million weights ×  2 bits = 0.15 GB   (+ a few full-precision layers ≈ 190 MB file)
                                  → 12× smaller download, accuracy within 0.3 points of the original
```

## Where it runs, and the numbers

| Encoder on | 6.06 s clip | Why |
|---|---|---|
| Browser CPU, 1 thread | 6.0 s | the 2-bit math has no fast WebAssembly path: ~3.3× slower than native |
| Browser CPU, 4 threads | 2.6–3.1 s | 4 workers, ~2× faster |
| **Browser GPU (WebGPU)** | **~180 ms** | the GTX 1050 has 640 small cores doing multiplications in parallel |

The encoder is ~95% of the work: on the GPU the whole transcription of a 2–4 s utterance takes 0.13–0.5 s.

## The surprising part

The GPU version first refused to load with *"no implementation for Cast"*. The culprit wasn't the encoder:
it was the **preprocessor**, which converts audio to 64-bit floats for precision, and the GPU build of ONNX
Runtime doesn't include 64-bit math at all. The fix was to load ONNX Runtime twice: the CPU build for the
preprocessor and decoder, the GPU build for the encoder. A 140 KB file was blocking a 15× speed-up.

**Bottom line:** a spectrogram goes in, a 24-layer encoder on the GPU summarizes it in ~0.2 s, and a ten-line
"type or jump" loop on the CPU writes the words.
