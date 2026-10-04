# 1. The big picture

**Core idea:** a voice agent is four specialists passing a note down a line. One decides when you've finished
talking, one writes down what you said, one writes a reply, and one reads the reply out loud.

## The analogy: a relay race where runners can start early

Think of a relay race with four runners. In a normal relay, each runner waits for the baton. The total time is
the sum of all four legs. The trick in this project is that **runners start moving before the baton fully
arrives**: the writer starts transcribing while we're still checking that you've stopped, and the reader starts
speaking the first sentence while the reply is still being written. That overlap is most of why the agent
feels fast.

## The pieces and the connections

```
┌──────────────┐
│  Microphone  │
└──────┬───────┘
       │ sound, cut into 32 ms slices (512 numbers each)
       ▼
┌────────────────────────────────────┐
│ 1. Silero VAD                [CPU] │  "is this slice speech?"
│    • 160 ms of quiet → start step 2 early (speculation)
│    • 500 ms of quiet → your turn is over
└──────┬─────────────────────────────┘
       │ your utterance: 16,000 numbers per second of audio
       ▼
┌────────────────────────────────────┐
│ 2. Parakeet speech-to-text         │
│    • preprocessor: sound → picture of frequencies   [CPU]
│    • encoder: picture → meaning, 24 layers          [GPU]  ← 95% of the work
│    • decoder loop (hand-written): meaning → tokens  [CPU]
└──────┬─────────────────────────────┘
       │ text: "What is the capital of France?"
       ▼
┌────────────────────────────────────┐
│ 3. LFM2 language model       [GPU] │  writes the reply one token at a time
└──────┬─────────────────────────────┘
       │ tokens as they're produced
       ▼
┌────────────────────────────────────┐
│ Sentence splitter                  │  waits for "." (or a comma in a long first sentence)
└──────┬─────────────────────────────┘
       │ one sentence at a time
       ▼
┌────────────────────────────────────┐
│ 4. Kokoro text-to-speech     [GPU] │  sentence → 24,000 numbers per second of audio
└──────┬─────────────────────────────┘
       │ audio chunks, queued back to back
       ▼
┌──────────────┐
│   Speakers   │
└──────────────┘
```

Two feedback rules sit on top:

```
  you speak again BEFORE the agent makes a sound ──▶ cancel the reply, merge both utterances
  the agent is speaking ──▶ ignore the mic (otherwise it hears itself)
                            unless "Let me interrupt" is on (headphones)
```

## Where the files come from

```
GitHub Pages  ──▶  the page itself (one HTML file + a 20-line service worker)
jsDelivr      ──▶  the libraries (ONNX Runtime Web, Transformers.js, kokoro-js)
Hugging Face  ──▶  model weights, ~830 MB the first time, then from the browser's cache
After loading, a conversation needs no server at all. Your audio never leaves the tab.
```

## The real numbers (one turn, through the microphone)

```
you stop talking
│◀──────── 500 ms silence wait ─────────▶│◀ LLM ~0.2 s ▶│◀── voice ~1 s ──▶│ 🔊 first sound
│◀ 160 ms ▶│◀ STT ~0.2 s ▶│ finished inside the wait, so it costs 0 ms
                                                                    total ≈ 1.4–2.1 s
```

Without the overlap, the same turn would be 0.5 + 0.2 + 0.2 + 1.0 ≈ 1.9 s *plus* waiting for the whole reply
before speaking (another 1–3 s). The overlap saves roughly a third to a half.

## The surprising part

The models are not the hard part. They're all off the shelf. The difference between a 27-second reply
(the first version) and a 1.5-second reply (the final one) came from **where each model runs** (CPU or GPU) and
**how much the stages overlap**. See [Latency tricks](07-latency-tricks.md).

**Bottom line:** four small models, mostly on the GPU, passing work along early instead of waiting for each
other, give a ~2-second conversation inside a browser tab.
