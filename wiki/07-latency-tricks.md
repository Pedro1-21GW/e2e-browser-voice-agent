# 7. Latency tricks

**Core idea:** the user only waits for the parts of the work that happen *after* they stop talking and
*before* the first sound. Every trick here either moves work out of that window or shrinks the first chunk of
work inside it.

## The analogy: a restaurant kitchen during a dinner rush

A slow kitchen cooks each order from scratch after it's placed, and serves the whole table at once. A fast one
starts the obvious prep while the customer is still deciding, and sends out the first dish the moment it's
ready instead of waiting for the main course. Same cooks, same recipes, much shorter wait.

## The five tricks, with what each one bought

| # | Trick | Kitchen version | Measured effect |
|---|---|---|---|
| 1 | Put each model on the right processor | give the onion-chopping to the prep line | first audio: 3.8–23.5 s → ~1 s; speech-to-text: 1.7–3.5 s → 0.13–0.53 s |
| 2 | Speak sentence by sentence while the LLM keeps writing | serve the starter while the main cooks | the user waits for one sentence, not the whole reply |
| 3 | If the first sentence is long, speak up to the first comma | send out the bread now | first audio sooner on long openings |
| 4 | Transcribe speculatively after 160 ms of quiet | start prep while they're still deciding | speech-to-text costs **0 ms** of waiting through the mic |
| 5 | Warm up the GPU before the first turn | preheat the oven | first-turn encoder: 2.7 s → ~0.18 s |

## Trick 4 in detail: guessing early

The end-of-turn rule waits for 500 ms of quiet. But most of the time, 160 ms of quiet already *is* the end.
So at 160 ms the page starts transcribing what it has. Then:

```
case A: you really stopped (most turns)
  speech │◀ 160 ms ▶│ start STT ──▶ done (0.2 s)
         │◀────────── 500 ms wait ──────────▶│ end of turn → transcript is already there
                                               cost of STT: 0 ms

case B: you were only pausing
  speech │◀ 160 ms ▶│ start STT ...  │ you speak again → throw the guess away
                                     (wasted a little compute, nobody waited)
```

Measured through the microphone: "STT wait" was **0 ms** in every turn; the speculative transcript was always
ready before the 500 ms window closed.

## Where the remaining time goes

A typical turn through the microphone, final version:

```
│ silence wait  500 ms │ LLM to first sentence  ~0.2 s │ voice first chunk  ~1.0 s │ 🔊   ≈ 1.8 s
```

The voice's first chunk is now the biggest piece. It's slow partly because of a fixed cost per sentence
(0.72 s even for "Hello!") and partly because the GPU is shared with the language model, which is still writing.

## The surprising part

None of these tricks made any model faster. The models ran at the same speed throughout. The latency went from
**8–27 s to 1.4–2.1 s** almost entirely by rearranging *when* and *where* work happens.

**Bottom line:** overlap the stages, guess early, ship the first sentence (or clause) as soon as it exists, and
pre-warm the GPU: the user then waits for a fraction of the actual compute.
