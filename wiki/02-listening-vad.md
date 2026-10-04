# 2. Listening: voice activity detection (VAD)

**Core idea:** before anything can be transcribed, something has to decide *when you've finished talking*.
That's a tiny model that looks at the sound 31 times per second and answers one question: "is this speech?"

## The analogy: a referee with a stopwatch

Picture a referee who glances at you every 32 milliseconds. Each glance, they write down "talking" or "quiet".
They don't care what you say, only whether you're saying something. When they've seen enough "quiet" in a row,
they blow the whistle: your turn is over.

The whole game is choosing **how much quiet** counts as the end of a turn:
- Too short and the referee cuts you off mid-sentence, every time you take a breath.
- Too long and every reply starts with an awkward silence.

## The real numbers

The microphone delivers 16,000 numbers per second (16 kHz). The VAD looks at slices of 512 numbers:

```
512 samples ÷ 16,000 samples/s = 0.032 s = 32 ms per glance  →  ~31 glances per second
```

Each glance, the model (Silero VAD, 2 MB) outputs a probability between 0 and 1:

```
probability ≥ 0.50  →  speech starts
probability < 0.35  →  counts as quiet   (two thresholds, so it doesn't flicker on the border)

quiet glances needed to end the turn:  500 ms ÷ 32 ms ≈ 16 glances
measured end-of-turn delay:            509–520 ms   ✓
```

It also remembers what you said *just before* it noticed speech (the last 10 glances = 320 ms), so the first
syllable isn't clipped, and it ignores anything shorter than 6 glances (~190 ms): coughs, clicks, a door.

```
mic ─▶ [512 samples] ─▶ Silero VAD ─▶ p = 0.92 talking
       [512 samples] ─▶ Silero VAD ─▶ p = 0.88 talking
       [512 samples] ─▶ Silero VAD ─▶ p = 0.12 quiet   ┐
            ...                             quiet       │ 5 quiet glances (160 ms):
       [512 samples] ─▶ Silero VAD ─▶ p = 0.05 quiet   ┘ start transcribing early (a guess)
            ...                             quiet         16 quiet glances (512 ms):
                                                          whistle, end of turn
```

## The surprising part

Half a second sounds short, but people pause that long mid-sentence all the time. Our test voice said
"Hi there." and paused for over 500 ms before continuing "Can you tell me…". The first version blew the whistle
after "Hi there." and threw away the rest of the question, because it ignored the mic while busy.

The fix: **if you start talking again before the agent has made a sound, the whistle was wrong**. The pending
reply is cancelled and both parts are merged into one turn. Once the agent is audibly speaking, the mic is
ignored, because otherwise the agent would hear its own voice through the speakers and "reply" to itself.

The cost of the 500 ms wait is also mostly hidden: transcription starts after only 160 ms of quiet, as a guess.
If you keep talking, the guess is thrown away. See [Latency tricks](07-latency-tricks.md).

**Bottom line:** a 2 MB model glancing at the sound every 32 ms decides when your turn ends; the 500 ms wait is
the biggest fixed cost in the pipeline, so the design works around it rather than shrinking it.
