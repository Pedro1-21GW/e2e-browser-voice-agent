# Wiki: how a voice agent fits in a browser tab

This wiki explains, in plain language, everything that happens between you saying something and the agent
answering out loud. Every page follows the same recipe (the Feynman method, written down as the
[feynman-explain skill](../.claude/skills/feynman-explain/SKILL.md)):

1. **The core idea first**, in one or two sentences with no jargon.
2. **One analogy** that matches the real mechanism.
3. **The real numbers**, measured in this repo, so you can check the reasoning yourself.
4. **The surprising part**, called out explicitly.
5. **A one-line bottom line.**

## Reading order

| # | Page | The question it answers |
|---|---|---|
| 1 | [The big picture](01-big-picture.md) | What are the pieces and how are they connected? |
| 2 | [Listening: voice activity detection](02-listening-vad.md) | How does it know you stopped talking? |
| 3 | [Speech to text: Parakeet](03-speech-to-text.md) | How does sound become words, and what is "TDT"? |
| 4 | [The language model](04-language-model.md) | How does it write the reply, and why so fast? |
| 5 | [Text to speech: Kokoro](05-text-to-speech.md) | How do words become a voice? |
| 6 | [The browser as a computer](06-browser-as-a-computer.md) | WebAssembly, WebGPU, threads, caches: what are they? |
| 7 | [Latency tricks](07-latency-tricks.md) | Why it answers in ~2 s instead of ~5 s |
| 8 | [GPU kernels (H2)](08-gpu-kernels.md) | What does "writing our own engine" actually mean? |
| – | [Glossary](glossary.md) | Every term in one line |

## The whole thing in one paragraph

Your microphone produces sound. A tiny model checks every 32 ms whether that sound is speech. When you've been
quiet for half a second, the recording of what you said goes to a speech-recognition model that turns it into
text. The text goes to a small language model that writes a reply one word-piece at a time. As soon as the first
sentence of the reply exists, a speech-synthesis model turns it into audio and plays it, while the language
model keeps writing the rest. All four models run inside the browser tab, mostly on the graphics card.

## Where the experiments live

- [H1: browser libraries + cache](../h1-browser-libs/README.md): built, measured, confirmed with WebGPU.
- [H2: our own inference engine](../h2-own-inference-engine/README.md): in progress.
