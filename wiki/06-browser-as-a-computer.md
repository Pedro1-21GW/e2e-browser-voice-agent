# 6. The browser as a computer

**Core idea:** a browser tab can run two kinds of heavy code: **WebAssembly** on the processor (CPU) and
**WebGPU** on the graphics card (GPU). Getting good speed means putting each job on the right one, and
unlocking a few features (threads, caching) that browsers keep locked by default.

## The analogy: a kitchen with a few chefs and a huge prep line

- The **CPU** is a handful of excellent chefs (this laptop: 4 cores). Each can do anything, including
  complicated, step-by-step recipes like the speech decoder's "type or jump" loop.
- The **GPU** is a prep line of hundreds of kitchen hands (GTX 1050: 640 cores). Each is simple, but give them
  "chop these 10,000 onions the same way" and they finish long before the chefs. Neural networks are mostly
  that kind of job: huge piles of identical multiplications.

```
             CPU (WebAssembly)                      GPU (WebGPU)
       ┌───┐ ┌───┐ ┌───┐ ┌───┐            ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
       │ C │ │ C │ │ C │ │ C │            ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤  640 simple cores
       └───┘ └───┘ └───┘ └───┘            └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
  good at: loops, decisions, small     good at: the same multiply on millions
  models (VAD, decoder loop)           of numbers (encoder, LLM, TTS)
```

Measured: the speech encoder takes 2.6–3.1 s on the 4 CPU cores and ~0.18 s on the GPU.

## The locks we had to open

### 1. Threads need "cross-origin isolation"

Using all 4 CPU cores from WebAssembly requires `SharedArrayBuffer` (memory several threads can share).
Browsers only enable it when the page is served with two security headers (COOP and COEP), to block a class
of timing attacks. GitHub Pages doesn't let you set headers.

The workaround is a 20-line **service worker** ([sw.js](../h1-browser-libs/sw.js)): a small script the browser
runs between the page and the network. It adds the two headers to the page's own responses. On the first visit
the page registers it and reloads once; from then on the page is isolated and threads work.

```
browser ──request──▶ service worker ──▶ GitHub Pages
browser ◀──page + COOP/COEP headers── service worker ◀── page (no headers)
```

Effect, measured: the speech encoder on the CPU goes from 6.0 s (1 thread) to 2.6–3.1 s (4 threads).

### 2. Caching ~830 MB

The models are downloaded from Hugging Face once and stored with the browser's **Cache API**. The second visit
loads them from disk: no download, but they still need to be placed in memory and the GPU programs compiled.

### 3. Warm-up: the GPU compiles its programs on first use

WebGPU turns each operation into a small GPU program (a "shader") the first time it runs. The first encoder
run took **2.7 s**; the next ones took **~0.18 s**. So right after loading, the page runs everything once on
dummy input (silence, "Hi", "Hello there.") so the user never pays that cost.

## The surprising part

There are **four copies of ONNX Runtime** in this page: the CPU build (preprocessor, decoder, VAD), the GPU
build (speech encoder), and the ones bundled inside Transformers.js (LLM) and inside kokoro-js (TTS, which ships
its own older Transformers.js). Each library brought its own engine. It works, but it's wasteful, and it's one
of the motivations for [H2](../h2-own-inference-engine/README.md): one engine we understand end to end.

**Bottom line:** the browser is a real computer with a GPU, but you have to ask for threads (a service-worker
trick on GitHub Pages), cache the models yourself, and warm up the GPU before the user speaks.
