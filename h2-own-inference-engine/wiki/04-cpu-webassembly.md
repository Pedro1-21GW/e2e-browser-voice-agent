# 4. The CPU: C, WebAssembly, SIMD, threads

**Core idea:** without a GPU, the same model runs as C code compiled to WebAssembly. Three things make it fast:
doing the multiplications on 8-bit integers (16 at a time), using several cores that share one memory, and not
using more cores than the memory can feed. The result writes **28–33 tokens/s** where the library writes 5.

## The analogy: a kitchen with one door to the pantry

The cores are cooks; the weights are ingredients in a pantry (main memory) behind one narrow door. This laptop has
a single 8 GB memory stick, so the door is single-width: **19.2 GB/s** on paper, **15–17 GB/s** measured from
WebAssembly. One or two cooks already keep the door busy:

```
measured read speed from WebAssembly:  1 thread 14.6 GB/s · 2 threads 16.8 · 4 threads 14.2 · 8 threads 15.2
345 MB per token ÷ ~15 GB/s ≈ 23 ms  →  ~43 tokens/s at best;  we get 28–33
```

For a prompt it's the other way around: every ingredient that comes through the door is used for many tokens, so
more cooks (all 8 hardware threads) help. The engine uses **8 threads for prompts and 4 for writing**.

## The integer trick (the same one llama.cpp uses)

Instead of multiplying 4-bit weights by 32-bit floats, the input is also rounded to 8-bit integers (one scale per
32 values). Then the inner loop is a pure integer dot product:

```
32 weights (16 bytes)      32 inputs (32 bytes)
  │ split into low/high      │
  │ nibbles (0..15)          │
  └──────── relaxed dot product: 16 int8 multiply-adds per instruction ────────┘
                         │ × (weight scale · input scale)
                         ▼
                      partial sum
```

"Relaxed SIMD" (Chrome 114+) has an instruction that multiplies 16 pairs of bytes and adds them in groups of 4. Its
second operand must be 0..127, so the weights stay unsigned (0..15) and the engine subtracts `8 × sum(inputs)`
afterwards, which is the same as using the signed weights (−8..7).

## Would C/C++ help? Measured.

| Engine (CPU) | Prompt, 256 tokens | Writing |
|---|---|---|
| ONNX Runtime Web (C++ compiled to WebAssembly), 4 threads, via Transformers.js | 3.95 s | 5.0 tokens/s |
| Ours, 1 thread, plain SIMD | 10.9 s | 12.9 tokens/s |
| Ours, 1 thread, relaxed SIMD | 7.1 s | 16.6 tokens/s |
| Ours, 4 threads | 3.0 s | 31 tokens/s |
| **Ours, 8 threads for prompts, 4 for writing** | **2.6 s** | **28–33 tokens/s** |

ONNX Runtime is also C++, so "being C++" isn't what makes the difference. **One thread of ours writes 3× faster than
four of the library's.** What matters is the kernel: 4-bit weights times 8-bit inputs with no conversion to floats,
a forward pass with no per-operation framework overhead, and the cache kept between calls.

## Threads in a browser

```
main thread ──message──▶ leader worker (thread 0) ──shared WebAssembly memory──▶ helper workers 1..7
                          runs the forward pass        each runs the same C function on its share of rows
```

Every operation splits its rows between threads, and a **barrier** (all threads wait until everyone is done)
separates operations: ~9 per layer, ~216 per token. A self-test runs 20,000 barriers with 2–8 threads and checks
nothing ran ahead: 0 errors, **0.3–0.7 µs per barrier**. The main thread can't block on a barrier (browsers forbid
it), which is why thread 0 is a worker too. Threads need `SharedArrayBuffer`, which needs two HTTP headers
(COOP/COEP): [serve.py](../../serve.py) sets them locally, and H1's service worker sets them on GitHub Pages.

## The surprising parts (two bugs worth remembering)

1. **Fast math made NaNs.** With `-ffast-math`, the relaxed-SIMD build printed `!!!!!!` (every score NaN): the
   compiler is allowed to swap in imprecise relaxed instructions elsewhere. Without the flag it is correct and still
   1.6× faster than plain SIMD on one thread.
2. **A race after the last barrier.** With 3 or 4 threads the output was garbage, but 2 threads looked fine. A
   helper read the "compute the final scores?" flag *after* the last barrier of the layers. By then the leader could
   already be setting up the next pass. The helper then joined a barrier from the wrong pass, and the counts got
   mixed up. The fix: every thread copies the job's parameters into local variables before the first barrier. Since
   then, the scores are **bit-identical for 1, 2, 3 and 4 threads**, whether the prompt arrives in one call, two
   calls, or token by token.

**Bottom line:** on this laptop the CPU engine writes at 60–75% of what its single memory channel allows, 6× faster
than the library on the same model, and thread count should follow the job: all threads for prompts, a few for
writing.
