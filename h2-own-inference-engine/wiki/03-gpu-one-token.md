# 3. One token on the GPU

**Core idea:** writing one token means reading all 345 MB of weights once and doing very little math with each
number. So the GPU's speed limit is its memory, and the job of a kernel is to keep memory busy. We got the GPU
from 15 tokens/s (the library) to 98 tokens/s, then found that the rest is mostly fixed cost per kernel launch.

## The analogy: a warehouse and its loading dock

The weights are boxes in a warehouse (GPU memory). The cores are workers who open each box, do one quick thing with
it, and are done. However many workers you hire, you can't go faster than the dock can bring boxes out. Measured on
this GPU, the dock moves **76 GB/s** (the spec sheet says 112):

```
345 MB of weights ÷ 76 GB/s = 4.5 ms per token  →  ~220 tokens/s at best
we measure 10 ms per token                       →   98 tokens/s
```

## Where the 10 ms go (timestamp queries, one token)

| Kernel | Time | Bytes | Speed |
|---|---|---|---|
| gate+up matrix × vector (24×) | 2.6 ms | 118 MB | 46 GB/s |
| output table, 8-bit (1×) | 2.2–2.5 ms | 153 MB | 61–68 GB/s |
| down matrix × vector (24×) | 1.5 ms | 59 MB | 40 GB/s |
| qkv, o matrix × vector (48×) | 1.1 ms | 25 MB | 23 GB/s |
| attention, RoPE, small steps | ~1.5 ms | – | – |

A kernel that does nothing at all still costs **5–10 µs** to launch and finish, and one token needs about 240
launches. That is ~2 ms of the 10 ms spent on launches alone.

## What each kernel attempt taught

| Version | Idea | gate+up time | What we learned |
|---|---|---|---|
| v1 | one thread per 4-byte weight word; each thread does R rows; input read from cache | 134 µs (R=4) → ~100 µs (R=16) | More rows per thread = more loads in flight. Total 10.2 → 8.0 ms. |
| v2 | input staged in shared memory; each lane reads a whole 18-byte block | **242 µs, slower** | Lane *i* read shared memory at `8·i`: all 32 lanes hit the same memory bank and queue up (32-way bank conflict). |
| v3 | input rounded to 8-bit; `dot4I8Packed` does 4 multiply-adds per instruction (DP4A); quantized inside every workgroup | **219 µs, slower** | 1,216 workgroups each redid the same normalize-and-quantize setup before reading a single weight. |
| v4 | the same integer math, but the input is quantized **once** by a small kernel and read from cache | 107 µs | Integer math works, but the total stayed at 8.5–9 ms: the arithmetic was never the limit. |

The engine uses v4: it's the approach llama.cpp's CUDA backend uses, and it fuses RMSNorm and SwiGLU into the
quantize step. On this GPU it ties with v1.

```
v4, per layer (one token):
quantize(+norm) ─▶ matvec qkv ─▶ RoPE+cache ─▶ attention ─▶ quantize ─▶ matvec o (+= x)
               ─▶ quantize(+norm) ─▶ matvec gate+up ─▶ quantize(+silu·up) ─▶ matvec down (+= x)
```

## The surprising part

Two things we assumed were the bottleneck weren't: **arithmetic** (integer dot products didn't change the total) and
**reading the result back to JavaScript** (the GPU's own time, 10.16 ms, equals the wall time, 10.13 ms). What's
left is launch overhead and kernels too small to reach full memory speed. The next step there is fewer, bigger
kernels (fusing RoPE into attention, the quantize steps into their producers), worth up to ~25%.

Prompt processing on the GPU still runs the token kernels 8 tokens at a time: 766 ms for 256 tokens, about the
same as the library (702 ms). A tiled matrix-matrix kernel is the obvious next piece; a voice turn rarely needs it,
because the cache leaves only ~18 new tokens per turn (69 ms).

**Bottom line:** the GPU engine is correct and writes 6.5× faster than the library; it reads weights at 50–90% of
what the memory can deliver, and the remaining time is mostly the fixed cost of ~240 kernel launches per token.
