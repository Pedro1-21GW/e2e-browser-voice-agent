# 8. GPU kernels: what H2 is building

**Core idea:** almost all the work in all four models is one operation: multiplying a list of numbers by a big
table of learned weights. A "kernel" is a small program that tells the GPU's hundreds of cores how to split
that multiplication between them. How you split it decides whether the GPU is busy or mostly waiting.

## The analogy: a library and its readers

The weight table lives in the library's stacks (the GPU's main memory: big, but far away). Each core is a
reader at a desk. Each desk group (a "workgroup") also has a small shelf right next to it (shared memory: tiny,
but instant to reach).

There are two very different jobs:

```
Job 1: one question, every book (matrix × VECTOR, the LLM writing one token)

   input: 1,024 numbers          weights: 4,096 rows × 1,024 numbers
   ┌────┐                        ┌──────────────────────────┐
   │ x  │  ── each row · x ──▶   │  every weight is used    │  ──▶ 4,096 answers
   └────┘                        │  ONCE, then never again  │
                                 └──────────────────────────┘
   Speed limit: how fast books come out of the stacks (memory bandwidth).

Job 2: many questions, same books (matrix × MATRIX, the speech encoder: 76 positions at once)

   input: 80 × 1,024             weights: 4,096 × 1,024
   every weight is used 80 times  ──▶  fetch a shelf-full once, reuse it
   Speed limit: how fast the readers can do arithmetic (FLOP/s).
```

## The real numbers (GTX 1050)

**Job 1, the speed limit is memory.** The GTX 1050 can move 112 GB/s. Our fp32 kernel reads 16.8 MB of weights
in 0.25 ms:

```
16.8 MB ÷ 0.25 ms = 67 GB/s  →  60% of the 112 GB/s limit
```

Storing the weights in 4 bits shrinks them to 2.6 MB, so it "should" take 2.6 ÷ 67 GB/s ≈ 0.04 ms. It takes
0.11 ms. Why? Every one of the 4,096 rows also reads the same 1,024-number input:

```
4,096 rows × 1,024 inputs × 4 bytes = 16.8 MB of input reads
```

That's as much traffic as the uncompressed weights. We compressed the books but kept fetching the question
4,096 times. The fix: put the question on the desk shelf once and answer several rows from it.

(We first guessed the floor was the fixed cost of starting a kernel. Measuring an empty kernel showed that cost
is only 0.024 ms, so the guess was wrong. That's why every claim here gets measured.)

**Job 2, the speed limit is arithmetic, if the books are close.** The naive kernel makes every reader fetch
every number from the stacks: 31 GFLOP/s. The tiled kernel makes each desk group fetch a 16×16 block onto its
shelf once and reuse it 16 times: 128 GFLOP/s, 4× faster from the same arithmetic.

```
naive:  reader ──▶ stacks ──▶ multiply ──▶ stacks ──▶ multiply ...      31 GFLOP/s
tiled:  group ──▶ stacks: fetch 16×16 block onto the shelf
        16 readers ──▶ shelf ──▶ multiply ×16 ──▶ next block            128 GFLOP/s
```

The GTX 1050's arithmetic limit is ~1.9 TFLOP/s (640 cores × ~1.5 GHz × 2 operations per multiply-add), so
128 GFLOP/s is 7%. H1's library reaches roughly 500 GFLOP/s on the speech encoder (estimated), so there's
room to learn.

## The surprising part

Making weights smaller (4-bit, 2-bit) only speeds things up if **nothing else** is the bottleneck. Here the
real bottleneck was re-reading the input, which compression doesn't touch. The same thing happened in H1 with
Kokoro on the CPU, where 8-bit was slower than 32-bit. "Compressed" and "fast" are different properties.

**Bottom line:** a kernel's job is to keep data close to the cores that use it; our first kernels are correct,
reach 60% of memory speed for fp32, and showed exactly which data movement to fix next.
