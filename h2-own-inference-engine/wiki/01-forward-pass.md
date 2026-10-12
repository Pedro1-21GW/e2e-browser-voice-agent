# 1. Anatomy of a forward pass

**Core idea:** a language model is a fixed recipe of about 300 steps, mostly "multiply this list of numbers by a
big table". Running it once (a forward pass) turns the tokens so far into a score for every possible next token.
Writing a reply means running it once per new token.

## The analogy: an assembly line with 24 identical stations

A token enters as a list of 896 numbers (its "embedding"). It goes through 24 identical stations (layers). Each
station does two things: it looks back at what earlier tokens left on the shelf (attention), and it reshapes the
numbers (the feed-forward block). At the end, one huge table turns the 896 numbers into 151,936 scores, one per
token in the vocabulary. The highest score is the next token.

```
token id ──▶ embedding (896 numbers)
                │
   ┌────────────▼─────────────┐  ×24 layers
   │ RMSNorm                  │
   │ Q, K, V = W·x + b        │  1152 rows × 896   (4-bit)
   │ rotate Q, K by position  │  (RoPE)
   │ store K, V on the shelf  │  (KV cache)
   │ attention over the shelf │  14 query heads share 2 K/V heads
   │ x += W_o · attention     │  896 × 896
   │ RMSNorm                  │
   │ gate, up = W·x           │  9728 × 896
   │ x += W_down · silu(gate)·up │ 896 × 4864
   └────────────┬─────────────┘
                ▼
     RMSNorm ─▶ output table (151,936 × 896, 8-bit) ─▶ scores ─▶ next token
```

## Two very different jobs

| | Prompt (prefill) | Writing the reply (decode) |
|---|---|---|
| Tokens per pass | many (the whole prompt) | one |
| Each weight is used | once per prompt token | once |
| What limits speed | arithmetic | reading the weights from memory |

The weights of Qwen2.5 0.5B in this file are **345 MB**, and writing one token reads all of them once. On the
GPU's memory (76 GB/s measured) that alone takes 4.5 ms, so ~220 tokens/s is the ceiling. On the laptop's main
memory (15 GB/s measured) it takes 23 ms: ~43 tokens/s.

## The shelf (KV cache) makes a conversation cheap

Each layer stores the K and V vectors of every token it has seen. A new token only computes its own K and V and
reads the old ones; nothing is recomputed. The engine keeps the shelf between turns of a conversation and, for a
new turn, only runs the tokens that differ from what it has already seen:

```
turn 1:  [system prompt][user: hi][assistant: hello!]          → 84 tokens computed
turn 2:  [system prompt][user: hi][assistant: hello!][user: …]  → only the new 18 tokens computed
```

That is why a voice turn's first token takes **69 ms on the GPU and 163 ms on the CPU**, even though processing
256 tokens from scratch takes 0.8 s and 2.6 s.

## The surprising part

The output table is the single biggest piece of work when writing a token: 151,936 rows, stored in 8 bits, is
**153 MB, 44% of all bytes read per token**, bigger than all 24 layers' feed-forward blocks together. The model is
"small" (0.5 B parameters), but its vocabulary is huge because it covers dozens of languages.

**Bottom line:** running a model is a long, fixed list of matrix-times-vector steps; writing a reply is limited by
how fast the weights can be read, and the cache keeps a conversation from paying twice for the same tokens.
