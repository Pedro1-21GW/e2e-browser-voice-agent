# 2. Reading the model file

**Core idea:** a GGUF file is a table of contents followed by the weights as plain bytes. Reading it needs no
library: a few hundred lines of JavaScript parse it, and a hand-written tokenizer turns text into token ids using
the vocabulary stored inside the same file.

## The analogy: a parts catalogue

The file opens with a catalogue: "the model is a qwen2, 24 layers, width 896, here are the 151,936 words of the
vocabulary and the 151,387 rules for gluing word pieces together". Then a list of parts: "blk.0.ffn_up.weight,
4864 rows × 896, type Q4_0, starts at byte 232,073,056". Then the parts themselves.

```
"GGUF" │ version │ metadata (key → value) │ tensor table (name, shape, type, offset) │ padding │ weights…
```

## How 4-bit weights are packed (Q4_0)

Every 32 weights share one scale. A block is 18 bytes:

```
┌────────┬──────────────────────────────────────────┐
│ scale  │ 16 bytes = 32 values of 4 bits (0..15)    │   weight = (value − 8) × scale
│ fp16   │ byte j: low half = weight j, high = j+16  │
└────────┴──────────────────────────────────────────┘
```

So a 4-bit weight costs 18 / 32 = **0.56 bytes**. The GPU can only read 4-byte-aligned words, so at load time the
engine splits each block into its 16 value bytes and its scale (converted to a 32-bit float). The CPU build does
the same, so its SIMD loads are aligned.

In this file, the layers are Q4_0 and the output table is **Q8_0** (one byte per weight plus a scale per 32), which
is why that one table is 153 MB.

## Text to tokens: byte-level BPE

```
"I'm sure."
   │  split on special tokens (<|im_start|> …), NFC-normalize
   │  pre-split with a regex: words, numbers, punctuation, spaces → ["I", "'m", " sure", "."]
   │  each piece as bytes, each byte shown as one printable character
   │  repeatedly glue the pair with the lowest rank in the merge list
   ▼
[40, 2776, 2704, 13]
```

The tokenizer ([tokenizer.js](../engine/tokenizer.js)) is ~120 lines. It reproduces Hugging Face's tokenizer on all
13 test strings (accents, emoji, Japanese, code, special tokens, trailing spaces) and on the three chat prompts. The
H2 plan had allowed the tokenizer as a temporary library exception; it turned out not to be needed.

## How we know the engine is right

[m1-llm/reference.py](../m1-llm/reference.py) loads the **same GGUF file** with Hugging Face Transformers (an
independent implementation), and records, for three chat prompts, the top-5 scores of the first step and 32 greedy
tokens. The engine must produce the same tokens:

| Backend | Top-5 scores vs reference | Greedy tokens |
|---|---|---|
| GPU | within 0.0005 | identical up to the end of the reply, all 3 prompts |
| CPU | within 0.11–0.25 (inputs rounded to 8 bits, as in llama.cpp) | identical up to the end of the reply, all 3 prompts |

## The surprising part

The reference kept writing after the end-of-reply token (`<|im_end|>`), so "8 of 32 tokens agree" looked like a
failure at first. The first 8 were the whole answer; what follows the end token is meaningless for both. The check
now compares up to the end of the reply.

**Bottom line:** the model file is self-describing, and both the weights and the tokenizer can be read with a few
hundred lines of plain code that match the reference exactly.
