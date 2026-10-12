# Hypo 2 wiki: building our own inference engine

This part of the wiki explains, in plain language, how the engine in
[h2-own-inference-engine/engine/](https://github.com/Pedro1-21GW/e2e-browser-voice-agent/tree/main/h2-own-inference-engine/engine) runs a language model inside a browser tab with no ML library:
hand-written WebGPU kernels for the GPU, and C compiled to WebAssembly for the CPU. Every page uses the numbers
measured in this repo, on one laptop (Intel i5-9300HF, 4 cores / 8 threads, one 8 GB DDR4-2400 stick, GTX 1050
3 GB, Chrome 154).

> **Experimental.** This is a research experiment, not a product: tested on one laptop (GTX 1050, Chrome on Windows), with three test prompts and clean synthetic audio. Expect rough edges on other hardware and browsers.

## The result in one table

Qwen2.5 0.5B Instruct, same model and same laptop, our engine against the libraries H1 uses (Transformers.js +
ONNX Runtime Web):

| | Prompt, 256 tokens | Writing the reply | Voice-turn first token |
|---|---|---|---|
| Libraries, GPU (4-bit) | 702 ms | 15 tokens/s | – |
| **Our engine, GPU** | 766 ms | **98 tokens/s** | **69 ms** |
| Libraries, CPU (8-bit, 4 threads) | 3.95 s | 5 tokens/s | – |
| **Our engine, CPU** | **2.6 s** | **28–33 tokens/s** | **163 ms** |

Inside the voice agent, with no GPU at all, the agent now starts answering **0.6–1.9 s** after you stop talking.
H1 needed 12–18 s for the same model on the CPU.

## Reading order

| # | Page | The question it answers |
|---|---|---|
| 1 | [Anatomy of a forward pass](01-forward-pass.md) | What does "running a language model" actually do, step by step? |
| 2 | [Reading the model file](02-reading-the-files.md) | What is inside a GGUF file, and how do text and tokens convert? |
| 3 | [One token on the GPU](03-gpu-one-token.md) | Why is writing a reply about memory, not arithmetic, and what did each kernel attempt teach? |
| 4 | [The CPU: C, WebAssembly, SIMD, threads](04-cpu-webassembly.md) | How does a laptop CPU get 6× faster than the library, and would "C++" help? |
| 5 | [Inside the voice agent](05-in-the-voice-agent.md) | What changed when the engine met speech recognition and a voice? |
| 6 | [Prior work](06-prior-work.md) | Who has done this before, and what did we borrow? |

## Words used on these pages

| Word | Meaning here |
|---|---|
| **Token** | A piece of text the model reads or writes (a word, part of a word, or a symbol). |
| **Forward pass** | One run of the model over some tokens, producing the scores for the next token. |
| **Prefill / prompt processing** | The forward pass over the whole prompt at once. |
| **Decode / generation** | One forward pass per new token. |
| **KV cache** | What each layer remembers about earlier tokens, so they are not recomputed. |
| **Kernel** | A small program that runs on many GPU (or CPU) cores at once. |
| **Dispatch** | One launch of a kernel on the GPU. |
| **Q4_0 / Q8_0** | Weights stored as 4-bit or 8-bit integers, with one scale per group of 32. |
| **SIMD** | One instruction that works on 4–16 numbers at once. |
| **Bandwidth** | How many bytes per second memory can deliver. |
