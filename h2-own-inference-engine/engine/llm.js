// The engine's public face: load a GGUF model on WebGPU or on the CPU, then chat with it.
//
//   const llm = await LLM.load({ url, backend: "webgpu" | "cpu", onProgress });
//   for await (const piece of llm.chat(messages, { maxTokens: 128 })) speak(piece);
//
// The conversation's KV cache is kept between turns: a new turn only runs the tokens that differ from what the
// model has already seen (usually the new user message), so the system prompt and history are never recomputed.

import { parseGGUF, qwen2Config, fetchModel } from "./gguf.js";
import { Tokenizer, chatPrompt } from "./tokenizer.js";

export class LLM {
  static async load({ url, backend = "webgpu", threads, ctx = 2048, onProgress = () => {}, ...opts }) {
    const t0 = performance.now();
    const bytes = await fetchModel(url, (got, total, cached) => onProgress({ stage: cached ? "cache" : "download", got, total }));
    const tDownload = performance.now() - t0;
    const gguf = parseGGUF(bytes);
    const cfg = qwen2Config(gguf.meta);
    if (cfg.arch !== "qwen2") throw new Error(`unsupported architecture ${cfg.arch}`);
    const tok = new Tokenizer(gguf.meta);
    const t1 = performance.now();
    let be;
    if (backend === "webgpu") {
      const { createWebGPU } = await import("./webgpu.js");
      be = await createWebGPU(gguf, bytes, cfg, { ctx, onProgress: (l, n) => onProgress({ stage: "upload", got: l, total: n }), ...opts });
    } else {
      const { createCPU } = await import("./cpu.js");
      be = await createCPU(gguf, bytes, cfg, { ctx, threads, onProgress: (l, n) => onProgress({ stage: "upload", got: l, total: n }), ...opts });
    }
    const llm = new LLM(be, tok, cfg);
    llm.loadTimes = { download: tDownload, prepare: performance.now() - t1 };
    return llm;
  }

  constructor(backend, tokenizer, cfg) {
    this.be = backend; this.tok = tokenizer; this.cfg = cfg;
    this.seen = [];                     // tokens whose K/V are in the cache, in order
    this.stats = null;
    this.stopIds = new Set([cfg.eos, tokenizer.ids.get("<|endoftext|>")]);
  }

  get backend() { return this.be.name; }

  // Process `ids` as a continuation of the cache, reusing the longest common prefix. Returns the next token.
  async feed(ids, sampling) {
    let common = 0;
    while (common < ids.length && common < this.seen.length && ids[common] === this.seen[common]) common++;
    if (common === ids.length) common--;           // always run at least one token to get fresh logits
    this.seen = ids.slice(0, common);
    const fresh = ids.slice(common);
    const next = await this.be.forward(fresh, common, sampling);
    this.seen.push(...fresh);
    return { next, prefilled: fresh.length, reused: common };
  }

  // Stream the assistant's reply as text pieces.
  async *chat(messages, { maxTokens = 256, temperature = 0, seed = (Math.random() * 2 ** 32) >>> 0, signal } = {}) {
    // Keep the prompt plus the reply inside the context: drop the oldest exchanges (never the system prompt).
    let msgs = messages, ids = this.tok.encode(chatPrompt(msgs));
    while (ids.length + maxTokens > this.be.ctx - 8 && msgs.length > 2) {
      msgs = msgs[0].role === "system" ? [msgs[0], ...msgs.slice(3)] : msgs.slice(2);
      ids = this.tok.encode(chatPrompt(msgs));
    }
    const dec = this.tok.streamDecoder();
    const t0 = performance.now();
    let { next, prefilled, reused } = await this.feed(ids, { temperature, seed });
    const tFirst = performance.now() - t0;
    let n = 0;
    const out = [];
    while (!this.stopIds.has(next) && n < maxTokens && !signal?.aborted) {
      out.push(next); n++;
      const piece = dec(next);
      if (piece) yield piece;
      const pos = this.seen.length;
      next = await this.be.forward([next], pos, { temperature, seed });
      this.seen.push(out[out.length - 1]);
    }
    // Keep the end-of-turn token in the history so the next turn's prompt matches the cache.
    if (this.stopIds.has(next)) { await this.be.forward([next], this.seen.length, { temperature, seed }); this.seen.push(next); }
    const tTotal = performance.now() - t0;
    this.stats = { promptTokens: ids.length, prefilled, reused, firstTokenMs: tFirst, tokens: n,
                   decodeTokPerS: n > 1 ? (n - 1) / ((tTotal - tFirst) / 1000) : 0, totalMs: tTotal, ids: out };
  }

  // Greedy continuation of raw token ids, for correctness checks. Resets the cache.
  async generateIds(ids, steps) {
    this.seen = [];
    const out = [];
    let { next } = await this.feed(ids, { temperature: 0 });
    for (let i = 0; i < steps; i++) {
      out.push(next);
      if (i === steps - 1) break;
      next = await this.be.forward([next], this.seen.length, { temperature: 0 });
      this.seen.push(out[out.length - 1]);
    }
    return out;
  }

  reset() { this.seen = []; }
}
