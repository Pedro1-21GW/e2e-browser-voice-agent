// Worker side of the CPU backend.
//
//   main thread ──postMessage──▶ leader worker (thread 0) ──shared memory──▶ helper workers (threads 1..n-1)
//
// The leader owns the model: it copies the weights into the WebAssembly memory, starts the helpers, and runs each
// forward pass. The helpers park inside worker_loop() in C and wake up (atomics) whenever the leader starts a pass.
// The browser's main thread may not block on atomics, which is why thread 0 is a worker too.

const TYPE = { F32: 0, Q4_0: 2, Q8_0: 8 };

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "helper") {
    const inst = await WebAssembly.instantiate(msg.module, { env: { memory: msg.memory, now: () => performance.now() } });
    inst.exports.__stack_pointer.value = msg.stackTop;
    self.postMessage({ type: "helper-ready" });
    inst.exports.worker_loop(msg.tid);                    // never returns
  } else if (msg.type === "init") {
    try { await init(msg); } catch (err) { self.postMessage({ type: "error", message: String(err && err.stack || err) }); }
  } else if (msg.type === "read-test") {
    const out = [];
    for (const nth of msg.threads) {
      let best = 1e9;
      for (let r = 0; r < 5; r++) { const t0 = performance.now(); ex.run_read_test(nth); best = Math.min(best, performance.now() - t0); }
      out.push({ threads: nth, ms: best, gbps: cfg.vocab * cfg.dim / best / 1e6 });
    }
    self.postMessage({ type: "token", out });
  } else if (msg.type === "barrier-test") {
    const t0 = performance.now(); const errors = ex.run_barrier_test(msg.iters, threads);
    self.postMessage({ type: "token", errors, ms: performance.now() - t0 });
  } else if (msg.type === "profile") {
    const names = ["embed", "norm+quantize", "matvec qkv", "rope+kv", "attention", "quantize", "matvec o", "matvec gate+up", "swiglu+quantize", "matvec down", "matvec output (Q8)"];
    ex.profile_start();
    const r = forward(msg);
    ex.profile_stop();
    const by = {}; names.forEach((n, i) => { by[n] = ex.profile_get(i); });
    self.postMessage({ type: "token", ...r, profile: by });
  } else if (msg.type === "forward") {
    try { self.postMessage({ type: "token", ...forward(msg) }); }
    catch (err) { self.postMessage({ type: "error", message: String(err && err.stack || err) }); }
  }
};

let ex, mem, cfg, V, threads, decodeThreads, MT, ctx;

async function init({ bytes, tensors, config, wasmUrl, threads: nThreads, decodeThreads: nDecode, ctx: nCtx, mt, shared }) {
  cfg = config; threads = shared ? nThreads : 1; decodeThreads = Math.min(threads, nDecode || threads); MT = mt; ctx = nCtx;
  const { dim: D, layers: NL, ffn: F, heads: H, kvHeads: KVH, vocab, ropeTheta, eps } = cfg;
  const HD = D / H, QKV = D + 2 * KVH * HD;
  const module = await WebAssembly.compileStreaming(fetch(wasmUrl));

  // ---- memory layout, decided here (sizes in bytes)
  const T = tensors;
  const q4 = (count) => ({ qs: count / 2, d: (count / 32) * 4 });
  const plan = [];
  const want = (key, bytes) => plan.push([key, bytes]);
  for (let l = 0; l < NL; l++) {
    const p = `blk.${l}.`;
    const sz = (names) => q4(names.reduce((n, k) => n + T[p + k].count, 0));
    for (const [kind, names] of [[0, ["attn_q.weight", "attn_k.weight", "attn_v.weight"]], [1, ["attn_output.weight"]],
                                 [2, ["ffn_gate.weight", "ffn_up.weight"]], [3, ["ffn_down.weight"]]]) {
      const s = sz(names); want(`L${l}.${kind}.qs`, s.qs); want(`L${l}.${kind}.d`, s.d);
    }
    want(`L${l}.4`, D * 4); want(`L${l}.5`, D * 4); want(`L${l}.6`, QKV * 4);
  }
  const e = q4(T["token_embd.weight"].count); want("embd.qs", e.qs); want("embd.d", e.d);
  want("out.qs", T["output.weight"].count); want("out.d", (T["output.weight"].count / 32) * 4);
  want("out_norm", D * 4);
  const bufs = {
    x: MT * D * 4, xn: MT * D * 4, qkv: MT * QKV * 4, att: MT * D * 4, gu: MT * 2 * F * 4, act: MT * F * 4,
    logits: vocab * 4, kc: NL * KVH * ctx * HD * 4, vc: NL * KVH * ctx * HD * 4, scores: threads * ctx * 4,
    aq: MT * F, ad: (MT * F / 32) * 4, as: (MT * F / 32) * 4, rope: ctx * HD * 4, tokens: (ctx + 1) * 4,
  };
  for (const [k, v] of Object.entries(bufs)) want(k, v);
  for (let t = 0; t < threads; t++) want(`stack${t}`, 256 * 1024);

  // __heap_base isn't known before instantiating, and instantiating needs the memory: reserve 1 MiB for the
  // module's own data and stack, which is far more than this tiny module uses (checked below).
  const base = 1 << 20;
  let top = base;
  const at = {};
  for (const [k, b] of plan) { top = Math.ceil(top / 64) * 64; at[k] = top; top += b; }
  const pages = Math.ceil(top / 65536);
  mem = new WebAssembly.Memory({ initial: pages, maximum: shared ? 65535 : pages, shared });
  const inst = await WebAssembly.instantiate(module, { env: { memory: mem, now: () => performance.now() } });
  ex = inst.exports;
  if (ex.__heap_base.value > base) throw new Error("module data larger than reserved");
  if (ex.__stack_pointer) ex.__stack_pointer.value = at.stack0 + 256 * 1024;
  const u8 = new Uint8Array(mem.buffer), src = new Uint8Array(bytes);
  const f32 = (off, n) => new Float32Array(mem.buffer, off, n);

  // ---- weights: GGUF blocks (scale + values interleaved) → separate value and scale arrays
  const F16 = new Float32Array(65536);
  { const h = new Uint16Array(1), f = new Float32Array(1); // fp16 → fp32 table
    for (let i = 0; i < 65536; i++) { const s = i & 0x8000 ? -1 : 1, ex2 = (i >> 10) & 31, m = i & 1023;
      F16[i] = ex2 === 0 ? s * m * 2 ** -24 : ex2 === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (ex2 - 15); } }
  const copyBlocks = (t, qsOff, dOff, blockBytes, valBytes) => {
    const nb = t.count / 32, d = f32(dOff, nb);
    for (let b = 0, o = t.start; b < nb; b++, o += blockBytes) {
      d[b] = F16[src[o] | (src[o + 1] << 8)];
      u8.set(src.subarray(o + 2, o + 2 + valBytes), qsOff + b * valBytes);
    }
    return nb;
  };
  const f32Tensor = (t, off) => { if (t.type !== TYPE.F32) throw new Error(t.name); u8.set(src.subarray(t.start, t.start + t.bytes), off); };
  for (let l = 0; l < NL; l++) {
    const p = `blk.${l}.`;
    for (const [kind, names] of [[0, ["attn_q.weight", "attn_k.weight", "attn_v.weight"]], [1, ["attn_output.weight"]],
                                 [2, ["ffn_gate.weight", "ffn_up.weight"]], [3, ["ffn_down.weight"]]]) {
      let qs = at[`L${l}.${kind}.qs`], d = at[`L${l}.${kind}.d`];
      for (const n of names) {
        if (T[p + n].type !== TYPE.Q4_0) throw new Error(`${p + n}: expected Q4_0`);
        const nb = copyBlocks(T[p + n], qs, d, 18, 16); qs += nb * 16; d += nb * 4;
      }
      ex.set_tensor(kind, l, at[`L${l}.${kind}.qs`], at[`L${l}.${kind}.d`]);
    }
    f32Tensor(T[p + "attn_norm.weight"], at[`L${l}.4`]); ex.set_tensor(4, l, 0, at[`L${l}.4`]);
    f32Tensor(T[p + "ffn_norm.weight"], at[`L${l}.5`]); ex.set_tensor(5, l, 0, at[`L${l}.5`]);
    f32Tensor(T[p + "attn_q.bias"], at[`L${l}.6`]);
    f32Tensor(T[p + "attn_k.bias"], at[`L${l}.6`] + D * 4);
    f32Tensor(T[p + "attn_v.bias"], at[`L${l}.6`] + (D + KVH * HD) * 4);
    ex.set_tensor(6, l, 0, at[`L${l}.6`]);
    self.postMessage({ type: "progress", got: l + 1, total: NL });
  }
  copyBlocks(T["token_embd.weight"], at["embd.qs"], at["embd.d"], 18, 16); ex.set_tensor(7, 0, at["embd.qs"], at["embd.d"]);
  if (T["output.weight"].type !== TYPE.Q8_0) throw new Error("output.weight: expected Q8_0");
  copyBlocks(T["output.weight"], at["out.qs"], at["out.d"], 34, 32); ex.set_tensor(8, 0, at["out.qs"], at["out.d"]);
  f32Tensor(T["output_norm.weight"], at.out_norm); ex.set_tensor(9, 0, 0, at.out_norm);

  // RoPE table: cos/sin of pos·θ_i for every position (computed once, in double precision)
  const rope = f32(at.rope, ctx * HD);
  for (let pos = 0; pos < ctx; pos++) for (let i = 0; i < HD / 2; i++) {
    const a = pos / Math.pow(ropeTheta, (2 * i) / HD);
    rope[pos * HD + 2 * i] = Math.cos(a); rope[pos * HD + 2 * i + 1] = Math.sin(a);
  }
  ex.set_config(D, NL, F, H, KVH, vocab, ctx, MT, eps);
  ex.set_buffers(at.x, at.xn, at.qkv, at.att, at.gu, at.act, at.logits, at.kc, at.vc, at.scores, at.aq, at.ad, at.as, at.rope, at.tokens);
  V = { logits: f32(at.logits, vocab), tokens: new Int32Array(mem.buffer, at.tokens, ctx + 1) };

  // ---- helper threads
  if (threads > 1) {
    const ready = [];
    for (let t = 1; t < threads; t++) {
      const w = new Worker(self.location.href, { type: "module" });
      ready.push(new Promise((res, rej) => { w.onmessage = (m) => m.data.type === "helper-ready" && res(); w.onerror = rej; }));
      w.postMessage({ type: "helper", module, memory: mem, tid: t, stackTop: at[`stack${t}`] + 256 * 1024 });
    }
    await Promise.all(ready);
  }
  self.postMessage({ type: "ready", bytes: top - base, threads, decodeThreads });
}

// Sampling on the logits: greedy, or softmax(logits / temperature) with a seeded random number.
function sample(temperature, seed) {
  const lg = V.logits;
  let best = 0;
  for (let i = 1; i < lg.length; i++) if (lg[i] > lg[best]) best = i;
  if (!(temperature > 0)) return best;
  let s = seed >>> 0;
  const rand = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const mx = lg[best] / temperature;
  let sum = 0;
  for (let i = 0; i < lg.length; i++) sum += Math.exp(lg[i] / temperature - mx);
  let r = rand() * sum;
  for (let i = 0; i < lg.length; i++) { r -= Math.exp(lg[i] / temperature - mx); if (r <= 0) return i; }
  return best;
}

function forward({ ids, pos, temperature = 0, seed = 1, wantLogits = false, nth }) {
  // Several tokens (a prompt) are limited by arithmetic: use every thread. One token (generation) is limited by
  // memory bandwidth, which a few threads already saturate; more just fight over it.
  nth ??= ids.length > 1 ? threads : decodeThreads;
  if (pos + ids.length >= ctx) throw new Error("context full");
  V.tokens.set(ids, pos);
  const t0 = performance.now();
  for (let c = 0; c < ids.length; c += MT) {
    const n = Math.min(MT, ids.length - c);
    ex.run(pos + c, n, c + n >= ids.length ? 1 : 0, nth);
  }
  const next = sample(temperature, (seed + pos) >>> 0);
  return { next, ms: performance.now() - t0, logits: wantLogits ? V.logits.slice() : null };
}
