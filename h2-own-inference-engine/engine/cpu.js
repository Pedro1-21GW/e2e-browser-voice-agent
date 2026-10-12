// CPU backend (main-thread side): starts the leader worker and forwards requests to it.
// Picks the best WebAssembly build the browser supports: threads need cross-origin isolation (COOP/COEP headers),
// relaxed SIMD needs a recent Chrome/Edge/Firefox.

const RELAXED_PROBE = new Uint8Array([   // a module with one i32x4.relaxed_dot_i8x16_i7x16_add_s instruction
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 33, 1, 31, 0,
  253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 0, 253, 17, 65, 0, 253, 17, 253, 147, 2, 11,
]);

export function cpuCapabilities() {
  const threads = typeof SharedArrayBuffer !== "undefined" && self.crossOriginIsolated === true;
  let relaxed = false;
  try { relaxed = WebAssembly.validate(RELAXED_PROBE); } catch { /* old browser */ }
  return { threads, relaxed, cores: navigator.hardwareConcurrency || 4 };
}

export async function createCPU(gguf, bytes, cfg, opts = {}) {
  const cap = cpuCapabilities();
  const shared = cap.threads && opts.threads !== 1;
  const relaxed = opts.relaxed ?? cap.relaxed;
  // Default: every logical core for prompts, half of them (≈ the physical cores) for generating tokens.
  const threads = shared ? (opts.threads || cap.cores) : 1;
  const decodeThreads = shared ? (opts.decodeThreads || Math.max(1, Math.min(threads, Math.ceil(cap.cores / 2)))) : 1;
  const file = `kernels-${shared ? "mt" : "st"}${relaxed ? "-relaxed" : ""}.wasm`;
  const worker = new Worker(new URL("./cpu-worker.js", import.meta.url), { type: "module" });

  let pending = null;
  const ready = new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") opts.onProgress?.(m.got, m.total);
      else if (m.type === "ready") resolve(m);
      else if (m.type === "error") { (pending ? pending.reject : reject)(new Error(m.message)); pending = null; }
      else if (m.type === "token") { const p = pending; pending = null; p.resolve(m); }
    };
    worker.onerror = (e) => reject(new Error(e.message));
  });
  worker.postMessage({
    type: "init", bytes, tensors: gguf.tensors, config: cfg, wasmUrl: new URL(file, import.meta.url).href,
    threads, decodeThreads, ctx: opts.ctx || 2048, mt: opts.prefillChunk || 16, shared,
  }, [bytes]);
  const info = await ready;

  const call = (msg) => new Promise((resolve, reject) => { pending = { resolve, reject }; worker.postMessage(msg); });
  let lastLogits = null;
  return {
    name: "cpu", ctx: opts.ctx || 2048, threads: info.threads, build: file, memoryBytes: info.bytes,
    info: info.threads > 1 ? `${info.threads} threads for prompts, ${info.decodeThreads} for generation, ${relaxed ? "relaxed SIMD" : "SIMD128"}` : `1 thread, ${relaxed ? "relaxed SIMD" : "SIMD128"}`,
    async forward(ids, pos, { temperature = 0, seed = 1, wantLogits = false } = {}) {
      const r = await call({ type: "forward", ids, pos, temperature, seed, wantLogits: wantLogits || !!opts.keepLogits });
      lastLogits = r.logits;
      return r.next;
    },
    async readLogits() {
      return lastLogits;
    },
    rawForward: (ids, pos, nth) => call({ type: "forward", ids, pos, nth, wantLogits: true }),
    profile: async (ids, pos) => (await call({ type: "profile", ids, pos })).profile,
    readTest: async (threads) => (await call({ type: "read-test", threads })).out,
    barrierTest: (iters) => call({ type: "barrier-test", iters }),
    destroy: () => worker.terminate(),
  };
}
