// GGUF reader: the file format llama.cpp uses for model weights.
//
// Layout:  "GGUF" | version | tensor count | metadata (key → value) | tensor table (name, shape, type, offset) | data
// We read it as plain bytes; no ML library involved.

export const TYPE = { F32: 0, F16: 1, Q4_0: 2, Q8_0: 8 };
const BLOCK = { [TYPE.Q4_0]: [32, 18], [TYPE.Q8_0]: [32, 34], [TYPE.F32]: [1, 4], [TYPE.F16]: [1, 2] };

export function tensorBytes(type, count) {
  const b = BLOCK[type];
  if (!b) throw new Error(`unsupported tensor type ${type}`);
  return (count / b[0]) * b[1];
}

export function parseGGUF(buffer) {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  const dec = new TextDecoder();
  let p = 0;
  const u32 = () => { const v = view.getUint32(p, true); p += 4; return v; };
  const u64 = () => { const v = Number(view.getBigUint64(p, true)); p += 8; return v; };
  const str = () => { const n = u64(); const s = dec.decode(bytes.subarray(p, p + n)); p += n; return s; };

  const readValue = (type) => {
    switch (type) {
      case 0: return view.getUint8(p++);
      case 1: return view.getInt8(p++);
      case 2: { const v = view.getUint16(p, true); p += 2; return v; }
      case 3: { const v = view.getInt16(p, true); p += 2; return v; }
      case 4: return u32();
      case 5: { const v = view.getInt32(p, true); p += 4; return v; }
      case 6: { const v = view.getFloat32(p, true); p += 4; return v; }
      case 7: return view.getUint8(p++) !== 0;
      case 8: return str();
      case 9: {
        const t = u32(), n = u64();
        if (t === 5) { const a = new Int32Array(n); for (let i = 0; i < n; i++) a[i] = view.getInt32(p + 4 * i, true); p += 4 * n; return a; }
        if (t === 6) { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = view.getFloat32(p + 4 * i, true); p += 4 * n; return a; }
        const a = new Array(n); for (let i = 0; i < n; i++) a[i] = readValue(t); return a;
      }
      case 10: return u64();
      case 11: { const v = Number(view.getBigInt64(p, true)); p += 8; return v; }
      case 12: { const v = view.getFloat64(p, true); p += 8; return v; }
      default: throw new Error(`unknown GGUF value type ${type}`);
    }
  };

  if (u32() !== 0x46554747) throw new Error("not a GGUF file");
  const version = u32();
  const nTensors = u64(), nKV = u64();
  const meta = {};
  for (let i = 0; i < nKV; i++) { const k = str(); meta[k] = readValue(u32()); }

  const tensors = {};
  for (let i = 0; i < nTensors; i++) {
    const name = str();
    const nd = u32();
    const shape = []; for (let d = 0; d < nd; d++) shape.push(u64());   // shape[0] = row length (fastest)
    const type = u32(), offset = u64();
    tensors[name] = { name, shape, type, offset, count: shape.reduce((a, b) => a * b, 1) };
  }
  const align = meta["general.alignment"] || 32;
  const dataStart = Math.ceil(p / align) * align;
  for (const t of Object.values(tensors)) {
    t.start = dataStart + t.offset;
    t.bytes = tensorBytes(t.type, t.count);
  }
  return { version, meta, tensors, dataStart };
}

// Qwen2-family hyperparameters from the metadata.
export function qwen2Config(meta) {
  const a = meta["general.architecture"];
  const g = (k) => meta[`${a}.${k}`];
  return {
    arch: a,
    dim: g("embedding_length"),
    layers: g("block_count"),
    ffn: g("feed_forward_length"),
    heads: g("attention.head_count"),
    kvHeads: g("attention.head_count_kv"),
    ropeTheta: g("rope.freq_base") || 10000,
    eps: g("attention.layer_norm_rms_epsilon"),
    vocab: meta["tokenizer.ggml.tokens"].length,
    eos: meta["tokenizer.ggml.eos_token_id"],
    contextMax: g("context_length"),
  };
}

// Download a model file with progress, keeping a copy in the browser's Cache Storage so the next visit is instant.
export async function fetchModel(url, onProgress = () => {}) {
  const cache = typeof caches !== "undefined" ? await caches.open("h2-models") : null;
  let res = cache && (await cache.match(url));
  const fromCache = !!res;
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`download failed: ${res.status} ${url}`);
  }
  // Read the body once, straight into one buffer (no clone: a cloned response keeps a second full copy in memory).
  const total = Number(res.headers.get("content-length")) || 0;
  let out = new Uint8Array(total || 64 << 20), got = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > out.length) {             // unknown or wrong length: grow
      const bigger = new Uint8Array(Math.max(out.length * 2, got + value.length));
      bigger.set(out.subarray(0, got)); out = bigger;
    }
    out.set(value, got); got += value.length;
    onProgress(got, total || got, fromCache);
  }
  if (got !== out.length) out = out.slice(0, got);
  if (cache && !fromCache) {
    try { await cache.put(url, new Response(out, { headers: { "content-length": String(got) } })); } catch { /* quota: no cache */ }
  }
  return out.buffer;
}
