// WebGPU backend for Qwen2-style models stored as GGUF Q4_0 (layers) + Q8_0 (output layer).
//
// One forward pass = one command buffer. For each of the 24 layers:
//
//   x ─▶ RMSNorm ─▶ [Q|K|V] = W·x + b ─▶ RoPE, append K,V to cache ─▶ attention ─▶ x += Wo·att
//     ─▶ RMSNorm ─▶ [gate|up] = W·x ─▶ silu(gate)·up ─▶ x += Wdown·(…)
//
// then the output layer (151,936 rows) and a sampler that writes the next token id into a GPU buffer, so the
// next step can start without the CPU reading anything back first.

import { TYPE } from "./gguf.js";

const U = GPUBufferUsage;

// ------------------------------------------------------------------ weight repacking (GGUF blocks → GPU arrays)
// Q4_0 block (18 bytes): fp16 scale + 16 bytes holding 32 4-bit values (low nibbles = values 0..15,
// high nibbles = values 16..31). We split it into a u32 array of the 16 bytes and an f32 array of scales,
// because WGSL can only read 4-byte-aligned words.
function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
const F16 = new Float32Array(65536);
for (let i = 0; i < 65536; i++) F16[i] = f16(i);

function repackQ4(bytes, t) {                         // → { words: Uint32Array, scales: Float32Array }
  const nb = t.count / 32;
  const words = new Uint32Array(nb * 4), scales = new Float32Array(nb);
  const src = new Uint8Array(bytes, t.start, t.bytes);
  const out8 = new Uint8Array(words.buffer);
  for (let b = 0; b < nb; b++) {
    const o = b * 18;
    scales[b] = F16[src[o] | (src[o + 1] << 8)];
    out8.set(src.subarray(o + 2, o + 18), b * 16);
  }
  return { words, scales };
}
function repackQ8(bytes, t) {                         // Q8_0 block (34 bytes): fp16 scale + 32 int8
  const nb = t.count / 32;
  const words = new Uint32Array(nb * 8), scales = new Float32Array(nb);
  const src = new Uint8Array(bytes, t.start, t.bytes);
  const out8 = new Uint8Array(words.buffer);
  for (let b = 0; b < nb; b++) {
    const o = b * 34;
    scales[b] = F16[src[o] | (src[o + 1] << 8)];
    out8.set(src.subarray(o + 2, o + 34), b * 32);
  }
  return { words, scales };
}
function f32Tensor(bytes, t) {
  if (t.type !== TYPE.F32) throw new Error(`${t.name}: expected F32`);
  return new Float32Array(bytes.slice(t.start, t.start + t.bytes));
}
function concatQ4(parts) {
  const words = new Uint32Array(parts.reduce((n, p) => n + p.words.length, 0));
  const scales = new Float32Array(parts.reduce((n, p) => n + p.scales.length, 0));
  let w = 0, s = 0;
  for (const p of parts) { words.set(p.words, w); scales.set(p.scales, s); w += p.words.length; s += p.scales.length; }
  return { words, scales };
}

// ------------------------------------------------------------------ WGSL kernels
const HEADER = `enable subgroups;\nrequires packed_4x8_integer_dot_product;\n`;

// Per-step parameters, written before each submit.
const PARAMS = `struct P { pos0: u32, M: u32, inOff: u32, seed: u32, temperature: f32, layer: u32, pad0: u32, pad1: u32 };`;

// RMSNorm: out = x / sqrt(mean(x²) + eps) * w, one workgroup per token.
const RMSNORM = (D, eps) => `${HEADER}${PARAMS}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, 32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(subgroup_size) sgs: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let m = wg.x; if (m >= p.M) { return; }
  let base = p.inOff + m * ${D}u;
  var ss = 0.0;
  for (var k = li; k < ${D}u; k += 128u) { let v = x[base + k]; ss += v * v; }
  ss = subgroupAdd(ss);
  if (lane == 0u) { red[li / sgs] = ss; }
  workgroupBarrier();
  var tot = 0.0;
  for (var i = 0u; i < 128u / sgs; i++) { tot += red[i]; }
  let r = inverseSqrt(tot / ${D}.0 + ${eps});
  for (var k = li; k < ${D}u; k += 128u) { y[m * ${D}u + k] = x[base + k] * r * w[k]; }
}`;

// y[m, n] (op)= W[n, :] · x[m, :] for M tokens, Q4_0 weights.
// Each workgroup computes RW rows; each thread walks the row in 8-value words (one u32 of packed weights),
// so neighbouring threads read neighbouring words (coalesced). Partial sums are added with subgroupAdd,
// then across subgroups through shared memory.
const GEMV_Q4 = ({ K, MT, RW, WG, epi }) => `${HEADER}${PARAMS}
@group(0) @binding(0) var<storage, read> Wq: array<u32>;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<uniform> shape: vec4<u32>;     // (N, ...)
${epi === "bias" ? "@group(0) @binding(6) var<storage, read> bias: array<f32>;" : ""}
var<workgroup> red: array<f32, ${(WG / 16) * RW * MT}>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(subgroup_size) sgs: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let N = shape.x; let row0 = wg.x * ${RW}u;
  var acc: array<f32, ${RW * MT}>;
  for (var j = li; j < ${K / 8}u; j += ${WG}u) {
    let blk = j / 4u; let w = j % 4u;
    var xl: array<vec4<f32>, ${MT}>; var xh: array<vec4<f32>, ${MT}>;
    for (var m = 0u; m < ${MT}u; m++) {
      if (m < p.M) {
        let xb = (m * ${K}u + blk * 32u) / 4u + w;
        xl[m] = x[xb]; xh[m] = x[xb + 4u];
      }
    }
    for (var r = 0u; r < ${RW}u; r++) {
      let n = row0 + r;
      if (n < N) {
        let q = Wq[n * ${K / 8}u + j]; let s = S[n * ${K / 32}u + blk];
        let lo = vec4<f32>(unpack4xU8(q & 0x0F0F0F0Fu)) - 8.0;
        let hi = vec4<f32>(unpack4xU8((q >> 4u) & 0x0F0F0F0Fu)) - 8.0;
        for (var m = 0u; m < ${MT}u; m++) { acc[r * ${MT}u + m] += s * (dot(lo, xl[m]) + dot(hi, xh[m])); }
      }
    }
  }
  let nsg = ${WG}u / sgs; let sg = li / sgs;
  for (var i = 0u; i < ${RW * MT}u; i++) {
    let v = subgroupAdd(acc[i]);
    if (lane == 0u) { red[sg * ${RW * MT}u + i] = v; }
  }
  workgroupBarrier();
  if (li < ${RW * MT}u) {
    let r = li / ${MT}u; let m = li % ${MT}u; let n = row0 + r;
    if (n < N && m < p.M) {
      var v = 0.0;
      for (var g = 0u; g < nsg; g++) { v += red[g * ${RW * MT}u + li]; }
      ${epi === "bias" ? "y[m * N + n] = v + bias[n];" : epi === "residual" ? "y[m * N + n] += v;" : "y[m * N + n] = v;"}
    }
  }
}`;

// Single-token path, version 4 (the decode hot path), in two kernels:
//
//  QUANT: one small workgroup turns the input into int8 (one scale per 32 values), applying RMSNorm or
//         silu(gate)·up on the way, so those steps need no dispatch of their own.
//  GEMV1: the multiply is integer: dot4I8Packed does 4 int8 multiply-adds in one instruction (DP4A on NVIDIA).
//         4-bit weights become signed bytes with ((q & 0x0F0F0F0F) + 0x78787878) ^ 0x80808080 (= q − 8 per byte,
//         no carries between bytes). Neighbouring threads read neighbouring weight words (coalesced); each thread
//         works on R rows at once so several loads are in flight. The int8 input (1 KB) stays in cache.
//
// (Version 2 staged the input in shared memory per workgroup and gave each lane a whole block: slower, because
//  of 32-way shared-memory bank conflicts. Version 3 also quantized per workgroup: slower, because 1,216
//  workgroups each redid the same setup. Measured, both in the H2 README.)
const QUANT = ({ K, input, eps }) => `${HEADER}${PARAMS}
@group(0) @binding(0) var<storage, read> xin: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xd: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${input === "norm" ? "@group(0) @binding(4) var<storage, read> nw: array<vec4<f32>>;" : ""}
var<workgroup> red: array<f32, 64>;
fn xval(i: u32, rn: f32) -> vec4<f32> {
  ${input === "norm" ? "return xin[p.inOff / 4u + i] * rn * nw[i];" : input === "swiglu" ? `let g = xin[p.inOff / 4u + i]; return g / (1.0 + exp(-g)) * xin[p.inOff / 4u + ${K / 4}u + i];` : "return xin[p.inOff / 4u + i];"}
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(subgroup_size) sgs: u32, @builtin(subgroup_invocation_id) lane: u32) {
  var rn = 1.0;
  ${input === "norm" ? `
  var ss = 0.0;
  for (var i = li; i < ${K / 4}u; i += 256u) { let v = xin[p.inOff / 4u + i]; ss += dot(v, v); }
  ss = subgroupAdd(ss);
  if (lane == 0u) { red[li / sgs] = ss; }
  workgroupBarrier();
  var tot = 0.0; for (var i = 0u; i < 256u / sgs; i++) { tot += red[i]; }
  rn = inverseSqrt(tot / ${K}.0 + ${eps});` : ""}
  for (var b = li; b < ${K / 32}u; b += 256u) {
    var v: array<vec4<f32>, 8>; var mx = 0.0;
    for (var j = 0u; j < 8u; j++) { v[j] = xval(8u * b + j, rn); let a = abs(v[j]); mx = max(mx, max(max(a.x, a.y), max(a.z, a.w))); }
    let d = mx / 127.0; let id = select(0.0, 1.0 / d, d > 0.0);
    xd[b] = d;
    for (var j = 0u; j < 8u; j++) { xq[8u * b + j] = pack4xI8(vec4<i32>(round(v[j] * id))); }
  }
}`;

const GEMV1_Q4 = ({ K, R, WG, epi }) => `${HEADER}${PARAMS}
@group(0) @binding(0) var<storage, read> Wq: array<u32>;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> xq: array<u32>;
@group(0) @binding(3) var<storage, read> xd: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> shape: vec4<u32>;
${epi === "bias" ? "@group(0) @binding(6) var<storage, read> bias: array<f32>;" : ""}
var<workgroup> red: array<f32, ${(WG / 16) * R}>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(subgroup_size) sgs: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let N = shape.x; let row0 = wg.x * ${R}u;
  var acc: array<f32, ${R}>;
  for (var j = li; j < ${K / 8}u; j += ${WG}u) {      // j = weight word: 8 values of block j/4
    let blk = j / 4u; let w = j % 4u;
    let xl = xq[8u * blk + w]; let xh = xq[8u * blk + 4u + w]; let dx = xd[blk];
    for (var r = 0u; r < ${R}u; r++) {
      let n = row0 + r;
      if (n < N) {
        let q = Wq[n * ${K / 8}u + j];
        let lo = ((q & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u;
        let hi = (((q >> 4u) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u;
        acc[r] += S[n * ${K / 32}u + blk] * dx * f32(dot4I8Packed(lo, xl) + dot4I8Packed(hi, xh));
      }
    }
  }
  let sg = li / sgs;
  for (var r = 0u; r < ${R}u; r++) {
    let v = subgroupAdd(acc[r]);
    if (lane == 0u) { red[sg * ${R}u + r] = v; }
  }
  workgroupBarrier();
  if (li < ${R}u) {
    let n = row0 + li;
    if (n < N) {
      var v = 0.0; for (var g = 0u; g < ${WG}u / sgs; g++) { v += red[g * ${R}u + li]; }
      ${epi === "bias" ? "y[n] = v + bias[n];" : epi === "residual" ? "y[n] += v;" : "y[n] = v;"}
    }
  }
}`;

// Output layer, Q8_0 weights, one token. Same structure; a word is 4 int8 values.
const GEMV_Q8 = ({ K, RW, WG }) => `${HEADER}${PARAMS}
@group(0) @binding(0) var<storage, read> Wq: array<u32>;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: vec4<u32>;
var<workgroup> red: array<f32, ${(WG / 16) * RW}>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(subgroup_size) sgs: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let N = shape.x; let row0 = (wg.y * 65535u + wg.x) * ${RW}u;
  var acc: array<f32, ${RW}>;
  for (var j = li; j < ${K / 4}u; j += ${WG}u) {
    let xv = x[j];
    for (var r = 0u; r < ${RW}u; r++) {
      let n = row0 + r;
      if (n < N) { acc[r] += S[n * ${K / 32}u + j / 8u] * dot(vec4<f32>(unpack4xI8(Wq[n * ${K / 4}u + j])), xv); }
    }
  }
  let nsg = ${WG}u / sgs; let sg = li / sgs;
  for (var r = 0u; r < ${RW}u; r++) {
    let v = subgroupAdd(acc[r]);
    if (lane == 0u) { red[sg * ${RW}u + r] = v; }
  }
  workgroupBarrier();
  if (li < ${RW}u) {
    let n = row0 + li;
    if (n < N) { var v = 0.0; for (var g = 0u; g < nsg; g++) { v += red[g * ${RW}u + li]; } y[n] = v; }
  }
}`;

// RoPE on Q and K (rotate pairs (i, i+32) by pos·θ_i), then append K and V to this layer's cache.
const ROPE_KV = ({ H, KVH, HD, CTX, theta }) => `${HEADER}${PARAMS}
@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<f32>;
@group(0) @binding(2) var<storage, read_write> vc: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(${(H + KVH) * (HD / 2)})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = wg.x; if (m >= p.M) { return; }
  let pos = p.pos0 + m;
  let h = li / ${HD / 2}u; let i = li % ${HD / 2}u;           // head (14 query + 2 key), pair index
  let base = m * ${(H + 2 * KVH) * HD}u + h * ${HD}u;
  let freq = pow(${theta}, -f32(2u * i) / ${HD}.0);
  let turns = f32(pos) * (freq / 6.2831853); let ang = 6.2831853 * (fract(turns + 0.5) - 0.5);
  let c = cos(ang); let s = sin(ang);
  let a = qkv[base + i]; let b = qkv[base + i + ${HD / 2}u];
  let ra = a * c - b * s; let rb = b * c + a * s;
  if (h < ${H}u) { qkv[base + i] = ra; qkv[base + i + ${HD / 2}u] = rb; }
  else {
    let kh = h - ${H}u;
    let cb = ((p.layer * ${KVH}u + kh) * ${CTX}u + pos) * ${HD}u;
    kc[cb + i] = ra; kc[cb + i + ${HD / 2}u] = rb;
    let vb = m * ${(H + 2 * KVH) * HD}u + ${(H + KVH) * HD}u + kh * ${HD}u;
    vc[cb + i] = qkv[vb + i]; vc[cb + i + ${HD / 2}u] = qkv[vb + i + ${HD / 2}u];
  }
}`;

// Attention for one (head, token): scores over all cached positions up to this token, softmax, weighted sum of V.
const ATTN = ({ H, KVH, HD, CTX }) => `${HEADER}${PARAMS}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> att: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> sc: array<f32, ${CTX}>;
var<workgroup> q: array<vec4<f32>, ${HD / 4}>;
var<workgroup> red: array<f32, 32>;
var<workgroup> part: array<f32, ${2 * HD}>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(subgroup_size) sgs: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let h = wg.x; let m = wg.y; if (m >= p.M) { return; }
  let T = p.pos0 + m + 1u;
  let kvh = h / ${H / KVH}u;
  let cb = (p.layer * ${KVH}u + kvh) * ${CTX}u;
  if (li < ${HD / 4}u) {
    let qb = m * ${(H + 2 * KVH) * HD}u + h * ${HD}u + li * 4u;
    q[li] = vec4<f32>(qkv[qb], qkv[qb + 1u], qkv[qb + 2u], qkv[qb + 3u]) * ${1 / Math.sqrt(HD)};
  }
  workgroupBarrier();
  var mx = -1e30;
  for (var t = li; t < T; t += 128u) {
    var s = 0.0;
    for (var d = 0u; d < ${HD / 4}u; d++) { s += dot(q[d], kc[(cb + t) * ${HD / 4}u + d]); }
    sc[t] = s; mx = max(mx, s);
  }
  mx = subgroupMax(mx);
  if (lane == 0u) { red[li / sgs] = mx; }
  workgroupBarrier();
  mx = -1e30; for (var i = 0u; i < 128u / sgs; i++) { mx = max(mx, red[i]); }
  workgroupBarrier();
  var sum = 0.0;
  for (var t = li; t < T; t += 128u) { let e = exp(sc[t] - mx); sc[t] = e; sum += e; }
  sum = subgroupAdd(sum);
  if (lane == 0u) { red[li / sgs] = sum; }
  workgroupBarrier();
  sum = 0.0; for (var i = 0u; i < 128u / sgs; i++) { sum += red[i]; }
  let d = li % ${HD}u; let half = li / ${HD}u;               // 2 halves × 64 dims
  var o = 0.0;
  for (var t = half; t < T; t += 2u) { o += sc[t] * vc[(cb + t) * ${HD}u + d]; }
  part[li] = o;
  workgroupBarrier();
  if (li < ${HD}u) { att[m * ${H * HD}u + h * ${HD}u + li] = (part[li] + part[li + ${HD}u]) / sum; }
}`;

const SWIGLU = (F) => `${HEADER}${PARAMS}
@group(0) @binding(0) var<storage, read> gu: array<f32>;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.M * ${F}u) { return; }
  let m = i / ${F}u; let k = i % ${F}u;
  let gt = gu[m * ${2 * F}u + k]; let up = gu[m * ${2 * F}u + ${F}u + k];
  a[i] = gt / (1.0 + exp(-gt)) * up;
}`;

// Sampling: pick argmax of logits/T + Gumbel noise (= a sample from softmax(logits/T)); T = 0 → greedy.
// Stage 1: each workgroup reduces a slice; stage 2: one workgroup reduces the partials and writes the token.
const ARGMAX1 = (V, WG) => `${HEADER}${PARAMS}
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> rv: array<f32, ${WG}>; var<workgroup> ri: array<u32, ${WG}>;
fn hash(x: u32) -> u32 { var v = x; v ^= v >> 16u; v *= 0x7feb352du; v ^= v >> 15u; v *= 0x846ca68bu; v ^= v >> 16u; return v; }
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let i = g.x;
  var v = -1e30;
  if (i < ${V}u) {
    v = logits[i];
    if (p.temperature > 0.0) {
      let u = (f32(hash(i ^ hash(p.seed + p.pos0)) >> 8u) + 0.5) / 16777216.0;
      v = v / p.temperature - log(-log(u));
    }
  }
  rv[li] = v; ri[li] = i;
  workgroupBarrier();
  for (var off = ${WG / 2}u; off > 0u; off >>= 1u) {
    if (li < off && rv[li + off] > rv[li]) { rv[li] = rv[li + off]; ri[li] = ri[li + off]; }
    workgroupBarrier();
  }
  if (li == 0u) { part[wg.x] = vec2<f32>(rv[0], bitcast<f32>(ri[0])); }
}`;
const ARGMAX2 = (NP) => `${HEADER}${PARAMS}
@group(0) @binding(0) var<storage, read> part: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> rv: array<f32, 256>; var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var v = -1e30; var idx = 0u;
  for (var i = li; i < ${NP}u; i += 256u) { let e = part[i]; if (e.x > v) { v = e.x; idx = bitcast<u32>(e.y); } }
  rv[li] = v; ri[li] = idx;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off >>= 1u) {
    if (li < off && rv[li + off] > rv[li]) { rv[li] = rv[li + off]; ri[li] = ri[li + off]; }
    workgroupBarrier();
  }
  if (li == 0u) { tokens[p.pos0 + p.M] = ri[0]; }
}`;

// ------------------------------------------------------------------ backend
export async function createWebGPU(gguf, bytes, cfg, opts = {}) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU not available");
  if (!adapter.features.has("subgroups")) throw new Error("this GPU/browser lacks WebGPU subgroups");
  const L = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures: ["subgroups", ...(adapter.features.has("timestamp-query") ? ["timestamp-query"] : [])],
    requiredLimits: {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: Math.min(1024, L.maxComputeInvocationsPerWorkgroup),
      maxComputeWorkgroupSizeX: Math.min(1024, L.maxComputeWorkgroupSizeX),
      maxStorageBuffersPerShaderStage: Math.min(10, L.maxStorageBuffersPerShaderStage),
    },
  });
  device.lost.then((i) => console.error("WebGPU device lost:", i.message));

  const { dim: D, layers: NL, ffn: F, heads: H, kvHeads: KVH, vocab: V, eps, ropeTheta } = cfg;
  const HD = D / H, QKV = D + 2 * KVH * HD;
  const CTX = opts.ctx || 4096;
  const MT = opts.prefillChunk || 8;                  // tokens per prefill dispatch
  const T = gguf.tensors;

  const buf = (data, usage = U.STORAGE) => {
    const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: usage | U.COPY_DST });
    device.queue.writeBuffer(b, 0, data.buffer || data, data.byteOffset || 0, data.byteLength);
    return b;
  };
  const scratch = (bytes, extra = 0) => device.createBuffer({ size: Math.max(16, bytes), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST | extra });

  // ---- weights
  let uploaded = 0;
  const q4 = (name) => { const r = repackQ4(bytes, T[name]); uploaded += r.words.byteLength + r.scales.byteLength; return r; };
  const up = (r) => ({ w: buf(r.words), s: buf(r.scales) });
  // The embedding table (85 MB in Q4_0) stays in JavaScript: each step only needs the rows of its tokens, and GPU
  // memory is the scarce resource when speech recognition, the LLM and the voice share a 3 GB card.
  const embd = repackQ4(bytes, T["token_embd.weight"]);
  const embedRows = (ids) => {
    const out = new Float32Array(ids.length * D), u8 = new Uint8Array(embd.words.buffer);
    ids.forEach((t, m) => {
      for (let b = 0; b < D / 32; b++) {
        const blk = t * (D / 32) + b, s = embd.scales[blk], o = m * D + 32 * b;
        for (let j = 0; j < 16; j++) { const q = u8[blk * 16 + j]; out[o + j] = ((q & 15) - 8) * s; out[o + 16 + j] = ((q >> 4) - 8) * s; }
      }
    });
    return out;
  };
  const outW = repackQ8(bytes, T["output.weight"]); uploaded += outW.words.byteLength + outW.scales.byteLength;
  const out = { w: buf(outW.words), s: buf(outW.scales) };
  const outNorm = buf(f32Tensor(bytes, T["output_norm.weight"]));
  const layers = [];
  for (let l = 0; l < NL; l++) {
    const p = `blk.${l}.`;
    const bias = new Float32Array(QKV);
    bias.set(f32Tensor(bytes, T[p + "attn_q.bias"]), 0);
    bias.set(f32Tensor(bytes, T[p + "attn_k.bias"]), D);
    bias.set(f32Tensor(bytes, T[p + "attn_v.bias"]), D + KVH * HD);
    layers.push({
      attnNorm: buf(f32Tensor(bytes, T[p + "attn_norm.weight"])),
      ffnNorm: buf(f32Tensor(bytes, T[p + "ffn_norm.weight"])),
      qkv: up(concatQ4([q4(p + "attn_q.weight"), q4(p + "attn_k.weight"), q4(p + "attn_v.weight")])),
      bqkv: buf(bias),
      o: up(q4(p + "attn_output.weight")),
      gu: up(concatQ4([q4(p + "ffn_gate.weight"), q4(p + "ffn_up.weight")])),
      down: up(q4(p + "ffn_down.weight")),
      lp: buf(new Uint32Array([0, 0, 0, 0, 0, l, 0, 0]), U.UNIFORM),
    });
    if (opts.onProgress) opts.onProgress(l + 1, NL);
  }

  // ---- activations and cache
  const x = scratch(MT * D * 4), xn = scratch(MT * D * 4), qkv = scratch(MT * QKV * 4), att = scratch(MT * D * 4);
  const gu = scratch(MT * 2 * F * 4), act = scratch(MT * F * 4), logits = scratch(V * 4);
  const kc = scratch(NL * KVH * CTX * HD * 4), vc = scratch(NL * KVH * CTX * HD * 4);
  const tokens = scratch((CTX + 1) * 4);
  const AWG = 256, NP = Math.ceil(V / AWG);
  const part = scratch(NP * 8);
  const P = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
  const shapes = {};
  const shape = (n) => (shapes[n] ||= buf(new Uint32Array([n, 0, 0, 0]), U.UNIFORM));

  // ---- pipelines (one set for single-token decode, one for prefill chunks of MT tokens)
  const pipes = new Map();
  const pipe = (code) => {
    if (!pipes.has(code)) pipes.set(code, device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }));
    return pipes.get(code);
  };
  const RW = opts.rowsPerWG || 4, WG = opts.gemvWG || 128;
  const K = {
    norm: pipe(RMSNORM(D, eps)),
    rope: pipe(ROPE_KV({ H, KVH, HD, CTX, theta: ropeTheta.toFixed(1) })),
    attn: pipe(ATTN({ H, KVH, HD, CTX })),
    swiglu: pipe(SWIGLU(F)),
    out: pipe(GEMV_Q8({ K: D, RW: opts.outRowsPerWG || 8, WG })),
    am1: pipe(ARGMAX1(V, AWG)),
    am2: pipe(ARGMAX2(NP)),
  };
  const gemv = (Kd, mt, epi) => pipe(GEMV_Q4({ K: Kd, MT: mt, RW, WG, epi }));
  const R1 = opts.rows1 || 4, WG1 = opts.wg1 || 128;
  const gemv1 = (Kd, epi) => pipe(GEMV1_Q4({ K: Kd, R: R1, WG: WG1, epi }));
  const quant = (Kd, input) => pipe(QUANT({ K: Kd, input, eps }));
  const xq = scratch(F), xd = scratch(F / 32 * 4);         // the current matrix input, as int8 + scales

  const bg = (p, ...res) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: res.map((r, i) => ({ binding: i, resource: { buffer: r } })) });
  const lp = (l) => layers[l].lp;
  // Every layer's P differs only in `layer`; we keep one uniform per layer and copy pos0/M/... into it per step.

  // Build the command list for a forward pass of `mt`-token chunks (bind groups cached per variant).
  const plans = new Map();
  function plan(mt, withOutput) {
    const key = mt + ":" + withOutput;
    if (plans.has(key)) return plans.get(key);
    const ops = [];
    for (let l = 0; l < NL; l++) {
      const Ly = layers[l];
      if (mt === 1 && !opts.unfused) {            // decode: 4 fused matrix-vector kernels + RoPE + attention
        const qn = quant(D, "norm"), qp = quant(D, "plain"), qs = quant(F, "swiglu");
        const gq = gemv1(D, "bias"), go = gemv1(D, "residual"), gg = gemv1(D, "store"), gd = gemv1(F, "residual");
        ops.push([qn, bg(qn, x, xq, xd, lp(l), Ly.attnNorm), [1], "quantize (+norm)"]);
        ops.push([gq, bg(gq, Ly.qkv.w, Ly.qkv.s, xq, xd, qkv, shape(QKV), Ly.bqkv), [Math.ceil(QKV / R1)], "matvec qkv"]);
        ops.push([K.rope, bg(K.rope, qkv, kc, vc, lp(l)), [1], "rope+kv"]);
        ops.push([K.attn, bg(K.attn, qkv, kc, vc, att, lp(l)), [H, 1], "attention"]);
        ops.push([qp, bg(qp, att, xq, xd, lp(l)), [1], "quantize"]);
        ops.push([go, bg(go, Ly.o.w, Ly.o.s, xq, xd, x, shape(D)), [Math.ceil(D / R1)], "matvec o"]);
        ops.push([qn, bg(qn, x, xq, xd, lp(l), Ly.ffnNorm), [1], "quantize (+norm)"]);
        ops.push([gg, bg(gg, Ly.gu.w, Ly.gu.s, xq, xd, gu, shape(2 * F)), [Math.ceil(2 * F / R1)], "matvec gate+up"]);
        ops.push([qs, bg(qs, gu, xq, xd, lp(l)), [1], "quantize (+swiglu)"]);
        ops.push([gd, bg(gd, Ly.down.w, Ly.down.s, xq, xd, x, shape(D)), [Math.ceil(D / R1)], "matvec down"]);
        continue;
      }
      const gq = gemv(D, mt, "bias"), go = gemv(D, mt, "residual"), gg = gemv(D, mt, "store"), gd = gemv(F, mt, "residual");
      ops.push([K.norm, bg(K.norm, x, Ly.attnNorm, xn, lp(l)), [mt], "rmsnorm"]);
      ops.push([gq, bg(gq, Ly.qkv.w, Ly.qkv.s, xn, qkv, lp(l), shape(QKV), Ly.bqkv), [Math.ceil(QKV / RW)], "matvec qkv"]);
      ops.push([K.rope, bg(K.rope, qkv, kc, vc, lp(l)), [mt], "rope+kv"]);
      ops.push([K.attn, bg(K.attn, qkv, kc, vc, att, lp(l)), [H, mt], "attention"]);
      ops.push([go, bg(go, Ly.o.w, Ly.o.s, att, x, lp(l), shape(D)), [Math.ceil(D / RW)], "matvec o"]);
      ops.push([K.norm, bg(K.norm, x, Ly.ffnNorm, xn, lp(l)), [mt], "rmsnorm"]);
      ops.push([gg, bg(gg, Ly.gu.w, Ly.gu.s, xn, gu, lp(l), shape(2 * F)), [Math.ceil(2 * F / RW)], "matvec gate+up"]);
      ops.push([K.swiglu, bg(K.swiglu, gu, act, lp(l)), [Math.ceil(mt * F / 256)], "swiglu"]);
      ops.push([gd, bg(gd, Ly.down.w, Ly.down.s, act, x, lp(l), shape(D)), [Math.ceil(D / RW)], "matvec down"]);
    }
    if (withOutput) {
      const ORW = opts.outRowsPerWG || 8, nwg = Math.ceil(V / ORW);
      ops.push([K.norm, bg(K.norm, x, outNorm, xn, P), [1], "rmsnorm"]);              // P.inOff selects the last token
      ops.push([K.out, bg(K.out, out.w, out.s, xn, logits, shape(V)), [Math.min(nwg, 65535), Math.ceil(nwg / 65535)], "matvec output (Q8)"]);
      ops.push([K.am1, bg(K.am1, logits, part, P), [NP], "sample"]);
      ops.push([K.am2, bg(K.am2, part, tokens, P), [1], "sample"]);
    }
    plans.set(key, ops);
    return ops;
  }

  const staging = device.createBuffer({ size: 16, usage: U.MAP_READ | U.COPY_DST });
  let lastLogitsStep = null;

  // Run tokens[pos .. pos+n) through the model; returns the sampled next token (written to tokens[pos+n]).
  async function forward(ids, pos, { temperature = 0, seed = 1 } = {}) {
    if (pos + ids.length >= CTX) throw new Error("context full");
    device.queue.writeBuffer(tokens, pos * 4, new Uint32Array(ids));
    for (let c = 0; c < ids.length; c += MT) {
      const n = Math.min(MT, ids.length - c), last = c + n >= ids.length;
      const mt = n === 1 ? 1 : MT;
      device.queue.writeBuffer(x, 0, embedRows(ids.slice(c, c + n)));
      const params = new Uint32Array(8);
      params.set([pos + c, n, last ? (n - 1) * D : 0, seed >>> 0]);
      new Float32Array(params.buffer)[4] = temperature;
      device.queue.writeBuffer(P, 0, params);
      params[2] = 0;
      for (let l = 0; l < NL; l++) { params[5] = l; device.queue.writeBuffer(layers[l].lp, 0, params); }
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const [pp, g, grid] of plan(mt, last)) { pass.setPipeline(pp); pass.setBindGroup(0, g); pass.dispatchWorkgroups(...grid); }
      pass.end();
      if (last) enc.copyBufferToBuffer(tokens, (pos + ids.length) * 4, staging, 0, 4);
      device.queue.submit([enc.finish()]);
    }
    await staging.mapAsync(GPUMapMode.READ);
    const next = new Uint32Array(staging.getMappedRange().slice(0))[0];
    staging.unmap();
    return next;
  }

  // Per-kernel GPU time for one forward pass of `n` tokens at position `pos` (needs the timestamp-query feature;
  // Chrome rounds timestamps to 100 µs unless started with --enable-webgpu-developer-features).
  async function profile(n = 1, pos = 64) {
    if (!device.features.has("timestamp-query")) return null;
    const ops = plan(n === 1 ? 1 : MT, true);
    const qs = device.createQuerySet({ type: "timestamp", count: 2 * ops.length });
    const res = device.createBuffer({ size: 16 * ops.length, usage: U.QUERY_RESOLVE | U.COPY_SRC });
    const rd = device.createBuffer({ size: 16 * ops.length, usage: U.MAP_READ | U.COPY_DST });
    const params = new Uint32Array(8); params.set([pos, n, (n - 1) * D, 1]);
    device.queue.writeBuffer(x, 0, embedRows(new Array(n).fill(0)));
    device.queue.writeBuffer(P, 0, params);
    params[2] = 0; for (let l = 0; l < NL; l++) { params[5] = l; device.queue.writeBuffer(layers[l].lp, 0, params); }
    const enc = device.createCommandEncoder();
    ops.forEach(([pp, g, grid], i) => {
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      pass.setPipeline(pp); pass.setBindGroup(0, g); pass.dispatchWorkgroups(...grid); pass.end();
    });
    enc.resolveQuerySet(qs, 0, 2 * ops.length, res, 0);
    enc.copyBufferToBuffer(res, 0, rd, 0, 16 * ops.length);
    device.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(rd.getMappedRange().slice(0)); rd.unmap();
    const by = {};
    ops.forEach((op, i) => { const us = Number(t[2 * i + 1] - t[2 * i]) / 1000; (by[op[3]] ||= { us: 0, count: 0 }); by[op[3]].us += us; by[op[3]].count++; });
    qs.destroy(); res.destroy(); rd.destroy();
    return by;
  }

  async function readLogits() {
    const st = device.createBuffer({ size: V * 4, usage: U.MAP_READ | U.COPY_DST });
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(logits, 0, st, 0, V * 4); device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const outv = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
    return outv;
  }

  // Warm-up: compile every pipeline variant once.
  await forward([0], 0); await forward([0, 0], 0);

  return {
    name: "webgpu", device, ctx: CTX, forward, readLogits, profile, uploadedBytes: uploaded,
    info: `${adapter.info.vendor} ${adapter.info.architecture}`,
    destroy: () => device.destroy(),
  };
}
