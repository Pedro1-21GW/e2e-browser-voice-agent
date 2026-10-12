// CPU kernels for the H2 engine, compiled to WebAssembly with 128-bit SIMD (and threads in the MT build).
//
// Build: python h2-own-inference-engine/engine/build.py   (uses Zig's clang; see build.py for the flags)
//
// Same model as the WebGPU backend (Qwen2 layout, GGUF Q4_0 layers + Q8_0 output layer). The trick that makes
// the CPU fast is the one llama.cpp uses: also quantize the *activations* to 8-bit integers (one scale per 32
// values), so every inner loop is an integer dot product: 16 multiply-adds per SIMD instruction instead of 4.
//
// Threads: every worker runs forward(tid, nth) on the same shared memory; each operation splits its rows
// between threads, and a barrier separates operations. JavaScript only starts a pass and reads the logits.

#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

#define EXPORT(name) __attribute__((export_name(#name)))
#define MAXL 64

// Optional profiling: thread 0 notes the time after each operation's barrier (clock imported from JavaScript).
__attribute__((import_module("env"), import_name("now"))) extern double js_now(void);
enum { P_EMBED, P_NORM_Q, P_QKV, P_ROPE, P_ATTN, P_QUANT, P_O, P_GU, P_SWIGLU, P_DOWN, P_OUT, P_N };
static double prof[P_N]; static int profiling;
static double prof_t;
static inline void tick(int tid, int k) { if (profiling && tid == 0) { double t = js_now(); prof[k] += t - prof_t; prof_t = t; } }

typedef struct { const uint8_t* qs; const float* d; } Q4;   // qs: 16 bytes per 32 values; d: one scale per 32
typedef struct { const int8_t* qs; const float* d; } Q8;
typedef struct { const float *attn_norm, *ffn_norm, *bqkv; Q4 qkv, o, gu, down; } Layer;

static struct {
  int dim, layers, ffn, heads, kvh, hd, vocab, ctx, mt;
  float eps;
  Q4 embd; Q8 out; const float* out_norm;
  Layer L[MAXL];
  const float* rope;                  // [ctx][hd/2][2] = cos, sin
  // activations, for up to mt tokens
  float *x, *xn, *qkv, *att, *gu, *act, *logits, *kc, *vc, *scores;
  int8_t* aq; float *ad, *as;         // quantized activations: int8 values, scale per block, scale·sum per block
  const int* tokens;
  int pos0, M, want_logits, mode, test_iters;
} g;

// ------------------------------------------------------------------ threads
#ifdef MT
static volatile int bar_count, bar_gen, job_gen, job_nth;
static void barrier(int nth) {
  if (nth == 1) return;
  int gen = __atomic_load_n(&bar_gen, __ATOMIC_ACQUIRE);
  if (__atomic_add_fetch(&bar_count, 1, __ATOMIC_ACQ_REL) == nth) {
    __atomic_store_n(&bar_count, 0, __ATOMIC_RELAXED);
    __atomic_add_fetch(&bar_gen, 1, __ATOMIC_RELEASE);
    __builtin_wasm_memory_atomic_notify((int*)&bar_gen, -1);
  } else {
    for (int spin = 0; __atomic_load_n(&bar_gen, __ATOMIC_ACQUIRE) == gen; spin++) {
      if (spin > 200000) __builtin_wasm_memory_atomic_wait32((int*)&bar_gen, gen, 1000000);   // 1 ms
    }
  }
}
#else
static void barrier(int nth) { (void)nth; }
#endif

static inline void split(int n, int tid, int nth, int* a, int* b) {
  int per = (n + nth - 1) / nth;
  *a = tid * per; *b = *a + per > n ? n : *a + per;
}

// ------------------------------------------------------------------ math helpers
static inline float fast_exp(float x) {           // e^x via 2^n · poly(r); ~1e-7 relative error
  if (x < -87.0f) return 0.0f;
  if (x > 88.0f) x = 88.0f;
  float n = __builtin_floorf(x * 1.44269504f + 0.5f);
  float r = x - n * 0.693145752f - n * 1.42860677e-6f;
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.166666672f + r * (0.0416666679f + r * (0.00833333377f + r * 0.00138888892f)))));
  union { float f; int32_t i; } u = { .f = p };
  u.i += (int32_t)n << 23;
  return u.f;
}

static void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
  v128_t s = wasm_f32x4_splat(0);
  for (int i = 0; i < n; i += 4) { v128_t v = wasm_v128_load(x + i); s = wasm_f32x4_add(s, wasm_f32x4_mul(v, v)); }
  float ss = wasm_f32x4_extract_lane(s, 0) + wasm_f32x4_extract_lane(s, 1) + wasm_f32x4_extract_lane(s, 2) + wasm_f32x4_extract_lane(s, 3);
  v128_t r = wasm_f32x4_splat(1.0f / __builtin_sqrtf(ss / n + eps));
  for (int i = 0; i < n; i += 4) wasm_v128_store(out + i, wasm_f32x4_mul(wasm_f32x4_mul(wasm_v128_load(x + i), r), wasm_v128_load(w + i)));
}

// Quantize n floats to int8 in blocks of 32: q = round(x / d), d = max|x| / 127. Also keep d·sum(q) per block,
// which the relaxed-SIMD kernel needs to undo the +8 offset of the 4-bit weights.
static void quantize_row(int8_t* q, float* d, float* s, const float* x, int n) {
  for (int b = 0; b < n / 32; b++) {
    const float* xb = x + 32 * b;
    v128_t mx = wasm_f32x4_splat(0);
    for (int i = 0; i < 32; i += 4) mx = wasm_f32x4_max(mx, wasm_f32x4_abs(wasm_v128_load(xb + i)));
    float l0 = wasm_f32x4_extract_lane(mx, 0), l1 = wasm_f32x4_extract_lane(mx, 1);
    float l2 = wasm_f32x4_extract_lane(mx, 2), l3 = wasm_f32x4_extract_lane(mx, 3);
    float amax = l0 > l1 ? l0 : l1; if (l2 > amax) amax = l2; if (l3 > amax) amax = l3;
    float dd = amax / 127.0f, id = dd ? 1.0f / dd : 0.0f;
    v128_t vid = wasm_f32x4_splat(id);
    int sum = 0;
    for (int i = 0; i < 32; i += 16) {
      v128_t a = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + i), vid)));
      v128_t b2 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + i + 4), vid)));
      v128_t c = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + i + 8), vid)));
      v128_t e = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + i + 12), vid)));
      v128_t s01 = wasm_i16x8_narrow_i32x4(a, b2), s23 = wasm_i16x8_narrow_i32x4(c, e);
      v128_t q8 = wasm_i8x16_narrow_i16x8(s01, s23);
      wasm_v128_store(q + 32 * b + i, q8);
      v128_t t = wasm_i32x4_add(wasm_i32x4_add(a, b2), wasm_i32x4_add(c, e));
      sum += wasm_i32x4_extract_lane(t, 0) + wasm_i32x4_extract_lane(t, 1) + wasm_i32x4_extract_lane(t, 2) + wasm_i32x4_extract_lane(t, 3);
    }
    d[b] = dd; s[b] = dd * (float)sum;
  }
}

static inline float hsum(v128_t v) {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

// Dot product of one Q4_0 row (nb blocks) with one quantized activation row.
static inline float dot_q4_q8(const uint8_t* wq, const float* wd, const int8_t* xq, const float* xd, const float* xs, int nb) {
  v128_t acc = wasm_f32x4_splat(0);
  const v128_t m4 = wasm_i8x16_splat(0x0F);
#ifndef RELAXED
  const v128_t s8 = wasm_i8x16_splat(8);
  (void)xs;
  for (int b = 0; b < nb; b++) {
    v128_t qs = wasm_v128_load(wq + 16 * b);
    v128_t lo = wasm_i8x16_sub(wasm_v128_and(qs, m4), s8);           // values 0..15  → -8..7
    v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(qs, 4), s8);           // values 16..31 → -8..7
    v128_t x0 = wasm_v128_load(xq + 32 * b), x1 = wasm_v128_load(xq + 32 * b + 16);
    v128_t p = wasm_i32x4_add(
      wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(lo), wasm_i16x8_extend_low_i8x16(x0)),
                     wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(lo), wasm_i16x8_extend_high_i8x16(x0))),
      wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(hi), wasm_i16x8_extend_low_i8x16(x1)),
                     wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(hi), wasm_i16x8_extend_high_i8x16(x1))));
    acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(p), wasm_f32x4_splat(wd[b] * xd[b])));
  }
  return hsum(acc);
#else
  // Relaxed SIMD: one instruction multiplies 16 int8 pairs and adds groups of 4 into an int32x4.
  // Its second operand must be 0..127, so the 4-bit weights stay unsigned (0..15) and we subtract 8·sum(x) after.
  float corr = 0.0f;
  for (int b = 0; b < nb; b++) {
    v128_t qs = wasm_v128_load(wq + 16 * b);
    v128_t lo = wasm_v128_and(qs, m4), hi = wasm_u8x16_shr(qs, 4);
    v128_t x0 = wasm_v128_load(xq + 32 * b), x1 = wasm_v128_load(xq + 32 * b + 16);
    v128_t p = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(x0, lo, wasm_i32x4_splat(0));
    p = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(x1, hi, p);
    acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(p), wasm_f32x4_splat(wd[b] * xd[b])));
    corr += wd[b] * xs[b];
  }
  return hsum(acc) - 8.0f * corr;
#endif
}

static inline float dot_q8_q8(const int8_t* wq, const float* wd, const int8_t* xq, const float* xd, int nb) {
  v128_t acc = wasm_f32x4_splat(0);
  for (int b = 0; b < nb; b++) {
    v128_t w0 = wasm_v128_load(wq + 32 * b), w1 = wasm_v128_load(wq + 32 * b + 16);
    v128_t x0 = wasm_v128_load(xq + 32 * b), x1 = wasm_v128_load(xq + 32 * b + 16);
    v128_t p = wasm_i32x4_add(
      wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(w0), wasm_i16x8_extend_low_i8x16(x0)),
                     wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(w0), wasm_i16x8_extend_high_i8x16(x0))),
      wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(w1), wasm_i16x8_extend_low_i8x16(x1)),
                     wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(w1), wasm_i16x8_extend_high_i8x16(x1))));
    acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(p), wasm_f32x4_splat(wd[b] * xd[b])));
  }
  return hsum(acc);
}

// One Q4_0 block (32 weights, unpacked into lo = values 0..15 and hi = values 16..31) · 32 int8 inputs → int32x4.
#ifdef RELAXED
#define Q4_UNPACK(qs, lo, hi) do { lo = wasm_v128_and(qs, m4); hi = wasm_u8x16_shr(qs, 4); } while (0)
#define Q4_BDOT(lo, hi, x0, x1) wasm_i32x4_relaxed_dot_i8x16_i7x16_add(x1, hi, wasm_i32x4_relaxed_dot_i8x16_i7x16_add(x0, lo, wasm_i32x4_splat(0)))
#define CORR(c, v) (c += (v))          // relaxed: weights kept as 0..15, so subtract 8·Σx afterwards
#else
#define Q4_UNPACK(qs, lo, hi) do { lo = wasm_i8x16_sub(wasm_v128_and(qs, m4), s8); hi = wasm_i8x16_sub(wasm_u8x16_shr(qs, 4), s8); } while (0)
static inline v128_t q4_bdot(v128_t lo, v128_t hi, v128_t x0, v128_t x1) {
  return wasm_i32x4_add(
    wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(lo), wasm_i16x8_extend_low_i8x16(x0)),
                   wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(lo), wasm_i16x8_extend_high_i8x16(x0))),
    wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(hi), wasm_i16x8_extend_low_i8x16(x1)),
                   wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(hi), wasm_i16x8_extend_high_i8x16(x1))));
}
#define Q4_BDOT q4_bdot
#define CORR(c, v) ((void)0)
#endif

enum { STORE, BIAS, ADD };
static inline void put(float* y, int N, int m, int n, float v, int op, const float* bias) {
  float* o = y + (size_t)m * N + n;
  if (op == ADD) *o += v; else if (op == BIAS) *o = v + bias[n]; else *o = v;
}

// y[m][n] (op)= W[n] · x[m] for rows [a, b).
//  - one token (generation): 4 rows per pass, so each input block is loaded once for 4 rows and the horizontal
//    sums happen once per row instead of once per block;
//  - several tokens (prompt): tiles of 2 rows × 4 tokens, so each weight block is unpacked once and used 4 times.
static void matmul_q4(Q4 W, int K, int N, int a, int b, int M, const int8_t* xq, const float* xd, const float* xs,
                      float* y, int op, const float* bias) {
  const int nb = K / 32;
  const v128_t m4 = wasm_i8x16_splat(0x0F);
#ifndef RELAXED
  const v128_t s8 = wasm_i8x16_splat(8);
#endif
  (void)xs;
  int n = a;
  if (M == 1) {
    for (; n + 4 <= b; n += 4) {
      const uint8_t* w = W.qs + (size_t)n * nb * 16; const float* d = W.d + (size_t)n * nb;
      v128_t a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0;
      float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
      for (int k = 0; k < nb; k++) {
        v128_t x0 = wasm_v128_load(xq + 32 * k), x1 = wasm_v128_load(xq + 32 * k + 16), lo, hi;
        float dx = xd[k];
#define ROW(r, acc, c) { Q4_UNPACK(wasm_v128_load(w + ((size_t)(r) * nb + k) * 16), lo, hi); float dw = d[(r) * nb + k]; \
          acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(Q4_BDOT(lo, hi, x0, x1)), wasm_f32x4_splat(dw * dx))); CORR(c, dw * xs[k]); }
        ROW(0, a0, c0) ROW(1, a1, c1) ROW(2, a2, c2) ROW(3, a3, c3)
#undef ROW
      }
#ifdef RELAXED
      put(y, N, 0, n, hsum(a0) - 8 * c0, op, bias); put(y, N, 0, n + 1, hsum(a1) - 8 * c1, op, bias);
      put(y, N, 0, n + 2, hsum(a2) - 8 * c2, op, bias); put(y, N, 0, n + 3, hsum(a3) - 8 * c3, op, bias);
#else
      put(y, N, 0, n, hsum(a0), op, bias); put(y, N, 0, n + 1, hsum(a1), op, bias);
      put(y, N, 0, n + 2, hsum(a2), op, bias); put(y, N, 0, n + 3, hsum(a3), op, bias);
#endif
    }
  } else {
    for (; n + 2 <= b; n += 2) {
      const uint8_t* w0 = W.qs + (size_t)n * nb * 16; const uint8_t* w1 = w0 + (size_t)nb * 16;
      const float* d0 = W.d + (size_t)n * nb; const float* d1 = d0 + nb;
      int m = 0;
      for (; m + 4 <= M; m += 4) {
        v128_t r00 = wasm_f32x4_splat(0), r01 = r00, r02 = r00, r03 = r00, r10 = r00, r11 = r00, r12 = r00, r13 = r00;
        float c00 = 0, c01 = 0, c02 = 0, c03 = 0, c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        for (int k = 0; k < nb; k++) {
          v128_t lo0, hi0, lo1, hi1;
          Q4_UNPACK(wasm_v128_load(w0 + 16 * k), lo0, hi0);
          Q4_UNPACK(wasm_v128_load(w1 + 16 * k), lo1, hi1);
          float e0 = d0[k], e1 = d1[k];
#define TOK(t, r0, r1, cc0, cc1) { const int8_t* xp = xq + (size_t)(m + (t)) * K + 32 * k; \
            v128_t x0 = wasm_v128_load(xp), x1 = wasm_v128_load(xp + 16); float dx = xd[(m + (t)) * nb + k]; \
            r0 = wasm_f32x4_add(r0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(Q4_BDOT(lo0, hi0, x0, x1)), wasm_f32x4_splat(e0 * dx))); \
            r1 = wasm_f32x4_add(r1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(Q4_BDOT(lo1, hi1, x0, x1)), wasm_f32x4_splat(e1 * dx))); \
            CORR(cc0, e0 * xs[(m + (t)) * nb + k]); CORR(cc1, e1 * xs[(m + (t)) * nb + k]); }
          TOK(0, r00, r10, c00, c10) TOK(1, r01, r11, c01, c11) TOK(2, r02, r12, c02, c12) TOK(3, r03, r13, c03, c13)
#undef TOK
        }
#ifdef RELAXED
        put(y, N, m, n, hsum(r00) - 8 * c00, op, bias); put(y, N, m + 1, n, hsum(r01) - 8 * c01, op, bias);
        put(y, N, m + 2, n, hsum(r02) - 8 * c02, op, bias); put(y, N, m + 3, n, hsum(r03) - 8 * c03, op, bias);
        put(y, N, m, n + 1, hsum(r10) - 8 * c10, op, bias); put(y, N, m + 1, n + 1, hsum(r11) - 8 * c11, op, bias);
        put(y, N, m + 2, n + 1, hsum(r12) - 8 * c12, op, bias); put(y, N, m + 3, n + 1, hsum(r13) - 8 * c13, op, bias);
#else
        put(y, N, m, n, hsum(r00), op, bias); put(y, N, m + 1, n, hsum(r01), op, bias);
        put(y, N, m + 2, n, hsum(r02), op, bias); put(y, N, m + 3, n, hsum(r03), op, bias);
        put(y, N, m, n + 1, hsum(r10), op, bias); put(y, N, m + 1, n + 1, hsum(r11), op, bias);
        put(y, N, m + 2, n + 1, hsum(r12), op, bias); put(y, N, m + 3, n + 1, hsum(r13), op, bias);
#endif
      }
      for (; m < M; m++) {                  // leftover tokens
        put(y, N, m, n, dot_q4_q8(w0, d0, xq + (size_t)m * K, xd + m * nb, xs + m * nb, nb), op, bias);
        put(y, N, m, n + 1, dot_q4_q8(w1, d1, xq + (size_t)m * K, xd + m * nb, xs + m * nb, nb), op, bias);
      }
    }
  }
  for (; n < b; n++) {                      // leftover rows
    const uint8_t* wq = W.qs + (size_t)n * nb * 16; const float* wd = W.d + (size_t)n * nb;
    for (int m = 0; m < M; m++) put(y, N, m, n, dot_q4_q8(wq, wd, xq + (size_t)m * K, xd + m * nb, xs + m * nb, nb), op, bias);
  }
}

static void quantize_tokens(const float* x, int K, int M, int tid, int nth) {
  for (int m = tid; m < M; m += nth) quantize_row(g.aq + m * K, g.ad + m * (K / 32), g.as + m * (K / 32), x + m * K, K);
}

// ------------------------------------------------------------------ barrier self-test (debug)
static volatile int test_slot[64], test_errors;
static void barrier_test(int tid, int nth) {
  for (int i = 1; i <= g.test_iters; i++) {
    test_slot[tid] = i;
    barrier(nth);
    for (int t = 0; t < nth; t++) if (test_slot[t] != i) __atomic_add_fetch(&test_errors, 1, __ATOMIC_RELAXED);
    barrier(nth);
  }
}

// ------------------------------------------------------------------ the forward pass
static void forward_pass(int tid, int nth);
// Memory read test (debug): every thread sums its slice of the output layer's weights.
static volatile int bw_sink;
static void read_test(int tid, int nth) {
  int a, b; size_t n16 = (size_t)g.vocab * g.dim / 16;
  split((int)n16, tid, nth, &a, &b);
  const v128_t* p = (const v128_t*)g.out.qs;
  v128_t s0 = wasm_i32x4_splat(0), s1 = s0, s2 = s0, s3 = s0;
  int i = a;
  for (; i + 4 <= b; i += 4) { s0 = wasm_v128_xor(s0, p[i]); s1 = wasm_v128_xor(s1, p[i + 1]); s2 = wasm_v128_xor(s2, p[i + 2]); s3 = wasm_v128_xor(s3, p[i + 3]); }
  v128_t s = wasm_v128_xor(wasm_v128_xor(s0, s1), wasm_v128_xor(s2, s3));
  bw_sink += wasm_i32x4_extract_lane(s, 0);
  barrier(nth);
}
static void forward(int tid, int nth) { int mode = g.mode; if (mode == 1) barrier_test(tid, nth); else if (mode == 2) read_test(tid, nth); else forward_pass(tid, nth); }
static void forward_pass(int tid, int nth) {
  const int D = g.dim, F = g.ffn, H = g.heads, KVH = g.kvh, HD = g.hd, QKV = D + 2 * KVH * HD, M = g.M;
  // Read the job's parameters once, now: after the final barrier the leader may already be writing the next job's.
  const int pos0 = g.pos0, want_logits = g.want_logits;
  int a, b;

  // embedding lookup (Q4_0 rows) for the M new tokens
  for (int m = tid; m < M; m += nth) {
    int t = g.tokens[pos0 + m];
    const uint8_t* q = g.embd.qs + (size_t)t * (D / 32) * 16; const float* d = g.embd.d + (size_t)t * (D / 32);
    float* xr = g.x + m * D;
    for (int blk = 0; blk < D / 32; blk++)
      for (int j = 0; j < 16; j++) {
        xr[32 * blk + j] = ((q[16 * blk + j] & 15) - 8) * d[blk];
        xr[32 * blk + 16 + j] = ((q[16 * blk + j] >> 4) - 8) * d[blk];
      }
  }
  barrier(nth); tick(tid, P_EMBED);

  for (int l = 0; l < g.layers; l++) {
    const Layer* L = &g.L[l];
    // attention block
    for (int m = tid; m < M; m += nth) {
      rmsnorm(g.xn + m * D, g.x + m * D, L->attn_norm, D, g.eps);
      quantize_row(g.aq + m * D, g.ad + m * (D / 32), g.as + m * (D / 32), g.xn + m * D, D);
    }
    barrier(nth); tick(tid, P_NORM_Q);
    split(QKV, tid, nth, &a, &b);
    matmul_q4(L->qkv, D, QKV, a, b, M, g.aq, g.ad, g.as, g.qkv, BIAS, L->bqkv);
    barrier(nth); tick(tid, P_QKV);
    // RoPE on Q and K heads, then append K and V to the cache
    for (int task = tid; task < M * (H + KVH); task += nth) {
      int m = task / (H + KVH), h = task % (H + KVH), pos = pos0 + m;
      float* v = g.qkv + m * QKV + h * HD;
      const float* cs = g.rope + (size_t)pos * HD;
      for (int i = 0; i < HD / 2; i++) {
        float c = cs[2 * i], s = cs[2 * i + 1], x0 = v[i], x1 = v[i + HD / 2];
        v[i] = x0 * c - x1 * s; v[i + HD / 2] = x1 * c + x0 * s;
      }
      if (h >= H) {
        int kh = h - H;
        float* kd = g.kc + (((size_t)l * KVH + kh) * g.ctx + pos) * HD;
        float* vd = g.vc + (((size_t)l * KVH + kh) * g.ctx + pos) * HD;
        const float* vs = g.qkv + m * QKV + D + KVH * HD + kh * HD;
        for (int i = 0; i < HD; i++) { kd[i] = v[i]; vd[i] = vs[i]; }
      }
    }
    barrier(nth); tick(tid, P_ROPE);
    // attention: one task per (token, head)
    for (int task = tid; task < M * H; task += nth) {
      int m = task / H, h = task % H, T = pos0 + m + 1, kh = h / (H / KVH);
      const float* q = g.qkv + m * QKV + h * HD;
      const float* kb = g.kc + ((size_t)l * KVH + kh) * g.ctx * HD;
      const float* vb = g.vc + ((size_t)l * KVH + kh) * g.ctx * HD;
      float* sc = g.scores + (size_t)tid * g.ctx;
      float mx = -1e30f, scale = 1.0f / __builtin_sqrtf((float)HD);
      for (int t = 0; t < T; t++) {
        v128_t s = wasm_f32x4_splat(0);
        for (int i = 0; i < HD; i += 4) s = wasm_f32x4_add(s, wasm_f32x4_mul(wasm_v128_load(q + i), wasm_v128_load(kb + (size_t)t * HD + i)));
        float v = hsum(s) * scale; sc[t] = v; if (v > mx) mx = v;
      }
      float sum = 0;
      for (int t = 0; t < T; t++) { float e = fast_exp(sc[t] - mx); sc[t] = e; sum += e; }
      float* o = g.att + m * D + h * HD;
      v128_t acc[16];
      for (int i = 0; i < HD / 4; i++) acc[i] = wasm_f32x4_splat(0);
      for (int t = 0; t < T; t++) {
        v128_t p = wasm_f32x4_splat(sc[t] / sum);
        for (int i = 0; i < HD / 4; i++) acc[i] = wasm_f32x4_add(acc[i], wasm_f32x4_mul(p, wasm_v128_load(vb + (size_t)t * HD + 4 * i)));
      }
      for (int i = 0; i < HD / 4; i++) wasm_v128_store(o + 4 * i, acc[i]);
    }
    barrier(nth); tick(tid, P_ATTN);
    quantize_tokens(g.att, D, M, tid, nth);
    barrier(nth); tick(tid, P_QUANT);
    split(D, tid, nth, &a, &b);
    matmul_q4(L->o, D, D, a, b, M, g.aq, g.ad, g.as, g.x, ADD, 0);
    barrier(nth); tick(tid, P_O);
    // feed-forward block
    for (int m = tid; m < M; m += nth) {
      rmsnorm(g.xn + m * D, g.x + m * D, L->ffn_norm, D, g.eps);
      quantize_row(g.aq + m * D, g.ad + m * (D / 32), g.as + m * (D / 32), g.xn + m * D, D);
    }
    barrier(nth); tick(tid, P_NORM_Q);
    split(2 * F, tid, nth, &a, &b);
    matmul_q4(L->gu, D, 2 * F, a, b, M, g.aq, g.ad, g.as, g.gu, STORE, 0);
    barrier(nth); tick(tid, P_GU);
    for (int task = tid; task < M; task += nth) {          // silu(gate) · up, then quantize
      float* gu = g.gu + (size_t)task * 2 * F; float* ac = g.act + (size_t)task * F;
      for (int i = 0; i < F; i++) { float gt = gu[i]; ac[i] = gt / (1.0f + fast_exp(-gt)) * gu[F + i]; }
      quantize_row(g.aq + task * F, g.ad + task * (F / 32), g.as + task * (F / 32), ac, F);
    }
    barrier(nth); tick(tid, P_SWIGLU);
    split(D, tid, nth, &a, &b);
    matmul_q4(L->down, F, D, a, b, M, g.aq, g.ad, g.as, g.x, ADD, 0);
    barrier(nth); tick(tid, P_DOWN);
  }

  if (want_logits) {
    if (tid == 0) {
      rmsnorm(g.xn, g.x + (M - 1) * D, g.out_norm, D, g.eps);
      quantize_row(g.aq, g.ad, g.as, g.xn, D);
    }
    barrier(nth);
    split(g.vocab, tid, nth, &a, &b);
    int nb = D / 32;
    for (int n = a; n < b; n++) g.logits[n] = dot_q8_q8(g.out.qs + (size_t)n * D, g.out.d + (size_t)n * nb, g.aq, g.ad, nb);
    barrier(nth); tick(tid, P_OUT);
  }
}

// ------------------------------------------------------------------ exports
EXPORT(set_config) void set_config(int dim, int layers, int ffn, int heads, int kvh, int vocab, int ctx, int mt, float eps) {
  g.dim = dim; g.layers = layers; g.ffn = ffn; g.heads = heads; g.kvh = kvh; g.hd = dim / heads;
  g.vocab = vocab; g.ctx = ctx; g.mt = mt; g.eps = eps;
}
// kind: 0 qkv, 1 o, 2 gu, 3 down (Q4); 4 attn_norm, 5 ffn_norm, 6 bqkv (f32); 7 embd (Q4); 8 output (Q8); 9 output_norm
EXPORT(set_tensor) void set_tensor(int kind, int layer, void* qs, float* d) {
  Layer* L = &g.L[layer];
  switch (kind) {
    case 0: L->qkv = (Q4){ qs, d }; break;   case 1: L->o = (Q4){ qs, d }; break;
    case 2: L->gu = (Q4){ qs, d }; break;    case 3: L->down = (Q4){ qs, d }; break;
    case 4: L->attn_norm = d; break;         case 5: L->ffn_norm = d; break;   case 6: L->bqkv = d; break;
    case 7: g.embd = (Q4){ qs, d }; break;   case 8: g.out = (Q8){ qs, d }; break;   case 9: g.out_norm = d; break;
  }
}
// One call with every scratch buffer (allocated by JavaScript inside this module's memory).
EXPORT(set_buffers) void set_buffers(float* x, float* xn, float* qkv, float* att, float* gu, float* act, float* logits,
                                     float* kc, float* vc, float* scores, int8_t* aq, float* ad, float* as, const float* rope, const int* tokens) {
  g.x = x; g.xn = xn; g.qkv = qkv; g.att = att; g.gu = gu; g.act = act; g.logits = logits; g.kc = kc; g.vc = vc;
  g.scores = scores; g.aq = aq; g.ad = ad; g.as = as; g.rope = rope; g.tokens = tokens;
}

#ifdef MT
// Helper threads park here forever, waking for each new job.
EXPORT(worker_loop) void worker_loop(int tid) {
  int seen = __atomic_load_n(&job_gen, __ATOMIC_ACQUIRE);
  for (;;) {
    for (int spin = 0; __atomic_load_n(&job_gen, __ATOMIC_ACQUIRE) == seen; spin++)
      if (spin > 200000) __builtin_wasm_memory_atomic_wait32((int*)&job_gen, seen, -1);
    seen = __atomic_load_n(&job_gen, __ATOMIC_ACQUIRE);
    if (tid < job_nth) forward(tid, job_nth);
  }
}
#endif

// Run tokens[pos0 .. pos0+M) through the model, using nth threads (thread 0 is the caller).
EXPORT(run_barrier_test) int run_barrier_test(int iters, int nth);
EXPORT(run) void run(int pos0, int M, int want_logits, int nth) {
  g.pos0 = pos0; g.M = M; g.want_logits = want_logits; g.mode = 0;
  if (profiling) prof_t = js_now();
#ifdef MT
  job_nth = nth;
  __atomic_add_fetch(&job_gen, 1, __ATOMIC_RELEASE);
  __builtin_wasm_memory_atomic_notify((int*)&job_gen, -1);
#else
  nth = 1;
#endif
  forward(0, nth);
}

EXPORT(run_barrier_test) int run_barrier_test(int iters, int nth) {
  g.mode = 1; g.test_iters = iters; test_errors = 0;
#ifdef MT
  job_nth = nth;
  __atomic_add_fetch(&job_gen, 1, __ATOMIC_RELEASE);
  __builtin_wasm_memory_atomic_notify((int*)&job_gen, -1);
#else
  nth = 1;
#endif
  forward(0, nth);
  g.mode = 0;
  return test_errors;
}

EXPORT(profile_start) void profile_start(void) { for (int i = 0; i < P_N; i++) prof[i] = 0; profiling = 1; prof_t = js_now(); }
EXPORT(profile_get) double profile_get(int k) { return prof[k]; }
EXPORT(profile_stop) void profile_stop(void) { profiling = 0; }

EXPORT(run_read_test) void run_read_test(int nth) {
  g.mode = 2;
#ifdef MT
  job_nth = nth;
  __atomic_add_fetch(&job_gen, 1, __ATOMIC_RELEASE);
  __builtin_wasm_memory_atomic_notify((int*)&job_gen, -1);
#else
  nth = 1;
#endif
  forward(0, nth);
  g.mode = 0;
}
