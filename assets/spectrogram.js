// Shared with pedro1-21gw.github.io: a live, scrolling spectrogram of synthetic speech.
// ---------- Live spectrogram of synthetic speech
// Each new column on the right is one moment of a made-up voice: a pitch that drifts, its harmonics
// shaped by two moving formants (vowels), syllable on/off envelopes, and bursts of hiss (s, f, sh).
// The picture scrolls left at a constant speed, so it changes smoothly as time goes on.
export function startSpectrogram(canvas) {
  const ctx = canvas.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SPEED = 60;          // CSS pixels per second (1 px per frame at 60 Hz)
  const SIM = 0.0055;        // seconds of simulated speech per CSS pixel: ~4 s across the page, played at ~1/3 speed
  const FMAX = 5000;         // Hz shown at the top (higher, single pixel rows would be wider than the harmonic spacing)
  let W = 0, H = 0, dpr = 1, column, t = 0, pending = 0, last = 0;

  // amber colormap: black → deep red → orange → pale yellow
  const STOPS = [[0, 10, 8, 7], [0.25, 70, 18, 8], [0.5, 190, 70, 20], [0.75, 255, 157, 77], [1, 255, 236, 200]];
  const LUT = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const v = i / 255;
    let k = 0; while (k < STOPS.length - 2 && v > STOPS[k + 1][0]) k++;
    const [a, ...ca] = STOPS[k], [b, ...cb] = STOPS[k + 1], f = (v - a) / (b - a);
    for (let c = 0; c < 3; c++) LUT[i * 3 + c] = ca[c] + (cb[c] - ca[c]) * f;
  }

  const mel = (f) => 2595 * Math.log10(1 + f / 700);
  const imel = (m) => 700 * (10 ** (m / 2595) - 1);
  let rowFreq = [];
  const gauss = (x, w) => Math.exp(-(x * x) / (2 * w * w));
  const smooth = (x) => x * x * (3 - 2 * x);
  // deterministic pseudo-random value per (integer step, channel) for syllable patterns
  const hash = (n, c) => { const x = Math.sin(n * 127.1 + c * 311.7) * 43758.5453; return x - Math.floor(x); };
  const at = (time, rate, c) => { const n = Math.floor(time * rate), f = smooth(time * rate - n); return hash(n, c) * (1 - f) + hash(n + 1, c) * f; };

  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  function drawColumn(x) {
    const phrase = smooth(clamp01((at(t, 0.45, 1) - 0.22) * 5));       // speaking vs. pausing, smooth in and out
    const syll = smooth(clamp01((at(t, 4.5, 7) - 0.28) * 2.2));         // ~4 syllables/s with soft onsets
    const voiced = phrase * syll;
    const f0 = 110 + 50 * at(t, 0.9, 2) + 6 * Math.sin(t * 9);         // pitch contour with a little vibrato
    const F1 = 320 + 480 * at(t, 3.2, 3), F2 = 950 + 1350 * at(t, 2.6, 4), F3 = 2500 + 350 * at(t, 1.4, 5);
    const hiss = phrase * (1 - syll) * smooth(clamp01((at(t, 3.6, 6) - 0.5) * 3));   // s / f / sh between vowels
    const img = column.data;
    for (let y = 0; y < H; y++) {
      const f = rowFreq[y];
      let e = 0;
      if (voiced > 0.01) {
        const k = Math.max(1, Math.round(f / f0)), d = f - k * f0;       // distance to the nearest harmonic
        const harm = gauss(d, 9 + 0.006 * f);
        const env = 0.015 + gauss(f - F1, 120) + 0.6 * gauss(f - F2, 170) + 0.25 * gauss(f - F3, 240);
        e += voiced * harm * env * Math.exp(-f / 1300);                 // energy falls off with frequency
      }
      if (hiss > 0.01) e += hiss * 0.005 * smooth(clamp01((f - 2000) / 3000)) * (0.4 + 0.6 * Math.random());
      e += 0.00035 * (0.5 + Math.random());                             // faint noise floor
      const db = 10 * Math.log10(e + 1e-5);                             // ≈ -36 dB (floor) .. 0 dB (formant peaks)
      const v = Math.max(0, Math.min(255, ((db + 42) / 42) * 255)) | 0;
      const i = y * 4;
      img[i] = LUT[v * 3]; img[i + 1] = LUT[v * 3 + 1]; img[i + 2] = LUT[v * 3 + 2]; img[i + 3] = 255;
    }
    ctx.putImageData(column, x, 0);
  }

  function resize() {
    dpr = Math.min(2, devicePixelRatio || 1);
    W = Math.round(canvas.clientWidth * dpr); H = Math.round(canvas.clientHeight * dpr);
    if (!W || !H) return;
    canvas.width = W; canvas.height = H;
    column = ctx.createImageData(1, H);
    const mTop = mel(FMAX);
    rowFreq = Array.from({ length: H }, (_, y) => imel(mTop * (1 - y / (H - 1))));   // mel scale, low notes at the bottom
    const step = SIM / dpr;
    t -= W * step;
    for (let x = 0; x < W; x++) { drawColumn(x); t += step; }                     // fill the screen once
  }

  function frame(now) {
    const dt = Math.min(0.1, (now - (last || now)) / 1000); last = now;
    pending += dt * SPEED * dpr;
    const n = Math.floor(pending);
    if (n > 0 && W) {
      pending -= n;
      ctx.drawImage(canvas, n, 0, W - n, H, 0, 0, W - n, H);                     // scroll left
      for (let i = n; i > 0; i--) { t += SIM / dpr; drawColumn(W - i); }
    }
    requestAnimationFrame(frame);
  }

  resize();
  let lastWidth = canvas.clientWidth;
  addEventListener("resize", () => { if (canvas.clientWidth !== lastWidth) { lastWidth = canvas.clientWidth; resize(); } });
  if (!reduce) requestAnimationFrame(frame);
}
