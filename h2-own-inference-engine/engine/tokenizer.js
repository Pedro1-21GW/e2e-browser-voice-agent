// Byte-level BPE tokenizer (GPT-2 style, as used by Qwen2), built from the vocabulary and merge list stored
// in the GGUF file. No tokenizer library.
//
// text ──NFC──▶ split on special tokens ──▶ pre-split with a regex (words, numbers, punctuation, spaces)
//      ──▶ each piece as bytes, each byte shown as one printable character ──▶ merge pairs by rank ──▶ ids

// GPT-2's byte → printable character table: bytes that are already printable map to themselves,
// the rest (spaces, control bytes) are shifted to code points 256+ so every byte is a visible character.
function byteToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const map = new Array(256);
  bs.forEach((b, i) => { map[b] = String.fromCodePoint(cs[i]); });
  return map;
}

// Qwen2's pre-tokenizer pattern (same as in its tokenizer.json): contractions, words with an optional leading
// symbol, single digits, punctuation runs, newlines, and whitespace.
const PRE = /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

export class Tokenizer {
  constructor(meta) {
    this.tokens = meta["tokenizer.ggml.tokens"];
    const types = meta["tokenizer.ggml.token_type"];
    this.ids = new Map(this.tokens.map((t, i) => [t, i]));
    this.ranks = new Map(meta["tokenizer.ggml.merges"].map((m, i) => [m, i]));
    this.b2u = byteToUnicode();
    this.u2b = new Map(this.b2u.map((c, b) => [c, b]));
    // Control / user-defined tokens (<|im_start|>, <|im_end|>, <|endoftext|>, ...) are matched literally.
    this.special = this.tokens.filter((t, i) => types && (types[i] === 3 || types[i] === 4));
    this.specialRe = this.special.length
      ? new RegExp("(" + this.special.sort((a, b) => b.length - a.length).map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")")
      : null;
    this.cache = new Map();
    this.enc = new TextEncoder();
  }

  bpe(piece) {
    let hit = this.cache.get(piece);
    if (hit) return hit;
    let syms = Array.from(this.enc.encode(piece), (b) => this.b2u[b]);
    while (syms.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < syms.length - 1; i++) {
        const r = this.ranks.get(syms[i] + " " + syms[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = syms[best], b = syms[best + 1], merged = [];
      for (let i = 0; i < syms.length; i++) {
        if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) { merged.push(a + b); i++; }
        else merged.push(syms[i]);
      }
      syms = merged;
    }
    hit = syms.map((s) => {
      const id = this.ids.get(s);
      if (id === undefined) throw new Error(`no token for ${JSON.stringify(s)}`);
      return id;
    });
    if (this.cache.size < 50000) this.cache.set(piece, hit);
    return hit;
  }

  encode(text) {
    const out = [];
    const parts = this.specialRe ? text.normalize("NFC").split(this.specialRe) : [text.normalize("NFC")];
    for (const part of parts) {
      if (!part) continue;
      const sid = this.specialRe && this.ids.get(part);
      if (sid !== undefined && this.special.includes(part)) { out.push(sid); continue; }
      for (const m of part.matchAll(PRE)) out.push(...this.bpe(m[0]));
    }
    return out;
  }

  // Bytes of one token (special tokens as their literal text).
  tokenBytes(id) {
    const t = this.tokens[id];
    if (this.special.includes(t)) return this.enc.encode(t);
    return Uint8Array.from(t, (c) => this.u2b.get(c));
  }

  decode(ids) {
    const chunks = ids.map((id) => this.tokenBytes(id));
    const all = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
    let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
    return new TextDecoder().decode(all);
  }

  // Streaming decoder: a character can be split across tokens (emoji, accents), so keep partial bytes.
  streamDecoder() {
    const td = new TextDecoder();
    return (id) => td.decode(this.tokenBytes(id), { stream: true });
  }
}

// Qwen2.5's chat format (ChatML). Matches tokenizer.chat_template for plain text messages.
export function chatPrompt(messages, addGenerationPrompt = true) {
  let s = "";
  if (messages[0]?.role !== "system") s += "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
  for (const m of messages) s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  if (addGenerationPrompt) s += "<|im_start|>assistant\n";
  return s;
}
