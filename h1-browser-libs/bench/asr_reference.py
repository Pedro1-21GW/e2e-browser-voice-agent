"""Reference Parakeet Redux TDT pipeline in Python (onnxruntime, CPU).

Used to validate the greedy TDT decoding loop before porting it to JavaScript,
and as the native-speed baseline the browser numbers are compared against.

Usage:
    pip install onnxruntime numpy soundfile
    python h1-browser-libs/bench/asr_reference.py h1-browser-libs/samples/test.wav
    T=1 python ...   # limit onnxruntime to 1 thread
"""
import os, sys, time, urllib.request
import numpy as np, soundfile as sf, onnxruntime as rt

HF = "https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-redux-onnx/resolve/main/"
FILES = {"nemo128.onnx": "nemo128.onnx", "vocab.txt": "vocab.txt",
         "encoder.onnx": "w2a8/encoder-model.w2a8.onnx", "decoder.onnx": "int8/decoder_joint-model.int8.onnx"}
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(DIR, exist_ok=True)
for name, remote in FILES.items():
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        print("downloading", remote)
        urllib.request.urlretrieve(HF + remote, path)

opts = rt.SessionOptions()
opts.intra_op_num_threads = int(os.environ.get("T", "0"))
P = ["CPUExecutionProvider"]
pre = rt.InferenceSession(os.path.join(DIR, "nemo128.onnx"), opts, providers=P)
enc = rt.InferenceSession(os.path.join(DIR, "encoder.onnx"), opts, providers=P)
dec = rt.InferenceSession(os.path.join(DIR, "decoder.onnx"), opts, providers=P)
vocab = {}
for line in open(os.path.join(DIR, "vocab.txt"), encoding="utf-8"):
    tok, i = line.rstrip("\n").split(" ")
    vocab[int(i)] = tok.replace("▁", " ")
V = len(vocab)  # 8193 tokens, the last one is <blk>
BLANK = 8192


def transcribe(wav):
    t0 = time.perf_counter()
    feats, flens = pre.run(None, {"waveforms": wav[None], "waveforms_lens": np.array([len(wav)], np.int64)})
    t1 = time.perf_counter()
    out, olens = enc.run(None, {"audio_signal": feats, "length": flens})
    t2 = time.perf_counter()
    e = out[0].T  # [frames, 1024]
    s1 = np.zeros((2, 1, 640), np.float32); s2 = np.zeros((2, 1, 640), np.float32)
    toks = []; t = 0; emitted = 0; steps = 0
    while t < olens[0]:
        o, ns1, ns2 = dec.run(["outputs", "output_states_1", "output_states_2"], {
            "encoder_outputs": e[t][None, :, None],
            "targets": np.array([[toks[-1] if toks else BLANK]], np.int32),
            "target_length": np.array([1], np.int32),
            "input_states_1": s1, "input_states_2": s2})
        steps += 1
        o = o.reshape(-1)
        tok = int(o[:V].argmax())   # which token (or blank)
        dur = int(o[V:].argmax())   # how many frames to skip: 0..4
        if tok != BLANK:
            s1, s2 = ns1, ns2; toks.append(tok); emitted += 1
        if dur > 0:
            t += dur; emitted = 0
        elif tok == BLANK or emitted == 10:
            t += 1; emitted = 0
    t3 = time.perf_counter()
    text = "".join(vocab[i] for i in toks).strip()
    return text, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, steps, olens[0]


for f in sys.argv[1:]:
    wav, sr = sf.read(f, dtype="float32"); assert sr == 16000
    transcribe(wav)  # warm-up
    text, a, b, c, steps, T = transcribe(wav)
    print(f"{f}: {len(wav)/sr:.2f}s audio | pre {a:.0f}ms enc {b:.0f}ms dec {c:.0f}ms "
          f"({steps} decoder steps, {T} encoder frames) | {len(wav)/sr/((a+b+c)/1e3):.1f}x real time")
    print("  ->", text)
