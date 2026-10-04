# e2e-browser-voice-agent

An end-to-end voice agent (listen → understand → reply → speak) that runs **entirely inside a browser tab**:
no server, no API keys, the user's audio never leaves their device.

The repository is organized as a series of **hypotheses**. Each one is a question that gets a folder, a
working build, measurements, and a verdict. Later hypotheses build on what the earlier ones taught.

| # | Hypothesis | Status | Try it | Write-up |
|---|---|---|---|---|
| **H1** | Can a single HTML file, using existing browser ML libraries and the browser cache, make a voice agent with low latency? | ✅ **Confirmed with WebGPU**: 1.4–2.1 s from the end of speech to the agent's first sound, on a laptop with a GTX 1050. ❌ **Not without WebGPU**: 12–18 s. | [demo](https://pedro1-21gw.github.io/e2e-browser-voice-agent/h1-browser-libs/) | [h1-browser-libs/README.md](h1-browser-libs/README.md) |
| **H2** | Can we write the inference engine ourselves (our own GPU kernels) and still run it all in one HTML file? | 🔬 Started: first hand-written WebGPU kernels are correct; fp32 matrix-vector at 60% of memory bandwidth, matrix-matrix at 7% of peak (next: close the gap) | [kernel lab](https://pedro1-21gw.github.io/e2e-browser-voice-agent/h2-own-inference-engine/m0-kernels/) | [h2-own-inference-engine/README.md](h2-own-inference-engine/README.md) |

## Learn how it works

The [wiki](wiki/Home.md) explains every piece in plain language, with box-and-arrow diagrams and the real
numbers from the experiments. Start at [wiki/Home.md](wiki/Home.md).

## How each hypothesis is written up

Every hypothesis folder has a README with the same sections, so they can be compared:

1. **Hypothesis**: the question, in one sentence.
2. **Why it matters**: what we'd learn or unlock.
3. **Setup**: models, libraries, hardware.
4. **Method**: what was built and how it was measured.
5. **Results**: tables of measured numbers, including the failed attempts.
6. **Verdict**: confirmed, rejected, or "it depends", in plain words.
7. **Limitations** and **what's next**.

## Run it locally

```bash
# from the repo root (use 127.0.0.1, not localhost)
python -m http.server 8765 --bind 127.0.0.1
# then open http://127.0.0.1:8765/h1-browser-libs/ in Chrome or Edge
```

Benchmarks drive a real Chrome through Playwright:

```bash
pip install -r requirements.txt
python h1-browser-libs/bench/latency_test.py            # end-to-end latency on test clips
MIC=h1-browser-libs/bench/mic.wav python h1-browser-libs/bench/latency_test.py   # through a fake microphone
```

## Test machine

All numbers in this repo come from one laptop unless stated otherwise: Intel i5-9300HF (4 cores / 8 threads),
NVIDIA GTX 1050 with 3 GB (no fp16 shader support), 8 GB RAM, Windows 11, Chrome 154.
That's a modest, 2019-era machine; a recent laptop should be faster.

## License

The code in this repository is [MIT](LICENSE). The models it downloads are not part of the repository and keep
their own licenses, listed below.

## Models and credits

| Role | Model | License |
|---|---|---|
| Voice activity detection | [Silero VAD](https://github.com/snakers4/silero-vad) | MIT |
| Speech to text | NVIDIA [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) → Moondream [Parakeet Redux](https://huggingface.co/moondream/parakeet-redux) (ternary encoder) → [ONNX export](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-redux-onnx) | CC BY 4.0 |
| Language model | Liquid AI [LFM2 350M/700M](https://huggingface.co/onnx-community/LFM2-350M-ONNX), Alibaba [Qwen2.5 0.5B](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct) | LFM Open License v1.0 / Apache 2.0 |
| Text to speech | [Kokoro 82M](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) via [kokoro-js](https://www.npmjs.com/package/kokoro-js) | Apache 2.0 |
| Runtimes | [ONNX Runtime Web](https://onnxruntime.ai), [Transformers.js](https://github.com/huggingface/transformers.js) | MIT / Apache 2.0 |
