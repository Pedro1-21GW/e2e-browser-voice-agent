# Glossary

| Term | In one line |
|---|---|
| **ASR / STT** | Automatic speech recognition / speech-to-text: audio in, words out. |
| **Cache API** | Browser storage for downloaded files; keeps the ~830 MB of models between visits. |
| **COOP / COEP** | Two HTTP security headers; with both, the page is "cross-origin isolated" and may use threads. |
| **Decode (LLM)** | Writing the reply one token at a time; sets the typing speed (tokens/s). |
| **Encoder** | The big part of a speech model that turns the spectrogram into a summary every 80 ms. |
| **Endpointing** | Deciding that the speaker's turn is over (here: 500 ms of quiet). |
| **fp32 / fp16** | 32-bit / 16-bit decimal numbers. The GTX 1050 can't do fp16 in WebGPU shaders. |
| **GGUF** | A file format for quantized model weights, used by llama.cpp, whisper.cpp and parakeet.cpp. |
| **Kernel / shader** | A small program that runs on the GPU, usually one operation like "multiply these matrices". |
| **Mel spectrogram** | A picture of sound: loudness of 128 frequency bands, one column every 10 ms. |
| **ONNX / ONNX Runtime Web** | A standard model file format / the engine that runs ONNX files in the browser. |
| **Phonemes** | Pronunciation symbols ("ðə" for "the"); what the TTS actually reads. |
| **Prefill (LLM)** | Reading the whole prompt in one pass; sets the time to the first token. |
| **Quantization (q4, q8, 2-bit)** | Storing weights with fewer bits to shrink downloads and memory. Not always faster. |
| **Real-time factor (× real time)** | Seconds of audio produced (or processed) per second of work. Above 1 = keeps up. |
| **Service worker** | A script the browser runs between the page and the network; here it adds COOP/COEP headers. |
| **SharedArrayBuffer** | Memory that several threads can use at once; required for multithreaded WebAssembly. |
| **Speculative transcription** | Starting speech-to-text after a short pause, and discarding it if the user keeps talking. |
| **TDT** | Token-and-Duration Transducer: at each step the decoder picks a token *and* how many frames to skip. |
| **Ternary weights** | Weights restricted to −1, 0, +1 (times a scale); ~1.58 bits of information, stored in 2. |
| **Token** | A word-piece; roughly 1.3 tokens per English word. |
| **TTS** | Text-to-speech. |
| **VAD** | Voice activity detection: "is this slice of audio speech?" |
| **Voice-to-voice latency** | Time from the end of the user's speech to the agent's first audible sound. |
| **WebAssembly (WASM)** | A fast, compact code format browsers run on the CPU. |
| **WebGPU** | The browser API for running computations on the graphics card. |
