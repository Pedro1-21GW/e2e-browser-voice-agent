"""Compile kernels.c to WebAssembly (four variants) with Zig's bundled clang:  pip install ziglang

  kernels-st.wasm           one thread, SIMD128                (works everywhere WebAssembly SIMD does)
  kernels-st-relaxed.wasm   one thread, + relaxed SIMD int8 dot product
  kernels-mt.wasm           threads (shared memory + atomics), SIMD128
  kernels-mt-relaxed.wasm   threads, + relaxed SIMD

Usage:  python h2-own-inference-engine/engine/build.py
"""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
common = ["-target", "wasm32-freestanding", "-O3", "-nostdlib", "-fno-math-errno",
          "-Wl,--no-entry", "-Wl,--import-memory", "-Wl,--export=__heap_base", "-Wl,--max-memory=4294901760",
          "-Wl,-z,stack-size=262144", "-msimd128", "-mbulk-memory", "-mnontrapping-fptoint", "-msign-ext"]
variants = {
    "kernels-st.wasm": [],
    "kernels-st-relaxed.wasm": ["-mrelaxed-simd", "-DRELAXED"],
    "kernels-mt.wasm": ["-matomics", "-mmutable-globals", "-DMT", "-Wl,--shared-memory", "-Wl,--export=__stack_pointer"],
    "kernels-mt-relaxed.wasm": ["-matomics", "-mmutable-globals", "-DMT", "-Wl,--shared-memory", "-Wl,--export=__stack_pointer", "-mrelaxed-simd", "-DRELAXED"],
}
for out, extra in variants.items():
    cmd = [sys.executable, "-m", "ziglang", "cc", *common, *extra, "-o", os.path.join(here, out), os.path.join(here, "kernels.c")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr); sys.exit(1)
    print(f"{out:26} {os.path.getsize(os.path.join(here, out)):7,} bytes")
