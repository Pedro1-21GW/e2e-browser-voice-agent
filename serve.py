"""Local static server for this repo, with the headers multithreaded WebAssembly needs.

SharedArrayBuffer (threads) requires the page to be "cross-origin isolated", which takes two response headers
(COOP + COEP). GitHub Pages can't set them, so the pages also ship a service worker (h1-browser-libs/sw.js) that
adds them; this server sets them directly, which is simpler for local work and benchmarks.

Usage:  python serve.py [port]      then open http://127.0.0.1:8765/
"""
import http.server, sys, os

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *args):
        pass

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"serving {os.getcwd()} on http://127.0.0.1:{port}/")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
