// Adds the cross-origin isolation headers GitHub Pages can't set, so the page
// gets SharedArrayBuffer and the CPU engine can use multithreaded WebAssembly. (Same file as h1-browser-libs/sw.js.)
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (new URL(req.url).origin !== self.location.origin) return;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  e.respondWith(
    fetch(req).then((res) => {
      if (res.status === 0) return res;
      const headers = new Headers(res.headers);
      headers.set("Cross-Origin-Embedder-Policy", "credentialless");
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    })
  );
});
