// Minimal static server for web/site (no dependencies):  node web/serve.mjs [port]   or   cd web && npm start
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("site/", import.meta.url)));
const PORT = Number(process.argv[2] || process.env.PORT || 8000);
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".json": "application/json", ".webp": "image/webp", ".png": "image/png", ".svg": "image/svg+xml" };

createServer(async (req, res) => {
  try {
    let path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^([/\\])+/, "");
    let file = join(ROOT, path);
    if (!file.startsWith(ROOT)) { res.writeHead(403).end(); return; }
    if ((await stat(file).catch(() => null))?.isDirectory()) file = join(file, "index.html");
    const body = await readFile(file);
    const ext = extname(file);
    // forecast data changes every cycle: no caching of JSON
    res.writeHead(200, { "Content-Type": TYPES[ext] || "application/octet-stream", "Cache-Control": ext === ".json" || ext === ".html" ? "no-cache" : "max-age=300" });
    res.end(body);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain" }).end("Not found");
  }
}).listen(PORT, () => console.log(`Windcast: http://localhost:${PORT}  (serving ${ROOT})`));
