"""在本机提供只读的 CUDA kernel 迭代监控。"""

import argparse
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .dashboard_local import campaign_view, file_for

LOCAL_PAGE = Path(__file__).with_name("dashboard_local.html")


def serve(port, campaigns):
    local_page = LOCAL_PAGE.read_bytes()
    roots = tuple(Path(p).resolve() for p in campaigns)

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.respond(200, "text/html; charset=utf-8", local_page)
            elif path == "/api/local":
                try:
                    data = dict(now=time.time(), campaigns=[campaign_view(r) for r in roots])
                    self.respond(200, "application/json; charset=utf-8", json.dumps(
                        data, ensure_ascii=False, allow_nan=False, default=str).encode())
                except (OSError, ValueError) as exc:
                    print("读取本地评测状态失败", type(exc).__name__, file=sys.stderr, flush=True)
                    self.respond(503, "application/json", b'{"error":"campaign_unavailable"}')
            elif path.startswith("/api/campaign/"):
                parts = path.split("/")
                if len(parts) != 6 or not parts[3].isascii() or not parts[3].isdigit() or len(parts[3]) > 8:
                    self.respond(404, "text/plain", b"Not found")
                    return
                i, kind, key = int(parts[3]), parts[4], unquote(parts[5])
                try:
                    target = file_for(roots[i], kind, key) if i < len(roots) else None
                    if target is None:
                        self.respond(404, "text/plain", b"Not found")
                        return
                    data = target.read_bytes()
                    if kind == "source" and hashlib.sha256(data).hexdigest() != key:
                        self.respond(409, "text/plain", b"Source digest mismatch")
                        return
                except (OSError, ValueError):
                    self.respond(503, "text/plain", b"Artifact unavailable")
                    return
                mime = "application/json" if kind == "report" else "text/plain"
                self.respond(200, mime + "; charset=utf-8", data)
            else:
                self.respond(404, "text/plain; charset=utf-8", b"Not found")

        def respond(self, status, content_type, data):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self' 'unsafe-inline'; "
                             "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; "
                             "font-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--campaign", type=Path, action="append", required=True, help="评测工作目录，可重复")
    args = parser.parse_args()
    with serve(args.port, args.campaign) as server:
        print(f"Kernel Evolution Workbench http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
