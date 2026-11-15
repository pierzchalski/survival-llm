"""A scripted HTTP server on 127.0.0.1 for the backend tests: it records every request and answers each with the next
reply queued for its path. No model and no network: replies are written out by the test.

Not a test itself (run_all.py runs only *_test.py)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Reply:
    """status, headers, and the body as a list of byte strings written one at a time. pause: seconds to wait before
    each part after the first; wait_for: an Event to wait on after the first part (so a test can act mid-stream)."""

    def __init__(self, parts, status=200, content_type="application/json", pause=0.0, wait_for=None):
        self.parts = [p if isinstance(p, bytes) else p.encode() for p in parts]
        self.status, self.content_type, self.pause, self.wait_for = status, content_type, pause, wait_for


def json_reply(obj, status=200):
    return Reply([json.dumps(obj)], status=status)


def ndjson_reply(chunks, **kw):
    return Reply([json.dumps(c) + "\n" for c in chunks], content_type="application/x-ndjson", **kw)


def sse_reply(events, **kw):
    """events: dicts (sent as "data: {json}"), ("name", dict) pairs (sent with an "event:" line first), or raw
    strings sent as they are (comments such as ": keepalive\\n\\n", or "data: [DONE]\\n\\n")."""
    parts = []
    for ev in events:
        if isinstance(ev, str):
            parts.append(ev)
        elif isinstance(ev, tuple):
            parts.append(f"event: {ev[0]}\ndata: {json.dumps(ev[1])}\n\n")
        else:
            parts.append(f"data: {json.dumps(ev)}\n\n")
    return Reply(parts, content_type="text/event-stream", **kw)


class Stub:
    def __init__(self):
        self.replies = {}  # (method, path) -> [Reply, ...]
        self.requests = []  # {"method", "path", "headers", "body"} in arrival order
        self.broken = []  # paths whose client went away before the reply was written out
        self.always = {}  # (method, path) -> Reply sent whenever that path's queue is empty (a status poll, say)
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _serve(self, method):
                size = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(size) if size else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw
                stub.requests.append({"method": method, "path": self.path, "headers": self.headers,  # any case
                                      "body": body})
                queue = stub.replies.get((method, self.path)) or []
                reply = queue.pop(0) if queue else stub.always.get((method, self.path)) or \
                    Reply(['{"error": "no reply scripted"}'], status=599)
                try:
                    self.send_response(reply.status)
                    self.send_header("Content-Type", reply.content_type)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for n, part in enumerate(reply.parts):
                        if n == 1 and reply.wait_for is not None:
                            reply.wait_for.wait(5)
                        if n and reply.pause:
                            time.sleep(reply.pause)
                        self.wfile.write(part)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    stub.broken.append(self.path)

            def do_GET(self):
                self._serve("GET")

            def do_POST(self):
                self._serve("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def queue(self, method, path, *replies):
        self.replies.setdefault((method, path), []).extend(replies)

    def bodies(self, path):
        return [r["body"] for r in self.requests if r["path"] == path]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def checker():
    """(check, fails): check(name, ok, detail) prints PASS/FAIL the way the other tests do."""
    fails = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {detail}"))
        if not ok:
            fails.append(name)
    return check, fails
