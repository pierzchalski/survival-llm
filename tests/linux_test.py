"""Linux: the AI engine (an ollama on PATH, or one answering at OLLAMA_HOST, and plain words when there is neither),
the models it offers, and updates.
No Ollama: a stand-in answers at OLLAMA_HOST, and a stand-in ollama command is put on PATH. Nothing is downloaded.

Usage: .venv/bin/python tests/linux_test.py"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if not sys.platform.startswith("linux"):
    print("not Linux: nothing to check")
    sys.exit(0)


class FakeOllama(BaseHTTPRequestHandler):
    models = ["granite4.2:8b"]

    def do_GET(self):
        body = json.dumps({"models": [{"name": m} for m in self.models]} if self.path == "/api/tags" else {}).encode()
        self.send_response(200 if self.path == "/api/tags" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


fake = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
threading.Thread(target=fake.serve_forever, daemon=True).start()

TMP = tempfile.mkdtemp(prefix="linux-test-")
os.environ.update(SURVIVAL_LLM_APP_DIR=os.path.join(TMP, "settings"), SURVIVAL_LLM_FILES=os.path.join(TMP, "files"),
                  SURVIVAL_LLM_LIBRARY=os.path.join(TMP, "library"), XDG_DATA_HOME=os.path.join(TMP, "data"),
                  OLLAMA_HOST=f"127.0.0.1:{fake.server_address[1]}")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import server  # noqa: E402

FAILS = []
PATH = os.environ["PATH"]


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {detail}"))
    if not ok:
        FAILS.append(name)


def wait_for(test, seconds=10):
    end = time.time() + seconds
    while time.time() < end:
        if test():
            return True
        time.sleep(0.05)
    return False


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


def command(folder, name, script):
    """A stand-in program on PATH."""
    path = write(os.path.join(folder, name), "#!/bin/sh\n" + script)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


bin_dir = os.path.join(TMP, "bin")  # stand-in programs

# ---------- the AI engine: an Ollama answering at OLLAMA_HOST is used, with no ollama command needed ----------
real_locations, real_start = server.ollama_locations, server.start_ollama
started = []
server.ollama_locations = lambda: (None, None)
server.start_ollama = lambda: started.append(1)
check("Ollama at OLLAMA_HOST: its models are listed", server.installed_models() == ["granite4.2:8b"],
      server.installed_models())
check("Ollama at OLLAMA_HOST: setup has nothing to do for a model it has",
      server.setup_step({"pack": None, "model": "granite4.2:8b"}) is None)
check("Ollama at OLLAMA_HOST: setup downloads another model through it",
      (server.setup_step({"pack": None, "model": "gemma4"}) or ())[:2] == ("model", "model:gemma4"))
fake.shutdown()
fake.server_close()
check("no Ollama running and no ollama command: setup asks for Ollama, and downloads no engine itself",
      server.setup_step({"pack": None, "model": "gemma4"}) == ("wait", "install-engine")
      and server.ENGINE is None and server.engine_info() is None and not started)
server.ollama_locations, server.start_ollama = real_locations, real_start

# ---------- ... and an ollama command on PATH is found and started ----------
args_file = os.path.join(TMP, "ollama-args")
ollama = command(bin_dir, "ollama", f'echo "$@" > "{args_file}"\n')
os.environ["PATH"] = bin_dir
try:
    found = server.ollama_locations()
finally:
    os.environ["PATH"] = PATH
check("an ollama on PATH is found", found == (None, ollama), found)
server.ollama_locations = lambda: (None, ollama)
server.ollama_started_at[0] = 0
server.start_ollama()
check("and started with serve", wait_for(lambda: os.path.exists(args_file) and open(args_file).read() == "serve\n"))
check("setup waits for it while it starts", server.setup_step({"pack": None, "model": "gemma4"})
      == ("wait", "engine-starting"))
server.ollama_locations = real_locations

# ---------- the models on offer: the PC builds, as on Windows ----------
server.ram_gb, server.disk_free_gb = (lambda: 32), (lambda path=None: 100)
rows = server.machine_catalog([])
check("Linux gets the PC builds", server.system_key() == "win" and all(r["model"] == next(
    c for c in server.CATALOG if c["key"] == r["key"])["win"] for r in rows), rows)
check("32 GB: Nemotron Lightning is recommended", next(r["model"] for r in rows if r["recommended"])
      == "nemotron-3.5-lightning:30b-a3b")
server.ram_gb = lambda: 256
check("Nemotron Super isn't offered, as on Windows",
      "nemotron-super" not in [r["key"] for r in server.machine_catalog([])])

# ---------- updates: there is no Linux download, so a copy run from source says git pull ----------
server.updates["data"] = {"version": "99.0", "mac": {"url": "https://x", "bytes": 1, "sha256": "a" * 64},
                          "windows": {"url": "https://x", "bytes": 1, "sha256": "a" * 64}}
check("no update is offered for Linux", server.platform_key() is None and server.newer_app() is None)
check("and why it can't update itself names git pull", server.DEV is not None
      or "git pull" in (server.update_blocker() or ""), server.update_blocker())


shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all Linux checks passed'}")
sys.exit(1 if FAILS else 0)
