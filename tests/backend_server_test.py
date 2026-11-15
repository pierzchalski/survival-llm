"""The app window (app/server.py) with a server other than Ollama: what the status says, what setup does, which
requests are refused, and that no path finds, starts, installs, warms or downloads through Ollama. The remote server
is tests/stub_server.py; Ollama is a second stub that records anything sent to it, plus an `ollama` command on PATH
that leaves a mark if anything runs it. The Anthropic shape and plain Ollama run in child processes, since the
backend is chosen when core is imported.

Usage: .venv/bin/python tests/backend_server_test.py"""
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from stub_server import Stub, json_reply, sse_reply  # noqa: E402

KEY = "sk-gui-SECRET-5555555555"
CHILD = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("--child=") else None
TMP = tempfile.mkdtemp(prefix="backend-server-")
MARK = os.path.join(TMP, "ollama-ran")
os.makedirs(os.path.join(TMP, "bin"))
with open(os.path.join(TMP, "bin", "ollama"), "w") as f:  # what ollama_locations would find, if it looked
    f.write(f"#!/bin/sh\necho \"$@\" >> '{MARK}'\n")
os.chmod(os.path.join(TMP, "bin", "ollama"), 0o755)

OLLAMA = Stub()  # stands in for Ollama: nothing should reach it while another server is chosen
REMOTE = Stub()
for k in [k for k in os.environ if k.startswith("SURVIVAL_LLM_")]:
    del os.environ[k]
os.environ.update(SURVIVAL_LLM_APP_DIR=os.path.join(TMP, "settings"), SURVIVAL_LLM_FILES=os.path.join(TMP, "files"),
                  SURVIVAL_LLM_LIBRARY=os.path.join(TMP, "library"), OLLAMA_HOST=OLLAMA.url.removeprefix("http://"),
                  PATH=os.path.join(TMP, "bin") + os.pathsep + os.environ.get("PATH", ""))
kind = CHILD.split("=", 1)[1] if CHILD else "openai"
if kind != "ollama":
    os.environ.update(SURVIVAL_LLM_BACKEND=kind, SURVIVAL_LLM_BASE_URL=REMOTE.url, SURVIVAL_LLM_API_KEY=KEY)
REMOTE.always[("GET", "/v1/models")] = json_reply({"data": [{"id": "qwen3.6-colibri"}, {"id": "my model@2"}, {"id": "qwen3.8:27b"}]})
OLLAMA.always[("GET", "/api/tags")] = json_reply({"models": [{"name": "qwen3.5:4b"}]})

started = []  # every process the app starts
_Popen = subprocess.Popen


class Recorder(_Popen):
    def __init__(self, args, *a, **kw):
        if any(w in str(args) for w in ("ollama", "serve", "open")):  # not this test's own children
            started.append(args)
        super().__init__(args, *a, **kw)


subprocess.Popen = Recorder
sys.path.insert(0, os.path.join(ROOT, "app"))
import server  # noqa: E402

jobs = []  # every background job the app starts (downloads, the engine install)
_start_job = server.start_job
server.start_job = lambda key, *a: jobs.append(key) or _start_job(key, *a)
httpd = server.make_server(0)
server.PORT = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()


def call(path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.PORT, timeout=30)
    conn.request("POST" if body is not None else "GET", path, json.dumps(body) if body is not None else None,
                 {"Host": f"127.0.0.1:{server.PORT}", "X-Token": server.TOKEN, "Content-Type": "application/json"})
    resp = conn.getresponse()
    text = resp.read().decode()
    return resp.status, text


def status():
    code, text = call("/api/status")
    return json.loads(text), text


if CHILD:  # report the status for the parent to check
    print(json.dumps({"status": status()[0], "ollama_requests": [r["path"] for r in OLLAMA.requests],
                      "started": [str(a) for a in started], "jobs": jobs}))
    sys.exit(0)

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {str(detail)[:600]}"))
    if not ok:
        FAILS.append(name)


def wait_for(test, seconds=10):
    end = time.time() + seconds
    while time.time() < end:
        if test():
            return True
        time.sleep(0.05)
    return False


# ---------- the status, for the OpenAI shape ----------
_engine = (server.ENGINE, server.engine_info)  # as on Windows, where the app can download Ollama itself
server.ENGINE, server.engine_info = os.path.join(TMP, "engine", "ollama.exe"), lambda: {"bytes": 2e9}
s, text = status()
server.ENGINE, server.engine_info = _engine
b = s.get("backend") or {}
check("the window app starts with another server", True)
check("status names the server", b.get("kind") == "openai" and b.get("base_url") == REMOTE.url + "/v1"
      and b.get("label") == "OpenAI-compatible server", b)
check("and what it can do", b.get("features", {}).get("manage_models") is False
      and b.get("features", {}).get("list_models") is True, b)
check("the models are the server's, none marked tested", [m["model"] for m in s["models"]] ==
      ["qwen3.6-colibri", "my model@2", "qwen3.8:27b"] and not any(m["tested"] for m in s["models"]), s["models"])
check("no catalogue to download from", s["catalog"] == [], s["catalog"][:1])
check("no Ollama to install or download", s["ollama_installed"] is False and s["engine"]["downloadable"] is False
      and s["engine"]["running"] is None, (s["ollama_installed"], s["engine"]))
check("the server's models count as loaded", s["loaded"] == ["qwen3.6-colibri", "my model@2", "qwen3.8:27b"], s["loaded"])
check("a server on this computer gets no badge", b.get("answers_from") is None, b)
check("the key is not in the status", KEY not in text)
n = len([r for r in REMOTE.requests if r["path"] == "/v1/models"])
status()
check("polling the status doesn't ask the server every time",
      len([r for r in REMOTE.requests if r["path"] == "/v1/models"]) == n, len(REMOTE.requests))

# ---------- refused and allowed requests ----------
for path in ("/api/pull", "/api/engine", "/api/start-ollama"):
    code, text = call(path, {"model": "qwen3.6-colibri"})
    check(f"{path} is refused", code == 409 and "doesn't download or start" in text and KEY not in text, (code, text))
code, _ = call("/api/warm", {"model": "my model@2"})
check("a server's own model name is accepted", code == 200, code)
code, _ = call("/api/warm", {"model": "bad\nname"})
check("control characters in a model name are not", code == 400, code)

# ---------- setup ----------
code, text = call("/api/setup", {"model": "nemotron-3.5-lightning:30b-a3b"})
check("setup refuses a model the server doesn't serve", code == 400, (code, text))
code, text = call("/api/setup", {"model": "qwen3.6-colibri"})
check("setup takes a model the server serves", code == 200, (code, text))
check("and finishes with nothing to download", wait_for(lambda: server.setup_plan() is None), server.setup_status())
check("no download or engine job was started", jobs == [], jobs)
check("the model step is nothing to do", server.setup_step({"model": "qwen3.6-colibri"}) is None)

n = len(REMOTE.bodies("/v1/models"))
code, _ = call("/api/setup/retry", {})
status()
check("Try again asks the server again", code == 200 and len(REMOTE.bodies("/v1/models")) == n + 1,
      (code, n, len(REMOTE.bodies("/v1/models"))))

# ---------- errors, in plain words, and a question asked through the page's own API ----------
E = server.BackendError
where = f"OpenAI-compatible server at {REMOTE.url}/v1"
cases = [
    (E(f"said: bad key {KEY[:4]}[key]", 401, "auth"), "didn't accept the API key"),
    (E("said: no such model", 404, "not_found"), "doesn't serve gpt-x. It serves qwen3.6-colibri, my model@2"),
    (E("said: context_length_exceeded", 400, "context"), "too long for the AI server. Start a new chat"),
    (E("said: overloaded", 529, "busy"), f"The {where} is busy"),
    (urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")), f"Can't reach the {where}"),
    (TimeoutError("timed out"), f"The {where} took too long"),
    (OSError(28, "No space left on device"), "The disk is full"),
    (server.ConfigError("qwen3.6-colibri needs to reason a little"), "The AI server settings can't be used: qwen3.6"),
]
for exc, want in cases:
    got = server.friendly_error(exc, "gpt-x")
    check(f"{type(exc).__name__} {getattr(exc, 'kind', '')}: {want[:40]}", want in got and KEY not in got, got)
got = server.download_error(urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")))
check("a refused library download is the internet, not the AI engine", "AI engine" not in got[0], got)


def chat(model, message="How do I purify water?"):
    code, text = call("/api/chat", {"model": model, "message": message})
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
    return code, text, events


DONE = "data: [DONE]\n\n"
REMOTE.queue("POST", "/v1/chat/completions", sse_reply([
    {"choices": [{"index": 0, "delta": {"content": "Boil it for one minute."}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, DONE]))
code, text, events = chat("my model@2")
check("a question is answered through /api/chat", code == 200 and "Boil it for one minute." in
      "".join(e.get("text", "") for e in events if e["type"] == "token") and any(e["type"] == "done" for e in events),
      text[-400:])
REMOTE.queue("POST", "/v1/chat/completions", json_reply({"error": {"message": f"Incorrect API key {KEY}",
                                                                   "type": "invalid_request_error"}}, status=401))
code, text, events = chat("my model@2", "And snow?")
errors = [e["message"] for e in events if e["type"] == "error"]
check("a refused key comes back as a readable chat error", errors and "didn't accept the API key" in errors[0]
      and KEY not in text, text[-400:])
n = len(REMOTE.bodies("/v1/chat/completions"))
code, text, events = chat("qwen3.6-colibri")
check("the reasoning floor refuses with the reason, before asking anything", code == 400
      and "settings can't be used: qwen3.6-colibri needs to reason" in json.loads(text).get("error", "")
      and len(REMOTE.bodies("/v1/chat/completions")) == n, (code, text[:300]))

# ---------- a server with no /models: say to name the model, and don't ask it on every poll ----------
listing = REMOTE.always[("GET", "/v1/models")]
REMOTE.always[("GET", "/v1/models")] = json_reply({"error": {"message": "Not Found", "type": "not_found"}}, 404)
server.models_seen[0] = 0
n = len(REMOTE.bodies("/v1/models"))
for _ in range(3):
    s, _text = status()
check("no model list: says to name the model", "doesn't list its models" in (s["backend"].get("error") or "")
      and "SURVIVAL_LLM_MODEL" in s["backend"]["error"], s["backend"].get("error"))
check("and asks the server once for three polls", len(REMOTE.bodies("/v1/models")) == n + 1,
      len(REMOTE.bodies("/v1/models")) - n)
REMOTE.always[("GET", "/v1/models")] = listing

# ---------- the server can't be reached ----------
REMOTE.close()
server.models_seen[0] = 0
s, text = status()
check("unreachable: the status says the server isn't answering", s["ollama"] is False and s["models"] == [], s["ollama"])
check("and why, in plain words", (s["backend"].get("error") or "").startswith(f"Can't reach the {where}"), s["backend"])
check("setup waits for the server rather than for Ollama", server.setup_step({"model": "qwen3.6-colibri"}) ==
      ("wait", "backend-unreachable"), server.setup_step({"model": "qwen3.6-colibri"}))
server.friendly_error(urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")))
server.download_error(urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")))
server.warm_model("qwen3.6-colibri")
check("warm-up starts nothing", not server.warming, server.warming)
try:
    server.pull_worker("model:x", "x")
    check("a download through Ollama can't be started any other way", False)
except Exception as exc:  # noqa: BLE001 - which one is the check
    check("a download through Ollama can't be started any other way", type(exc) is RuntimeError
          and not OLLAMA.bodies("/api/pull"), repr(exc))

# ---------- the Anthropic shape, and Ollama as before, in their own processes ----------
subprocess.Popen = _Popen  # from here on the processes started are this test's own
children = {}
for child in ("anthropic", "ollama"):
    out = subprocess.run([sys.executable, __file__, f"--child={child}"], capture_output=True, text=True, timeout=120,
                         env={**os.environ})
    try:
        children[child] = json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        children[child] = {}
        check(f"{child} child ran", False, out.stdout[-300:] + out.stderr[-600:])
a = (children.get("anthropic") or {}).get("status") or {}
check("Anthropic shape: status names it", (a.get("backend") or {}).get("kind") == "anthropic"
      and a.get("catalog") == [] and a.get("ollama_installed") is False, a.get("backend"))
check("Anthropic shape: nothing reached Ollama or started", not children.get("anthropic", {}).get("ollama_requests")
      and not children.get("anthropic", {}).get("started"), children.get("anthropic"))
o = (children.get("ollama") or {}).get("status") or {}
check("Ollama: status as before, with the catalogue", (o.get("backend") or {}).get("kind") == "ollama"
      and len(o.get("catalog") or []) > 0 and o.get("ollama") is True
      and (o.get("backend") or {}).get("answers_from") is None, (o.get("backend"), len(o.get("catalog") or [])))
check("Ollama: still finds the ollama command", o.get("ollama_installed") is True, o.get("ollama_installed"))

# ---------- unusable settings, opened as the app: the reason is shown in a window ----------
FAKE = os.path.join(TMP, "fakewebview")
os.makedirs(FAKE)
with open(os.path.join(FAKE, "webview.py"), "w") as f:  # records what it was asked to show
    f.write("import os\ndef create_window(title, html='', **kw):\n"
            "    open(os.environ['SHOWN'], 'w').write(html)\ndef start(*a, **kw):\n    pass\n")
shown = os.path.join(TMP, "shown.html")
env = {k: v for k, v in os.environ.items() if not k.startswith("SURVIVAL_LLM_")}
env.update(SURVIVAL_LLM_APP_DIR=os.path.join(TMP, "settings2"), SURVIVAL_LLM_BACKEND="openai",
           SURVIVAL_LLM_BASE_URL="127.0.0.1:8000?x=1", PYTHONPATH=FAKE, SHOWN=shown)
out = subprocess.run([sys.executable, os.path.join(ROOT, "app", "server.py")], env=env, capture_output=True,
                     text=True, timeout=60)
page = open(shown).read() if os.path.exists(shown) else ""
check("bad settings: the app stops and shows why in a window", out.returncode == 1
      and "settings can't be used" in page and "query string" in page, (out.returncode, page))
os.remove(shown)
out = subprocess.run([sys.executable, os.path.join(ROOT, "app", "server.py"), "--browser"], env=env,
                     capture_output=True, text=True, timeout=60)
check("but not with --browser, which asks for no window", out.returncode == 1 and not os.path.exists(shown)
      and "query string" in out.stderr, (out.returncode, out.stderr[-200:]))

# ---------- the badge: when questions leave this computer ----------
for url, want in [("http://127.0.0.1:8000/v1", None), ("http://localhost:8000/v1", None),
                  ("http://10.1.2.3:8000/v1", None), ("http://192.168.1.20:8000/v1", None),
                  ("http://[::1]:8000/v1", None), ("http://[fd00::5]:8000/v1", None),
                  ("https://api.example.com/v1", "api.example.com"), ("http://8.8.8.8/v1", "8.8.8.8"),
                  ("http://nas.lan:8000/v1", "nas.lan")]:
    check(f"badge for {url}: {want}", server.answers_from(url) == want, server.answers_from(url))

# ---------- the proof: nothing went to Ollama, ran it, or installed it ----------
check("no request reached Ollama", OLLAMA.requests == [], [r["path"] for r in OLLAMA.requests])
check("no process was started", started == [], started)
check("the ollama command was never run", not os.path.exists(MARK))
check("no download or engine job, on any path", jobs == [], jobs)

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.stdout.flush()  # os._exit skips it, and run_all reads stdout through a pipe
os._exit(1 if FAILS else 0)
