"""Choosing the server: Ollama by default, otherwise from backend.json and SURVIVAL_LLM_* variables; and chat.py
answering through an OpenAI- or Anthropic-shape server. A scripted HTTP server on 127.0.0.1 stands in for the AI
server (tests/stub_server.py).

Usage: .venv/bin/python tests/backends_config_test.py"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from backends import AnthropicBackend, ConfigError, OllamaBackend, OpenAIBackend  # noqa: E402
from backends.config import from_config, normalise_base  # noqa: E402
from stub_server import Stub, checker, json_reply, sse_reply  # noqa: E402

check, FAILS = checker()
KEY = "sk-config-SECRET-5555555555"
TMP = tempfile.mkdtemp(prefix="backends-config-")
OLLAMA = "http://127.0.0.1:11434"
NO_FILE = os.path.join(TMP, "none.json")


def write(name, obj, mode=0o600):
    path = os.path.join(TMP, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj))
    os.chmod(path, mode)
    return path


def refused(env=None, path=NO_FILE):
    try:
        from_config(OLLAMA, env or {}, path)
    except ConfigError as exc:
        return str(exc)
    return None


# ---- the default ----
b = from_config(OLLAMA, {}, NO_FILE)
check("no settings: Ollama at OLLAMA_HOST", isinstance(b, OllamaBackend) and b.base_url == OLLAMA, b)
check("backend.json lives in the app's settings folder", isinstance(from_config(
    OLLAMA, {"SURVIVAL_LLM_APP_DIR": TMP}), OllamaBackend))
check("SURVIVAL_LLM_BACKEND=ollama is the same", isinstance(from_config(OLLAMA, {"SURVIVAL_LLM_BACKEND": "ollama"},
                                                                        NO_FILE), OllamaBackend))
check("Ollama with remote settings is refused", "OLLAMA_HOST" in (refused({"SURVIVAL_LLM_BASE_URL": "x:1"}) or ""))

# ---- addresses ----
for given, want in (("127.0.0.1:8000", "http://127.0.0.1:8000/v1"), ("http://h:1/v1/", "http://h:1/v1"),
                    ("http://h:1/v1/chat/completions", "http://h:1/v1"), ("https://api.example.com", "https://api.example.com/v1"),
                    ("http://h:1/v1/messages", "http://h:1/v1"), ("http://h/prefix", "http://h/prefix/v1"),
                    ("http://[::1]:8000", "http://[::1]:8000/v1"), ("http://user:pw@h:1", ""), ("ftp://h", ""), ("", "")):
    check(f"normalise_base({given!r})", normalise_base(given) == want, normalise_base(given))

# ---- from the environment ----
env = {"SURVIVAL_LLM_BACKEND": "openai", "SURVIVAL_LLM_BASE_URL": "127.0.0.1:8000", "SURVIVAL_LLM_API_KEY": KEY,
       "SURVIVAL_LLM_MODEL": "qwen3-8b", "SURVIVAL_LLM_THINKING": "reasoning_effort",
       "SURVIVAL_LLM_MAX_TOKENS": "4096"}
b = from_config(OLLAMA, env, NO_FILE)
check("openai from the environment", isinstance(b, OpenAIBackend) and b.base_url == "http://127.0.0.1:8000/v1"
      and b.fixed_model == "qwen3-8b" and b.thinking == "reasoning_effort" and b.max_tokens == 4096, repr(b))
check("its repr has no key", KEY not in repr(b))
check("caps default to tools and thinking", b.capabilities("x") == {"tools", "thinking"})
for caps, want in (("none", set()), ("tools", {"tools"}), (" Tools , thinking ", {"tools", "thinking"})):
    got = from_config(OLLAMA, {**env, "SURVIVAL_LLM_CAPS": caps}, NO_FILE).capabilities("x")
    check(f"caps {caps!r}", got == want, got)
b = from_config(OLLAMA, {"SURVIVAL_LLM_BACKEND": "anthropic", "SURVIVAL_LLM_BASE_URL": "https://api.example.com"},
                NO_FILE)
check("anthropic from the environment", isinstance(b, AnthropicBackend) and b.base_url == "https://api.example.com/v1"
      and b.thinking == "anthropic", repr(b))
for bad, says in (({"SURVIVAL_LLM_BACKEND": "llamafile"}, "backend must be"),
                  ({"SURVIVAL_LLM_BACKEND": "openai"}, "needs base_url"),
                  ({**env, "SURVIVAL_LLM_BASE_URL": f"http://me:{KEY}@h:1"}, "needs base_url"),
                  ({**env, "SURVIVAL_LLM_THINKING": "anthropic"}, "thinking must be"),
                  ({**env, "SURVIVAL_LLM_CAPS": "tools,vision"}, "caps may name"),
                  ({**env, "SURVIVAL_LLM_MAX_TOKENS": "lots"}, "max_tokens must be")):
    said = refused(bad)
    check(f"refused: {says}", said is not None and says in said and KEY not in said, said)

# ---- from backend.json, with the environment over it ----
path = write("backend.json", {"backend": "anthropic", "base_url": "http://127.0.0.1:9100", "api_key": KEY,
                              "caps": ["tools"]})
b = from_config(OLLAMA, {}, path)
check("anthropic from backend.json", isinstance(b, AnthropicBackend) and b.base_url == "http://127.0.0.1:9100/v1"
      and b.capabilities("x") == {"tools"}, repr(b))
b = from_config(OLLAMA, {"SURVIVAL_LLM_BASE_URL": "http://127.0.0.1:9200"}, path)
check("an environment variable overrides one setting", isinstance(b, AnthropicBackend)
      and b.base_url == "http://127.0.0.1:9200/v1", repr(b))
os.chmod(path, 0o644)
said = refused(path=path)
check("a key file others can read is refused", said is not None and "chmod 600" in said and KEY not in said, said)
check("a file without a key may be readable", isinstance(from_config(OLLAMA, {}, write(
    "open.json", {"backend": "openai", "base_url": "h:1"}, 0o644)), OpenAIBackend))
check("an unknown setting is refused", "doesn't know: base" in (refused(path=write("typo.json", {"base": "h:1"})) or ""))
check("broken JSON is refused", "Can't read" in (refused(path=write("broken.json", "{nope")) or ""))
key_file = write("key.txt", KEY + "\n")
b = from_config(OLLAMA, {"SURVIVAL_LLM_BACKEND": "openai", "SURVIVAL_LLM_BASE_URL": "h:1",
                         "SURVIVAL_LLM_API_KEY_FILE": key_file}, NO_FILE)
check("the key can come from a file", b.client.scrub(KEY) == "[key]", repr(b))
check("but not from both", "not both" in (refused(path=write("both.json", {
    "backend": "openai", "base_url": "h:1", "api_key": "a", "api_key_file": key_file})) or ""))
b = from_config(OLLAMA, {"SURVIVAL_LLM_API_KEY_FILE": key_file}, write("both2.json", {
    "backend": "openai", "base_url": "h:1", "api_key": "from-file"}))
check("a key in the environment replaces the file's", b.client.scrub(KEY) == "[key]"
      and b.client.scrub("from-file") == "from-file")
said = refused({"SURVIVAL_LLM_BACKEND": "openai", "SURVIVAL_LLM_BASE_URL": "h:1",
                "SURVIVAL_LLM_API_KEY_FILE": os.path.join(TMP, "missing.txt")})
check("a missing key file is refused", said is not None and "Can't read the API key file" in said, said)


# ---- core and chat.py ----
def run(args, extra, stdin=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SURVIVAL_LLM_")}
    env.update(SURVIVAL_LLM_APP_DIR=os.path.join(TMP, "app"), SURVIVAL_LLM_LIBRARY=os.path.join(TMP, "library"),
               OLLAMA_HOST="127.0.0.1:9", **extra)
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
                          input=stdin)


probe = ["-c", "import core; print(type(core.BACKEND).__name__, core.BACKEND.base_url)"]
out = run(probe, {})
check("core uses Ollama by default", out.stdout.strip() == "OllamaBackend http://127.0.0.1:9", out.stdout + out.stderr)
out = run(probe, {"SURVIVAL_LLM_BACKEND": "openai", "SURVIVAL_LLM_BASE_URL": "127.0.0.1:8000"})
check("core uses the configured backend", out.stdout.strip() == "OpenAIBackend http://127.0.0.1:8000/v1",
      out.stdout + out.stderr)
out = run(probe, {"SURVIVAL_LLM_BACKEND": "nonsense"})
check("bad settings stop core loading", out.returncode != 0 and "ConfigError" in out.stderr, out.stderr[-300:])
out = run(["chat.py", "hi"], {"SURVIVAL_LLM_BACKEND": "nonsense"})
check("chat.py says what's wrong with the settings", out.returncode != 0 and "settings can't be used: backend must be"
      in out.stderr and "Traceback" not in out.stderr, out.stderr[-300:])

stub = Stub()
stub.queue("POST", "/v1/chat/completions", sse_reply([
    {"choices": [{"index": 0, "delta": {"content": "Outdoors, "}, "finish_reason": None}]},
    {"choices": [{"index": 0, "delta": {"content": "20 feet away."}, "finish_reason": "stop"}]},
    {"choices": [], "usage": {"completion_tokens": 5}}, "data: [DONE]\n\n"]))
out = run(["chat.py", "Where can I run a generator?"], {
    "SURVIVAL_LLM_BACKEND": "openai", "SURVIVAL_LLM_BASE_URL": stub.url, "SURVIVAL_LLM_API_KEY": KEY,
    "SURVIVAL_LLM_MODEL": "qwen3-8b", "SURVIVAL_LLM_CAPS": "none"})
req = next((r for r in stub.requests if r["path"] == "/v1/chat/completions"), None)
check("chat.py answers through an OpenAI-shape server", out.returncode == 0
      and "Outdoors, 20 feet away." in out.stdout and "[5 tokens" in out.stdout, out.stdout + out.stderr)
check("with the configured model and key", req is not None and req["body"]["model"] == "qwen3-8b"
      and req["headers"].get("Authorization") == "Bearer " + KEY, req and req["body"])
check("the key isn't printed", KEY not in out.stdout + out.stderr)
stub.queue("GET", "/v1/models", json_reply({"data": []}))  # no models: chat.py has none to pick
out = run(["chat.py", "hi"], {"SURVIVAL_LLM_BACKEND": "anthropic", "SURVIVAL_LLM_BASE_URL": stub.url})
check("with no model it says where it looked", out.returncode != 0 and f"No AI model found at {stub.url}/v1"
      in out.stderr, out.stdout + out.stderr)
stub.queue("POST", "/v1/messages", sse_reply([("error", {"type": "error", "error": {
    "type": "authentication_error", "message": f"bad key {KEY}"}})]))
out = run(["chat.py"], {"SURVIVAL_LLM_BACKEND": "anthropic", "SURVIVAL_LLM_BASE_URL": stub.url,
                        "SURVIVAL_LLM_API_KEY": KEY, "SURVIVAL_LLM_MODEL": "claude-x"}, stdin="hello\n/bye\n")
check("in a chat, a server error is shown without the key", "bad key [key]" in out.stdout and KEY not in out.stdout
      + out.stderr and out.returncode == 0, out.stdout[-400:] + out.stderr[-400:])
check("and the banner names the server", f"Anthropic-compatible server at {stub.url}/v1" in out.stdout,
      out.stdout[:300])
stub.close()

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
