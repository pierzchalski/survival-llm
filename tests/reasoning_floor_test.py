"""The reasoning floor on remote servers: a model that must reason a little before it searches (Qwen, or one named in
reasoning_floor) is refused when the settings give no way to ask the server for reasoning, unless
SURVIVAL_LLM_ALLOW_NO_REASONING is set; with a way to ask, Reasoning off sends a brief think. Ollama is unchanged.
A scripted HTTP server on 127.0.0.1 stands in for the AI server (tests/stub_server.py).

Usage: .venv/bin/python tests/reasoning_floor_test.py"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import backends  # noqa: E402
import core  # noqa: E402
from backends import ConfigError, OllamaBackend  # noqa: E402
from backends.config import from_config  # noqa: E402
from stub_server import Stub, checker, json_reply, sse_reply  # noqa: E402

check, FAILS = checker()
TMP = tempfile.mkdtemp(prefix="reasoning-floor-")
NO_FILE = os.path.join(TMP, "none.json")
OPENAI = {"SURVIVAL_LLM_BACKEND": "openai", "SURVIVAL_LLM_BASE_URL": "http://127.0.0.1:9"}
ANTHROPIC = {"SURVIVAL_LLM_BACKEND": "anthropic", "SURVIVAL_LLM_BASE_URL": "http://127.0.0.1:9"}


def refusal(env, model, path=NO_FILE):
    """core.capabilities' ConfigError for this model with these settings, or None."""
    core.BACKEND = from_config("http://127.0.0.1:9", env, path)
    try:
        core.capabilities(model)
    except ConfigError as exc:
        return str(exc)
    return None


def off(env, model):
    """What core asks for at Reasoning off with these settings."""
    core.BACKEND = from_config("http://127.0.0.1:9", env, NO_FILE)
    return core.think_setting(model, core.BACKEND.capabilities(model), "off")


# ---- Qwen with no way to ask for reasoning: refused, with the setting to add ----
said = refusal(OPENAI, "qwen3.6-colibri")
check("openai, qwen, no thinking setting: refused", said is not None, said)
check("the message names the setting and the presets", said and "SURVIVAL_LLM_THINKING" in said
      and "reasoning_effort, enable_thinking, chat_template_kwargs" in said, said)
check("and the override", said and "SURVIVAL_LLM_ALLOW_NO_REASONING=1" in said, said)
check("and the model", said and said.startswith("qwen3.6-colibri "), said)
said = refusal({**OPENAI, "SURVIVAL_LLM_THINKING": "none"}, "Qwen3.8-27B")
check("thinking none, name in capitals: refused", said is not None and "SURVIVAL_LLM_THINKING" in said, said)
said = refusal({**OPENAI, "SURVIVAL_LLM_THINKING": "reasoning_effort", "SURVIVAL_LLM_CAPS": "tools"}, "qwen3.6")
check("a thinking setting but caps without thinking: refused, naming caps", said is not None
      and "SURVIVAL_LLM_CAPS" in said and "SURVIVAL_LLM_THINKING (" not in said, said)
said = refusal({**ANTHROPIC, "SURVIVAL_LLM_THINKING": "none"}, "qwen3.8-27b-colibri")
check("anthropic with thinking none: refused, its preset named", said is not None and "setting: anthropic." in said,
      said)

# ---- not refused ----
for name, env, model in [
    ("a thinking setting", {**OPENAI, "SURVIVAL_LLM_THINKING": "reasoning_effort"}, "qwen3.6-colibri"),
    ("enable_thinking", {**OPENAI, "SURVIVAL_LLM_THINKING": "enable_thinking"}, "qwen3.6-colibri"),
    ("anthropic's default thinking", ANTHROPIC, "qwen3.8-27b-colibri"),
    ("the override", {**OPENAI, "SURVIVAL_LLM_ALLOW_NO_REASONING": "1"}, "qwen3.6-colibri"),
    ("the override, true", {**OPENAI, "SURVIVAL_LLM_ALLOW_NO_REASONING": "true"}, "qwen3.6-colibri"),
    ("a model with no floor", OPENAI, "granite4.2:8b"),
    ("no tools offered (core adds excerpts itself)", {**OPENAI, "SURVIVAL_LLM_CAPS": "none"}, "qwen3-8b"),
]:
    said = refusal(env, model)
    check(f"not refused: {name}", said is None, said)
said = refusal({**OPENAI, "SURVIVAL_LLM_ALLOW_NO_REASONING": "0"}, "qwen3.6-colibri")
check("override 0 is no override", said is not None, said)

# ---- what Reasoning off sends ----
check("qwen, thinking setting: off is a brief think", off({**OPENAI, "SURVIVAL_LLM_THINKING": "reasoning_effort"},
                                                          "qwen3.6-colibri") == "low")
check("qwen with the override: nothing asked", off({**OPENAI, "SURVIVAL_LLM_ALLOW_NO_REASONING": "1"},
                                                   "qwen3.6-colibri") is None)
check("no floor: off is off", off({**OPENAI, "SURVIVAL_LLM_THINKING": "reasoning_effort"}, "granite4.2:8b") is False)

# ---- a floor declared for another model ----
FLOOR = {**OPENAI, "SURVIVAL_LLM_REASONING_FLOOR": "Mistral, glm"}
said = refusal(FLOOR, "mistral-small-3.2")
check("declared floor: refused without a thinking setting", said is not None and "SURVIVAL_LLM_THINKING" in said, said)
check("declared floor, second name", refusal(FLOOR, "GLM-5.2-colibri") is not None)
check("declared floor: other models untouched", refusal(FLOOR, "granite4.2:8b") is None)
check("declared floor: off is a brief think",
      off({**FLOOR, "SURVIVAL_LLM_THINKING": "reasoning_effort"}, "mistral-small-3.2") == "low")
check("declared floor: other models still fully off",
      off({**FLOOR, "SURVIVAL_LLM_THINKING": "reasoning_effort"}, "granite4.2:8b") is False)
path = os.path.join(TMP, "backend.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump({"backend": "openai", "base_url": "http://127.0.0.1:9", "reasoning_floor": ["mistral"]}, f)
check("backend.json: reasoning_floor as a list", refusal({}, "mistral-small", path) is not None)
with open(path, "w", encoding="utf-8") as f:
    json.dump({"backend": "openai", "base_url": "http://127.0.0.1:9", "allow_no_reasoning": True}, f)
check("backend.json: allow_no_reasoning true", refusal({}, "qwen3.6", path) is None)

# ---- bad settings ----
for name, env, want in [
    ("allow_no_reasoning=maybe", {**OPENAI, "SURVIVAL_LLM_ALLOW_NO_REASONING": "maybe"}, "allow_no_reasoning must be"),
    ("ollama with reasoning_floor", {"SURVIVAL_LLM_REASONING_FLOOR": "mistral"}, "for the openai and anthropic"),
    ("ollama with allow_no_reasoning", {"SURVIVAL_LLM_ALLOW_NO_REASONING": "1"}, "for the openai and anthropic"),
]:
    try:
        from_config("http://127.0.0.1:9", env, NO_FILE)
        said = None
    except ConfigError as exc:
        said = str(exc)
    check(f"refused: {name}", said is not None and want in said, said)

# ---- Ollama: no check, and the same think values as before ----
calls = []
real_check = backends.check_reasoning
backends.check_reasoning = lambda *a: calls.append(a)
ollama = OllamaBackend("http://127.0.0.1:9")
ollama.capabilities = lambda model: {"tools", "thinking"}  # no Ollama running
core.BACKEND = ollama
check("ollama: capabilities as before", core.capabilities("qwen3.6:35b") == {"tools", "thinking"})
check("ollama: the floor check isn't run", calls == [], calls)
backends.check_reasoning = real_check
for model, want in [("qwen3.6:35b", "low"), ("granite4.2:8b", False), ("nemotron-3.5", False)]:
    got = core.think_setting(model, {"tools", "thinking"}, "off")
    check(f"ollama: off for {model} is {want!r}", got == want, got)


# ---- chat.py ----
def run(args, extra, stdin=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SURVIVAL_LLM_")}
    env.update(SURVIVAL_LLM_APP_DIR=os.path.join(TMP, "app"), SURVIVAL_LLM_LIBRARY=os.path.join(TMP, "library"),
               OLLAMA_HOST="127.0.0.1:9", **extra)
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
                          input=stdin)


ANSWER = sse_reply([{"choices": [{"index": 0, "delta": {"content": "Boil it."}, "finish_reason": "stop"}]},
                    "data: [DONE]\n\n"])
stub = Stub()
remote = {"SURVIVAL_LLM_BACKEND": "openai", "SURVIVAL_LLM_BASE_URL": stub.url}
out = run(["chat.py", "How do I make creek water safe?"], {**remote, "SURVIVAL_LLM_MODEL": "qwen3.6-colibri"})
check("chat.py refuses to start, with the message", out.returncode != 0 and "settings can't be used: qwen3.6-colibri"
      in out.stderr and "SURVIVAL_LLM_THINKING" in out.stderr and "Traceback" not in out.stderr, out.stderr[-500:])
check("and asks the model nothing", not stub.bodies("/v1/chat/completions"), stub.bodies("/v1/chat/completions"))
stub.queue("GET", "/v1/models", json_reply({"data": [{"id": "qwen3.8-27b-colibri"}]}))
out = run(["chat.py", "hi"], remote)
check("chat.py refuses a model picked from the server's list", out.returncode != 0
      and "qwen3.8-27b-colibri needs to reason" in out.stderr, out.stderr[-500:])
stub.queue("POST", "/v1/chat/completions", ANSWER)
out = run(["chat.py", "How do I make creek water safe?"], {**remote, "SURVIVAL_LLM_MODEL": "qwen3.6-colibri",
                                                           "SURVIVAL_LLM_ALLOW_NO_REASONING": "1"})
sent = stub.bodies("/v1/chat/completions")
check("with the override chat.py answers", out.returncode == 0 and "Boil it." in out.stdout, out.stdout + out.stderr)
check("and sends no reasoning field", sent and not {"reasoning_effort", "enable_thinking", "chat_template_kwargs"}
      & set(sent[-1]), sent and sorted(sent[-1]))
stub.queue("POST", "/v1/chat/completions", ANSWER)
out = run(["chat.py", "How do I make creek water safe?"], {**remote, "SURVIVAL_LLM_MODEL": "mistral-small",
                                                           "SURVIVAL_LLM_REASONING_FLOOR": "mistral",
                                                           "SURVIVAL_LLM_THINKING": "reasoning_effort"})
sent = stub.bodies("/v1/chat/completions")
check("a declared floor with a thinking setting: Reasoning off asks for low", out.returncode == 0 and sent
      and sent[-1].get("reasoning_effort") == "low", (out.stderr[-300:], sent and sent[-1].get("reasoning_effort")))
stub.close()

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
