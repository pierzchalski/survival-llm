"""Which server answers: Ollama unless configured otherwise.

Settings come from backend.json in the app's settings folder, and environment variables override them one by one
(SURVIVAL_LLM_BACKEND naming another backend than the file's sets the whole file aside, so SURVIVAL_LLM_BACKEND=ollama
always gives Ollama):

  backend.json      environment variable            meaning
  "backend"         SURVIVAL_LLM_BACKEND            ollama (default), openai or anthropic
  "base_url"        SURVIVAL_LLM_BASE_URL           the server, e.g. http://127.0.0.1:8000 (/v1 is added)
  "api_key"         SURVIVAL_LLM_API_KEY            sent to the server only; never shown or logged
  "api_key_file"    SURVIVAL_LLM_API_KEY_FILE       a file holding the key, instead of api_key
  "model"           SURVIVAL_LLM_MODEL              the one model to use (the server may serve several)
  "thinking"        SURVIVAL_LLM_THINKING           how to ask for reasoning (openai_chat.THINKING,
                                                    anthropic_messages.THINKING)
  "max_tokens"      SURVIVAL_LLM_MAX_TOKENS         a ceiling on reply length
  "caps"            SURVIVAL_LLM_CAPS               what the model can do: "tools,thinking", "tools" or "none"
  "reasoning_floor" SURVIVAL_LLM_REASONING_FLOOR    more models that must reason a little even at Reasoning off,
                                                    by part of their name: "mistral,glm" (Qwen always does)
  "allow_no_reasoning" SURVIVAL_LLM_ALLOW_NO_REASONING  1: start even when such a model can't be asked to reason

A model with a reasoning floor (core.THINK_OFF, or reasoning_floor) searches the library only if it reasons a little
first, so at Reasoning off core still asks for a brief think. That needs a "thinking" setting the server accepts; with
none (or caps without thinking), check_reasoning() refuses the model rather than run it fully off, unless
allow_no_reasoning is set.

Everything but "backend" is for openai and anthropic: Ollama's address stays OLLAMA_HOST. The settings are read once,
when core.py is imported. Only the terminal chat (chat.py) can use openai or anthropic so far: the app window
(app/server.py) refuses to start with them. A backend.json holding an api_key must not be readable by other users
(chmod 600).
"""
import json
import os
import re
import stat
import sys
from urllib.parse import urlsplit

from .anthropic_messages import AnthropicBackend
from .ollama import OllamaBackend
from .openai_chat import OpenAIBackend

FIELDS = ("backend", "base_url", "api_key", "api_key_file", "model", "thinking", "max_tokens", "caps",
          "reasoning_floor", "allow_no_reasoning")
KINDS = {"ollama": OllamaBackend, "openai": OpenAIBackend, "anthropic": AnthropicBackend}
HOST_PORT = re.compile(r"^[A-Za-z0-9._\-]+(:\d{1,5})?$|^\[[0-9a-fA-F:.]+\](:\d{1,5})?$")  # host[:port], nothing else


class ConfigError(ValueError):
    """The backend settings can't be used. The message names the setting, never the key."""


def app_dir(env=None):
    """The app's settings folder, worked out the way app/server.py does."""
    env = os.environ if env is None else env
    if env.get("SURVIVAL_LLM_APP_DIR"):
        return env["SURVIVAL_LLM_APP_DIR"]
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "Survival LLM")
    if os.name == "nt":
        return os.path.join(env.get("APPDATA", os.path.expanduser("~")), "Survival LLM")
    return os.path.join(os.path.expanduser("~"), ".survival-llm")


def normalise_base(url):
    """The address requests go under ("{base}/chat/completions", "{base}/models"), from what the person wrote. A bare
    "host:port" or "http://host:port" gets /v1, the usual place. An address with a path is used as given, less a
    trailing slash or an endpoint ("/chat/completions", "/messages", "/models"); so a server whose API sits at the
    root is given as its full endpoint, e.g. "https://example.com/chat/completions". "" when it isn't an http(s)
    address with a plain host (no user name or password in it). A query string is refused: none would be sent."""
    url = str(url or "").strip()
    if "?" in url or "#" in url:
        raise ConfigError("base_url can't have a query string (?...) or #...: the app would not send it. Give the "
                          "address without it")
    url = url.rstrip("/")
    if url and not re.match(r"(?i)^https?://", url):
        url = "http://" + url
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not HOST_PORT.fullmatch(parts.netloc):
        return ""
    path, endpoint = parts.path, False
    for tail in ("/chat/completions", "/completions", "/messages", "/models"):
        if path.endswith(tail):
            path, endpoint = path[:-len(tail)], True
            break
    return f"{parts.scheme.lower()}://{parts.netloc}{path or ('' if endpoint else '/v1')}"


def settings(env=None, path=None):
    """The settings as a dict: backend.json, then the environment over it. When the environment names a different
    backend from the file's, the file is left out altogether: its settings are for another server."""
    env = os.environ if env is None else env
    path = path or os.path.join(app_dir(env), "backend.json")
    from_env = {f: env[f"SURVIVAL_LLM_{f.upper()}"] for f in FIELDS if env.get(f"SURVIVAL_LLM_{f.upper()}")}
    found = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                found = json.load(f)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"Can't read {path}: {getattr(exc, 'strerror', None) or exc}") from None
        if not isinstance(found, dict):
            raise ConfigError(f"{path} must hold a JSON object")
        kind = lambda conf: str(conf.get("backend") or "ollama").strip().lower()  # noqa: E731
        if "backend" in from_env and kind(from_env) != kind(found):
            return from_env
        unknown = sorted(set(found) - set(FIELDS))
        if unknown:
            raise ConfigError(f"{path} has settings this version doesn't know: {', '.join(unknown)}")
        if found.get("api_key") and os.name != "nt" and os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ConfigError(f"{path} holds an API key but other users can read it. Run: chmod 600 '{path}'")
    if {"api_key", "api_key_file"} & set(from_env):  # a key from the environment replaces the file's
        found = {k: v for k, v in found.items() if k not in ("api_key", "api_key_file")}
    return {**found, **from_env}


def _number(value, name):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    if number <= 0:
        raise ConfigError(f"{name} must be a whole number above 0")
    return number


def from_config(ollama_url, env=None, path=None):
    """The backend the settings ask for. ollama_url: Ollama's address (core.HOST), used when they ask for none."""
    conf = settings(env, path)
    kind = str(conf.get("backend") or "ollama").strip().lower()
    if kind not in KINDS:
        raise ConfigError(f"backend must be one of {', '.join(KINDS)}")
    if kind == "ollama":
        extra = sorted(f for f in FIELDS if f != "backend" and conf.get(f) not in (None, ""))
        if extra:
            raise ConfigError(f"{', '.join(extra)} {'is' if len(extra) == 1 else 'are'} for the openai and anthropic "
                              "backends only; Ollama's address is set with OLLAMA_HOST")
        return OllamaBackend(ollama_url)
    base = normalise_base(conf.get("base_url"))
    if not base:
        raise ConfigError(f"The {kind} backend needs base_url (SURVIVAL_LLM_BASE_URL): an http(s) address such as "
                          "http://127.0.0.1:8000, with no user name or password in it")
    if conf.get("api_key") and conf.get("api_key_file"):
        raise ConfigError("Set api_key or api_key_file, not both")
    key = str(conf.get("api_key") or "")
    if conf.get("api_key_file"):
        try:
            with open(os.path.expanduser(str(conf["api_key_file"])), encoding="utf-8") as f:
                key = f.read().strip()
        except OSError as exc:
            raise ConfigError(f"Can't read the API key file {conf['api_key_file']}: {exc.strerror}") from None
    caps = conf.get("caps")
    if caps is not None:
        names = [str(c).strip().lower() for c in (caps.split(",") if isinstance(caps, str) else
                                                   caps if isinstance(caps, list) else [caps])]
        caps = {c for c in names if c and c != "none"}
        if caps - {"tools", "thinking"}:
            raise ConfigError("caps may name only tools and thinking (or none)")
    options = {"api_key": key, "model": str(conf.get("model") or "") or None, "caps": caps,
               "max_tokens": _number(conf["max_tokens"], "max_tokens") if conf.get("max_tokens") else None}
    if conf.get("thinking"):
        options["thinking"] = str(conf["thinking"]).strip().lower()
    try:
        backend = KINDS[kind](base, **options)
    except ValueError as exc:  # an unknown thinking setting
        raise ConfigError(str(exc)) from None
    floor = conf.get("reasoning_floor") or []
    floor = floor.split(",") if isinstance(floor, str) else floor if isinstance(floor, list) else None
    if floor is None or not all(isinstance(f, str) for f in floor):
        raise ConfigError('reasoning_floor must name models by part of their name, e.g. "mistral,glm"')
    backend.reasoning_floor = tuple(f.strip().lower() for f in floor if f.strip())
    allow = conf.get("allow_no_reasoning")
    allow = allow if isinstance(allow, bool) else str(allow or "0").strip().lower()
    if allow not in (True, False, "1", "true", "yes", "0", "false", "no"):
        raise ConfigError("allow_no_reasoning must be 1 or 0")
    backend.allow_no_reasoning = allow in (True, "1", "true", "yes")
    return backend


def check_reasoning(backend, model, floor):
    """ConfigError when model has a reasoning floor (floor: core.THINK_OFF or the reasoning_floor setting) but these
    settings give no way to ask the server for reasoning, so Reasoning off would run it with none at all, unless
    allow_no_reasoning is set. Nothing to check for Ollama, which says itself whether a model can reason, or for a
    model not given the library's tools (core then adds the excerpts itself, so there is no search to skip)."""
    caps = backend.capabilities(model) if backend.kind != "ollama" else set()
    if not floor or backend.allow_no_reasoning or "tools" not in caps:
        return
    if backend.thinking != "none" and "thinking" in caps:
        return
    presets = ", ".join(t for t in backend.THINKING if t != "none")
    if backend.thinking == "none":
        fix = (f"Set SURVIVAL_LLM_THINKING (\"thinking\" in backend.json) to the way this server takes a reasoning "
               f"setting: {presets}")
    else:
        fix = "Add thinking to SURVIVAL_LLM_CAPS (\"caps\" in backend.json)"
    raise ConfigError(f"{model} needs to reason a little before it answers, or it may answer from memory without "
                      f"searching the library, but these settings can't ask the {backend.label} for any reasoning. "
                      f"{fix}. To run it without reasoning anyway, or if this model has no reasoning "
                      "mode at all, set SURVIVAL_LLM_ALLOW_NO_REASONING=1 (\"allow_no_reasoning\": true in "
                      "backend.json).")
