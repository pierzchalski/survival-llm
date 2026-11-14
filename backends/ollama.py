"""Ollama's native API (/api/chat, /api/show, /api/tags): the app's default backend.

The request bodies and error handling are the ones core.py used before backends existed, moved here unchanged.
"""
import json
import urllib.request

from .base import Backend, Features


class OllamaBackend(Backend):
    kind = "ollama"
    label = "Ollama"
    features = Features(list_models=True, manage_models=True, loaded_state=True, warm=True, engine_lifecycle=True,
                        version=True)

    def __init__(self, base_url):
        self.base_url = base_url  # "http://127.0.0.1:11434", no trailing slash
        self._shown = {}  # model -> Ollama's /api/show answer

    def _post(self, path, body, stream=False):
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=900)
        return resp if stream else json.load(resp)

    def show(self, model):
        if model not in self._shown:
            try:
                self._shown[model] = self._post("/api/show", {"model": model})
            except (OSError, ValueError):
                return {}  # not cached: Ollama may just not be running yet
        return self._shown[model]

    def capabilities(self, model):
        return set(self.show(model).get("capabilities") or [])

    def family_hint(self, model):
        details = self.show(model).get("details") or {}
        return " ".join([details.get("family") or ""] + (details.get("families") or []))

    def models(self):
        try:
            with urllib.request.urlopen(self.base_url + "/api/tags", timeout=5) as resp:
                return [m["name"] for m in json.load(resp).get("models", [])]
        except OSError:
            return None  # Ollama is not running

    def chat_stream(self, model, messages, tools, think, options):
        body = {"model": model, "messages": messages, "stream": True, "keep_alive": "15m", "options": options}
        if tools:
            body["tools"] = tools
        if think is not None:
            body["think"] = think
        with self._post("/api/chat", body, stream=True) as resp:
            for line in resp:
                yield json.loads(line)
