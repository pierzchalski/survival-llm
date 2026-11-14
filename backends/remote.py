"""What the OpenAI- and Anthropic-shape backends share: a configured server, an optional fixed model, a ceiling on
reply length, and capabilities taken from the configuration (these APIs don't say what a model can do)."""
from ._http import BackendError, Client
from .base import Backend, Features


def model_names(answer):
    """Model names from a /models answer: most servers list {"id"} under "data", some {"name"} under "models", a
    few a plain list of names."""
    items = (answer.get("data") or answer.get("models") or []) if isinstance(answer, dict) else answer
    names = []
    for item in items if isinstance(items, list) else []:
        name = (item.get("id") or item.get("name")) if isinstance(item, dict) else item
        if isinstance(name, str) and name:
            names.append(name)
    return names


class RemoteBackend(Backend):
    features = Features(list_models=True)
    THINKING = ("none",)  # the ways of asking for reasoning this wire format knows

    def __init__(self, base_url, api_key="", model=None, thinking="none", max_tokens=None, caps=None):
        """base_url: up to and including /v1, e.g. "http://127.0.0.1:8000/v1". caps: the capabilities to report
        for every model (default: tools, plus thinking unless thinking is "none")."""
        if thinking not in self.THINKING:
            raise ValueError(f"thinking must be one of {', '.join(self.THINKING)}")
        self.base_url, self.fixed_model, self.thinking, self.max_tokens = base_url, model or None, thinking, max_tokens
        self.client = Client(base_url, {**self.HEADERS, **(self.auth_headers(api_key) if api_key else {})}, api_key)
        self._caps = set(caps) if caps is not None else {"tools"} | ({"thinking"} if thinking != "none" else set())
        self.last_error = None

    HEADERS = {}  # sent with every request

    def auth_headers(self, api_key):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.base_url!r}, model={self.fixed_model!r}, thinking={self.thinking!r})"

    def models(self):
        """The server's models, or just the configured one if there is one. None when the server can't be reached,
        or (with no configured model) refused to list them; last_error then says why."""
        self.last_error = None
        try:
            names = model_names(self.client.request("/models", timeout=5))
        except BackendError as exc:
            names, self.last_error = None, exc  # it answered, with an error: a wrong key, or no /models
        except (OSError, ValueError) as exc:
            self.last_error = exc
            return None
        return [self.fixed_model] if self.fixed_model else names

    def capabilities(self, model):
        return set(self._caps)

    def max_reply(self, options, default=None):
        """max_tokens for a request: the app's length limit, held to the configured ceiling; with no limit, the
        ceiling, else default."""
        limit, cap = options.get("num_predict"), self.max_tokens
        limit = limit if isinstance(limit, int) and limit > 0 else None
        return min(n for n in (limit, cap) if n) if limit or cap else default
