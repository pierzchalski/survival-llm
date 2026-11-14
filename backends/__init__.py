"""Model backends: the servers the app can talk to, each behind the same small interface (base.Backend)."""
from ._http import BackendError
from .base import Backend, Features
from .ollama import OllamaBackend
from .openai_chat import OpenAIBackend

__all__ = ["Backend", "BackendError", "Features", "OllamaBackend", "OpenAIBackend"]
