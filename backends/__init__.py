"""Model backends: the servers the app can talk to, each behind the same small interface (base.Backend)."""
from .base import Backend, Features
from .ollama import OllamaBackend

__all__ = ["Backend", "Features", "OllamaBackend"]
