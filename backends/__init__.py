"""Model backends: the servers the app can talk to, each behind the same small interface (base.Backend).
config.from_config() picks one from the person's settings; Ollama is the default."""
from ._http import BackendError
from .base import Backend, Features
from .ollama import OllamaBackend
from .openai_chat import OpenAIBackend
from .anthropic_messages import AnthropicBackend
from .config import ConfigError, from_config

__all__ = ["AnthropicBackend", "Backend", "BackendError", "ConfigError", "Features", "OllamaBackend", "OpenAIBackend",
           "from_config"]
