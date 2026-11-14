"""What every model backend offers core.py.

The chat loop in core.py reads the model's answer as Ollama's /api/chat chunks: {"message": {"content", "thinking",
"tool_calls"}} while it streams, then a last chunk with "done": True, "done_reason", "eval_count" and "eval_duration"
(nanoseconds). Each backend turns its own wire format into those chunks, so the chat loop and its tests stay the same
whichever server answers.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Features:
    """What the app may do with this backend beyond chatting."""
    list_models: bool = False  # it can say which models it serves
    manage_models: bool = False  # it can download and delete models (Ollama's pull)
    loaded_state: bool = False  # it can say which models are loaded in memory (Ollama's /api/ps)
    warm: bool = False  # it can load a model ahead of the first question
    engine_lifecycle: bool = False  # the app may find, start and stop the engine itself
    version: bool = False  # it reports an engine version the app can check


class Backend:
    kind = ""  # "ollama", "openai" or "anthropic"
    label = ""  # what the person is shown
    base_url = ""
    fixed_model = None  # the one model this backend serves, when it was configured with one
    features = Features()

    def chat_stream(self, model, messages, tools, think, options):
        """Yields Ollama-shaped chunks (see the module docstring). Closing the generator closes the connection,
        which is how Stop ends a generation."""
        raise NotImplementedError

    def models(self):
        """The names of the models it serves, or None when it can't be reached."""
        raise NotImplementedError

    def capabilities(self, model):
        """A subset of {"tools", "thinking"}."""
        raise NotImplementedError

    def family_hint(self, model):
        """Text naming the model's family, for core.family() when the model's name doesn't say; "" if unknown."""
        return ""
