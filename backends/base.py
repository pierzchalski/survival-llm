"""What every model backend offers core.py.

The chat loop in core.py reads the model's answer as Ollama's /api/chat chunks: {"message": {"content", "thinking",
"tool_calls"}} while it streams, then a last chunk with "done": True, "done_reason", "eval_count" and "eval_duration"
(nanoseconds). Each backend turns its own wire format into those chunks, so the chat loop and its tests stay the same
whichever server answers.
"""
import json
import re
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


def parse_arguments(text):
    """A tool call's arguments as a dict. Fixes slips models make in the JSON they write (a code fence around it, a
    comma before a closing bracket, words before or after it), the same tricks core.memory_plan uses. Returns the
    text unchanged when it still isn't a JSON object, so core can answer the call with BAD_ARGS."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return {}
    unfenced = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text)
    for attempt in (text, unfenced, re.sub(r",\s*([}\]])", r"\1", unfenced)):
        start = attempt.find("{")  # only the first object: one nested in a cut-off call is not the call
        try:
            found = json.JSONDecoder().raw_decode(attempt, start)[0] if start >= 0 else None
        except ValueError:
            continue
        if isinstance(found, dict):
            return found
    return text


STOPPED = "No result: the lookup was stopped before it ran."


def paired(messages):
    """Each message with the wire ids that go with it, as (message, ids): for an assistant message, a list with one id
    per tool call (the id the server gave, else a made-up one); for a tool result, the id of the call it answers,
    pairing results with calls in order as core._used_ids does (None when there is no call left to answer); else
    None. The stored chat keeps Ollama's shape, which has no ids.

    A turn stopped while its tools ran (Stop, or Ctrl+C in chat.py) leaves calls with no result, which the
    OpenAI- and Anthropic-shape APIs don't allow; each such call gets a STOPPED result here, after the ones it has."""
    pending, made = [], 0
    for m in messages:
        role = m.get("role")
        if role != "tool":
            for call_id in pending:
                yield {"role": "tool", "content": STOPPED}, call_id
            pending = []
        if role == "assistant":
            ids = []
            for call in m.get("tool_calls") or []:
                made += 1
                ids.append(str(call.get("id") or f"call_sllm_{made}"))
            pending = list(ids)
            yield m, ids
        elif role == "tool":
            yield m, pending.pop(0) if pending else None
        else:
            yield m, None
    for call_id in pending:
        yield {"role": "tool", "content": STOPPED}, call_id
