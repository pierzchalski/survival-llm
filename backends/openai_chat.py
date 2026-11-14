"""Servers that speak OpenAI's chat-completions API (POST {base}/chat/completions with "stream": true, answered with
server-sent events; GET {base}/models), turned into the Ollama-shaped chunks core.py reads.

Only fields of that API are sent: Ollama's own options (num_ctx, keep_alive, top_k) are left out, since a server may
refuse a request with a field it doesn't know. How to ask the model to reason or not has no standard field, so it is
configured per server (THINKING).
"""
import json
import time

from ._http import BackendError, Client, sse_data
from .base import Backend, Features, paired, parse_arguments

# How core's "think" setting (None, False, True, "low" or "xhigh") is sent:
#   none                  nothing is sent, and the model isn't offered the app's Reasoning setting
#   reasoning_effort      "reasoning_effort": False -> "none", "low", True -> "medium", "xhigh"
#   enable_thinking       "enable_thinking": true/false
#   chat_template_kwargs  "chat_template_kwargs": {"enable_thinking": true/false}
THINKING = ("none", "reasoning_effort", "enable_thinking", "chat_template_kwargs")
EFFORT = {False: "none", "low": "low", True: "medium", "xhigh": "xhigh"}


def to_openai(messages):
    """The stored chat (Ollama's shape) as chat-completions messages: tool calls get ids and their arguments as JSON
    text, and each tool result names the call it answers."""
    out = []
    for m, ids in paired(messages):
        role, content = m.get("role"), m.get("content") or ""
        if role == "tool":
            # A result with no call to answer can't be sent as a tool message; it still says something useful.
            out.append({"role": "tool", "tool_call_id": ids, "content": content} if ids else
                       {"role": "user", "content": content})
        elif role == "assistant" and ids:
            calls = []
            for call, call_id in zip(m["tool_calls"], ids):
                fn = call.get("function") or {}
                args = fn.get("arguments")
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": fn.get("name") or "", "arguments": args if isinstance(args, str) else json.dumps(args or {})}})
            out.append({"role": "assistant", "content": content, "tool_calls": calls})
        else:
            out.append({"role": role, "content": content})
    return out


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


class OpenAIBackend(Backend):
    kind = "openai"
    label = "OpenAI-compatible server"
    features = Features(list_models=True)

    def __init__(self, base_url, api_key="", model=None, thinking="none", max_tokens=None, caps=None):
        """base_url: up to and including /v1, e.g. "http://127.0.0.1:8000/v1". caps: the capabilities to report
        for every model (default: tools, plus thinking unless thinking is "none")."""
        if thinking not in THINKING:
            raise ValueError(f"thinking must be one of {', '.join(THINKING)}")
        self.base_url, self.fixed_model, self.thinking, self.max_tokens = base_url, model or None, thinking, max_tokens
        self.client = Client(base_url, {"Authorization": "Bearer " + api_key} if api_key else {}, api_key)
        self._caps = set(caps) if caps is not None else {"tools"} | ({"thinking"} if thinking != "none" else set())
        self.last_error = None

    def __repr__(self):
        return f"OpenAIBackend({self.base_url!r}, model={self.fixed_model!r}, thinking={self.thinking!r})"

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

    def body(self, model, messages, tools, think, options):
        body = {"model": model, "messages": to_openai(messages), "stream": True,
                "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
        limit, cap = options.get("num_predict"), self.max_tokens
        limit = limit if isinstance(limit, int) and limit > 0 else None
        if limit or cap:
            body["max_tokens"] = min(n for n in (limit, cap) if n)
        for field in ("temperature", "top_p"):
            if field in options:
                body[field] = options[field]
        if think is not None and self.thinking == "reasoning_effort":
            body["reasoning_effort"] = EFFORT.get(think, think)
        elif think is not None and self.thinking == "enable_thinking":
            body["enable_thinking"] = bool(think)
        elif think is not None and self.thinking == "chat_template_kwargs":
            body["chat_template_kwargs"] = {"enable_thinking": bool(think)}
        return body

    def chat_stream(self, model, messages, tools, think, options):
        resp = self.client.request("/chat/completions", self.body(model, messages, tools, think, options), stream=True)
        with resp:
            yield from self.read(sse_data(resp))

    def read(self, events):
        """Ollama-shaped chunks from the text of each server-sent event (None for a keepalive comment)."""
        calls, usage, finish, pieces, started, heard = {}, {}, None, 0, None, False
        for data in events:
            if data is None:
                yield {"message": {}}  # a keepalive: nothing to show, but the server can notice a closed page
                continue
            try:
                event = json.loads(data)
            except ValueError:
                raise BackendError(f"The AI server at {self.base_url} sent something that isn't JSON: {data[:80]!r}",
                                   kind="stream") from None
            if not isinstance(event, dict):
                continue
            heard = True
            if event.get("error"):
                raise self.client.error(event)
            usage = event.get("usage") or usage
            choice = next((c for c in event.get("choices") or [] if isinstance(c, dict)), {})
            delta = choice.get("delta") or {}
            msg = {}
            thought = next((delta[k] for k in ("reasoning_content", "reasoning", "thinking")
                            if isinstance(delta.get(k), str) and delta[k]), "")
            if thought:
                msg["thinking"] = thought
            if isinstance(delta.get("content"), str) and delta["content"]:
                msg["content"] = delta["content"]
            for part in delta.get("tool_calls") or []:
                if not isinstance(part, dict):
                    continue
                # Arguments may arrive whole or in fragments: fragments of one call share its index.
                slot = calls.setdefault(part.get("index", len(calls)), {"id": None, "name": "", "args": ""})
                fn = part.get("function") or {}
                slot["id"] = part.get("id") or slot["id"]
                slot["name"] = slot["name"] or fn.get("name") or ""
                args = fn.get("arguments")
                if isinstance(args, dict):
                    slot["args"] = args
                elif isinstance(args, str) and isinstance(slot["args"], str):
                    slot["args"] += args
            if msg:
                pieces += 1
                started = started or time.monotonic()
            yield {"message": msg}
            finish = choice.get("finish_reason") or finish
        if not heard:
            raise BackendError(f"The AI server at {self.base_url} ended the answer without sending anything.",
                               kind="stream")
        made = []
        for slot in calls.values():
            if slot["name"]:  # a call with no name can't be run or sent back
                call = {"function": {"name": slot["name"], "arguments": parse_arguments(slot["args"])}}
                made.append({"id": slot["id"], **call} if slot["id"] else call)
        yield {"message": {"tool_calls": made} if made else {}, "done": True,
               "done_reason": "length" if finish == "length" else "stop",
               "eval_count": usage.get("completion_tokens") or pieces,
               "eval_duration": int((time.monotonic() - started) * 1e9) if started else 0}
