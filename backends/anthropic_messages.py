"""Servers that speak Anthropic's Messages API (POST {base}/messages with "stream": true, answered with server-sent
events; GET {base}/models), turned into the Ollama-shaped chunks core.py reads.

The wire format differs from the stored chat (Ollama's shape) in ways the conversion handles: the system prompt is a
top-level field; turns alternate user/assistant, so tool results become tool_result blocks in a user turn, merged
with any note from the app that follows them; tool calls are tool_use blocks with ids. max_tokens is required.
"""
import json
import time

from ._http import BackendError, sse_data
from .base import paired, parse_arguments
from .remote import RemoteBackend

VERSION = "2023-06-01"  # the anthropic-version header
# How core's "think" setting is sent. "anthropic": False -> {"type": "disabled"}, otherwise {"type": "enabled",
# "budget_tokens": BUDGET[think]}; "none": nothing, and the model isn't offered the app's Reasoning setting.
THINKING = ("anthropic", "none")
BUDGET = {"low": 1024, True: 4096, "xhigh": 16384}
DEFAULT_MAX_TOKENS = 16384  # max_tokens is required: used when the app sets no length limit and none is configured


def to_anthropic(messages):
    """(system, messages) for the Messages API from the stored chat."""
    system, turns = [], []

    def add(role, blocks):
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": list(blocks)})

    for m, ids in paired(messages):
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            if content:
                system.append(content)
        elif role == "tool":
            add("user", [{"type": "tool_result", "tool_use_id": ids, "content": content}] if ids else
                [{"type": "text", "text": content}] if content else [])
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            # Reasoning that led to a tool call goes back with it (see read()).
            blocks = list(calls[0].get("thinking_blocks") or []) if calls else []
            if content.strip():
                blocks.append({"type": "text", "text": content})
            for call, call_id in zip(calls, ids):
                fn = call.get("function") or {}
                args = fn.get("arguments")
                blocks.append({"type": "tool_use", "id": call_id, "name": fn.get("name") or "",
                               "input": args if isinstance(args, dict) else {}})
            add("assistant", blocks)
        else:
            add("user", [{"type": "text", "text": content}] if content else [])
    return "\n\n".join(system), turns


def to_tools(tools):
    """The app's tool list (OpenAI's function shape) as Messages API tools."""
    out = []
    for tool in tools:
        fn = tool.get("function") or {}
        out.append({"name": fn.get("name"), "description": fn.get("description") or "",
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}})
    return out


class AnthropicBackend(RemoteBackend):
    kind = "anthropic"
    label = "Anthropic-compatible server"
    THINKING = THINKING
    HEADERS = {"anthropic-version": VERSION}

    def __init__(self, base_url, api_key="", model=None, thinking="anthropic", max_tokens=None, caps=None):
        super().__init__(base_url, api_key, model, thinking, max_tokens, caps)
        self.last_tools = None  # the tool list last sent, for rounds that offer none (body())

    def auth_headers(self, api_key):
        return {"x-api-key": api_key}

    def body(self, model, messages, tools, think, options):
        system, turns = to_anthropic(messages)
        body = {"model": model, "max_tokens": self.max_reply(options, DEFAULT_MAX_TOKENS), "messages": turns,
                "stream": True}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = self.last_tools = to_tools(tools)
        elif any(b.get("type") in ("tool_use", "tool_result") for t in turns for b in t["content"]):
            # A round that offers no tools (the last one, or the app's retries) still has this question's lookups in
            # it. The reviewer recalls Anthropic's API refusing tool blocks without tools; this keeps the tools
            # declared but lets the model call none. With no list sent yet, the tools the chat used are declared.
            body["tools"] = self.last_tools or [
                {"name": name, "description": "", "input_schema": {"type": "object", "properties": {}}}
                for name in dict.fromkeys(b["name"] for t in turns for b in t["content"] if b["type"] == "tool_use")]
            body["tool_choice"] = {"type": "none"}
        thinking = think is not None and self.thinking == "anthropic"
        if thinking and think is False:
            body["thinking"] = {"type": "disabled"}
        elif thinking:
            budget = BUDGET.get(think, BUDGET[True])
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            if budget >= body["max_tokens"]:  # the budget counts toward max_tokens and must be below it
                body["max_tokens"] += budget
        # Only temperature is passed on, and not while reasoning: the design notes (unverified) that Anthropic's own
        # API refuses changed sampling with thinking on. top_p and top_k are never sent.
        if "temperature" in options and body.get("thinking", {}).get("type") != "enabled":
            body["temperature"] = options["temperature"]
        return body

    def chat_stream(self, model, messages, tools, think, options):
        resp = self.client.request("/messages", self.body(model, messages, tools, think, options), stream=True)
        with resp:
            yield from self.read(sse_data(resp))

    def read(self, events):
        """Ollama-shaped chunks from the text of each server-sent event (None for a keepalive comment)."""
        blocks, stop, out_tokens, pieces, started, heard = {}, None, 0, 0, None, False
        for data in events:
            if data is None:
                yield {"message": {}}
                continue
            try:
                event = json.loads(data)
            except ValueError:
                raise BackendError(f"The AI server at {self.base_url} sent something that isn't JSON: {data[:80]!r}",
                                   kind="stream") from None
            if not isinstance(event, dict):
                continue
            heard = True
            kind, msg = event.get("type"), {}
            if kind == "error":
                raise self.client.error(event)
            if kind == "content_block_start":
                block = dict(event.get("content_block") or {})
                blocks[event.get("index", len(blocks))] = {**block, "json": ""}
                if block.get("type") == "text" and block.get("text"):
                    msg["content"] = block["text"]
            elif kind == "content_block_delta":
                d, block = event.get("delta") or {}, blocks.setdefault(event.get("index", 0), {"json": ""})
                if d.get("type") == "text_delta" and d.get("text"):
                    msg["content"] = d["text"]
                elif d.get("type") == "thinking_delta" and d.get("thinking"):
                    msg["thinking"] = d["thinking"]
                    block["thinking"] = (block.get("thinking") or "") + d["thinking"]
                elif d.get("type") == "signature_delta":
                    block["signature"] = (block.get("signature") or "") + (d.get("signature") or "")
                elif d.get("type") == "input_json_delta":
                    block["json"] += d.get("partial_json") or ""
            elif kind == "message_delta":
                stop = (event.get("delta") or {}).get("stop_reason") or stop
                out_tokens = (event.get("usage") or {}).get("output_tokens") or out_tokens
            if msg:
                pieces += 1
                started = started or time.monotonic()
            yield {"message": msg}
        if not heard:
            raise BackendError(f"The AI server at {self.base_url} ended the answer without sending anything.",
                               kind="stream")
        made, kept = [], []
        for _, block in sorted(blocks.items()):
            if block.get("type") == "tool_use" and block.get("name"):
                args = parse_arguments(block["json"]) if block["json"] else block.get("input") or {}
                made.append({"id": block.get("id"), "function": {"name": block["name"], "arguments": args}}
                            if block.get("id") else {"function": {"name": block["name"], "arguments": args}})
            elif block.get("type") == "thinking" and block.get("signature"):
                kept.append({"type": "thinking", "thinking": block.get("thinking") or "",
                             "signature": block["signature"]})
            elif block.get("type") == "redacted_thinking":
                kept.append({"type": "redacted_thinking", "data": block.get("data")})
        if made and kept:
            # Kept on the first call so the reasoning goes back with the tool results (to_anthropic). The design
            # says Anthropic's own API requires this with thinking on; that is not verified here.
            made[0]["thinking_blocks"] = kept
        yield {"message": {"tool_calls": made} if made else {}, "done": True,
               "done_reason": "length" if stop == "max_tokens" else "stop",
               "eval_count": out_tokens or pieces,
               "eval_duration": int((time.monotonic() - started) * 1e9) if started else 0}
