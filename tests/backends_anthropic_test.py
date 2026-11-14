"""The Anthropic-shape backend: what it sends to /v1/messages, and how it turns the server-sent events it gets back
into the Ollama-shaped chunks core.py reads. A scripted HTTP server on 127.0.0.1 stands in for the AI server
(tests/stub_server.py); the replies are written by hand, not recorded from any real server.

Usage: .venv/bin/python tests/backends_anthropic_test.py"""
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import core  # noqa: E402
from backends import AnthropicBackend, BackendError  # noqa: E402
from backends.base import STOPPED  # noqa: E402
from stub_server import Reply, Stub, checker, json_reply, sse_reply  # noqa: E402

check, FAILS = checker()
KEY = "sk-ant-test-SECRET-0987654321"
stub = Stub()
V1 = stub.url + "/v1"
PATH = "/v1/messages"
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "Where can I run a generator?"}]
OPTIONS = {"num_ctx": 32768, "num_predict": 9192, "temperature": 0.7, "top_p": 0.8, "top_k": 20}


def start(output_tokens=1):
    return ("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "m", "stop_reason": None,
        "usage": {"input_tokens": 10, "output_tokens": output_tokens}}})


def block(index, **content_block):
    return ("content_block_start", {"type": "content_block_start", "index": index, "content_block": content_block})


def piece(index, **delta):
    return ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})


def stop_block(index):
    return ("content_block_stop", {"type": "content_block_stop", "index": index})


def end(reason="end_turn", output_tokens=7):
    return [("message_delta", {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
                                "usage": {"output_tokens": output_tokens}}),
            ("message_stop", {"type": "message_stop"})]


def text(*parts, reason="end_turn", output_tokens=7):
    return [start(), block(0, type="text", text=""), *[piece(0, type="text_delta", text=p) for p in parts],
            stop_block(0), *end(reason, output_tokens)]


def run(backend, events, tools=None, think=None, options=OPTIONS, messages=MESSAGES, model="m"):
    stub.queue("POST", PATH, sse_reply(events))
    return list(backend.chat_stream(model, messages, tools, think, options)), stub.requests[-1]


def final(chunks):
    return chunks[-1] if chunks and chunks[-1].get("done") else {}


def raised(fn):
    try:
        fn()
    except BackendError as exc:
        return exc
    return None


backend = AnthropicBackend(V1, api_key=KEY)

# ---- a plain answer ----
chunks, req = run(backend, text("Out", "doors."), tools=core.TOOLS)
body = req["body"]
check("posts to {base}/messages", req["path"] == PATH, req["path"])
check("sends the key as x-api-key, with a version", req["headers"].get("x-api-key") == KEY
      and req["headers"].get("anthropic-version") == "2023-06-01" and "Authorization" not in req["headers"])
check("the text streams as content chunks",
      [c["message"].get("content") for c in chunks if c["message"].get("content")] == ["Out", "doors."], chunks)
check("eval_count from usage.output_tokens", final(chunks).get("eval_count") == 7, final(chunks))
check("done_reason stop", final(chunks).get("done_reason") == "stop")
check("the system prompt is top-level", body.get("system") == "s" and all(m["role"] != "system"
                                                                          for m in body["messages"]), body)
check("only Messages API fields are sent", set(body) == {"model", "max_tokens", "messages", "stream", "system",
                                                         "tools", "temperature"}, sorted(body))
check("max_tokens is the length limit", body["max_tokens"] == 9192)
check("tools become name/description/input_schema",
      body["tools"][0] == {"name": "search_library", "description": core.TOOLS[0]["function"]["description"],
                           "input_schema": core.TOOLS[0]["function"]["parameters"]}, body["tools"][0])
check("user text goes as blocks", body["messages"] == [{"role": "user", "content": [
    {"type": "text", "text": "Where can I run a generator?"}]}], body["messages"])

chunks, req = run(backend, text("x"), options={"num_predict": -1})
check("max_tokens is sent even with no limit", req["body"]["max_tokens"] == 16384, req["body"])
chunks, req = run(AnthropicBackend(V1, max_tokens=2048), text("x"), options={"num_predict": -1})
check("as the configured ceiling when there is one", req["body"]["max_tokens"] == 2048)
chunks, req = run(backend, text("cut", reason="max_tokens"))
check("stop_reason max_tokens becomes done_reason length", final(chunks).get("done_reason") == "length")

# ---- thinking ----
check("thinking is offered by default", backend.capabilities("m") == {"tools", "thinking"})
chunks, req = run(backend, text("x"), think=False)
check("off sends thinking disabled", req["body"].get("thinking") == {"type": "disabled"}, req["body"])
check("and keeps temperature", req["body"].get("temperature") == 0.7)
chunks, req = run(backend, text("x"), think=True)
check("on sends a budget", req["body"].get("thinking") == {"type": "enabled", "budget_tokens": 4096}, req["body"])
check("and drops temperature", "temperature" not in req["body"], req["body"])
chunks, req = run(backend, text("x"), think="low", options={"num_predict": 600})
check("a budget above max_tokens raises max_tokens", req["body"]["thinking"]["budget_tokens"] == 1024
      and req["body"]["max_tokens"] == 1624, req["body"])
chunks, req = run(backend, text("x"), think=None)
check("think None sends no thinking", "thinking" not in req["body"])
plain = AnthropicBackend(V1, thinking="none")
chunks, req = run(plain, text("x"), think=True)
check("dialect none sends nothing and offers no thinking", "thinking" not in req["body"]
      and plain.capabilities("m") == {"tools"})
events = [start(), ": keepalive\n\n", ("ping", {"type": "ping"}), block(0, type="thinking", thinking="", signature=""),
          piece(0, type="thinking_delta", thinking="Let me think. "), piece(0, type="signature_delta", signature="sig"),
          stop_block(0), block(1, type="text", text=""), piece(1, type="text_delta", text="Outside."), stop_block(1),
          *end()]
chunks, req = run(backend, events, think=True)
check("thinking deltas become thinking chunks", [c["message"].get("thinking") for c in chunks
                                                 if c["message"].get("thinking")] == ["Let me think. "], chunks)
check("pings and comments become ticks", chunks[1] == {"message": {}} and chunks[2] == {"message": {}}, chunks[:3])
check("no tool call, no thinking blocks kept", final(chunks)["message"] == {}, final(chunks))

# ---- tool calls ----
tool_events = [start(), block(0, type="thinking", thinking="", signature=""),
               piece(0, type="thinking_delta", thinking="Search first."), piece(0, type="signature_delta",
                                                                               signature="c2lnbmF0dXJl"),
               stop_block(0), block(1, type="text", text=""), piece(1, type="text_delta", text="Let me look."),
               stop_block(1), block(2, type="tool_use", id="toolu_01", name="search_library", input={}),
               piece(2, type="input_json_delta", partial_json=""),
               piece(2, type="input_json_delta", partial_json='{"query": "gener'),
               piece(2, type="input_json_delta", partial_json='ator garage"}'), stop_block(2),
               block(3, type="tool_use", id="toolu_02", name="read_article", input={}),
               piece(3, type="input_json_delta", partial_json='{"id": 1403312}'), stop_block(3),
               *end("tool_use")]
chunks, req = run(backend, tool_events, tools=core.TOOLS, think=True)
calls = final(chunks)["message"].get("tool_calls")
check("input_json_delta pieces are joined and parsed", [c["function"] for c in calls] == [
    {"name": "search_library", "arguments": {"query": "generator garage"}},
    {"name": "read_article", "arguments": {"id": 1403312}}], calls)
check("ids are kept", [c["id"] for c in calls] == ["toolu_01", "toolu_02"], calls)
check("the signed reasoning rides on the first call", calls[0].get("thinking_blocks") == [
    {"type": "thinking", "thinking": "Search first.", "signature": "c2lnbmF0dXJl"}] and "thinking_blocks" not in calls[1],
      calls)
chunks, req = run(backend, [start(), block(0, type="tool_use", id="t", name="search_library", input={"query": "w"}),
                            stop_block(0), *end("tool_use")], tools=core.TOOLS)
check("input given whole in the start block", final(chunks)["message"]["tool_calls"][0]["function"]["arguments"]
      == {"query": "w"}, final(chunks))
chunks, req = run(backend, [start(), block(0, type="tool_use", id="t", name="search_library", input={}),
                            piece(0, type="input_json_delta", partial_json='{"query": "w'), stop_block(0),
                            *end("tool_use")], tools=core.TOOLS)
check("unreadable input stays text (core answers BAD_ARGS)",
      final(chunks)["message"]["tool_calls"][0]["function"]["arguments"] == '{"query": "w', final(chunks))

# ---- the stored chat goes out as alternating turns ----
history = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
           {"role": "assistant", "content": "Let me look.", "tool_calls": [
               {"id": "toolu_01", "function": {"name": "search_library", "arguments": {"query": "a"}},
                "thinking_blocks": [{"type": "thinking", "thinking": "t", "signature": "s1"}]},
               {"function": {"name": "read_article", "arguments": '{"id": 1'}}]},
           {"role": "tool", "tool_name": "search_library", "content": "ra"},
           {"role": "tool", "tool_name": "read_article", "content": core.BAD_ARGS},
           {"role": "user", "content": core.FINAL_NUDGE}]
chunks, req = run(backend, text("x"), messages=history)
sent = req["body"]["messages"]
check("roles alternate, starting with user", [m["role"] for m in sent] == ["user", "assistant", "user"], sent)
check("the reasoning goes back first, then text, then tool_use", [b["type"] for b in sent[1]["content"]]
      == ["thinking", "text", "tool_use", "tool_use"], sent[1])
made_up = sent[1]["content"][3]["id"]
check("a call without an id gets one; unreadable arguments go as {}", made_up and made_up != "toolu_01"
      and sent[1]["content"][3]["input"] == {}, sent[1]["content"][3])
check("tool results and the app's note share one user turn", sent[2]["content"] == [
    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "ra"},
    {"type": "tool_result", "tool_use_id": made_up, "content": core.BAD_ARGS},
    {"type": "text", "text": core.FINAL_NUDGE}], sent[2])
check("a round with no tools offered still declares the tools sent before, and lets it call none",
      req["body"].get("tools") == backend.last_tools and backend.last_tools[0]["name"] == "search_library"
      and req["body"].get("tool_choice") == {"type": "none"}, {k: req["body"].get(k) for k in ("tools", "tool_choice")})
chunks, req = run(AnthropicBackend(V1), text("x"), messages=history)
check("with no tools sent yet, the ones the chat used are declared", [t["name"] for t in req["body"].get("tools", [])]
      == ["search_library", "read_article"] and req["body"].get("tool_choice") == {"type": "none"}, req["body"])
chunks, req = run(backend, text("x"))
check("a chat with no lookups in it sends neither", "tools" not in req["body"] and "tool_choice" not in req["body"])
chunks, req = run(backend, text("x"), messages=history[:4] + [{"role": "user", "content": "never mind"}])
check("a call a stopped turn never answered gets a placeholder tool_result", req["body"]["messages"][2]["content"] == [
    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "ra"},
    {"type": "tool_result", "tool_use_id": made_up, "content": STOPPED},
    {"type": "text", "text": "never mind"}], req["body"]["messages"][2])
chunks, req = run(backend, text("x"), messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                                                {"role": "assistant", "content": ""}, {"role": "user", "content": "q2"}])
check("an empty answer leaves no empty block, and the users' turns merge",
      req["body"]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "q"},
                                                               {"type": "text", "text": "q2"}]}], req["body"]["messages"])

# ---- errors, and the key ----
stub.queue("POST", PATH, Reply([json.dumps({"type": "error", "error": {"type": "authentication_error",
                                                                       "message": f"invalid x-api-key {KEY}"}})],
                               status=401))
err = raised(lambda: list(backend.chat_stream("m", MESSAGES, None, None, OPTIONS)))
check("a 401 raises BackendError kind auth", err is not None and err.kind == "auth", repr(err))
check("with the server's words but not the key", err is not None and "invalid x-api-key" in str(err)
      and KEY not in str(err), str(err))
check("the backend's repr has no key", KEY not in repr(backend))
stub.queue("POST", PATH, Reply([json.dumps({"type": "error", "error": {"type": "overloaded_error",
                                                                       "message": "Overloaded"}})], status=529))
err = raised(lambda: list(backend.chat_stream("m", MESSAGES, None, None, OPTIONS)))
check("529 is busy", err is not None and err.kind == "busy", repr(err))
stub.queue("POST", PATH, Reply([json.dumps({"error": {"message": "`max_tokens` is required.",
                                                      "type": "invalid_request_error", "param": "max_tokens"}})],
                               status=400))
err = raised(lambda: list(backend.chat_stream("m", MESSAGES, None, None, OPTIONS)))
check("a 400 in OpenAI's error shape names the param", err is not None and err.param == "max_tokens", repr(err))
err = raised(lambda: run(backend, [start(), block(0, type="text", text=""), piece(0, type="text_delta", text="par"),
                                   ("error", {"type": "error", "error": {"type": "api_error", "message": "boom"}})]))
check("an error event mid-stream raises", err is not None and err.kind == "stream" and "boom" in str(err), repr(err))
err = raised(lambda: run(backend, [": keepalive\n\n"]))
check("a stream with nothing in it raises", err is not None and err.kind == "stream", repr(err))

# ---- /models ----
stub.queue("GET", "/v1/models", json_reply({"data": [{"id": "claude-x", "type": "model"}], "has_more": False}))
check("models come from /v1/models", backend.models() == ["claude-x"])
check("with the key and version", stub.requests[-1]["headers"].get("x-api-key") == KEY
      and stub.requests[-1]["headers"].get("anthropic-version") == "2023-06-01")


# ---- a whole turn through core.chat_events ----
class Lib:
    has_personal = False

    def labels(self):
        return ["Ready.gov"]

    def search(self, query, topic="any", limit=8):
        return [{"id": 1403312, "title": "Generator safety", "source": "Ready.gov", "official": True,
                 "snippet": "Never run a generator in a garage."}]

    def read(self, pid, max_chars=5000):
        return {"title": "Generator safety", "source": "Ready.gov", "official": True, "url": "",
                "text": "Never run a generator in a garage, even with the door open.", "next_id": None}


core.BACKEND = AnthropicBackend(V1, api_key=KEY)
stub.queue("POST", PATH, sse_reply(tool_events[:13] + end("tool_use")))
stub.queue("POST", PATH, sse_reply(text("Never in a garage.")))
messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "Can I run a generator in my garage?"}]
before = len(stub.requests)
events = list(core.chat_events("claude-x", Lib(), messages, reasoning="medium", caps={"tools", "thinking"}))
done = next(e for e in events if e["type"] == "done")
second = stub.requests[-1]["body"]
check("a tool round and an answer make two requests", len(stub.requests) - before == 2, len(stub.requests) - before)
check("the search ran", any(e["type"] == "search" and e["query"] == "generator garage" for e in events))
check("the reasoning was shown", any(e["type"] == "thinking" and e["text"] == "Search first." for e in events))
check("the second request alternates and carries the reasoning back",
      [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
      and second["messages"][1]["content"][0] == {"type": "thinking", "thinking": "Search first.",
                                                  "signature": "c2lnbmF0dXJl"}, second["messages"])
check("the result answers toolu_01", second["messages"][2]["content"][0].get("tool_use_id") == "toolu_01",
      second["messages"][2])
check("the answer arrives", done["answer"].startswith("Never in a garage."), done["answer"])

# The last round offers no tools, but the chat sent with it holds this question's tool_use and tool_result blocks.
core.MAX_TOOL_ROUNDS, rounds_before = 1, core.MAX_TOOL_ROUNDS
core.BACKEND = AnthropicBackend(V1)
stub.queue("POST", PATH, sse_reply(tool_events[:13] + end("tool_use")))
stub.queue("POST", PATH, sse_reply(text("Never in a garage.")))
before = len(stub.requests)
messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "Can I run a generator in my garage?"}]
events = list(core.chat_events("claude-x", Lib(), messages, caps={"tools"}))
sent = [r["body"] for r in stub.requests[before:]]
has_blocks = [any(b.get("type") in ("tool_use", "tool_result") for m in s["messages"] for b in m["content"]) for s in sent]
check("the last round sends tool blocks", len(sent) == 2 and has_blocks == [False, True], has_blocks)
check("and declares tools with tool_choice none", all("tools" in s for s, h in zip(sent, has_blocks) if h)
      and sent[-1].get("tool_choice") == {"type": "none"}, sent[-1].get("tool_choice"))
check("and is answered", next(e for e in events if e["type"] == "done")["answer"].startswith("Never in a garage."))
core.MAX_TOOL_ROUNDS = rounds_before

core.REASONING_LIMITS ={**core.REASONING_LIMITS, "short": 5}
go_on = threading.Event()
stub.queue("POST", PATH, sse_reply([start(), block(0, type="thinking", thinking="", signature="")]
                                   + [piece(0, type="thinking_delta", thinking=f"step {i}. ") for i in range(300)],
                                   pause=0.005, wait_for=go_on))
stub.queue("POST", PATH, sse_reply(text("Outdoors only.")))
stub.broken[:] = []
messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "Where can I run a generator?"}]
events = []
for event in core.chat_events("claude-x", None, messages, reasoning="short", caps={"thinking"}):
    events.append(event)
    if event["type"] == "think_limit":
        go_on.set()
for _ in range(300):
    if stub.broken:
        break
    time.sleep(0.01)
check("the reasoning cap stops the stream", any(e["type"] == "think_limit" for e in events))
check("and the server sees the connection close", stub.broken == [PATH], stub.broken)
check("the retry turns thinking off", stub.requests[-1]["body"].get("thinking") == {"type": "disabled"},
      stub.requests[-1]["body"])
check("and gets its answer", next(e for e in events if e["type"] == "done")["answer"] == "Outdoors only.")

stub.close()
print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
