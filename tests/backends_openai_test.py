"""The OpenAI-shape backend: what it sends to /v1/chat/completions, and how it turns the server-sent events it gets
back into the Ollama-shaped chunks core.py reads. A scripted HTTP server on 127.0.0.1 stands in for the AI server
(tests/stub_server.py); the replies are written by hand, not recorded from any real server.

Usage: .venv/bin/python tests/backends_openai_test.py"""
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import core  # noqa: E402
from backends import BackendError, OpenAIBackend  # noqa: E402
from backends.base import STOPPED, parse_arguments  # noqa: E402
from backends.openai_chat import to_openai  # noqa: E402
from stub_server import Reply, Stub, checker, json_reply, sse_reply  # noqa: E402

check, FAILS = checker()
KEY = "sk-test-SECRET-1234567890"
stub = Stub()
V1 = stub.url + "/v1"
PATH = "/v1/chat/completions"
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "Where can I run a generator?"}]
OPTIONS = {"num_ctx": 32768, "num_predict": 9192, "temperature": 0.7, "top_p": 0.8, "top_k": 20}


def delta(**fields):
    return {"choices": [{"index": 0, "delta": fields, "finish_reason": None}]}


def finish(reason="stop"):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}}


def run(backend, events, tools=None, think=None, options=OPTIONS, messages=MESSAGES, model="m"):
    stub.queue("POST", PATH, sse_reply(events))
    return list(backend.chat_stream(model, messages, tools, think, options)), stub.requests[-1]


def final(chunks):
    return chunks[-1] if chunks and chunks[-1].get("done") else {}


backend = OpenAIBackend(V1, api_key=KEY)

# ---- a plain answer ----
chunks, req = run(backend, [delta(role="assistant", content=""), delta(content="Out"), delta(content="doors."),
                            finish(), USAGE, "data: [DONE]\n\n"], tools=core.TOOLS)
body = req["body"]
check("posts to {base}/chat/completions", req["path"] == PATH, req["path"])
check("sends the key as a bearer token", req["headers"].get("Authorization") == "Bearer " + KEY)
check("the text streams as content chunks",
      [c["message"].get("content") for c in chunks if c["message"].get("content")] == ["Out", "doors."], chunks)
check("one done chunk, last", [c.get("done") for c in chunks].count(True) == 1 and chunks[-1].get("done"), chunks)
check("eval_count comes from usage", final(chunks).get("eval_count") == 7, final(chunks))
check("done_reason stop", final(chunks).get("done_reason") == "stop", final(chunks))
check("eval_duration is nanoseconds", isinstance(final(chunks).get("eval_duration"), int), final(chunks))
check("only chat-completions fields are sent",
      set(body) == {"model", "messages", "stream", "stream_options", "tools", "max_tokens", "temperature", "top_p"},
      sorted(body))
check("no Ollama options leak", not {"options", "num_ctx", "keep_alive", "top_k", "think"} & set(body), sorted(body))
check("max_tokens is the length limit", body["max_tokens"] == 9192, body.get("max_tokens"))
check("sampling is passed on", body["temperature"] == 0.7 and body["top_p"] == 0.8)
check("usage is asked for", body["stream_options"] == {"include_usage": True} and body["stream"] is True)
check("tools are passed as they are", body["tools"] == core.TOOLS)
check("messages are plain role/content", body["messages"] == MESSAGES, body["messages"])

chunks, req = run(backend, [delta(content="x"), finish()], options={"num_predict": -1, "temperature": 0.3})
check("no limit sends no max_tokens", "max_tokens" not in req["body"] and "top_p" not in req["body"], req["body"])
check("no usage: eval_count counts the pieces", final(chunks).get("eval_count") == 1, final(chunks))
capped = OpenAIBackend(V1, max_tokens=4096)
chunks, req = run(capped, [delta(content="x"), finish()], options={"num_predict": -1})
check("a configured ceiling replaces no limit", req["body"].get("max_tokens") == 4096, req["body"])
chunks, req = run(capped, [delta(content="x"), finish()], options={"num_predict": 600})
check("and a lower limit still wins", req["body"].get("max_tokens") == 600, req["body"])
check("no key, no Authorization header", "Authorization" not in req["headers"], req["headers"])

# ---- thinking ----
check("dialect none: no thinking capability", backend.capabilities("m") == {"tools"}, backend.capabilities("m"))
chunks, req = run(backend, [delta(content="x"), finish()], think=True)
check("dialect none sends nothing", not {"reasoning_effort", "enable_thinking", "chat_template_kwargs"} & set(req["body"]))
effort = OpenAIBackend(V1, thinking="reasoning_effort")
check("other dialects offer thinking", effort.capabilities("m") == {"tools", "thinking"})
for think, want in ((False, "none"), ("low", "low"), (True, "medium"), ("xhigh", "xhigh")):
    chunks, req = run(effort, [delta(content="x"), finish()], think=think)
    check(f"reasoning_effort for think={think!r} is {want}", req["body"].get("reasoning_effort") == want, req["body"])
chunks, req = run(effort, [delta(content="x"), finish()], think=None)
check("think None sends no reasoning_effort", "reasoning_effort" not in req["body"])
for dialect, field, value in (("enable_thinking", "enable_thinking", True),
                              ("chat_template_kwargs", "chat_template_kwargs", {"enable_thinking": True})):
    chunks, req = run(OpenAIBackend(V1, thinking=dialect), [delta(content="x"), finish()], think="low")
    check(f"{dialect} on", req["body"].get(field) == value, req["body"])
chunks, req = run(OpenAIBackend(V1, thinking="chat_template_kwargs"), [delta(content="x"), finish()], think=False)
check("chat_template_kwargs off", req["body"].get("chat_template_kwargs") == {"enable_thinking": False})
check("caps can be set outright", OpenAIBackend(V1, caps=set()).capabilities("m") == set())
try:
    OpenAIBackend(V1, thinking="sometimes")
    check("an unknown dialect is refused", False)
except ValueError:
    check("an unknown dialect is refused", True)

# ---- reasoning, keepalives, empty deltas ----
chunks, req = run(backend, [": keepalive\n\n", delta(reasoning_content="Let me "), delta(reasoning_content=""),
                            delta(reasoning="think. "), delta(thinking="Hmm. "), delta(content="Outside."),
                            finish(), "data: [DONE]\n\n"])
check("reasoning_content, reasoning and thinking become thinking chunks",
      [c["message"].get("thinking") for c in chunks if c["message"].get("thinking")] == ["Let me ", "think. ", "Hmm. "],
      chunks)
check("keepalives and empty reasoning become empty chunks (ticks)", chunks[0] == {"message": {}}
      and chunks[2] == {"message": {}}, chunks[:3])
check("pieces counted without usage include reasoning", final(chunks).get("eval_count") == 4, final(chunks))
chunks, req = run(backend, ["data:" + json.dumps(delta(content="tight")) + "\n", "data: " + json.dumps(finish()) + "\n"])
check("data lines without a space or blank line between them", [c["message"].get("content") for c in chunks][:1]
      == ["tight"] and final(chunks).get("done"), chunks)
chunks, req = run(backend, [delta(content="cut"), finish("length")])
check("finish length becomes done_reason length", final(chunks).get("done_reason") == "length", final(chunks))

# ---- tool calls ----
fragments = [delta(role="assistant", tool_calls=[{"index": 0, "id": "call_abc", "type": "function",
                                                  "function": {"name": "search_library", "arguments": ""}}]),
             delta(tool_calls=[{"index": 0, "function": {"arguments": '{"query": '}}]),
             delta(tool_calls=[{"index": 0, "function": {"arguments": '"generator garage"}'}}]),
             delta(tool_calls=[{"index": 1, "id": "call_def", "type": "function",
                                "function": {"name": "read_article", "arguments": '{"id": 1403312}'}}]),
             finish("tool_calls"), USAGE]
chunks, req = run(backend, fragments, tools=core.TOOLS)
calls = final(chunks).get("message", {}).get("tool_calls")
check("fragmented arguments are joined and parsed", calls == [
    {"id": "call_abc", "function": {"name": "search_library", "arguments": {"query": "generator garage"}}},
    {"id": "call_def", "function": {"name": "read_article", "arguments": {"id": 1403312}}}], calls)
check("calls only come in the done chunk", all(not c["message"].get("tool_calls") for c in chunks[:-1]))
whole = [delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "search_library",
                                                                 "arguments": '{"query": "water"}'}}]), finish()]
chunks, req = run(backend, whole, tools=core.TOOLS)
check("whole arguments in one piece", final(chunks)["message"]["tool_calls"][0]["function"]["arguments"]
      == {"query": "water"}, final(chunks))
bad = [delta(tool_calls=[{"index": 0, "function": {"name": "search_library", "arguments": '{"query": "wat'}}]),
       finish()]
chunks, req = run(backend, bad, tools=core.TOOLS)
call = final(chunks)["message"]["tool_calls"][0]
check("unreadable arguments stay text (core answers BAD_ARGS)", call["function"]["arguments"] == '{"query": "wat'
      and core._bad_args(call), call)
check("a call with no id gets none here", "id" not in call, call)
check("arguments in a code fence are repaired", parse_arguments('```json\n{"query": "a"}\n```') == {"query": "a"})
check("a trailing comma is repaired", parse_arguments('{"query": "a",}') == {"query": "a"})
check("words around it are skipped", parse_arguments('Call: {"id": 5} done') == {"id": 5})
check("a nested object in a cut-off call is not taken", parse_arguments('{"a": {"b": 1}') == '{"a": {"b": 1}')
check("empty arguments are an empty object", parse_arguments("") == {} and parse_arguments(None) == {})

# ---- the stored chat goes out with ids ----
history = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
           {"role": "assistant", "content": "", "tool_calls": [
               {"function": {"name": "search_library", "arguments": {"query": "a"}}},
               {"function": {"name": "search_library", "arguments": {"query": "b"}}}]},
           {"role": "tool", "tool_name": "search_library", "content": "ra"},
           {"role": "tool", "tool_name": "search_library", "content": "rb"},
           {"role": "assistant", "content": "", "tool_calls": [
               {"id": "srv_9", "function": {"name": "read_article", "arguments": '{"id": 1'}}]},
           {"role": "tool", "tool_name": "read_article", "content": core.BAD_ARGS},
           {"role": "assistant", "content": "answer"}, {"role": "user", "content": "more"}]
chunks, req = run(backend, [delta(content="x"), finish()], messages=history)
sent = req["body"]["messages"]
first_ids = [c["id"] for c in sent[2]["tool_calls"]]
check("calls without ids get distinct made-up ids", len(set(first_ids)) == 2 and all(first_ids), first_ids)
check("results answer their calls in order", [sent[3]["tool_call_id"], sent[4]["tool_call_id"]] == first_ids, sent[3:5])
check("a server's own id is kept", sent[5]["tool_calls"][0]["id"] == "srv_9" and sent[6]["tool_call_id"] == "srv_9")
check("arguments go as JSON text", sent[2]["tool_calls"][0]["function"]["arguments"] == '{"query": "a"}'
      and sent[5]["tool_calls"][0]["function"]["arguments"] == '{"id": 1', sent[2])
check("tool_name stays home", all("tool_name" not in m for m in sent), sent)
check("the made-up ids are the same next time", run(backend, [delta(content="x"), finish()], messages=history)[1]
      ["body"]["messages"][2]["tool_calls"][0]["id"] == first_ids[0])
stopped = history[:4] + [{"role": "user", "content": "never mind"}]  # stopped after the first of two lookups
sent = run(backend, [delta(content="x"), finish()], messages=stopped)[1]["body"]["messages"]
check("a call a stopped turn never answered gets a placeholder result", [(m["role"], m.get("tool_call_id"),
                                                                          m["content"]) for m in sent[3:]]
      == [("tool", first_ids[0], "ra"), ("tool", first_ids[1], STOPPED), ("user", None, "never mind")], sent[3:])
sent = run(backend, [delta(content="x"), finish()], messages=history[:3])[1]["body"]["messages"]
check("also when the chat ends on the calls", [m.get("tool_call_id") for m in sent[3:]] == first_ids, sent[3:])

# ---- errors, and the key ----
stub.queue("POST", PATH, Reply([json.dumps({"error": {"message": f"Incorrect API key provided: {KEY}",
                                                      "type": "invalid_request_error", "code": "invalid_api_key"}})],
                               status=401))
try:
    list(backend.chat_stream("m", MESSAGES, None, None, OPTIONS))
    err = None
except BackendError as exc:
    err = exc
check("a 401 raises BackendError kind auth", err is not None and err.kind == "auth" and err.status == 401, repr(err))
check("its message has the server's words", err is not None and "Incorrect API key provided" in str(err), str(err))
check("but never the key", err is not None and KEY not in str(err) and KEY not in repr(err.args), str(err))
check("the backend's repr has no key", KEY not in repr(backend) and KEY not in repr(backend.client), repr(backend))
stub.queue("POST", PATH, Reply([json.dumps({"error": {"message": "Unsupported parameter: top_p", "type":
                                                      "invalid_request_error", "param": "top_p", "code": None}})],
                               status=400))
try:
    list(backend.chat_stream("m", MESSAGES, None, None, OPTIONS))
    err = None
except BackendError as exc:
    err = exc
check("a 400 names the param", err is not None and err.kind == "bad_request" and err.param == "top_p", repr(err))
stub.queue("POST", PATH, Reply(['{"error": "model \\"m\\" not found"}'], status=404))  # Ollama's shape
try:
    list(backend.chat_stream("m", MESSAGES, None, None, OPTIONS))
    err = None
except BackendError as exc:
    err = exc
check("a 404 in Ollama's error shape", err is not None and err.kind == "not_found" and "not found" in str(err), err)
try:
    run(backend, [delta(content="partial "), {"error": {"message": "engine crashed", "type": "server_error"}}])
    err = None
except BackendError as exc:
    err = exc
check("an error event mid-stream raises", err is not None and err.kind == "stream" and "engine crashed" in str(err),
      repr(err))
try:
    run(backend, [": keepalive\n\n"])
    err = None
except BackendError as exc:
    err = exc
check("a stream that ends with nothing in it raises", err is not None and err.kind == "stream", repr(err))
try:
    run(backend, ["data: <html>oops</html>\n\n"])
    err = None
except BackendError as exc:
    err = exc
check("data that isn't JSON raises", err is not None and "isn't JSON" in str(err), repr(err))

# ---- /models ----
stub.queue("GET", "/v1/models", json_reply({"object": "list", "data": [{"id": "qwen3-8b"}, {"id": "gemma-4"}]}))
check("models come from /v1/models", backend.models() == ["qwen3-8b", "gemma-4"])
check("with the key", stub.requests[-1]["headers"].get("Authorization") == "Bearer " + KEY)
stub.queue("GET", "/v1/models", json_reply({"models": [{"name": "a"}]}))
check("or a models/name list", backend.models() == ["a"])
fixed = OpenAIBackend(V1, model="qwen3-8b")
stub.queue("GET", "/v1/models", json_reply({"data": [{"id": "qwen3-8b"}, {"id": "other"}]}))
check("a configured model is the only one offered", fixed.models() == ["qwen3-8b"])
stub.queue("GET", "/v1/models", Reply(['{"error": "no such route"}'], status=404))
check("even when the server has no /models", fixed.models() == ["qwen3-8b"])
stub.queue("GET", "/v1/models", Reply([json.dumps({"error": {"message": f"bad key {KEY}"}})], status=401))
check("a refused list is None", backend.models() is None)
check("and says why, without the key", backend.last_error.kind == "auth" and KEY not in str(backend.last_error))
gone = Stub()
gone.close()
check("unreachable is None", OpenAIBackend(gone.url + "/v1", model="x").models() is None)


# ---- whole turns through core.chat_events ----
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


core.BACKEND = OpenAIBackend(V1, api_key=KEY, thinking="reasoning_effort")
stub.queue("POST", PATH, sse_reply(fragments[:3] + [finish("tool_calls")]))
stub.queue("POST", PATH, sse_reply([delta(content="Never in a garage."), finish(), USAGE]))
messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "Can I run a generator in my garage?"}]
before = len(stub.requests)
events = list(core.chat_events("qwen3-8b", Lib(), messages, caps={"tools", "thinking"}))
done = next(e for e in events if e["type"] == "done")
second = stub.requests[-1]["body"]["messages"]
check("a tool round and an answer make two requests", len(stub.requests) - before == 2, len(stub.requests) - before)
check("the search ran with the joined arguments", any(e["type"] == "search" and e["query"] == "generator garage"
                                                      for e in events), events)
check("the result answers the call by its id", second[-1] == {"role": "tool", "tool_call_id": "call_abc",
                                                              "content": second[-1]["content"]}
      and second[-2]["tool_calls"][0]["id"] == "call_abc", second[-2:])
check("the answer arrives", done["answer"].startswith("Never in a garage."), done["answer"])
check("the stored chat keeps Ollama's shape", messages[-2].get("tool_name") == "search_library"
      and messages[-3]["tool_calls"][0]["function"]["arguments"] == {"query": "generator garage"}, messages[-3:])

core.REASONING_LIMITS = {**core.REASONING_LIMITS, "short": 5}
go_on = threading.Event()
stub.queue("POST", PATH, sse_reply([delta(reasoning_content=f"step {i}. ") for i in range(300)], pause=0.005,
                                   wait_for=go_on))
stub.queue("POST", PATH, sse_reply([delta(content="Outdoors only."), finish()]))
stub.broken[:] = []
messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "Where can I run a generator?"}]
events = []
for event in core.chat_events("qwen3-8b", None, messages, reasoning="short", caps={"thinking"}):
    events.append(event)
    if event["type"] == "think_limit":
        go_on.set()
for _ in range(300):
    if stub.broken:
        break
    time.sleep(0.01)
check("the reasoning cap stops the stream", any(e["type"] == "think_limit" for e in events), [e["type"] for e in events])
check("and the server sees the connection close", stub.broken == [PATH], stub.broken)
check("the retry asks for no reasoning", stub.requests[-1]["body"].get("reasoning_effort") == "none",
      stub.requests[-1]["body"])
check("and gets its answer", next(e for e in events if e["type"] == "done")["answer"] == "Outdoors only.")

# ---- reasoning before a tool call goes back with it (smoke test: colibri re-read the whole chat every round) ----
reasoned = [delta(reasoning_content="Need the "), delta(reasoning_content="library."),
            delta(tool_calls=[{"index": 0, "id": "c9", "type": "function",
                               "function": {"name": "search_library", "arguments": '{"query": "burn"}'}}]),
            finish("tool_calls")]
chunks, req = run(backend, reasoned)
made = final(chunks)["message"]["tool_calls"]
check("reasoning is kept on the first call", made[0].get("reasoning") == {"reasoning_content": "Need the library."},
      made)
history = MESSAGES + [{"role": "assistant", "content": "", "tool_calls": made},
                      {"role": "tool", "tool_name": "search_library", "content": "[id 1] Burns"}]
chunks, req = run(backend, [delta(content="Cool it."), finish()], messages=history)
sent = req["body"]["messages"][2]
check("and sent back as reasoning_content", sent.get("reasoning_content") == "Need the library.", sent)
check("not inside the call", "reasoning" not in sent["tool_calls"][0] and "reasoning_content" not in
      sent["tool_calls"][0], sent)
chunks, req = run(backend, [delta(tool_calls=[{"index": 0, "id": "c10", "type": "function",
                                               "function": {"name": "search_library", "arguments": "{}"}}]),
                            finish("tool_calls")])
check("no reasoning, nothing kept", "reasoning" not in final(chunks)["message"]["tool_calls"][0])
# A server that streams it as "reasoning" (Ollama's /v1, for one) gets it back as "reasoning", not as a field it
# never named
chunks, req = run(backend, [delta(reasoning="Search."), delta(tool_calls=[
    {"index": 0, "id": "c11", "type": "function", "function": {"name": "search_library", "arguments": "{}"}}]),
    finish("tool_calls")])
made = final(chunks)["message"]["tool_calls"]
chunks, req = run(backend, [delta(content="Cool it."), finish()], messages=MESSAGES + [
    {"role": "assistant", "content": "", "tool_calls": made}, {"role": "tool", "content": "[id 1] Burns"}])
sent = req["body"]["messages"][2]
check("streamed as reasoning, sent back as reasoning", sent.get("reasoning") == "Search."
      and "reasoning_content" not in sent, sent)

# ---- an answer's reasoning goes back with it on the next question (smoke test: colibri re-read the whole chat
# for a follow-up after a thinking turn) ----
fresh = OpenAIBackend(V1)
chunks, req = run(fresh, [delta(reasoning_content="Cold water. "), delta(content="\n\nCool it "),
                          delta(content="for 20 minutes."), finish()])
answer = "".join(c["message"].get("content") or "" for c in chunks)  # what core keeps as the answer's text
check("the answer's chunks carry no kept reasoning", all("reasoning" not in c["message"] for c in chunks), chunks)
later = MESSAGES + [{"role": "assistant", "content": answer}, {"role": "user", "content": "And blisters?"}]
chunks, req = run(fresh, [delta(content="Leave them."), finish()], messages=later)
sent = req["body"]["messages"][2]
check("a follow-up sends the answer back with its reasoning", sent == {
    "role": "assistant", "content": "\n\nCool it for 20 minutes.", "reasoning_content": "Cold water. "}, sent)
check("the stored chat itself keeps no reasoning", "reasoning" not in str(later), later)
chunks, req = run(fresh, [delta(content="x"), finish()], messages=later + [
    {"role": "assistant", "content": "Leave them."}, {"role": "user", "content": "Ok"}])
check("an answer streamed without reasoning goes back as it is", req["body"]["messages"][4] == {"role": "assistant", "content": "Leave them."},
      req["body"]["messages"][4])
check("an edited answer gets no reasoning", to_openai([{"role": "assistant", "content": "Cool it."}], fresh.answers)
      == [{"role": "assistant", "content": "Cool it."}])
for n in range(OpenAIBackend.ANSWERS_KEPT + 5):
    run(fresh, [delta(reasoning_content="r"), delta(content=f"answer {n}"), finish()])
check("only the latest answers are kept", len(fresh.answers) == OpenAIBackend.ANSWERS_KEPT
      and "answer 0" not in fresh.answers and f"answer {OpenAIBackend.ANSWERS_KEPT + 4}" in fresh.answers,
      len(fresh.answers))

stub.close()
print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
