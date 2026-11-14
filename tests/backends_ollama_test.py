"""The Ollama backend sends Ollama exactly what core.py sent before backends/ existed, and reads its answers the same
way. A scripted HTTP server on 127.0.0.1 stands in for Ollama (tests/stub_server.py); no model is run.

Usage: .venv/bin/python tests/backends_ollama_test.py"""
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import backends  # noqa: E402
import core  # noqa: E402
from stub_server import Reply, Stub, checker, json_reply, ndjson_reply  # noqa: E402

check, FAILS = checker()
stub = Stub()
core.HOST, core.BACKEND = stub.url, backends.OllamaBackend(stub.url)

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "Where can I run a generator?"}]
OPTIONS = {"num_ctx": 32768, "num_predict": 9192, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
# The /api/chat body core.py sent at upstream 7d1fe29, written out by hand: any change here changes what Ollama is
# asked, so it should be deliberate.
GOLDEN = {"model": "qwen3.6:27b", "messages": MESSAGES, "stream": True, "keep_alive": "15m", "options": OPTIONS,
          "tools": core.TOOLS, "think": "low"}
CHUNKS = [{"message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False},
          {"message": {"role": "assistant", "content": "Outdoors."}, "done": False},
          {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop", "eval_count": 2,
           "eval_duration": 1000}]

# ---- the request body ----
stub.queue("POST", "/api/chat", ndjson_reply(CHUNKS))
got = list(core._stream("qwen3.6:27b", MESSAGES, core.TOOLS, "low", OPTIONS))
req = stub.requests[-1]
check("chat goes to /api/chat", req["path"] == "/api/chat", req["path"])
check("the body is today's body", req["body"] == GOLDEN, json.dumps(req["body"])[:300])
check("in today's key order", list(req["body"]) == list(GOLDEN), list(req["body"]))
check("sent as JSON", req["headers"].get("Content-Type") == "application/json", req["headers"])
check("chunks come through unchanged", got == CHUNKS, got)

stub.queue("POST", "/api/chat", ndjson_reply(CHUNKS))
list(core._stream("granite4.2:8b", MESSAGES, None, None, OPTIONS))
body = stub.requests[-1]["body"]
check("no tools and no think when there are none", "tools" not in body and "think" not in body, list(body))
stub.queue("POST", "/api/chat", ndjson_reply(CHUNKS))
list(core._stream("granite4.2:8b", MESSAGES, None, False, OPTIONS))
check("think false is still sent", stub.requests[-1]["body"].get("think") is False, stub.requests[-1]["body"])

# ---- /api/show: capabilities and family ----
stub.queue("POST", "/api/show", json_reply({"capabilities": ["completion", "tools", "thinking"],
                                            "details": {"family": "granite", "families": ["granite"]}}))
caps = core.capabilities("mystery:1b")
check("capabilities come from /api/show", caps == {"completion", "tools", "thinking"}, caps)
check("and are asked with the model name", stub.requests[-1]["body"] == {"model": "mystery:1b"}, stub.requests[-1])
asked = len(stub.requests)
check("family falls back to what Ollama reports", core.family("mystery:1b") == "granite", core.family("mystery:1b"))
check("/api/show is asked once per model", len(stub.requests) == asked, len(stub.requests) - asked)
check("family from the name needs no request", core.family("gemma4:e4b") == "gemma" and len(stub.requests) == asked)
stub.queue("POST", "/api/show", Reply(['{"error": "boom"}'], status=500))
check("a failed /api/show gives no capabilities", core.capabilities("other:1b") == set())
stub.queue("POST", "/api/show", json_reply({"capabilities": ["tools"]}))
check("and is not cached", core.capabilities("other:1b") == {"tools"})

# ---- /api/tags ----
stub.queue("GET", "/api/tags", json_reply({"models": [{"name": "granite4.2:8b"}, {"name": "nomic-embed-text"}]}))
check("installed models come from /api/tags", core.installed_models() == ["granite4.2:8b", "nomic-embed-text"])
gone = Stub()
gone.close()  # nothing listens on its port now
core.BACKEND = backends.OllamaBackend(gone.url)
check("installed models is None when Ollama isn't running", core.installed_models() is None)
core.BACKEND = backends.OllamaBackend(stub.url)

# ---- Stop closes the connection ----
go_on = threading.Event()
stub.queue("POST", "/api/chat", ndjson_reply([{"message": {"content": f"word {i} "}, "done": False}
                                              for i in range(200)], pause=0.01, wait_for=go_on))
stream = core._stream("granite4.2:8b", MESSAGES, None, None, OPTIONS)
first = next(stream)
stream.close()
go_on.set()
for _ in range(300):
    if stub.broken:
        break
    time.sleep(0.01)
check("closing the stream closes the connection", stub.broken == ["/api/chat"], stub.broken)
check("after the first chunk arrived", first == {"message": {"content": "word 0 "}, "done": False}, first)


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


stub.queue("POST", "/api/chat", ndjson_reply([
    {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "search_library", "arguments": {"query": "generator garage"}}}]}, "done": False},
    {"message": {"content": ""}, "done": True, "done_reason": "stop", "eval_count": 5, "eval_duration": 10**9}]))
stub.queue("POST", "/api/chat", ndjson_reply([
    {"message": {"content": "Never in a garage."}, "done": False},
    {"message": {"content": ""}, "done": True, "done_reason": "stop", "eval_count": 4, "eval_duration": 10**9}]))
messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "Can I run a generator in my garage?"}]
before = len(stub.requests)
events = list(core.chat_events("granite4.2:8b", Lib(), messages, caps={"tools"}))
done = next(e for e in events if e["type"] == "done")
second = stub.requests[-1]["body"]
check("a tool round and an answer make two requests", len(stub.requests) - before == 2, len(stub.requests) - before)
check("the search ran", any(e["type"] == "search" and e["query"] == "generator garage" for e in events))
check("the tool result goes back with its tool_name",
      second["messages"][-1].get("role") == "tool" and second["messages"][-1].get("tool_name") == "search_library",
      second["messages"][-1])
check("the answer arrives", done["answer"].startswith("Never in a garage."), done["answer"])
check("counts add up across rounds", done["tokens"] == 9 and done["seconds"] == 2.0, (done["tokens"], done["seconds"]))

stub.close()
print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
