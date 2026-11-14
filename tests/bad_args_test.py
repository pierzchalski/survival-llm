"""A tool call whose arguments aren't valid JSON gets an error the model can act on (BAD_ARGS) and another round,
instead of being run with no arguments. Ollama parses arguments itself; servers that pass on the model's text don't.
No Ollama: a fake model (core._stream) and a fake library.

Usage: .venv/bin/python tests/bad_args_test.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {detail}"))
    if not ok:
        FAILS.append(name)


class Lib:
    has_personal = False

    def __init__(self):
        self.queries = []

    def labels(self):
        return ["Ready.gov"]

    def search(self, query, topic="any", limit=8):
        self.queries.append(query)
        return [{"id": 1403312, "title": "Generator safety", "source": "Ready.gov", "official": True,
                 "snippet": "Never run a generator in a garage."}]

    def read(self, pid, max_chars=5000):
        return {"title": "Generator safety", "source": "Ready.gov", "official": True, "url": "",
                "text": "Never run a generator in a garage, even with the door open.", "next_id": None}


rounds = []


def fake_stream(model, messages, tools, think, options):
    yield from rounds.pop(0)
    yield {"done": True, "eval_count": 3, "eval_duration": 1}


core._stream = fake_stream


def call(args, name="search_library"):
    return [{"message": {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}}]


def ask(script):
    rounds[:] = script
    lib = Lib()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "Can I run a generator in my garage?"}]
    evs = list(core.chat_events("nemotron", lib, messages, caps={"tools"}))
    return evs, messages, lib


answer = [{"message": {"content": "No, never in a garage."}}]

evs, messages, lib = ask([call('{"query": "generator garage"'), call('{"query": "generator garage"}'), answer])
results = [m["content"] for m in messages if m["role"] == "tool"]
check("unparseable arguments get BAD_ARGS", results[:1] == [core.BAD_ARGS], results)
check("and aren't run with no arguments", lib.queries == ["generator garage"], lib.queries)
check("the model's second try runs", len(results) == 2 and results[1].startswith("[id 1403312]"), results)
check("and it answers", next(e for e in evs if e["type"] == "done")["answer"].startswith("No, never in a garage."))

for bad in ("[1, 2]", "not json at all", '"just a string"', ["a list"], 7):
    evs, messages, lib = ask([call(bad), answer])
    results = [m["content"] for m in messages if m["role"] == "tool"]
    check(f"BAD_ARGS for {bad!r}", results == [core.BAD_ARGS] and lib.queries == [], (results, lib.queries))

for fine in ({"query": "generator"}, '{"query": "generator"}', "", None):
    evs, messages, lib = ask([call(fine), answer])
    results = [m["content"] for m in messages if m["role"] == "tool"]
    check(f"no BAD_ARGS for {fine!r}", results and results[0] != core.BAD_ARGS and len(lib.queries) == 1,
          (results, lib.queries))

check("BAD_ARGS counts as a tool error", core.BAD_ARGS in core.TOOL_ERRORS)
messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_article",
                                                                             "arguments": "{id: 5}"}}]},
            {"role": "tool", "tool_name": "read_article", "content": core.BAD_ARGS}]
check("a BAD_ARGS read is not counted as read", core._used_ids(messages) == set(), core._used_ids(messages))

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
