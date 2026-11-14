"""The terminal chat ends each answer with the answer core settled on, not only the text the model streamed: core
replaces a Sources line naming articles the model never opened, and adds one when the model named none, and the
app window shows that answer; the terminal used to show only what streamed, made-up sources included.
No Ollama: a fake model (core._stream) and a fake library.

Usage: .venv/bin/python tests/chat_answer_test.py"""
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chat  # noqa: E402
import core  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {detail!r}"))
    if not ok:
        FAILS.append(name)


class Lib:
    has_personal = False

    def labels(self):
        return ["Ready.gov"]

    def search(self, query, topic="any", limit=8):
        return [{"id": 1403312, "title": "Burns", "source": "MedlinePlus", "official": True, "snippet": "Cool it."}]

    def read(self, pid, max_chars=5000):
        return {"title": "Burns", "source": "MedlinePlus", "official": True, "url": "",
                "text": "Cool a burn under cool running water for 20 minutes.", "next_id": None}


rounds = []


def fake_stream(model, messages, tools, think, options):
    yield from rounds.pop(0)
    yield {"done": True, "eval_count": 5, "eval_duration": 1}


core._stream = fake_stream


def say(text):
    return [{"message": {"content": piece}} for piece in re.findall(r"\S+\s*|\s+", text)]  # a word at a time


def call(name, **args):
    return [{"message": {"tool_calls": [{"function": {"name": name, "arguments": args}}]}}]


def turn(question, script):
    rounds[:] = script
    out = io.StringIO()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": question}]
    with contextlib.redirect_stdout(out):
        chat.run_turn("nemotron", Lib(), messages, "off", {"tools"})
    return out.getvalue(), messages


def answer_part(printed):
    """The printed text before the closing "[N tokens ...]" line."""
    return printed.rsplit(f"\n{chat.DIM}[", 1)[0]


# Made-up sources, no lookup: the terminal must end with core's line, saying it was corrected and why
made_up = "Cool it under running water for 20 minutes.\n\nSources: Ready.gov, CDC, MedlinePlus"
printed, messages = turn("My son grabbed a hot pan. How should I treat it?", [say(made_up)])
check("made-up sources: the streamed text is shown", made_up in printed, printed)
shown = answer_part(printed)
check("made-up sources: the terminal ends with core's Sources line",
      shown.rstrip().endswith("Sources: none (from its own knowledge)"), shown)
check("made-up sources: says the app corrected it, and why", "corrected by the app: it named sources it never opened"
      in printed, printed)

# It read an article and named none: core's Sources line is added after the answer, with no correction note
printed, _ = turn("How do I treat a burn?", [call("search_library", query="burn"), call("read_article", id=1403312),
                                             say("Cool it under running water for 20 minutes.")])
shown = answer_part(printed)
check("no Sources line: core's line is added", shown.rstrip().endswith("Sources: Burns (MedlinePlus)"), shown)
check("no Sources line: no correction note", "corrected" not in printed, printed)
check("no Sources line: the answer is printed once", printed.count("Cool it under running water") == 1, printed)

# It named what it read: nothing more is printed
good = "Cool it under running water for 20 minutes.\n\nSources: Burns (MedlinePlus)"
printed, _ = turn("How do I treat a burn?", [call("search_library", query="burn"), call("read_article", id=1403312),
                                             say(good)])
check("right sources: printed as streamed, once", answer_part(printed).strip().endswith(good)
      and printed.count("Sources:") == 1 and "corrected" not in printed, printed)

# Text before a lookup isn't the answer: what streamed before the search doesn't count as shown
printed, _ = turn("How do I treat a burn?", [say("Let me check.") + call("search_library", query="burn"),
                                             call("read_article", id=1403312), say(good)])
check("text before a lookup: the answer still isn't corrected", "corrected" not in printed
      and answer_part(printed).strip().endswith(good), printed)

# Asked to look it up, it answered from memory first: after the retry, the answer it then gave is what counts
printed, _ = turn("Please look it up: how do I treat a burn?",
                  [say(made_up), call("search_library", query="burn"), call("read_article", id=1403312), say(good)])
check("after a retry: ends with the answer it gave, uncorrected", answer_part(printed).strip().endswith(good)
      and "corrected" not in printed, printed)

# A Sources line the model wrote inline is only moved onto its own line by core: nothing was corrected
printed, _ = turn("How do I treat a burn?", [call("search_library", query="burn"), call("read_article", id=1403312),
                                             say("Cool it under running water for 20 minutes. Sources: Burns (MedlinePlus)")])
check("inline Sources line: core only moves it, so no correction note", "corrected" not in printed, printed)
check("inline Sources line: the answer is printed once", printed.count("Cool it under running water") == 1, printed)

# corrected_tail itself
check("tail: same text", chat.corrected_tail("A.\n\nSources: X", "A.\n\nSources: X") == ("", False))
check("tail: an added line", chat.corrected_tail("A.", "A.\n\nSources: X") == ("\n\nSources: X", False))
check("tail: a replaced line starts at its line", chat.corrected_tail("A.\nB.\n\nSources: Y, Z", "A.\nB.\n\nSources: X")
      == ("Sources: X", True), chat.corrected_tail("A.\nB.\n\nSources: Y, Z", "A.\nB.\n\nSources: X"))
check("tail: a change in whitespace only is no change", chat.corrected_tail("A. Sources: X", "A.\n\nSources: X") == ("", False),
      chat.corrected_tail("A. Sources: X", "A.\n\nSources: X"))
check("tail: a change in the first line gives the whole answer", chat.corrected_tail("Boil it 1 min (id 1403312).",
      "Boil it 1 min.") == ("Boil it 1 min.", True))

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
