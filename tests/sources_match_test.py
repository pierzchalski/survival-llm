"""The Sources check against Sources lines models really wrote: names of articles they opened must pass, made-up ones
must still be caught.

The opened articles and Sources lines are copied from two live runs against a local OpenAI/Anthropic-compatible
server (Nov 14-15 2026): Qwen3.8-27B, whose answers named only articles it had opened but were "corrected" twice,
and Qwen3.6-35B-A3B with reasoning off, which opened nothing and named sources anyway.
No Ollama: a fake model (core._stream) and a fake library.

Usage: .venv/bin/python tests/sources_match_test.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {detail}"))
    if not ok:
        FAILS.append(name)


def unopened(line, read, read_before=()):
    return [n for n in core.cited_sources(line) or [] if not core._opened(n, read, read_before)]


# What the two runs opened (title, collection, address), as the library gives them.
MEDLINE = {"title": "Burn | First Degree Burn | Second Degree Burn", "source": "MedlinePlus (NIH)",
           "url": "https://medlineplus.gov/burns.html"}
WIKIMED = {"title": "Burn", "source": "WikiMed (Wikipedia medicine)", "url": "https://en.wikipedia.org/wiki/Burn"}
WIKIHOW = {"title": "How to Treat a Blistering Burn on Your Finger", "source": "wikiHow",
           "url": "https://www.wikihow.com/Treat-a-Blistering-Burn-on-Your-Finger"}
CERT = {"title": "FEMA CERT Basic Training Participant Manual", "source": "U.S. government survival and first-aid manuals",
        "url": "https://www.ready.gov/sites/default/files/2019.CERT_.Basic_.PM_FINAL_508c.pdf"}
ATP = {"title": "US Army ATP 3-50.21: Survival", "source": "U.S. government survival and first-aid manuals",
       "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/pdf/web/ARN12086_ATP%203-50x21%20FINAL%20WEB%202.pdf"}
WHOBEC = {"title": "WHO-ICRC Basic Emergency Care (2018)", "source": "IFRC and WHO first-aid and emergency care guides",
          "url": "https://iris.who.int/x"}
BURNS, WATER = [MEDLINE, WIKIMED, WIKIHOW], [CERT, ATP]

# ---- named only what it opened: no correction ----
for line, read in [
    ("Sources: Burn (WikiMed), How to Treat a Blistering Burn on Your Finger (wikiHow), MedlinePlus Burns", BURNS),
    ("Sources: FEMA CERT Basic Training manual (Ready.gov); US Army ATP 3-50.21 Survival.", WATER),
    # the other four answers of the Qwen3.8 run, which passed before and must still pass
    ("Sources: First aid for minor burns (Medical Sciences Q&A), MedlinePlus – Burns",
     [MEDLINE, {"title": "First aid for minor burns", "source": "Medical Sciences Q&A (Stack Exchange)"}]),
    ("Sources: FEMA CERT Basic Training Manual; US Army ATP 3-50.21 Survival", WATER),
    ("Sources: Burn (WikiMed), How to Treat a Blistering Burn on Your Finger (wikiHow)", [WIKIMED, WIKIHOW]),
]:
    got = unopened(line, read)
    check(f"opened, not flagged: {line!r}", got == [], got)

# The address matters only for the article it belongs to, and is kept for a follow-up question too.
earlier = core._read_earlier([{"role": "tool", "tool_name": "read_article",
                               "content": f"{CERT['title']} ({CERT['source']})\n{CERT['url']}\n\nWater ..."}])
check("an earlier read keeps its address", earlier and earlier[0].get("url") == CERT["url"], earlier)
got = unopened("Sources: FEMA CERT Basic Training manual (Ready.gov)", [], earlier)
check("follow-up naming an article read earlier, by its website", got == [], got)

# ---- named what it never opened: still caught ----
for line, read in [
    # the Qwen3.6 run, reasoning off: no lookup at all
    ("Sources: Ready.gov, CDC, MedlinePlus", []),
    ("Sources: Ready.gov, MedlinePlus, CDC", []),
    ("**Sources:** CDC Emergency Response, Ready.gov, WHO Guidelines for Drinking-water Quality", []),
    # the same lines when it had opened something: the parts it never opened still count
    ("Sources: Ready.gov, CDC, MedlinePlus", BURNS),
    ("**Sources:** CDC Emergency Response, Ready.gov, WHO Guidelines for Drinking-water Quality", WATER),
    # a collection it read from, with a title it never opened
    ("Sources: MedlinePlus Sunburn Treatment", BURNS),
    ("Sources: MedlinePlus First Aid", BURNS),
    ("Sources: WikiMed Scald Injury", BURNS),
    # a publisher in parentheses that is neither the article's collection nor its website
    ("Sources: FEMA CERT Basic Training manual (CDC)", WATER),
    ("Sources: Burns (Mayo Clinic)", BURNS),
    ("Sources: US Army ATP 3-50.21 Survival (Ready.gov)", WATER),  # Ready.gov is the other article's website
    ("Sources: Preventing and Treating Burns (Ready.gov)", WATER),  # a real Ready.gov page, but not one it opened
    # a collection's descriptive words ("first aid") plus a word of the opened title is not a name
    ("Sources: Army First Aid Manual", [ATP]),  # the library's "First Aid (FM 4-25.11)" was not opened
    ("Sources: Basic First Aid", [CERT]),
    ("Sources: FEMA First Aid", [CERT]),
    ("Sources: Government Training Manual", [CERT]),
    ("Sources: Basic First Aid Guidelines", [WHOBEC]),
    # a website named on its own is not an article
    ("Sources: Ready.gov", WATER),
]:
    got = unopened(line, read)
    check(f"made up, flagged: {line!r}", got != [], got)

# ---- through chat_events: the address comes from the library with the article ----
SCRIPT = []


def fake_stream(model, messages, tools, think, options):
    step = SCRIPT.pop(0) if SCRIPT else ("say", "OK.")
    if step[0] == "call":
        yield {"message": {"content": "", "tool_calls": [{"function": {"name": step[1], "arguments": step[2]}}]}}
    else:
        yield {"message": {"content": step[1]}}
    yield {"done": True, "eval_count": 1, "eval_duration": 1}


core._stream = fake_stream


class Lib:
    has_personal = False
    docs = {623294: CERT, 620768: ATP}

    def search(self, q, topic=None, limit=8):
        return [{"id": pid, "title": d["title"], "source": d["source"], "official": True, "snippet": "Boil it."}
                for pid, d in self.docs.items()]

    def read(self, pid, max_chars=5000):
        d = self.docs.get(pid)
        return d and {**d, "official": True, "text": "Boil water for one minute.", "next_id": None}


def answer(reply):
    SCRIPT[:] = [("call", "search_library", {"query": "creek water"}), ("call", "read_article", {"id": 623294}),
                 ("call", "read_article", {"id": 620768}), ("say", reply)]
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "How do I make creek water safe?"}]
    return next(e for e in core.chat_events("qwen3.8", Lib(), msgs, caps={"tools"}) if e["type"] == "done")


done = answer("Boil it for one minute.\n\nSources: FEMA CERT Basic Training manual (Ready.gov); US Army ATP 3-50.21 Survival.")
check("chat_events: website of an opened article, no correction", done["unopened"] == [], done["unopened"])
check("chat_events: the read event's article keeps its address", done["sources"][0].get("url") == CERT["url"],
      done["sources"])
done = answer("Boil it for one minute.\n\nSources: FEMA CERT Basic Training manual (CDC); US Army ATP 3-50.21 Survival.")
check("chat_events: wrong publisher still corrected", done["unopened"] == ["FEMA CERT Basic Training manual (CDC)"],
      done["unopened"])

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all passed'}")
sys.exit(1 if FAILS else 0)
