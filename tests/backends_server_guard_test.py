"""The app window (app/server.py) and backend settings: unusable settings stop it with a message before it does
anything; settings choosing another server than Ollama load it (what it then does is in backend_server_test.py).
Each case imports server.py in a fresh process; nothing is started.

Usage: .venv/bin/python tests/backends_server_guard_test.py"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from stub_server import checker  # noqa: E402

check, FAILS = checker()
TMP = tempfile.mkdtemp(prefix="backends-guard-")
KEY = "sk-guard-SECRET-4444444444"


LOG = os.path.join(TMP, "settings", "app.log")


def import_server(extra, no_stderr=False):
    """no_stderr: as under Windows' pythonw, which starts the packaged app with sys.stderr None."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SURVIVAL_LLM_")}
    env.update(SURVIVAL_LLM_APP_DIR=os.path.join(TMP, "settings"), SURVIVAL_LLM_FILES=os.path.join(TMP, "files"),
               SURVIVAL_LLM_LIBRARY=os.path.join(TMP, "library"), OLLAMA_HOST="127.0.0.1:9")
    env.update(extra)
    code = "import sys; sys.path.insert(0, 'app'); " + ("sys.stderr = None; " if no_stderr else "") + \
        "import server; print('loaded')"
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)


def log_text():
    try:
        with open(LOG, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


out = import_server({})
check("with no settings the app loads as before", out.returncode == 0 and out.stdout.strip() == "loaded",
      out.stdout + out.stderr[-500:])
out = import_server({"SURVIVAL_LLM_BACKEND": "openai", "SURVIVAL_LLM_BASE_URL": "127.0.0.1:8000",
                     "SURVIVAL_LLM_API_KEY": KEY})
check("another server loads too (backend_server_test checks what it then does)",
      out.returncode == 0 and out.stdout.strip() == "loaded", out.stdout + out.stderr[-500:])
check("without a traceback or the key", "Traceback" not in out.stderr and KEY not in out.stderr, out.stderr[-500:])
out = import_server({"SURVIVAL_LLM_BACKEND": "nonsense"})
check("unusable settings stop it with the reason", out.returncode == 1 and "settings can't be used: backend must be"
      in out.stderr and "Traceback" not in out.stderr, out.stderr[-500:])
check("the reason is kept in app.log", "settings can't be used: backend must be" in log_text()
      and KEY not in log_text(), log_text()[-500:])

if os.path.exists(LOG):
    os.remove(LOG)
out = import_server({"SURVIVAL_LLM_BACKEND": "anthropic", "SURVIVAL_LLM_BASE_URL": "127.0.0.1:8001?x=1"},
                    no_stderr=True)
check("with no stderr it still stops", out.returncode == 1 and "loaded" not in out.stdout and out.stderr == "",
      (out.returncode, out.stdout, out.stderr[-300:]))
check("and the reason is in app.log", "query string" in log_text(), log_text()[-300:])
os.makedirs(os.path.join(TMP, "blocked"))
with open(os.path.join(TMP, "blocked", "settings"), "w") as f:
    f.write("a file where the settings folder should be")
out = import_server({"SURVIVAL_LLM_BACKEND": "nonsense", "SURVIVAL_LLM_APP_DIR": os.path.join(TMP, "blocked",
                                                                                               "settings")})
check("a log it can't write doesn't hide the reason", out.returncode == 1 and "backend must be" in out.stderr
      and "Traceback" not in out.stderr, out.stderr[-300:])

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
