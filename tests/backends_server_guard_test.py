"""The app window (app/server.py) starts only with Ollama: unusable backend settings, or settings choosing another
server, stop it with a message before it does anything, instead of setup starting or downloading Ollama for a
server it doesn't use. Each case imports server.py in a fresh process; nothing is started.

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


def import_server(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SURVIVAL_LLM_")}
    env.update(SURVIVAL_LLM_APP_DIR=os.path.join(TMP, "settings"), SURVIVAL_LLM_FILES=os.path.join(TMP, "files"),
               SURVIVAL_LLM_LIBRARY=os.path.join(TMP, "library"), OLLAMA_HOST="127.0.0.1:9", **extra)
    return subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'app'); import server; print('loaded')"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)


out = import_server({})
check("with no settings the app loads as before", out.returncode == 0 and out.stdout.strip() == "loaded",
      out.stdout + out.stderr[-500:])
out = import_server({"SURVIVAL_LLM_BACKEND": "openai", "SURVIVAL_LLM_BASE_URL": "127.0.0.1:8000",
                     "SURVIVAL_LLM_API_KEY": KEY})
check("another server stops it", out.returncode == 1 and "loaded" not in out.stdout, out.stdout + out.stderr[-500:])
check("saying what was chosen and to use chat.py", "OpenAI-compatible server at http://127.0.0.1:8000/v1" in out.stderr
      and "works only with Ollama" in out.stderr and "chat.py" in out.stderr, out.stderr[-500:])
check("without a traceback or the key", "Traceback" not in out.stderr and KEY not in out.stderr, out.stderr[-500:])
out = import_server({"SURVIVAL_LLM_BACKEND": "nonsense"})
check("unusable settings stop it with the reason", out.returncode == 1 and "settings can't be used: backend must be"
      in out.stderr and "Traceback" not in out.stderr, out.stderr[-500:])

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
