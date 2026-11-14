"""HTTP plumbing shared by the OpenAI- and Anthropic-shape backends: JSON requests, server-sent events, and errors
that say what the server said without ever showing the API key."""
import json
import urllib.error
import urllib.request


class BackendError(RuntimeError):
    """The server answered with an error. status: the HTTP status (None for an error sent mid-stream); kind: "auth",
    "not_found", "context", "busy", "bad_request", "server" or "stream"; param: the request field the server named,
    if it named one. The message never holds the API key."""

    def __init__(self, message, status=None, kind="", param=None):
        super().__init__(message)
        self.status, self.kind, self.param = status, kind, param


def error_detail(body):
    """(message, code, param) from an error body in any of the shapes servers use: Ollama's {"error": "text"},
    OpenAI's {"error": {"message", "type", "param", "code"}}, Anthropic's {"type": "error", "error": {"type",
    "message"}}; or the text itself when it isn't JSON."""
    try:
        data = json.loads(body) if isinstance(body, (str, bytes)) else body
    except ValueError:
        return str(body if isinstance(body, str) else body.decode("utf-8", "replace")).strip()[:300], "", None
    err = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(err, str):
        return err, "", None
    if isinstance(err, dict):
        message = err.get("message") or err.get("detail") or json.dumps(err)[:300]
        return str(message), str(err.get("code") or err.get("type") or ""), err.get("param")
    return json.dumps(data)[:300], "", None


def _kind(status, code, message):
    low = f"{code} {message}".lower()
    if "context_length" in low or "context length" in low or "too many tokens" in low:
        return "context"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status in (429, 503, 529):
        return "busy"
    if status and status >= 500:
        return "server"
    return "bad_request" if status else "stream"


class Client:
    """Requests to one server. The key goes in a header and is scrubbed from every error message."""

    def __init__(self, base_url, headers, secret=""):
        self.base_url, self._headers, self._secret = base_url, dict(headers), secret

    def __repr__(self):
        return f"Client({self.base_url!r})"

    def scrub(self, text):
        return text.replace(self._secret, "[key]") if self._secret else text

    def error(self, body, status=None):
        message, code, param = error_detail(body)
        message = self.scrub(message)
        where = f"HTTP {status}" if status else "during the answer"
        return BackendError(f"The AI server at {self.base_url} said ({where}): {message}", status,
                            _kind(status, code, message), param)

    def request(self, path, body=None, stream=False, timeout=900):
        """The response (stream=True) or its JSON. Raises BackendError for an HTTP error, OSError when the server
        can't be reached."""
        req = urllib.request.Request(self.base_url + path, headers={"Content-Type": "application/json", **self._headers},
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                text = exc.read().decode("utf-8", "replace")
            except OSError:
                text = ""
            raise self.error(text or exc.reason, exc.code) from None
        if stream:
            return resp
        with resp:
            return json.load(resp)


def sse_data(resp):
    """The data of each server-sent event, as text, until "[DONE]" or the end of the stream; None for a comment
    line (": keepalive"), so the caller can still tick. Each "data:" line is taken as one event: the servers this
    is written for put each JSON event on one line, and this way a server that leaves out the blank line between
    events is still read correctly."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith(":"):
            yield None
        elif line.startswith("data:"):
            data = line[5:].lstrip(" ")
            if data == "[DONE]":
                return
            if data:
                yield data
