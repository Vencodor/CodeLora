"""Teacher boundary. No SDK-level invisible retries or guessed provider endpoints."""

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .ingest import is_link
from .prompts import Request

META_ENDPOINT = "https://api.meta.ai/v1/chat/completions"
META_MODEL = "muse-spark-1.3-contributor"


def model_api_key(env_file: Path) -> str | None:
    """Read only MODEL_API_KEY, without executing shell syntax or mutating os.environ.

    The CLI calls this only after real-API approval, never for mock or dry-run.
    Existing nonempty process environment takes precedence over the local file.
    """
    existing = os.environ.get("MODEL_API_KEY")
    if existing:
        return existing
    path = env_file.absolute()
    if not path.exists():
        return None
    if any(is_link(p) for p in (path, *path.parents)):
        raise ValueError("credential path must not contain symlinks or junctions")
    with path.open("rb") as handle:
        data = handle.read(65_537)
    if len(data) > 65_536:
        raise ValueError("local credential file is too large")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("local credential file must be UTF-8") from None
    key: str | None = None
    found = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if name.strip() != "MODEL_API_KEY":
            continue
        if not separator or found:
            raise ValueError("invalid local credential assignment")
        found = True
        value = value.strip()
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError("invalid local credential quoting")
            value = value[1:-1]
        if any(c.isspace() or ord(c) < 33 or ord(c) == 127 for c in value):
            raise ValueError("invalid local credential characters")
        key = value or None
    return key


@dataclass(frozen=True)
class Response:
    content: str
    finish_reason: str = "stop"
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    usage_kind: str = "provider_reported"


class TeacherError(Exception):
    """Messages are internal fixed codes, never server bodies or exception strings."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class SafeRetry(TeacherError):
    """Known not to have submitted a generation request."""


class Rejected(TeacherError):
    """Definitive terminal rejection; no automatic retry."""


class Uncertain(TeacherError):
    """May have reached provider: do not replay automatically."""


class Teacher(Protocol):
    def identity(self) -> dict: ...

    def generate(self, request: Request) -> Response: ...


class MockTeacher:
    """Deterministic plumbing fixture, not a semantic-quality benchmark."""

    def identity(self) -> dict:
        return {"provider": "mock", "model": "mock-v1"}

    def generate(self, request: Request) -> Response:
        qa: list[dict[str, str]] = []
        explanations: list[dict[str, str]] = []
        data = {"qa": qa, "explanations": explanations, "zero_reason": ""}
        lines = [
            line.strip()
            for line in request.chunk.text.splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "```"))
        ]
        if not lines:
            data["zero_reason"] = "The chunk contains only headings or delimiters."
        else:
            fact = lines[0]
            for i in range(request.config.qa_count):
                qa.append(
                    {
                        "question": f"Which source fact is recorded at line {request.chunk.start_line}"
                        f" in {request.chunk.path}? (mock focus {i + 1})",
                        "answer": fact[: request.config.qa_max_chars],
                    }
                )
            for i in range(request.config.explanation_count):
                explanations.append(
                    {
                        "question": f"Explain a recorded detail in {request.chunk.path} near line "
                        f"{request.chunk.start_line}. (mock focus {i + 1})",
                        "text": f"The source records: {fact}"[
                            : request.config.explanation_max_chars
                        ],
                    }
                )
        return Response(
            json.dumps(data), usage_kind="mock", request_id="mock-" + request.job_id[:12]
        )


def _tokens(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


class MetaTeacher:
    """Documented Meta Chat Completions contract; real calls require explicit CLI consent.

    Official references and verification limits are recorded in README.md.
    Request IDs are diagnostic correlation IDs, NOT idempotency keys.
    """

    def __init__(
        self, api_key: str | None = None, model: str = META_MODEL, opener: Callable | None = None
    ):
        self._key = api_key if api_key is not None else os.environ.get("MODEL_API_KEY", "")
        if not self._key:
            raise ValueError("MODEL_API_KEY is required for real generation")
        self.model = model
        self._opener = opener or urllib.request.build_opener(_NoRedirect()).open

    def identity(self) -> dict:
        return {
            "provider": "meta",
            "model": self.model,
            "endpoint": META_ENDPOINT,
            "contract": "chat-completions-v1",
        }

    def generate(self, request: Request) -> Response:
        body = json.dumps(
            {
                "model": self.model,
                "messages": request.messages,
                "max_tokens": request.config.max_output_tokens,
                "temperature": request.config.temperature,
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            META_ENDPOINT,
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer " + self._key,
                "Content-Type": "application/json",
            },
        )
        try:
            with self._opener(req, timeout=request.config.timeout) as response:
                # A finite response limit avoids accidental unbounded local storage.
                raw = response.read(8_000_001)
                if len(raw) > 8_000_000:
                    raise Uncertain("response_size_limit")
                obj = json.loads(raw)
                item = obj["choices"][0]
                content = item["message"]["content"]
                finish = item["finish_reason"]
                if not isinstance(content, str) or not isinstance(finish, str):
                    raise Uncertain("invalid_provider_envelope")
                usage = obj.get("usage") or {}
                if not isinstance(usage, dict):
                    usage = {}
                request_id = response.headers.get("x-request-id") or obj.get("id")
                if not isinstance(request_id, str):
                    request_id = None
                return Response(
                    content,
                    finish,
                    request_id,
                    _tokens(usage.get("prompt_tokens")),
                    _tokens(usage.get("completion_tokens")),
                )
        except urllib.error.HTTPError as exc:
            # 429/5xx may involve upstream execution; no idempotency guarantee was verified.
            if exc.code in {400, 401, 403, 404, 405, 413, 415, 422}:
                raise Rejected("http_rejected_" + str(exc.code)) from None
            raise Uncertain("http_uncertain_" + str(exc.code)) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.gaierror, ConnectionRefusedError)):
                raise SafeRetry("connection_not_established") from None
            raise Uncertain("transport_outcome_unknown") from None
        except (TimeoutError, ConnectionError, OSError):
            raise Uncertain("transport_outcome_unknown") from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise Uncertain("invalid_provider_envelope") from None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a bearer key or project source through a redirect.
        return None
