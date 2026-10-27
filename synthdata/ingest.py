"""Conservative UTF-8 ingestion; no symlinks or Windows junctions are traversed."""

import ast
import fnmatch
import hashlib
import json
import re
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config

EXTENSIONS = set(
    """.md .markdown .txt .rst .py .pyi .js .jsx .ts .tsx .mjs .cjs .java
.c .h .cpp .hpp .cc .cs .go .rs .rb .php .swift .kt .kts .scala .sh .ps1 .sql .r
.lua .ex .exs .erl .clj .cljs .vue .svelte .html .css .scss .json .toml .yaml .yml
.ini .cfg .conf .xml .properties .dockerfile""".split()
)
DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "target",
    "bin",
    "obj",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".nuxt",
    "coverage",
    ".idea",
    ".vscode",
    ".aws",
    ".ssh",
    ".codex",
    ".agents",
    ".test-tmp",
    "site-packages",
    "outputs",
    "synth-output",
}
SECRET_NAMES = {
    "credentials",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_ed25519",
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".git-credentials",
    "service-account.json",
    "service_account.json",
    "kubeconfig",
}
SECRET_EXTENSIONS = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"}
SECRET_CONTENT = re.compile(
    r"-----BEGIN (?:[A-Z ]*PRIVATE KEY)-----|"
    r"\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{20,}|"
    r"\bsk-[A-Za-z0-9_-]{20,}|"
    r"(?im:^\s*[\"']?(?:api[_-]?key|secret|password|access[_-]?token)[\"']?\s*[:=]\s*"
    r"[\"']?[^\s\"']{8,})"
)


def digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)
    )


def matches(path: str, patterns: tuple[str, ...]) -> bool:
    name = path.casefold()
    for pattern in patterns:
        pattern = pattern.replace("\\", "/").strip("/").casefold()
        if not pattern:
            continue
        if "/" not in pattern and any(fnmatch.fnmatchcase(p, pattern) for p in name.split("/")):
            return True
        if fnmatch.fnmatchcase(name, pattern) or name.startswith(pattern + "/"):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatchcase(name, pattern[3:]):
            return True
    return False


@dataclass(frozen=True)
class Chunk:
    path: str
    file_hash: str
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    section: str
    text: str
    chunk_id: str = ""

    def source(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "text"}


@dataclass
class Snapshot:
    snapshot_id: str
    files: list[dict]
    chunks: list[Chunk]
    skipped: list[dict] = field(default_factory=list)


def chunk_text(path: str, text: str, file_hash: str, limit: int) -> list[Chunk]:
    """Headings/paragraphs, Python top-level AST, generic code declaration boundaries.

    Oversize units split at exact character offsets, retaining line spans. No text lost.
    """
    lines = text.splitlines(keepends=True)
    boundaries = {0, len(lines)}
    sections: dict[int, str] = {0: path}
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        try:
            tree = ast.parse(text)
            for node in tree.body:
                start = (
                    min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
                )
                boundaries.add(start)
                sections[start] = getattr(node, "name", path)
        except (SyntaxError, ValueError):
            pass
    fence = False
    for index, line in enumerate(lines):
        if suffix in {".md", ".markdown", ".txt", ".rst"}:
            if line.lstrip().startswith(("```", "~~~")):
                fence = not fence
            if not fence and re.match(r"^#{1,6}\s+", line):
                boundaries.add(index)
                sections[index] = line.strip()
            elif not fence and not line.strip():
                boundaries.add(index + 1)
        elif suffix != ".py" and re.match(
            r"^(?:export\s+)?(?:async\s+)?(?:function|class|def|fn|func|interface|struct)\b", line
        ):
            boundaries.add(index)
            sections[index] = line.strip()[:160]
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    output = []
    section = path
    points = sorted(boundaries)
    for lo, hi in zip(points, points[1:]):
        section = sections.get(lo, section)
        start, end = offsets[lo], offsets[hi]
        while start < end:
            stop = min(end, start + limit)
            if stop < end:
                newline = text.rfind("\n", start, stop)
                if newline > start:
                    stop = newline + 1
            value = text[start:stop]
            if value.strip():
                start_line = text.count("\n", 0, start) + 1
                end_line = start_line + value.rstrip("\n").count("\n")
                key = digest([path, file_hash, start, stop])
                output.append(
                    Chunk(path, file_hash, start_line, end_line, start, stop, section, value, key)
                )
            start = stop
    return output


def scan(input_path: Path, config: Config, exclude_paths: tuple[Path, ...] = ()) -> Snapshot:
    original = input_path.absolute()
    # resolve() alone would silently follow a root link (or a link in an ancestor).
    if any(is_link(p) for p in (original, *original.parents)):
        raise ValueError("input path must not contain symlinks or junctions")
    root = original if original.is_dir() else original.parent
    if not original.exists():
        raise ValueError("input path does not exist")
    files: list[dict] = []
    chunks: list[Chunk] = []
    skipped: list[dict] = []
    total = 0
    excluded = tuple(p.resolve() for p in exclude_paths)

    def reason(path: Path) -> str | None:
        rel = path.relative_to(root).as_posix()
        # Include the selected root itself. Otherwise selecting evaluation/ or
        # node_modules/ directly would hide that excluded component from rel.
        parts = {root.name.casefold(), *(p.casefold() for p in path.relative_to(root).parts)}
        rooted_rel = root.name + "/" + rel
        name = path.name.casefold()
        if is_link(path):
            return "link_or_junction"
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()):
            return "outside_root"
        if any(resolved == p or resolved.is_relative_to(p) for p in excluded):
            return "output_directory"
        if parts & DIRECTORIES:
            return "build_dependency_or_private_directory"
        if (
            name.startswith(".env")
            or name in SECRET_NAMES
            or path.suffix.lower() in SECRET_EXTENSIONS
            or re.search(r"(?:^|[_.-])(?:secrets?|credentials?)(?:[_.-]|$)", name)
        ):
            return "secret_filename"
        if matches(rel, config.evaluation_exclude) or matches(
            rooted_rel, config.evaluation_exclude
        ):
            return "evaluation_artifact"
        if matches(rel, config.exclude) or matches(rooted_rel, config.exclude):
            return "user_exclusion"
        return None

    candidates = [original]
    while candidates:
        path = candidates.pop()
        rel = path.relative_to(root).as_posix()
        try:
            why = reason(path)
            if why:
                skipped.append({"path": rel, "reason": why})
                continue
            if path.is_dir():
                candidates.extend(sorted(path.iterdir(), reverse=True))
                continue
            if not path.is_file():
                skipped.append({"path": rel, "reason": "not_regular_file"})
                continue
            if path.suffix.lower() not in EXTENSIONS and path.name.lower() not in {
                "dockerfile",
                "makefile",
                "cmakelists.txt",
            }:
                skipped.append({"path": rel, "reason": "unsupported_extension"})
                continue
            if path.stat().st_size > config.max_file_bytes:
                skipped.append({"path": rel, "reason": "file_byte_limit"})
                continue
            # Recheck before opening; source roots must remain quiescent during scanning.
            if is_link(path):
                skipped.append({"path": rel, "reason": "link_or_junction"})
                continue
            with path.open("rb") as handle:
                data = handle.read(config.max_file_bytes + 1)
            if len(data) > config.max_file_bytes:
                skipped.append({"path": rel, "reason": "file_byte_limit"})
                continue
            if b"\0" in data:
                skipped.append({"path": rel, "reason": "binary"})
                continue
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                skipped.append({"path": rel, "reason": "not_utf8"})
                continue
            if any(ord(c) < 32 and c not in "\r\n\t\f" for c in text):
                skipped.append({"path": rel, "reason": "binary_control_characters"})
                continue
            if SECRET_CONTENT.search(text):
                skipped.append({"path": rel, "reason": "possible_secret_content"})
                continue
            if len(files) >= config.max_files or total + len(data) > config.max_input_bytes:
                raise ValueError(
                    "input budget exceeded; narrow input or raise max_files/max_input_bytes"
                )
            total += len(data)
            file_hash = hashlib.sha256(data).hexdigest()
            files.append({"path": rel, "file_hash": file_hash, "bytes": len(data)})
            chunks.extend(chunk_text(rel, text, file_hash, config.chunk_chars))
        except OSError:
            skipped.append({"path": rel, "reason": "unreadable"})
    snapshot_id = digest(files)
    return Snapshot(snapshot_id, files, chunks, skipped)
