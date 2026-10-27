import argparse
import json
import sqlite3
import sys
from dataclasses import fields
from pathlib import Path

from .config import Config
from .export import export_sft, sample
from .ingest import scan
from .pipeline import generate, plan
from .providers import META_ENDPOINT, META_MODEL, MetaTeacher, MockTeacher, Teacher, model_api_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local source-grounded dataset synthesis")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "generate"):
        sub = commands.add_parser(name)
        sub.add_argument("input", type=Path, help="UTF-8 file or directory")
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--config", type=Path, help="JSON Config fields; no credentials")
        sub.add_argument("--project")
        sub.add_argument("--provider", choices=("mock", "meta"), default="mock")
        sub.add_argument("--model", help="Meta model ID (default: verified Contributor ID)")
        sub.add_argument("--exclude", action="append", default=[])
        sub.add_argument("--evaluation-exclude", action="append", default=[])
        for flag in (
            "qa-count",
            "explanation-count",
            "qa-max-chars",
            "explanation-max-chars",
            "question-max-chars",
            "chunk-chars",
            "max-file-bytes",
            "max-input-bytes",
            "max-files",
            "max-calls",
            "max-records",
            "max-output-tokens",
            "max-total-output-tokens",
            "retries",
            "concurrency",
        ):
            sub.add_argument("--" + flag, type=int)
        sub.add_argument("--timeout", type=float)
        sub.add_argument("--retry-delay", type=float)
        sub.add_argument("--temperature", type=float)
        if name == "generate":
            sub.add_argument(
                "--env-file",
                type=Path,
                help="Local key file for approved Meta calls (default: .env in current directory)",
            )
            sub.add_argument(
                "--allow-external-api",
                action="store_true",
                help="Explicit approval of external source transfer, billing and training use",
            )
    sub = commands.add_parser("export", help="Separate transformation; never calls a teacher")
    sub.add_argument("output", type=Path)
    sub.add_argument("--destination", type=Path, required=True)
    sub.add_argument("--format", choices=("messages", "instruction"), default="messages")
    sub.add_argument("--include-explanations", action="store_true")
    sub.add_argument("--run-id")
    sub = commands.add_parser(
        "sample", help="Local human review: generated record beside saved source"
    )
    sub.add_argument("output", type=Path)
    sub.add_argument("--count", type=int, default=3)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--run-id")
    return parser


def load_config(args: argparse.Namespace) -> Config:
    data = {}
    if args.config:
        data = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("config must be a JSON object")
        if set(data) - {f.name for f in fields(Config)}:
            raise ValueError("unknown configuration field")
    for item in fields(Config):
        value = getattr(args, item.name, None)
        if item.name not in {"exclude", "evaluation_exclude"} and value is not None:
            data[item.name] = value
    for key in ("exclude", "evaluation_exclude"):
        initial = data.get(key, getattr(Config(), key))
        if not isinstance(initial, (tuple, list)) or not all(isinstance(p, str) for p in initial):
            raise ValueError("exclusions must be arrays of path patterns")
        data[key] = tuple(initial) + tuple(getattr(args, key))
    return Config(**data)


def emit(value: dict, stream=None) -> None:
    # ASCII escaping also works on Windows terminals with legacy encodings.
    print(
        json.dumps(value, ensure_ascii=True, sort_keys=True), file=stream or sys.stdout, flush=True
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "export":
            count = export_sft(
                args.output, args.destination, args.format, args.include_explanations, args.run_id
            )
            emit({"exported_records": count, "format": args.format})
            return 0
        if args.command == "sample":
            for row in sample(args.output, args.count, args.seed, args.run_id):
                emit(row)
            return 0
        config = load_config(args)
        source, output = args.input.resolve(), args.output.resolve()
        if source == output or source.is_relative_to(output):
            raise ValueError("output must not equal or contain the input path")
        snapshot = scan(args.input, config, exclude_paths=(args.output,))
        identity = (
            MockTeacher().identity()
            if args.provider == "mock"
            else {
                "provider": "meta",
                "model": args.model or META_MODEL,
                "endpoint": META_ENDPOINT,
                "contract": "chat-completions-v1",
            }
        )
        if args.command == "dry-run":
            emit(plan(snapshot, config, identity))
            return 0
        if args.provider == "meta":
            if not args.allow_external_api:
                raise ValueError(
                    "real generation requires --allow-external-api after separate approval"
                )
            key = model_api_key(args.env_file or Path.cwd() / ".env")
            teacher: Teacher = MetaTeacher(api_key=key, model=args.model or META_MODEL)
        else:
            teacher = MockTeacher()
        report = generate(
            snapshot, args.output, config, teacher, progress=lambda row: emit(row, sys.stderr)
        )
        emit(report)
        states = report["states"]
        if any(states.get(state, 0) for state in ("failed", "unknown", "invalid")):
            return 2
        if states.get("budget", 0):
            return 3
        return 0
    except KeyboardInterrupt:
        emit(
            {"error": "interrupted; resume preserves saved results and holds unknown dispatches"},
            sys.stderr,
        )
        return 130
    except (ValueError, TypeError, OSError, sqlite3.Error):
        # Do not echo arbitrary config content, provider body, key, or exception text.
        emit(
            {
                "error": "invalid configuration/input or local I/O failure; check paths, limits, "
                "output lock, and real-API approval/environment settings"
            },
            sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
