import random
from pathlib import Path

from .pipeline import materialize
from .storage import Store, atomic_jsonl, output_lock


def export_sft(
    output: Path,
    destination: Path,
    format: str = "messages",
    include_explanations: bool = False,
    run_id: str | None = None,
) -> int:
    if format not in {"messages", "instruction"}:
        raise ValueError("format must be messages or instruction")
    protected = {"state.sqlite3", ".writer.lock", "manifest.jsonl", "report.jsonl"}
    resolved = destination.resolve()
    if (
        destination.name in protected
        or resolved.is_relative_to((output / "raw").resolve())
        or destination.suffix.lower() != ".jsonl"
    ):
        raise ValueError("export destination must be a separate .jsonl file")
    with output_lock(output):
        store = Store(output)
        try:
            run = store.run(run_id)
            records = materialize(store, run["id"])
            rows = []
            for record in records:
                if record["duplicate_of"] or (
                    record["type"] == "explanation" and not include_explanations
                ):
                    continue
                answer = record.get("answer", record.get("text"))
                metadata = {k: record[k] for k in ("record_id", "snapshot_id", "source", "type")}
                row: dict
                if format == "messages":
                    row = {
                        "messages": [
                            {"role": "user", "content": record["question"]},
                            {"role": "assistant", "content": answer},
                        ],
                        "metadata": metadata,
                    }
                else:
                    row = {
                        "instruction": record["question"],
                        "response": answer,
                        "metadata": metadata,
                    }
                rows.append(row)
            atomic_jsonl(destination, rows)
            return len(rows)
        finally:
            store.close()


def sample(output: Path, count: int = 3, seed: int = 0, run_id: str | None = None) -> list[dict]:
    if count < 0:
        raise ValueError("sample count must be nonnegative")
    with output_lock(output):
        store = Store(output)
        try:
            run = store.run(run_id)
            jobs, _ = store.rows(run["id"])
            records = materialize(store, run["id"])
            sources = {job["id"]: job["text"] for job in jobs}
            selected = random.Random(seed).sample(records, min(count, len(records)))
            return [
                {"record": record, "original_source_text": sources[record["job_id"]]}
                for record in selected
            ]
        finally:
            store.close()
