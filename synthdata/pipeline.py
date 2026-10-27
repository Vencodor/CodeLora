import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .config import Config
from .ingest import Snapshot, digest
from .prompts import PROMPT_VERSION, InvalidOutput, Request, make_request, validate
from .providers import Rejected, SafeRetry, Teacher, Uncertain
from .storage import Store, atomic_jsonl, output_lock


def plan(snapshot: Snapshot, config: Config, identity: dict) -> dict:
    requests = [
        make_request(chunk, snapshot.snapshot_id, config, identity) for chunk in snapshot.chunks
    ]
    admitted = min(
        len(requests),
        config.max_calls,
        config.max_records // (config.qa_count + config.explanation_count),
        config.max_total_output_tokens // config.max_output_tokens,
    )
    return {
        "snapshot_id": snapshot.snapshot_id,
        "files": snapshot.files,
        "chunks": [chunk.source() for chunk in snapshot.chunks],
        "skipped": snapshot.skipped,
        "file_count": len(snapshot.files),
        "chunk_count": len(requests),
        "estimated_initial_calls_within_budgets": admitted,
        "call_attempt_upper_bound": min(
            admitted * (config.retries + 1),
            config.max_calls,
            config.max_total_output_tokens // config.max_output_tokens,
        ),
        "estimated_input_tokens_initial": sum(
            max(1, sum(len(m["content"]) for m in r.messages) // 4) for r in requests[:admitted]
        ),
        "estimate_note": "Characters/4 heuristic, not a tokenizer or price quote; excludes retries. "
        "Dry-run does not inspect saved state or call the teacher.",
        "requested_record_upper_bound": admitted * (config.qa_count + config.explanation_count),
    }


def generate(
    snapshot: Snapshot,
    output: Path,
    config: Config,
    teacher: Teacher,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    identity = teacher.identity()
    base_requests = [
        make_request(c, snapshot.snapshot_id, config, identity) for c in snapshot.chunks
    ]
    run_id = digest(
        [
            snapshot.snapshot_id,
            config.generation(),
            identity,
            PROMPT_VERSION,
            [r.messages for r in base_requests],
        ]
    )
    requests = [
        make_request(c, snapshot.snapshot_id, config, identity, run_id) for c in snapshot.chunks
    ]
    with output_lock(output):
        store = Store(output)
        try:
            store.setup(run_id, snapshot, config.to_dict(), identity, requests)

            def process(request: Request) -> None:
                while True:
                    attempt, state = store.claim(run_id, request)
                    if attempt is None:
                        if progress:
                            progress(
                                {
                                    "job_id": request.job_id,
                                    "state": state,
                                    "reused": state == "saved",
                                }
                            )
                        return
                    try:
                        response = teacher.generate(request)
                    except SafeRetry:
                        # Only the error CLASS affects retry safety. Never persist arbitrary messages.
                        store.fail(request, attempt, "retryable", "known_not_submitted")
                        used = sum(a["job_id"] == request.job_id for a in store.rows(run_id)[1])
                        if used <= config.retries:
                            time.sleep(min(30.0, config.retry_delay * 2 ** (used - 1)))
                        continue
                    except Rejected:
                        store.fail(request, attempt, "failed", "provider_rejected")
                    except Uncertain:
                        store.fail(request, attempt, "unknown", "provider_outcome_unknown")
                    except Exception:
                        # A custom adapter crash may occur AFTER send. Fail closed, no exception text.
                        store.fail(request, attempt, "unknown", "adapter_outcome_unknown")
                    else:
                        try:
                            records, reason = validate(
                                response.content, response.finish_reason, request
                            )
                        except InvalidOutput as exc:
                            store.save(request, attempt, response, [], "", invalid=str(exc))
                        else:
                            store.save(request, attempt, response, records, reason)
                    if progress:
                        job = store.job(request.job_id)
                        progress(
                            {"job_id": request.job_id, "state": job["state"], "code": job["code"]}
                        )
                    return

            with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
                # Consume results so storage/worker failures cannot silently disappear.
                for _ in pool.map(process, requests):
                    pass
            materialize(store, run_id)
            report = store.report(run_id)
            report["execution_config"] = config.to_dict()
            atomic_jsonl(output / "report.jsonl", [report])
            return report
        finally:
            store.close()


def materialize(store: Store, run_id: str) -> list[dict]:
    run = store.run(run_id)
    prompt_version = json.loads(run["manifest"])["prompt_version"]
    jobs, attempts = store.rows(run_id)
    records = []
    seen: dict[str, str] = {}
    for job in jobs:
        for index, record in enumerate(json.loads(job["records"] or "[]")):
            row = {
                **record,
                "record_id": digest([job["id"], index]),
                "job_id": job["id"],
                "snapshot_id": run["snapshot"],
                "source": json.loads(job["source"]),
                "teacher": json.loads(run["identity"]),
                "prompt_version": prompt_version,
                "prompt_hash": job["prompt_hash"],
            }
            answer = row.get("answer", row.get("text"))
            key = digest([row["type"], row["question"], answer])
            row["duplicate_of"] = seen.get(key)
            seen.setdefault(key, row["record_id"])
            records.append(row)
    raw = store.output / "raw"
    atomic_jsonl(raw / "qa.jsonl", (r for r in records if r["type"] == "qa"))
    atomic_jsonl(raw / "explanations.jsonl", (r for r in records if r["type"] == "explanation"))
    atomic_jsonl(raw / "responses.jsonl", ({**r, "snapshot_id": run["snapshot"]} for r in attempts))
    atomic_jsonl(
        raw / "outcomes.jsonl",
        ({k: v for k, v in job.items() if k not in {"records", "source", "text"}} for job in jobs),
    )
    atomic_jsonl(
        store.output / "manifest.jsonl",
        [
            {
                **json.loads(run["manifest"]),
                "run_id": run_id,
                "initial_config": json.loads(run["config"]),
                "teacher": json.loads(run["identity"]),
                "prompt_version": prompt_version,
            }
        ],
    )
    return records
