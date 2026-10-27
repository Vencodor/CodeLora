"""SQLite is the durable source of truth. JSONL files are rebuildable materialized views."""

import json
import os
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .ingest import Snapshot, digest
from .prompts import PROMPT_VERSION, Request
from .providers import Response


def encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@contextmanager
def output_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / ".writer.lock").open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    locked = False
    try:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError:
            raise ValueError("output directory is locked by another writer") from None
        yield
    finally:
        if locked:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(encoded(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Store:
    def __init__(self, output: Path):
        self.output = output
        self.lock = threading.RLock()
        self.db = sqlite3.connect(output / "state.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
          id TEXT PRIMARY KEY, snapshot TEXT NOT NULL, config TEXT NOT NULL,
          identity TEXT NOT NULL, manifest TEXT NOT NULL, created REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS prompts (
          job_id TEXT PRIMARY KEY, prompt_hash TEXT NOT NULL, messages TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
          run_id TEXT NOT NULL, chunk_id TEXT NOT NULL, source TEXT NOT NULL,
          text TEXT NOT NULL, PRIMARY KEY(run_id, chunk_id)
        );
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, run_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
          state TEXT NOT NULL, code TEXT, reason TEXT, records TEXT,
          reservation INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts (
          id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, run_id TEXT NOT NULL,
          state TEXT NOT NULL, code TEXT, max_output_tokens INTEGER NOT NULL,
          request_id TEXT, input_tokens INTEGER, output_tokens INTEGER, usage_kind TEXT,
          content TEXT, finish_reason TEXT, started REAL NOT NULL, finished REAL
        );
        CREATE INDEX IF NOT EXISTS attempts_by_run ON attempts(run_id);
        """)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def setup(
        self, run_id: str, snapshot: Snapshot, config: dict, identity: dict, requests: list[Request]
    ) -> None:
        with self.lock, self.db:
            manifest = {
                "snapshot_id": snapshot.snapshot_id,
                "files": snapshot.files,
                "skipped": snapshot.skipped,
                "chunk_count": len(snapshot.chunks),
                "prompt_version": PROMPT_VERSION,
            }
            self.db.execute(
                "INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?,?)",
                (
                    run_id,
                    snapshot.snapshot_id,
                    encoded(config),
                    encoded(identity),
                    encoded(manifest),
                    time.time(),
                ),
            )
            self.db.execute("INSERT OR REPLACE INTO settings VALUES('active_run',?)", (run_id,))
            for request in requests:
                self.db.execute(
                    "INSERT OR IGNORE INTO chunks VALUES(?,?,?,?)",
                    (
                        run_id,
                        request.chunk.chunk_id,
                        encoded(request.chunk.source()),
                        request.chunk.text,
                    ),
                )
                self.db.execute(
                    "INSERT OR IGNORE INTO jobs VALUES(?,?,?,'pending',NULL,NULL,NULL,0,?)",
                    (request.job_id, run_id, request.chunk.chunk_id, time.time()),
                )
                self.db.execute(
                    "INSERT OR IGNORE INTO prompts VALUES(?,?,?)",
                    (request.job_id, digest(request.messages), encoded(request.messages)),
                )
            # A dispatch marker was committed before sending. A dead process may have sent it.
            self.db.execute(
                "UPDATE jobs SET state='unknown',code='interrupted_after_dispatch' "
                "WHERE state='dispatched'"
            )
            self.db.execute(
                "UPDATE attempts SET state='unknown',code='interrupted_after_dispatch' "
                "WHERE state='dispatched'"
            )

    def job(self, key: str) -> dict:
        with self.lock:
            return dict(self.db.execute("SELECT * FROM jobs WHERE id=?", (key,)).fetchone())

    def claim(self, run_id: str, request: Request) -> tuple[int | None, str]:
        """Reserve conservative output bounds and commit dispatch before calling teacher."""
        config = request.config
        with self.lock, self.db:
            job = self.job(request.job_id)
            if job["state"] not in {"pending", "retryable", "budget"}:
                return None, job["state"]
            attempts = self.db.execute(
                "SELECT COUNT(*) FROM attempts WHERE job_id=?", (request.job_id,)
            ).fetchone()[0]
            if attempts >= config.retries + 1:
                self.db.execute(
                    "UPDATE jobs SET state='failed',code='retry_limit' WHERE id=?",
                    (request.job_id,),
                )
                return None, "failed"
            calls, reserved_tokens = self.db.execute(
                "SELECT COUNT(*),COALESCE(SUM(max_output_tokens),0) FROM attempts WHERE run_id=?",
                (run_id,),
            ).fetchone()
            reserved_records = self.db.execute(
                "SELECT COALESCE(SUM(reservation),0) FROM jobs WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
            additional = 0 if job["reservation"] else config.qa_count + config.explanation_count
            if (
                calls >= config.max_calls
                or reserved_tokens + config.max_output_tokens > config.max_total_output_tokens
                or reserved_records + additional > config.max_records
            ):
                self.db.execute(
                    "UPDATE jobs SET state='budget',code='budget_limit' WHERE id=?",
                    (request.job_id,),
                )
                return None, "budget"
            self.db.execute(
                "UPDATE jobs SET state='dispatched',code=NULL,reservation=? WHERE id=?",
                (job["reservation"] or additional, request.job_id),
            )
            cursor = self.db.execute(
                "INSERT INTO attempts(job_id,run_id,state,max_output_tokens,started) "
                "VALUES(?,?,'dispatched',?,?)",
                (request.job_id, run_id, config.max_output_tokens, time.time()),
            )
            return cursor.lastrowid, "dispatched"

    def fail(self, request: Request, attempt: int, state: str, code: str) -> None:
        with self.lock, self.db:
            self.db.execute(
                "UPDATE attempts SET state=?,code=?,finished=? WHERE id=?",
                (state, code, time.time(), attempt),
            )
            self.db.execute(
                "UPDATE jobs SET state=?,code=? WHERE id=?", (state, code, request.job_id)
            )

    def save(
        self,
        request: Request,
        attempt: int,
        response: Response,
        records: list[dict],
        reason: str,
        invalid: str | None = None,
    ) -> None:
        # Raw completion, usage, status and records commit together. No replay on export failure.
        state = "invalid" if invalid else "saved"
        with self.lock, self.db:
            self.db.execute(
                "UPDATE attempts SET state=?,code=?,request_id=?,input_tokens=?,output_tokens=?,"
                "usage_kind=?,content=?,finish_reason=?,finished=? WHERE id=?",
                (
                    state,
                    invalid,
                    response.request_id,
                    response.input_tokens,
                    response.output_tokens,
                    response.usage_kind,
                    response.content,
                    response.finish_reason,
                    time.time(),
                    attempt,
                ),
            )
            self.db.execute(
                "UPDATE jobs SET state=?,code=?,reason=?,records=? WHERE id=?",
                (state, invalid, reason, encoded(records), request.job_id),
            )

    def report(self, run_id: str) -> dict:
        with self.lock:
            states = dict(
                self.db.execute(
                    "SELECT state,COUNT(*) FROM jobs WHERE run_id=? GROUP BY state", (run_id,)
                ).fetchall()
            )
            rows = self.db.execute("SELECT * FROM attempts WHERE run_id=?", (run_id,)).fetchall()
            actual = [r for r in rows if r["usage_kind"] == "provider_reported"]
            records = [
                record
                for row in self.db.execute(
                    "SELECT records FROM jobs WHERE run_id=? AND state='saved'",
                    (run_id,),
                )
                for record in json.loads(row["records"] or "[]")
            ]
            unique = {
                digest([r["type"], r["question"], r.get("answer", r.get("text"))]) for r in records
            }
            return {
                "run_id": run_id,
                "states": states,
                "attempts_total": len(rows),
                "generated_records_raw": len(records),
                "unique_records": len(unique),
                "qa_records_raw": sum(r["type"] == "qa" for r in records),
                "explanation_records_raw": sum(r["type"] == "explanation" for r in records),
                "reserved_output_token_upper_bound": sum(r["max_output_tokens"] for r in rows),
                "provider_reported_input_tokens": sum(r["input_tokens"] or 0 for r in actual),
                "provider_reported_output_tokens": sum(r["output_tokens"] or 0 for r in actual),
                "attempts_with_complete_provider_usage": sum(
                    r["input_tokens"] is not None and r["output_tokens"] is not None for r in actual
                ),
                "attempts_without_complete_provider_usage": sum(
                    r["usage_kind"] != "provider_reported"
                    or r["input_tokens"] is None
                    or r["output_tokens"] is None
                    for r in rows
                ),
                "usage_note": "Provider-reported sums only; missing/unknown usage is not zero billing. "
                "Mock responses do not report real API tokens.",
            }

    def run(self, run_id: str | None = None) -> dict:
        with self.lock:
            if run_id:
                row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            else:
                row = self.db.execute(
                    "SELECT * FROM runs WHERE id=(SELECT value FROM settings "
                    "WHERE key='active_run')"
                ).fetchone()
            if row is None:
                raise ValueError("no matching saved run")
            return dict(row)

    def rows(self, run_id: str) -> tuple[list[dict], list[dict]]:
        with self.lock:
            jobs = [
                dict(r)
                for r in self.db.execute(
                    "SELECT jobs.*,chunks.source,chunks.text,prompts.prompt_hash FROM jobs JOIN chunks "
                    "ON jobs.run_id=chunks.run_id AND jobs.chunk_id=chunks.chunk_id "
                    "JOIN prompts ON jobs.id=prompts.job_id "
                    "WHERE jobs.run_id=? ORDER BY jobs.created,jobs.id",
                    (run_id,),
                )
            ]
            attempts = [
                dict(r)
                for r in self.db.execute(
                    "SELECT * FROM attempts WHERE run_id=? ORDER BY id",
                    (run_id,),
                )
            ]
            return jobs, attempts
