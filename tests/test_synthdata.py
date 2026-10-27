import io
import json
import os
import shutil
import socket
import stat
import tempfile
import threading
import time
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from synthdata.cli import main
from synthdata.config import Config
from synthdata.export import export_sft, sample
from synthdata.ingest import chunk_text, scan
from synthdata.pipeline import generate, plan
from synthdata.prompts import SYSTEM, make_request, validate
from synthdata.providers import (
    META_ENDPOINT,
    META_MODEL,
    MetaTeacher,
    MockTeacher,
    Rejected,
    Response,
    SafeRetry,
    Uncertain,
    model_api_key,
)
from synthdata.storage import Store, output_lock

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_project"


def workspace_temp():
    workspace = Path(__file__).resolve().parents[1]
    directory = workspace / ".test-tmp"
    directory.mkdir(exist_ok=True)
    if not directory.resolve().is_relative_to(workspace):
        raise ValueError("test temporary directory must stay inside workspace")
    return tempfile.TemporaryDirectory(dir=directory)


def answer(question="What mode does settings.txt specify?", text="offline", **kwargs):
    obj = {"qa": [{"question": question, "answer": text}], "explanations": [], "zero_reason": ""}
    return Response(json.dumps(obj), **kwargs)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ScriptTeacher:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []
        self.lock = threading.Lock()

    def identity(self):
        return {"provider": "script-mock-api", "model": "fixture-v1"}

    def generate(self, request):
        with self.lock:
            self.calls.append(request)
            value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = workspace_temp()
        self.root = Path(self.tmp.name)
        self.src = self.root / "source with spaces"
        self.src.mkdir()
        (self.src / "settings.txt").write_text("mode=offline\n", encoding="utf-8", newline="\n")
        self.out = self.root / "output with spaces"
        self.config = Config(
            project="Lantern", qa_count=1, explanation_count=0, retry_delay=0, concurrency=1
        )

    def tearDown(self):
        self.tmp.cleanup()

    def run_teacher(self, teacher, config=None):
        config = config or self.config
        snapshot = scan(self.src, config, (self.out,))
        return generate(snapshot, self.out, config, teacher)

    def test_fixture_end_to_end_and_independent_sft(self):
        self.assertTrue(self.src.resolve().is_relative_to(self.root.resolve()))
        shutil.rmtree(self.src)
        shutil.copytree(FIXTURE, self.src)
        config = replace(self.config, qa_count=2, explanation_count=1)
        snapshot = scan(self.src, config)
        self.assertEqual(len(snapshot.files), 3)
        self.assertTrue(any(s["reason"] == "evaluation_artifact" for s in snapshot.skipped))
        teacher = MockTeacher()
        result = generate(snapshot, self.out, config, teacher)
        self.assertGreater(result["states"]["saved"], 1)
        self.assertEqual(result["attempts_total"], len(snapshot.chunks))
        qa = read_jsonl(self.out / "raw" / "qa.jsonl")
        explanations = read_jsonl(self.out / "raw" / "explanations.jsonl")
        self.assertTrue(qa and explanations)
        for record in qa + explanations:
            self.assertEqual(record["snapshot_id"], snapshot.snapshot_id)
            self.assertIn(record["source"]["path"], {f["path"] for f in snapshot.files})
            self.assertIn("Project: Lantern; file:", record["question"])
            self.assertNotIn("source_text", record["question"])
            self.assertNotIn("HELD_OUT", record["question"])
        target = self.out / "sft.messages.jsonl"
        count = export_sft(self.out, target)
        self.assertEqual(count, len(qa))
        original_chunks = {c.text.strip() for c in snapshot.chunks}
        for row in read_jsonl(target):
            student = row["messages"][0]["content"]
            self.assertEqual(row["messages"][0]["role"], "user")
            self.assertNotIn("original_source_text", row)
            self.assertFalse(any(text and text in student for text in original_chunks))
        target2 = self.out / "sft.instruction.jsonl"
        self.assertEqual(export_sft(self.out, target2, "instruction", True), len(qa + explanations))
        self.assertIn("response", read_jsonl(target2)[0])
        reviewed = sample(self.out, 2)
        self.assertEqual(len(reviewed), 2)
        self.assertTrue(all("original_source_text" in item for item in reviewed))
        self.assertEqual(result["attempts_with_complete_provider_usage"], 0)

    def test_teacher_sees_source_but_no_evaluation(self):
        (self.src / "eval").mkdir()
        (self.src / "eval" / "answers.txt").write_text("HELD_OUT", encoding="utf-8")
        teacher = ScriptTeacher([answer()])
        self.run_teacher(teacher)
        payload = json.loads(teacher.calls[0].messages[1]["content"])
        self.assertEqual(payload["source_text"], "mode=offline\n")
        self.assertNotIn("HELD_OUT", json.dumps(teacher.calls[0].messages))
        for phrase in ("historical", "future plans", "Do not invent", "untrusted", "upper bounds"):
            self.assertIn(phrase, SYSTEM)

    def test_short_fact_is_not_discarded(self):
        (self.src / "settings.txt").write_text("x=1", encoding="utf-8")
        teacher = ScriptTeacher([answer(text="1")])
        result = self.run_teacher(teacher)
        self.assertEqual(result["states"], {"saved": 1})
        self.assertEqual(read_jsonl(self.out / "raw" / "qa.jsonl")[0]["answer"], "1")

    def test_normal_zero_and_reuse(self):
        zero = Response(
            json.dumps({"qa": [], "explanations": [], "zero_reason": "No behavior specified."})
        )
        teacher = ScriptTeacher([zero])
        first = self.run_teacher(teacher)
        second = self.run_teacher(teacher)
        self.assertEqual(first["states"], {"saved": 1})
        self.assertEqual(second["attempts_total"], 1)
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(read_jsonl(self.out / "raw" / "qa.jsonl"), [])
        self.assertEqual(
            read_jsonl(self.out / "raw" / "outcomes.jsonl")[0]["reason"], "No behavior specified."
        )

    def test_invalid_outputs_are_preserved_without_retry_or_refill(self):
        cases = [
            (Response(""), "empty_output"),
            (Response('{"qa":['), "malformed_json"),
            (Response("{}", "length"), "truncated_or_nonfinal"),
            (Response("{}"), "invalid_schema"),
            (Response('{"qa":[],"explanations":[],"zero_reason":""}'), "inconsistent_zero_reason"),
            (answer(question="What does the above code do?"), "dependent_or_code_question"),
            (answer(question="What does mode=offline mean?"), "source_in_question"),
            (answer(text=""), "empty_item"),
        ]
        for index, (response, code) in enumerate(cases):
            with self.subTest(code=code):
                self.out = self.root / f"invalid-{index}"
                teacher = ScriptTeacher([response])
                result = self.run_teacher(teacher)
                self.assertEqual(result["states"], {"invalid": 1})
                self.assertEqual(
                    read_jsonl(self.out / "raw" / "responses.jsonl")[0]["content"], response.content
                )
                self.assertEqual(read_jsonl(self.out / "raw" / "outcomes.jsonl")[0]["code"], code)
                self.run_teacher(teacher)
                self.assertEqual(len(teacher.calls), 1)

    def test_duplicates_preserved_raw_only_exact_removed_from_export(self):
        item = {"question": "What mode does settings.txt specify?", "answer": "offline"}
        variant = {
            "question": "Which operating mode is specified in settings.txt?",
            "answer": "offline",
        }
        response = Response(
            json.dumps({"qa": [item, item, variant], "explanations": [], "zero_reason": ""})
        )
        teacher = ScriptTeacher([response])
        self.run_teacher(teacher, replace(self.config, qa_count=3))
        raw = read_jsonl(self.out / "raw" / "qa.jsonl")
        self.assertEqual(len(raw), 3)
        self.assertIsNotNone(raw[1]["duplicate_of"])
        self.assertIsNone(raw[2]["duplicate_of"])
        self.assertEqual(export_sft(self.out, self.out / "sft.jsonl"), 2)

    def test_safe_retry_upper_bound_and_resume(self):
        teacher = ScriptTeacher([SafeRetry("private-message") for _ in range(4)])
        result = self.run_teacher(teacher)
        self.assertEqual(len(teacher.calls), 3)
        self.assertEqual(result["states"], {"failed": 1})
        self.run_teacher(teacher)
        self.assertEqual(len(teacher.calls), 3)
        self.assertNotIn("private-message", (self.out / "raw" / "responses.jsonl").read_text())

    def test_safe_retry_then_success_and_reported_tokens(self):
        teacher = ScriptTeacher([SafeRetry("ignored"), answer(input_tokens=123, output_tokens=45)])
        result = self.run_teacher(teacher)
        self.assertEqual(result["attempts_total"], 2)
        self.assertEqual(result["states"], {"saved": 1})
        self.assertEqual(result["provider_reported_input_tokens"], 123)
        self.assertEqual(result["provider_reported_output_tokens"], 45)
        self.assertEqual(result["attempts_without_complete_provider_usage"], 1)

    def test_uncertain_and_arbitrary_adapter_failures_never_replayed(self):
        for index, exc in enumerate([Uncertain("SECRET_BODY"), RuntimeError("SECRET_BODY")]):
            self.out = self.root / f"unknown-{index}"
            teacher = ScriptTeacher([exc])
            result = self.run_teacher(teacher)
            self.assertEqual(result["states"], {"unknown": 1})
            self.run_teacher(teacher)
            self.assertEqual(len(teacher.calls), 1)
            self.assertNotIn("SECRET_BODY", (self.out / "raw" / "responses.jsonl").read_text())

    def test_rejection_is_terminal_and_separate_from_unknown(self):
        teacher = ScriptTeacher([Rejected("SECRET_BODY")])
        result = self.run_teacher(teacher)
        self.assertEqual(result["states"], {"failed": 1})
        self.assertEqual(len(teacher.calls), 1)

    def test_crash_between_response_and_save_is_unknown_on_resume(self):
        teacher = ScriptTeacher([answer()])
        with patch.object(Store, "save", side_effect=OSError("simulated_disk_failure")):
            with self.assertRaises(OSError):
                self.run_teacher(teacher)
        result = self.run_teacher(teacher)
        self.assertEqual(result["states"], {"unknown": 1})
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(result["attempts_total"], 1)

    def test_crash_after_dispatch_before_send_is_conservatively_unknown(self):
        teacher = ScriptTeacher([KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.run_teacher(teacher)
        result = self.run_teacher(teacher)
        self.assertEqual(result["states"], {"unknown": 1})
        self.assertEqual(len(teacher.calls), 1)

    def test_jsonl_export_failure_after_commit_does_not_regenerate(self):
        teacher = ScriptTeacher([answer()])
        with patch("synthdata.pipeline.atomic_jsonl", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.run_teacher(teacher)
        result = self.run_teacher(teacher)
        self.assertEqual(result["states"], {"saved": 1})
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(len(read_jsonl(self.out / "raw" / "qa.jsonl")), 1)

    def test_budgeted_resume_and_success_reuse_when_operating_limits_change(self):
        (self.src / "second.txt").write_text("limit=8\n", encoding="utf-8")
        teacher = ScriptTeacher([answer(), answer()])
        first = self.run_teacher(teacher, replace(self.config, max_calls=1))
        self.assertEqual(first["states"], {"budget": 1, "saved": 1})
        second = self.run_teacher(teacher, replace(self.config, max_calls=2))
        self.assertEqual(second["states"], {"saved": 2})
        self.assertEqual(second["attempts_total"], 2)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(len(teacher.calls), 2)

    def test_token_and_record_budgets_respected(self):
        (self.src / "second.txt").write_text("limit=8\n", encoding="utf-8")
        for index, changed in enumerate(
            [
                replace(self.config, max_total_output_tokens=self.config.max_output_tokens),
                replace(self.config, max_records=1),
            ]
        ):
            self.out = self.root / f"cap-{index}"
            teacher = ScriptTeacher([answer()])
            result = self.run_teacher(teacher, changed)
            self.assertEqual(result["attempts_total"], 1)
            self.assertEqual(result["states"], {"budget": 1, "saved": 1})

    def test_changed_source_or_prompt_configuration_not_reused(self):
        teacher = ScriptTeacher([answer(), answer(), answer()])
        first = self.run_teacher(teacher)
        (self.src / "settings.txt").write_text("mode=online\n", encoding="utf-8")
        second = self.run_teacher(teacher)
        third = self.run_teacher(teacher, replace(self.config, qa_max_chars=99))
        self.assertEqual(len({first["run_id"], second["run_id"], third["run_id"]}), 3)
        self.assertEqual(len(teacher.calls), 3)

    def test_default_export_selects_most_recently_resumed_run(self):
        teacher = ScriptTeacher([answer(text="first"), answer(text="second")])
        first = self.run_teacher(teacher)
        self.run_teacher(teacher, replace(self.config, qa_max_chars=100))
        self.run_teacher(teacher)
        target = self.out / "selected.jsonl"
        export_sft(self.out, target, "instruction")
        self.assertEqual(read_jsonl(target)[0]["response"], "first")
        self.assertEqual(len(teacher.calls), 2)
        self.assertEqual(read_jsonl(self.out / "manifest.jsonl")[0]["run_id"], first["run_id"])

    def test_full_prompt_change_invalidates_run_even_if_version_is_unchanged(self):
        teacher = ScriptTeacher([answer(), answer()])
        first = self.run_teacher(teacher)
        with patch("synthdata.prompts.SYSTEM", SYSTEM + " Extra grounded constraint."):
            second = self.run_teacher(teacher)
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(len(teacher.calls), 2)

    def test_changed_chunk_plan_keeps_all_jobs(self):
        (self.src / "second.txt").write_text(
            "a=" + "x" * 100 + "\n\nb=" + "y" * 100, encoding="utf-8"
        )
        first = self.run_teacher(MockTeacher(), replace(self.config, chunk_chars=500))
        config = replace(self.config, chunk_chars=80)
        snapshot = scan(self.src, config)
        second = self.run_teacher(MockTeacher(), config)
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(second["states"]["saved"], len(snapshot.chunks))

    def test_concurrency_cap(self):
        for i in range(5):
            (self.src / f"f{i}.txt").write_text(f"limit={i}", encoding="utf-8")

        class ConcurrentTeacher(MockTeacher):
            def __init__(self):
                self.active = 0
                self.peak = 0
                self.lock = threading.Lock()

            def generate(inner, request):
                with inner.lock:
                    inner.active += 1
                    inner.peak = max(inner.peak, inner.active)
                time.sleep(0.02)
                with inner.lock:
                    inner.active -= 1
                return super().generate(request)

        teacher = ConcurrentTeacher()
        self.run_teacher(teacher, replace(self.config, concurrency=2))
        self.assertEqual(teacher.peak, 2)

    def test_sample_uses_saved_source_after_original_changes(self):
        self.run_teacher(ScriptTeacher([answer()]))
        (self.src / "settings.txt").write_text("changed", encoding="utf-8")
        self.assertEqual(sample(self.out, 1)[0]["original_source_text"], "mode=offline\n")

    def test_writer_lock_released_and_protects_second_writer(self):
        with output_lock(self.out):
            with self.assertRaises(ValueError):
                with output_lock(self.out):
                    self.fail("second writer acquired lock")
        self.run_teacher(MockTeacher())

    def test_export_cannot_overwrite_authoritative_storage(self):
        self.run_teacher(MockTeacher())
        for path in [self.out / "state.sqlite3", self.out / "raw" / "qa.jsonl"]:
            with self.assertRaises(ValueError):
                export_sft(self.out, path)

    def test_dry_run_never_constructs_real_teacher_or_reads_key(self):
        stdout = io.StringIO()
        with (
            patch("synthdata.cli.MetaTeacher", side_effect=AssertionError("network")),
            patch("synthdata.cli.model_api_key", side_effect=AssertionError("key read")),
            redirect_stdout(stdout),
        ):
            status = main(
                ["dry-run", str(self.src), "--output", str(self.out), "--provider", "meta"]
            )
        self.assertEqual(status, 0)
        self.assertFalse(self.out.exists())
        self.assertEqual(json.loads(stdout.getvalue())["estimated_initial_calls_within_budgets"], 1)

    def test_real_generation_requires_approval(self):
        with (
            patch("synthdata.cli.MetaTeacher", side_effect=AssertionError("network")),
            patch("synthdata.cli.model_api_key", side_effect=AssertionError("key read")),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                main(["generate", str(self.src), "--output", str(self.out), "--provider", "meta"]),
                1,
            )

    def test_cli_with_windows_style_spaces_mock_export_sample(self):
        with (
            patch("synthdata.cli.model_api_key", side_effect=AssertionError("key read")),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                main(
                    [
                        "generate",
                        str(self.src),
                        "--output",
                        str(self.out),
                        "--project",
                        "Lantern",
                        "--qa-count",
                        "1",
                        "--explanation-count",
                        "0",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(["export", str(self.out), "--destination", str(self.out / "data.jsonl")]), 0
            )
            self.assertEqual(main(["sample", str(self.out), "--count", "1"]), 0)

    def test_approved_meta_cli_loads_default_or_explicit_fake_env_without_logging_key(self):
        for index, explicit in enumerate((False, True)):
            with self.subTest(explicit=explicit):
                env_file = self.root / ("fake credentials.env" if explicit else ".env")
                env_file.write_text("MODEL_API_KEY=fake-test-key-only\n", encoding="utf-8")
                stdout, stderr = io.StringIO(), io.StringIO()
                out = self.root / f"fake-meta-{index}"
                args = [
                    "generate",
                    str(self.src),
                    "--output",
                    str(out),
                    "--provider",
                    "meta",
                    "--allow-external-api",
                    "--qa-count",
                    "1",
                    "--explanation-count",
                    "0",
                ]
                if explicit:
                    args.extend(["--env-file", str(env_file)])
                with (
                    patch("synthdata.providers.os.environ", {}),
                    patch("synthdata.cli.Path.cwd", return_value=self.root),
                    patch("synthdata.cli.MetaTeacher", return_value=MockTeacher()) as constructor,
                    redirect_stdout(stdout),
                    redirect_stderr(stderr),
                ):
                    self.assertEqual(main(args), 0)
                constructor.assert_called_once_with(api_key="fake-test-key-only", model=META_MODEL)
                self.assertNotIn("fake-test-key-only", stdout.getvalue() + stderr.getvalue())
                for path in out.rglob("*.jsonl"):
                    self.assertNotIn("fake-test-key-only", path.read_text(encoding="utf-8"))


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = workspace_temp()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def test_defaults_and_configurable_evaluation_exclusions(self):
        cases = {
            "doc.md": "Knowledge that is evaluated can be learned.",
            "tiny.txt": "x=1",
            ".env": "sensitive",
            "secret-config.json": "sensitive",
            "node_modules/a.js": "dependency",
            "build/a.py": "build",
            ".aws/config": "sensitive",
            "eval/q.txt": "HELD_OUT",
            "review_annotations/r.md": "HELD_OUT",
            "hidden-assessment/questions.txt": "HELD_OUT",
            "notes/private.md": "excluded",
            "ordinary.txt": "api_key=sk-" + "a" * 25,
        }
        for path, text in cases.items():
            self.write(path, text)
        (self.root / "binary.py").write_bytes(b"print\0binary")
        (self.root / "bad.txt").write_bytes(b"\xff\xfe")
        config = replace(
            Config(),
            exclude=("notes/**",),
            evaluation_exclude=Config().evaluation_exclude + ("hidden-assessment/**",),
        )
        snapshot = scan(self.root, config)
        self.assertEqual({f["path"] for f in snapshot.files}, {"doc.md", "tiny.txt"})
        self.assertEqual(len(snapshot.chunks), 2)
        self.assertIn("possible_secret_content", {r["reason"] for r in snapshot.skipped})

    def test_snapshot_hash_stable_and_changes_with_file_content(self):
        self.write("a.txt", "a=1")
        first = scan(self.root, Config())
        self.assertEqual(first.snapshot_id, scan(self.root, Config()).snapshot_id)
        self.write("eval/answers.txt", "not part of snapshot")
        self.assertEqual(first.snapshot_id, scan(self.root, Config()).snapshot_id)
        self.write("a.txt", "a=2")
        self.assertNotEqual(first.snapshot_id, scan(self.root, Config()).snapshot_id)

    def test_direct_input_root_or_file_cannot_bypass_default_exclusions(self):
        for directory, filename, why in [
            ("evaluation", "answers.txt", "evaluation_artifact"),
            ("EvAlUaTiOn", "questions.md", "evaluation_artifact"),
            ("node_modules", "library.js", "build_dependency_or_private_directory"),
            (".aws", "notes.txt", "build_dependency_or_private_directory"),
        ]:
            self.write(directory + "/" + filename, "must not reach the teacher")
            for target in (self.root / directory, self.root / directory / filename):
                with (
                    self.subTest(target=target),
                    patch.object(
                        Path,
                        "open",
                        side_effect=AssertionError("excluded input must not be opened"),
                    ),
                ):
                    snapshot = scan(target, Config())
                    self.assertEqual(snapshot.files, [])
                    self.assertEqual(snapshot.chunks, [])
                    self.assertEqual(snapshot.skipped[0]["reason"], why)

    def test_direct_root_respects_custom_exclusions_and_explicit_replacement(self):
        self.write("held-out/questions.txt", "HELD_OUT")
        custom = replace(Config(), evaluation_exclude=("held-out/**",))
        for target in (self.root / "held-out", self.root / "held-out" / "questions.txt"):
            with self.subTest(target=target):
                self.assertEqual(scan(target, custom).files, [])
        self.write("internal/detail.txt", "internal detail")
        self.assertEqual(
            scan(self.root / "internal", replace(Config(), exclude=("internal/**",))).files, []
        )
        self.write("evaluation/knowledge.txt", "deliberately allowed by explicit config")
        allowed = scan(self.root / "evaluation", replace(Config(), evaluation_exclude=()))
        self.assertEqual(len(allowed.files), 1)

    def test_chunk_offsets_and_python_symbols(self):
        source = "VALUE = 1\n\n@decorator\ndef check(x):\n    return x\n\nclass Gate:\n    pass\n"
        chunks = chunk_text("gate.py", source, "hash", 40)
        self.assertTrue(any(c.section == "check" for c in chunks))
        self.assertTrue(any(c.section == "Gate" for c in chunks))
        for chunk in chunks:
            self.assertEqual(source[chunk.start_char : chunk.end_char], chunk.text)
            self.assertEqual(chunk.start_line, source.count("\n", 0, chunk.start_char) + 1)
            self.assertEqual(chunk.file_hash, "hash")
            self.assertLessEqual(len(chunk.text), 40)

    def test_markdown_fences_and_long_line_offsets(self):
        source = "# Rules\n\n```python\n# not a heading\nx = 1\n```\n\n" + "x" * 300
        chunks = chunk_text("rules.md", source, "hash", 80)
        self.assertFalse(any(c.section == "# not a heading" for c in chunks))
        for chunk in chunks:
            self.assertEqual(source[chunk.start_char : chunk.end_char], chunk.text)
            self.assertLessEqual(len(chunk.text), 80)

    def test_single_file_paths_and_output_exclusion(self):
        self.write("single.txt", "mode=offline")
        snapshot = scan(self.root / "single.txt", Config())
        self.assertEqual(snapshot.files[0]["path"], "single.txt")
        self.write("custom-output/result.json", "generated")
        full = scan(self.root, Config(), (self.root / "custom-output",))
        self.assertEqual(len(full.files), 1)

    def test_symlinks_not_followed_even_outside_root(self):
        self.write("real.txt", "source")
        link = self.root / "link.txt"
        try:
            link.symlink_to(self.root / "real.txt")
        except OSError:
            self.skipTest(
                "Windows symlink privilege unavailable; junction test covers directory links"
            )
        snapshot = scan(self.root, Config())
        self.assertEqual({f["path"] for f in snapshot.files}, {"real.txt"})
        self.assertTrue(any(s["reason"] == "link_or_junction" for s in snapshot.skipped))
        with self.assertRaises(ValueError):
            scan(link, Config())

    @unittest.skipUnless(os.name == "nt", "Windows junction behavior")
    def test_external_directory_junction_not_followed(self):
        import _winapi

        source = self.root / "input"
        outside = self.root / "outside"
        source.mkdir()
        outside.mkdir()
        (outside / "hidden.txt").write_text("must never be read", encoding="utf-8")
        junction = source / "junction"
        try:
            _winapi.CreateJunction(str(outside), str(junction))
        except PermissionError:
            self.skipTest(
                "Windows sandbox denied junction creation; reparse-point guard tested with mocked stat"
            )
        snapshot = scan(source, Config())
        self.assertEqual(snapshot.files, [])
        self.assertEqual(snapshot.skipped, [{"path": "junction", "reason": "link_or_junction"}])
        with self.assertRaises(ValueError):
            scan(junction, Config())
        # Cleanup removes the junction entry only; the external source stays intact.
        junction.rmdir()
        self.assertTrue((outside / "hidden.txt").exists())

    def test_windows_reparse_point_guard_without_link_privileges(self):
        self.write("pretend-junction/hidden.txt", "must not reach teacher")
        junction = self.root / "pretend-junction"
        original_lstat = Path.lstat

        def lstat(path, *args, **kwargs):
            if path == junction:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=1024)
            return original_lstat(path, *args, **kwargs)

        with patch.object(Path, "lstat", lstat):
            snapshot = scan(self.root, Config())
        self.assertEqual(snapshot.files, [])
        self.assertEqual(
            snapshot.skipped, [{"path": "pretend-junction", "reason": "link_or_junction"}]
        )

    def test_budget_and_non_utf8_handling(self):
        self.write("large.txt", "x" * 100)
        self.assertEqual(len(scan(self.root, replace(Config(), max_file_bytes=10)).files), 0)
        self.write("small.txt", "y" * 5)
        with self.assertRaises(ValueError):
            scan(self.root, replace(Config(), max_input_bytes=50))
        with self.assertRaises(ValueError):
            scan(self.root, replace(Config(), max_files=1))

    def test_dry_run_is_an_estimate_not_actual_usage(self):
        self.write("settings.txt", "mode=offline")
        config = replace(Config(), max_calls=1, max_records=5)
        result = plan(scan(self.root, config), config, MockTeacher().identity())
        self.assertIn("estimated_input_tokens_initial", result)
        self.assertNotIn("provider_reported_input_tokens", result)
        self.assertIn("heuristic", result["estimate_note"])


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = workspace_temp()
        self.root = Path(self.tmp.name)
        self.path = self.root / ".env"

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_key_forms_without_expansion_or_environment_mutation(self):
        for assignment in (
            "MODEL_API_KEY=fake-only",
            "MODEL_API_KEY='fake-only'",
            'export MODEL_API_KEY = "fake-only"',
            "MODEL_API_KEY=$(fake-command)",
        ):
            self.path.write_text(
                "# local fixture\nUNRELATED=ignored\n" + assignment + "\n", encoding="utf-8"
            )
            env = {}
            with patch("synthdata.providers.os.environ", env):
                value = model_api_key(self.path)
            expected = "$(fake-command)" if "$(" in assignment else "fake-only"
            self.assertEqual(value, expected)
            self.assertEqual(env, {})

    def test_existing_process_key_wins_without_reading_file(self):
        with (
            patch("synthdata.providers.os.environ", {"MODEL_API_KEY": "fake-process-only"}),
            patch.object(Path, "open", side_effect=AssertionError("must not read file")),
        ):
            self.assertEqual(model_api_key(self.path), "fake-process-only")

    def test_absent_empty_and_unrelated_files_return_no_key(self):
        with patch("synthdata.providers.os.environ", {}):
            self.assertIsNone(model_api_key(self.path))
            for text in ("", "MODEL_API_KEY=\n", "OTHER=fake-only\n"):
                self.path.write_text(text, encoding="utf-8")
                self.assertIsNone(model_api_key(self.path))

    def test_invalid_credential_errors_do_not_echo_fake_values(self):
        for text in (
            "MODEL_API_KEY='fake-only",
            "MODEL_API_KEY=fake-only with-space",
            "MODEL_API_KEY=fake-only\nMODEL_API_KEY=fake-duplicate",
            "MODEL_API_KEY",
            "MODEL_API_KEY=fake-only\x00",
        ):
            self.path.write_text(text, encoding="utf-8")
            with (
                patch("synthdata.providers.os.environ", {}),
                self.assertRaises(ValueError) as caught,
            ):
                model_api_key(self.path)
            self.assertNotIn("fake-only", str(caught.exception))

    def test_credential_size_encoding_and_link_guards(self):
        with patch("synthdata.providers.os.environ", {}):
            for data in (b"x" * 65_537, b"\xff"):
                self.path.write_bytes(data)
                with self.assertRaises(ValueError):
                    model_api_key(self.path)
            self.path.write_text("MODEL_API_KEY=fake-only\n", encoding="utf-8")
            with (
                patch("synthdata.providers.is_link", return_value=True),
                self.assertRaises(ValueError),
            ):
                model_api_key(self.path)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        chunk = chunk_text("settings.txt", "mode=offline", "hash", 100)[0]
        self.request = make_request(chunk, "snapshot", Config(), {"provider": "test"})

    def test_meta_contract_through_mock_http_and_usage(self):
        captured = {}

        class HTTPResponse:
            headers = {"x-request-id": "request-123"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, size):
                return json.dumps(
                    {
                        "choices": [
                            {"message": {"content": answer().content}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 42, "completion_tokens": 19},
                    }
                ).encode()

        def opener(req, timeout):
            captured.update(
                url=req.full_url,
                payload=json.loads(req.data),
                timeout=timeout,
                authorization=req.get_header("Authorization"),
            )
            return HTTPResponse()

        teacher = MetaTeacher(api_key="fixture-key-not-real", opener=opener)
        response = teacher.generate(self.request)
        self.assertEqual(captured["url"], META_ENDPOINT)
        self.assertEqual(captured["payload"]["model"], META_MODEL)
        self.assertEqual(captured["payload"]["max_tokens"], self.request.config.max_output_tokens)
        self.assertEqual(captured["timeout"], self.request.config.timeout)
        self.assertEqual(response.input_tokens, 42)
        self.assertEqual(response.output_tokens, 19)
        self.assertEqual(response.request_id, "request-123")
        self.assertEqual(
            len(validate(response.content, response.finish_reason, self.request)[0]), 1
        )

    def test_transport_classification_and_redaction(self):
        failures = [
            (urllib.error.URLError(socket.gaierror("secret")), SafeRetry),
            (urllib.error.URLError(ConnectionRefusedError("secret")), SafeRetry),
            (TimeoutError("secret"), Uncertain),
            (urllib.error.URLError("secret"), Uncertain),
            (urllib.error.HTTPError(META_ENDPOINT, 401, "secret", {}, None), Rejected),
            (urllib.error.HTTPError(META_ENDPOINT, 429, "secret", {}, None), Uncertain),
            (urllib.error.HTTPError(META_ENDPOINT, 500, "secret", {}, None), Uncertain),
        ]
        for failure, expected in failures:
            with self.subTest(failure=type(failure).__name__, expected=expected.__name__):

                def opener(*args, **kwargs):
                    raise failure

                with self.assertRaises(expected) as caught:
                    MetaTeacher(api_key="fixture-key", opener=opener).generate(self.request)
                self.assertNotIn("secret", str(caught.exception))

    def test_invalid_envelope_unknown_and_no_redirect(self):
        from synthdata.providers import _NoRedirect

        self.assertIsNone(
            _NoRedirect().redirect_request(None, None, 307, "", {}, "https://other.invalid")
        )
        body = io.BytesIO(b"{}")
        body.headers = {}
        with self.assertRaises(Uncertain):
            MetaTeacher(api_key="fixture-key", opener=lambda *a, **kw: body).generate(self.request)

    def test_config_rejects_invalid_types_and_nonfinite_values(self):
        for kwargs in [
            {"max_calls": -1},
            {"concurrency": 0},
            {"qa_count": 0, "explanation_count": 0},
            {"qa_count": True},
            {"timeout": float("nan")},
            {"max_records": 1.5},
            {"project": 1},
            {"exclude": "private"},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
