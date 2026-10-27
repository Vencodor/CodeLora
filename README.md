# Local project data synthesis

Python CLI and reusable module for **local files → source chunks → teacher synthesis → raw JSONL**.
A separate command exports SFT data for a future Glimmer LoRA workflow. Runtime dependencies:
Python 3.11+ and its standard library. No upload receiver, UI, training, RAG, serving, reviewer,
evaluator, or generic replay-data collection is implemented.

## Quick start (PowerShell)

Run from the repository; installation is optional. Quote paths containing spaces.

```powershell
Set-Location -LiteralPath 'C:\Users\D.S Kim\Desktop\Meta Hackathon'
python -m synthdata dry-run '.\tests\fixtures\tiny_project' --output '.\outputs\mock demo' --config '.\examples\mock-config.json'
python -m synthdata generate '.\tests\fixtures\tiny_project' --output '.\outputs\mock demo' --config '.\examples\mock-config.json' --provider mock
python -m synthdata export '.\outputs\mock demo' --destination '.\outputs\mock demo\sft.messages.jsonl' --format messages
python -m synthdata export '.\outputs\mock demo' --destination '.\outputs\mock demo\sft.instruction.jsonl' --format instruction --include-explanations
python -m synthdata sample '.\outputs\mock demo' --count 3 --seed 7
```

`dry-run` reads local files and reports accepted/skipped files, exact chunk locations, estimated
initial calls, conservative attempt/record bounds, and a **characters/4 input-token estimate**.
It creates no output directory, reads no API key, does not consult saved state, and makes no
API calls. Estimates are neither actual usage nor price quotes. `mock` uses deterministic
fixture outputs to test plumbing; its questions/answers do not demonstrate synthesis quality.

Optional installation, within a virtual environment:

```powershell
python -m venv '.venv'
& '.\.venv\Scripts\python.exe' -m pip install -e '.'
& '.\.venv\Scripts\synthdata.exe' --help
```

In restricted environments, point process-local `TEMP`/`TMP` at an allowed working directory
before creating a virtual environment. The module itself needs no package download.

## Input, provenance, and exclusions

Use one file or a directory. Supported inputs include UTF-8 Markdown, text, common programming
languages and text configuration files. A file input uses its parent as the relative-path root;
a directory input uses that directory. Binary/control-byte and non-UTF-8 inputs are skipped
with a fixed reason. Empty files are tracked in the manifest, with no generation calls.

Markdown splits on headings and paragraphs while respecting fenced blocks. Python uses
top-level AST boundaries (including decorators); invalid Python falls back to bounded text.
Other code uses simple declaration boundaries. Oversized units split at line boundaries when
possible, then exact character offsets. There is no minimum content length: `mode=offline`
is retained. Source records carry relative path, SHA-256 of original bytes, chunk ID, section
or symbol, 1-based inclusive line spans, and 0-based half-open character spans in the decoded
text. Long lines can produce multiple chunks on the same source line. UTF-8 BOMs are removed
for decoding, while hashes still describe original bytes.

The input snapshot ID hashes the sorted accepted file manifest. Excluded artifacts do not
affect it. The SQLite store retains the exact source chunks used for synthesis, so human
samples remain reproducible after original files change.

Defaults exclude VCS/private folders, `.venv`, dependency/build directories (`node_modules`,
`vendor`, `dist`, `build`, `target`, etc.), `.env*`, secret/key filenames and common credential
patterns in content. **All symlinks and Windows reparse points/junctions are skipped**, including
links inside the input root. An input path that contains a link is rejected. The active output
directory is also excluded. Keep the source tree unchanged while scanning: the scanner is not
a hardened sandbox against a hostile process swapping files concurrently. Secret detection is
heuristic and may have false positives/negatives; review the dry-run manifest before real use.

Evaluation questions, answers, and review annotations must remain separate from synthesis.
Default directory exclusions include `eval`, `evaluation`, `evaluations`, `benchmarks`, and
`review_annotations`. Add your own locations:

```powershell
python -m synthdata dry-run 'C:\Projects\Example Project' --output '.\outputs\example' --project 'Example Project' --evaluation-exclude 'held-out/**' --evaluation-exclude '**/answer-key.json' --exclude 'internal/**'
```

Patterns are case-insensitive, use `/` separators, match relative paths, and can contain `*`
and `**`. A bare component like `eval` excludes that name at any depth. Excluding a directory
excludes its descendants.
The selected input root's name also participates in exclusion matching, so directly selecting
`evaluation/`, `node_modules/`, or a file in one of those roots does not bypass its exclusion.
Relative paths in stored provenance remain relative to the selected input root.
CLI patterns append to config patterns. A JSON config's
`evaluation_exclude` array replaces defaults, so include every required exclusion there.
Ordinary test source code is not blanket-excluded; specify evaluation artifacts explicitly.
Knowledge source documents may be used even if their knowledge is later evaluated. No random
split of synthetic QA is presented as a generalization evaluation.

## Generation and validation

Every teacher request contains the chunk itself, source labels, and generation constraints.
The teacher sees the primary source directly. The prompt treats source instructions as
untrusted data, requires grounded facts and standalone questions, distinguishes current,
historical and planned behavior, and forbids invented APIs/files/design intent. Questions
vary in focus; explanation records teach the source around a concrete question. There is no
evidence-card stage, full LLM verification, RAG-failure regeneration, or search-query training.

`qa_count` and `explanation_count` are independent **upper bounds per chunk**, not quotas.
Lengths are configurable character upper bounds, plus a provider output-token cap. Defaults
are convenience settings, not empirically optimal mixing ratios. A teacher can return fewer
items or normal zero with a nonempty `zero_reason`; no filler or refill calls are made.

Validation distinguishes malformed JSON/schema, empty output/items, non-final/truncated output,
length/count errors, and inconsistent zero reasons. It also rejects simple source-dependent
questions, fenced code, and direct full-source/long-source-line copies into questions. This
is a conservative guard, **not a factual accuracy or complete leakage guarantee**. Compare
samples to source before training. Exact duplicate type/question/response records stay in raw
data with `duplicate_of`; only export suppresses them. Semantic/wording variants are kept.

## Output and SFT contract

`state.sqlite3` is authoritative, with transactions and full synchronous writes. Keep its WAL
files together with it when backing up a running job. JSONL views are atomically replaced per
file after generation and can be rebuilt by repeating generation or running export. The group
of JSONL files is not a single filesystem transaction; use SQLite after interrupted exports.

| File | Contents |
| --- | --- |
| `manifest.jsonl` | Selected run, snapshot, accepted files/hashes, skips, initial settings/model |
| `raw/qa.jsonl` | Typed QA records, provenance, record/job IDs, duplicate markers |
| `raw/explanations.jsonl` | Original question-centered explanations, independently retained |
| `raw/responses.jsonl` | All attempts, original completion text (including invalid output), usage/status/request IDs |
| `raw/outcomes.jsonl` | Per-chunk status, fixed error codes, normal-zero reason, reservations |
| `report.jsonl` | Cumulative selected-run attempt/token report |
| User-selected SFT file | Separate `instruction`/`response` or `messages` export |

Raw/SQLite artifacts include source and teacher content. Keep them private and apply appropriate
local access controls. Console errors/progress never print API keys, provider response bodies,
or arbitrary exception strings. The explicit `sample` command prints selected source text
alongside generated records for human review; do not paste sensitive samples into public logs.

Example SFT instruction record (hashes shortened for illustration):

```json
{"instruction":"Project: Lantern; file: settings.txt\nWhich mode does Lantern configure in settings.txt?","response":"The mode is offline.","metadata":{"record_id":"...","snapshot_id":"...","type":"qa","source":{"path":"settings.txt","file_hash":"...","start_line":1,"end_line":1,"start_char":0,"end_char":13,"section":"settings.txt","chunk_id":"..."}}}
```

Student input consists solely of the generated question plus project/file and, when safe,
a bare symbol identifier. The exporter never appends the source body, a source system message,
or code context. Hashes are tracking metadata, not required question content. Metadata should
not be serialized into student prompts by downstream training loaders. By default SFT export
includes QA only. `--include-explanations` opts into converting explanations using their own
question and original text. Keeping original raw explanations supports later training formats.

Multiple input snapshots/settings can coexist in one SQLite store. JSONL views and default
export/sample select the most recently generated/resumed run. `--run-id` selects an older run
for export/sample. Changing source, prompt, model/adapter identity, or generation settings
creates distinct jobs. Operating budget, concurrency and timeout changes reuse saved results.

## Resume, budgets, and uncertain outcomes

Repeat the same `generate` command to resume. The job key fingerprints input snapshot, chunk,
the full prompt, model/adapter identity, generation settings and chunk plan. A writer lock
allows one process per output directory; worker concurrency is configurable. OS locks release
on process exit, so a leftover `.writer.lock` file alone does not block recovery.

| State | Resume behavior |
| --- | --- |
| `saved` | Reuse without teacher calls, including normal zero |
| `invalid` | Retain original completion and diagnosis; do not repeat |
| `pending` / `budget` | Eligible when budgets permit |
| `retryable` | Only failures known not to have submitted; bounded retries |
| `failed` | Definitive rejection or exhausted safe retries; terminal |
| `unknown` | May have executed at provider; terminal, held for manual reconciliation |

The dispatch marker is committed before every teacher call. A process interrupted after that
marker, after sending, after receiving, or before committing results becomes `unknown` on
resume. This conservative choice also holds a rare call interrupted before the actual send.
After a successful transaction, JSONL-output failure never causes teacher replay. Unknown jobs
have no automatic resend flag. Reconcile with the provider/account records first; a deliberate
new run/output or manual store reset can duplicate execution and billing.

Only DNS resolution failure/connection refusal before an HTTP connection is established are
automatically retried by the Meta adapter. Definitive HTTP request rejections are terminal.
Timeouts, disconnects, malformed API envelopes, redirects, HTTP 429/5xx and unexpected adapter
errors are conservatively uncertain. This sacrifices some recoverability to avoid unsafe replay.
No provider idempotency guarantee was verified. A **request ID is not an idempotency key**;
this module does not claim exactly-once execution or billing prevention.

`max_calls` caps cumulative attempts for the selected run, including failed/unknown attempts
and retries. `retries=2` permits at most three attempts per job. `max_records` reserves each
admitted chunk's requested QA+explanation maximum; unused/failed/normal-zero capacity is not
recycled to fill quotas. `max_total_output_tokens` reserves `max_output_tokens` for each attempt,
including unknown attempts. These conservative reservations bound submitted generation limits,
not monetary charges. Raise operating budgets on the same output to continue `budget` jobs;
already saved jobs remain cached. Changing input/model/generation settings starts a new run
with separate budgets. Input byte/file caps prevent accidentally ingesting a very large tree.

`report.jsonl` separates reserved limits and provider-reported token sums. Missing token usage,
unknown calls and mock usage are explicitly counted; missing usage does not mean zero billing.
Provider-reported output tokens may include reasoning under the provider's accounting rules.
`max_output_tokens` caps the requested provider generation, including any reasoning that the
provider counts toward it. There is no price estimator or monetary-cost guarantee.

Exit codes: `0` finished (including normal zero), `1` input/config/local I/O error, `2` failed,
unknown or invalid jobs, `3` budget-limited partial completion, `130` interrupted. Progress is
JSON on stderr, summary JSON on stdout; inspect outcomes for fixed per-job diagnoses.

## Real Meta adapter: separate approval required

**No real API calls, paid requests, or user-project source transfers were used for validation.**
The implementation was exercised with a fictional fixture and an injected mock HTTP API.
Contributor prompts and outputs **may be used to train future Meta models**. Obtain approval
for the exact files, transfer, provider-training use and costs before real generation; exclude
secrets, proprietary data that cannot be sent, and evaluation artifacts.

Official Meta documentation checked on 2026-10-27:

- [Models and Contributor tier](https://llama.developer.meta.com/docs/models): the search index
  explicitly lists `muse-spark-1.3-contributor` and the Contributor data-use tradeoff.
- [Quickstart](https://llama.developer.meta.com/docs/quickstart): API base
  `https://api.meta.ai/v1`, bearer authentication using `MODEL_API_KEY`, Chat Completions support.
- [Structured output / Chat Completions](https://llama.developer.meta.com/docs/features/structured-output):
  `POST https://api.meta.ai/v1/chat/completions` and JSON-string completion at
  `choices[0].message.content`.

Full official pages returned Markdown that the web reader could not render, and a direct
public-document fetch was unavailable in this environment. Verification therefore used the
official search index excerpts. The adapter implements the documented OpenAI-compatible chat
shape with `messages`, `model`, `max_tokens`, `temperature`, `stream=false`; it reads
`finish_reason`, optional `usage.prompt_tokens` / `usage.completion_tokens`, and optional
request IDs. **Actual account access, parameter support, response/usage behavior, rate limits,
prices, output quality and idempotency are unverified live.** Recheck official docs/account
settings before the separately approved minimal real test. No guessed fallback endpoint or
third-party gateway is used, and redirects are never followed with source/key data.

The repository root has an ignored `.env` prepared with an empty assignment:

```dotenv
MODEL_API_KEY=
```

Enter the key locally after the `=`. A nonempty process `MODEL_API_KEY` takes precedence.
For `generate --provider meta --allow-external-api`, the CLI automatically reads `.env` from
the current working directory, or the path provided with `--env-file 'C:\Local Settings\.env'`.
It reads only `MODEL_API_KEY`, accepts an optional `export` prefix and matching outer quotes,
and performs no shell execution, interpolation, or process-environment modification. Use one
assignment on one line, without inline comments. Duplicate assignments, whitespace/control
characters in a key, non-UTF-8/oversized files and linked credential paths are rejected.
Missing/empty keys fail before any request. No `.env` is read for dry-run, mock, export, sample,
or a Meta generation missing `--allow-external-api`. The reusable `MetaTeacher` accepts an
explicit key or process environment; `.env` loading is the CLI's responsibility.

`.env` is excluded from Git and source synthesis; do not force-add it. The CLI never prints
the key or stores it in dataset metadata. Do not put keys in config, source, shell history, or
this README. Preparing a local key does not approve external transfer or billing; separate
approval is still required. The following command **sends source
externally and may incur charges**; it is documentation, not a command executed here:

```powershell
python -m synthdata dry-run 'C:\Approved Data\Example Project' --output '.\outputs\approved real test' --provider meta --project 'Example Project' --max-calls 1 --max-records 5
python -m synthdata generate 'C:\Approved Data\Example Project' --output '.\outputs\approved real test' --provider meta --model 'muse-spark-1.3-contributor' --project 'Example Project' --max-calls 1 --max-records 5 --max-output-tokens 2048 --max-total-output-tokens 2048 --concurrency 1 --retries 0 --timeout 60 --allow-external-api
```

## Development and reusable API

```python
from pathlib import Path
from synthdata import Config, scan, generate
from synthdata.export import export_sft
from synthdata.providers import MockTeacher

config = Config(project="Lantern", qa_count=2, explanation_count=1)
output = Path("outputs") / "python example"
snapshot = scan(Path("tests/fixtures/tiny_project"), config, exclude_paths=(output,))
report = generate(snapshot, output, config, MockTeacher())
export_sft(output, output / "sft.jsonl", include_explanations=False)
```

Custom teachers implement `identity()` and `generate(Request) -> Response`. Identity must
change when model, endpoint, adapter contract or behavior changes. `SafeRetry` must only be
raised with positive evidence no generation was submitted. Other uncertain outcomes should
raise `Uncertain`. Raw exceptions are conservatively treated as unknown.

```powershell
python -m unittest discover -s tests -v
python -m compileall -q synthdata tests
# Optional development tools, installed only in a local virtual environment:
& '.\.venv\Scripts\python.exe' -m pip install ruff mypy
& '.\.venv\Scripts\python.exe' -m ruff check synthdata tests
& '.\.venv\Scripts\python.exe' -m mypy synthdata
```

Tests use `.test-tmp` inside this workspace for compatibility with restricted Windows temporary
folders. Symlink tests may skip when Windows denies link creation; junction coverage is tested
separately where available. No external API access is required.

Known limitations: local per-chunk context can miss cross-file relationships; no PDF/OCR,
whole-repository dependency graph, all-language AST engine, or automatic factual verification.
Large syntax units can be split and should be omitted by a teacher if context is insufficient.
The source-grounded prompt and heuristic validators reduce errors but do not prove correctness.
Mock tests prove software behavior, not model quality or downstream LoRA gains. Persisted local
data is not encrypted. No schema migration or provider-side reconciliation tool is supplied yet.

Design references: [SPA direct generation](https://github.com/Tangkexian/SPA/blob/main/src/generate_data.py)
and [SMT Appendix D / Focal Rewriting](https://arxiv.org/html/2603.23562v2#A4).
This is a small adaptation of direct source generation and question-centered diversification,
not a reproduction of the full papers, their optimal ratios, or their performance claims.
