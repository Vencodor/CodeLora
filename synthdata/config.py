import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    project: str = "project"
    qa_count: int = 3
    explanation_count: int = 2
    qa_max_chars: int = 1600
    explanation_max_chars: int = 4000
    question_max_chars: int = 600
    chunk_chars: int = 6000
    max_file_bytes: int = 1_000_000
    max_input_bytes: int = 20_000_000
    max_files: int = 1000
    max_calls: int = 100
    max_records: int = 500
    max_output_tokens: int = 4096
    max_total_output_tokens: int = 409_600
    timeout: float = 60.0
    retries: int = 2
    concurrency: int = 2
    retry_delay: float = 0.5
    temperature: float = 0.7
    exclude: tuple[str, ...] = ()
    evaluation_exclude: tuple[str, ...] = (
        "eval",
        "evaluation",
        "evaluations",
        "benchmarks",
        "review_annotations",
        "**/eval_questions*",
        "**/eval_answers*",
        "**/review_annotations*",
    )

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name in {"project", "exclude", "evaluation_exclude"}:
                continue
            if name in {"timeout", "retry_delay", "temperature"}:
                if type(value) not in {int, float} or not math.isfinite(value):
                    raise ValueError(f"{name} must be finite")
            elif type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if not isinstance(self.project, str):
            raise ValueError("project must be a string")
        for patterns in (self.exclude, self.evaluation_exclude):
            if not isinstance(patterns, (tuple, list)) or not all(
                isinstance(p, str) for p in patterns
            ):
                raise ValueError("exclusion patterns must be strings")
        positive = (
            "qa_max_chars",
            "explanation_max_chars",
            "question_max_chars",
            "chunk_chars",
            "max_file_bytes",
            "max_input_bytes",
            "max_files",
            "max_output_tokens",
            "timeout",
            "concurrency",
        )
        nonnegative = (
            "qa_count",
            "explanation_count",
            "max_calls",
            "max_records",
            "max_total_output_tokens",
            "retries",
            "retry_delay",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not self.qa_count + self.explanation_count:
            raise ValueError("at least one output type must be requested")
        if not self.project.strip() or len(self.project) > 160:
            raise ValueError("project must contain 1..160 characters")
        if self.concurrency > 32 or self.retries > 10 or self.retry_delay > 30:
            raise ValueError("concurrency <= 32, retries <= 10, retry_delay <= 30 required")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    def generation(self) -> dict:
        # Operational budgets do not invalidate an already saved generation.
        names = (
            "project",
            "qa_count",
            "explanation_count",
            "qa_max_chars",
            "explanation_max_chars",
            "question_max_chars",
            "max_output_tokens",
            "temperature",
        )
        return {name: getattr(self, name) for name in names}

    def to_dict(self) -> dict:
        return asdict(self)
