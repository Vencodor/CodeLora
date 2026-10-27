"""Local source ingestion and durable synthetic data generation."""

from .config import Config
from .ingest import scan
from .pipeline import generate, plan

__all__ = ["Config", "generate", "plan", "scan"]
