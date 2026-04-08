"""Base pipeline classes for synthetic medical-report generation.

This module defines the common pipeline infrastructure shared by both the
Brest and AP-HP pipelines. Concrete pipeline implementations should inherit
from :class:`BasePipeline` and override the specific methods as needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import polars as pl

from core.clients import AnthropicClient, MistralClient, OllamaClient
from core.logger import get_logger

REPORT_SCHEMA = {
    "generation_id": pl.Utf8,
    "scenario": pl.Utf8,
    "report": pl.Utf8,
    "model": pl.Utf8,
    "timestamp": pl.Datetime,
}


class BasePipeline(ABC):
    """Base class for synthetic medical-report generation pipelines.

    Provides shared building blocks (LLM wrappers, generation loop, Parquet
    persistence) and defines the interface that each concrete pipeline must
    implement.
    """

    name: str = "base"

    def __init__(self, config: dict, prompt: dict, servers: dict) -> None:
        self.config = config
        self.prompt = prompt
        self.servers = servers
        self.logger = get_logger(self.name)

    # -- Abstract interface ------------------------------------------------

    @abstractmethod
    def check_data(self) -> None:
        """Verify that source data is present and prepare it if needed."""

    @abstractmethod
    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Load prepared data as LazyFrames."""

    @abstractmethod
    def get_fictive(self, data: dict[str, pl.LazyFrame], **kwargs) -> pl.DataFrame:
        """Generate fictitious hospital stays from loaded data."""

    @abstractmethod
    def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform fictitious stays into text scenarios for the LLM."""

    # -- LLM report generation (shared) ------------------------------------
    @abstractmethod
    def get_report(
        self,
        df: pl.DataFrame,
        client: AnthropicClient | MistralClient | OllamaClient,
        model: str,
        batch_size: int = 1000,
    ) -> pl.DataFrame:
        """Generate one medical report per scenario via LLM calls.

        Results are flushed to timestamped Parquet files every
        *batch_size* rows in the configured output directory.

        Returns
        -------
        pl.DataFrame
            Columns: generation_id, scenario, report, model, timestamp.
        """


def flush_batch(batch: list[dict], output_dir: Path, count: int) -> None:
    """Write a batch of results to a timestamped Parquet file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"medical_reports_{count}_{ts}.parquet"
    pl.DataFrame(batch, schema=REPORT_SCHEMA).write_parquet(path)
