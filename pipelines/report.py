"""CRH generation — common primitives for LLM-based medical-report generation.

This module provides shared building blocks for generating synthetic
medical reports (CRH) from text scenarios using LLM calls. It is used
by both the Brest and AP-HP pipelines to ensure consistent LLM interaction
logic and reduce code duplication.

Responsibilities:
- Generate synthetic medical reports (CRH) from text scenarios using LLM calls.
- Ensure consistent LLM interaction logic across pipelines.
- Reduce code duplication between pipelines.

Public API:
- :func:`generate_reports`: Generate CRH reports for a given pipeline.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

from pipelines.pipeline import REPORT_SCHEMA, flush_batch

# ---------------------------------------------------------------------------
# Common CRH-generation helpers
# ---------------------------------------------------------------------------


def generate_reports(
    df: pl.DataFrame,
    client: Any,
    model: str,
    system_prompt: str,
    generate_fn: Callable[[dict, Any, str], dict],
    batch_size: int = 1000,
    output_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Generate one medical report per scenario via LLM calls.

    Parameters
    ----------
    df: DataFrame of scenarios
    client: LLM client instance
    model: model name
    generate_fn: function to generate a single report from a row
    batch_size: batch size
    output_dir: directory to flush results

    Returns
    -------
    pl.DataFrame
        Columns: generation_id, scenario, report, model, timestamp.
    """
    if "generation_id" not in df.columns:
        raise ValueError(
            "Le DataFrame doit contenir une colonne 'generation_id'. "
            "Assurez-vous de passer par get_fictive avant get_report."
        )

    if output_dir is None:
        output_dir = Path("./output")  # Default fallback
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    batch: list[dict] = []

    for df_row in tqdm(
        df.iter_rows(named=True), desc="Génération CRH", unit="crh", total=len(df)
    ):
        response = generate_fn(df_row, client, model, system_prompt)

        row = {
            "generation_id": df_row["generation_id"],
            "scenario": df_row["scenario"],
            "report": response["message"]["content"],
            "model": model,
            "timestamp": datetime.now(),
        }
        results.append(row)
        batch.append(row)

        if len(batch) >= batch_size:
            flush_batch(batch, output_dir, batch_size)
            batch = []

    if batch:
        flush_batch(batch, output_dir, len(batch))

    return pl.DataFrame(results, schema=REPORT_SCHEMA)
