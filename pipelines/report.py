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
    batch_size: int = 1000,
    output_dir: str | Path | None = None,
    pipeline_type: str = "brest",
) -> pl.DataFrame:
    """Generate one medical report per scenario via LLM calls.

    Parameters
    ----------
    df:
        DataFrame of scenarios (output of ``format_scenarios``). Must contain
        a ``generation_id`` column and either a ``scenario`` column (Brest) or
        both ``scenario`` and ``system_prompt`` columns (AP-HP).
    client:
        LLM client instance (e.g., ``AnthropicClient``, ``MistralClient``).
    model:
        Model name to use for generation.
    batch_size:
        Number of reports to generate before flushing to disk.
    output_dir:
        Directory to write batch Parquet files. If ``None``, uses the
        configured output directory from the pipeline.
    pipeline_type:
        Either "brest" or "aphp" to select the appropriate prompting logic.

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

    if pipeline_type == "aphp" and "system_prompt" not in df.columns:
        raise ValueError(
            "Le DataFrame doit contenir une colonne 'system_prompt' pour AP-HP. "
            "Assurez-vous de passer par get_scenario avant get_report."
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
        if pipeline_type == "aphp":
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": df_row["system_prompt"]},
                    {"role": "user", "content": df_row["scenario"]},
                ],
            )
        else:  # brest
            response = client.chat(
                model=model,
                messages=[
                    {"role": "user", "content": df_row["scenario"]},
                ],
            )

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
