from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import os
from mistralai import Mistral

import polars as pl


def generate_aphp_report(
    row: dict,
    client: Any,
    model: str,
    system_prompt: str = "",
) -> dict:
    """Generate one AP-HP report with the generic Stream client interface."""
    row_system_prompt = row.get("system_prompt") or system_prompt
    user_prompt = row.get("user_prompt") or row["scenario"]

    messages = [
        {"role": "system", "content": row_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return client.chat(model=model, messages=messages)


def generate_aphp_reports_mistral_batch(
    df: pl.DataFrame,
    client: Any,
    model: str,
    *,
    output_dir: Path,
    max_tokens: int = 128_000,
    poll_interval_seconds: int = 1,
) -> pl.DataFrame:
    """Generate AP-HP reports using Historical AP-HP Mistral batch method.

    This expects the DataFrame produced by fictomed to contain:
    system_prompt, user_prompt, prefix.
    """
    required = {"generation_id", "scenario", "system_prompt", "user_prompt", "prefix"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Le DataFrame AP-HP ne contient pas les colonnes attendues : "
            + ", ".join(sorted(missing))
        )

    batch_requests: list[dict] = []

    for idx, row in enumerate(df.iter_rows(named=True)):
        batch_requests.append(
            {
                "custom_id": str(idx),
                "generation_id": row.get("generation_id"),
                "system_prompt": row["system_prompt"],
                "user_prompt": row.get("user_prompt") or row["scenario"],
                "prefix": row.get("prefix") or "",
            }
        )

    responses = client.batch_chat(
        model=model,
        requests=batch_requests,
        max_tokens=max_tokens,
        poll_interval_seconds=poll_interval_seconds,
    )

    responses_by_idx = {int(response["custom_id"]): response for response in responses}

    timestamp = datetime.now()
    output_rows: list[dict] = []

    for idx, row in enumerate(df.iter_rows(named=True)):
        response = responses_by_idx.get(idx, {})
        content = response.get("content", "")
        error = response.get("error")
        raw = response.get("raw")

        out = {
            "generation_id": row["generation_id"],
            "scenario": row["scenario"],
            "report": content,
            "model": model,
            "timestamp": timestamp,
            "prompt_tokens": response.get("prompt_tokens"),
            "completion_tokens": response.get("completion_tokens"),
            "total_tokens": response.get("total_tokens"),
            "mistral_batch_error": (
                json.dumps(error, ensure_ascii=False) if error else ""
            ),
            "mistral_batch_raw": (json.dumps(raw, ensure_ascii=False) if raw else ""),
        }
        output_rows.append(out)

    out_df = pl.DataFrame(output_rows)

    output_path = output_dir / (
        f"aphp_mistral_batch_reports_{out_df.height}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    out_df.write_parquet(output_path)

    return out_df
