from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from fictomed.sites.aphp.prompt import make_final_user_prompt


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


def _json_or_empty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if value else ""


def generate_aphp_reports_mistral_batch_two_stage(
    df: pl.DataFrame,
    client: Any,
    *,
    summary_model: str,
    report_model: str,
    output_dir: Path,
    summary_max_tokens: int = 8_000,
    report_max_tokens: int = 128_000,
    poll_interval_seconds: int = 1,
) -> pl.DataFrame:
    """Generate an intermediate summary, then the final AP-HP report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "generation_id",
        "scenario",
        "user_prompt",
        "summary_system_prompt",
        "summary_prefix",
        "final_system_prompt",
        "final_prefix",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Le DataFrame AP-HP two_stage ne contient pas les colonnes attendues : "
            + ", ".join(sorted(missing))
        )

    rows = list(df.iter_rows(named=True))
    summary_requests = [
        {
            "custom_id": str(idx),
            "system_prompt": row["summary_system_prompt"],
            "user_prompt": row["user_prompt"],
            "prefix": row.get("summary_prefix") or "",
        }
        for idx, row in enumerate(rows)
    ]
    summary_responses = client.batch_chat(
        model=summary_model,
        requests=summary_requests,
        max_tokens=summary_max_tokens,
        poll_interval_seconds=poll_interval_seconds,
    )
    summaries_by_idx = {
        int(response["custom_id"]): response for response in summary_responses
    }

    report_requests: list[dict] = []
    for idx, row in enumerate(rows):
        summary_response = summaries_by_idx.get(idx, {})
        if summary_response.get("error") or not summary_response.get("content"):
            continue
        report_requests.append(
            {
                "custom_id": str(idx),
                "system_prompt": row["final_system_prompt"],
                "user_prompt": make_final_user_prompt(
                    row["user_prompt"], summary_response["content"]
                ),
                "prefix": row.get("final_prefix") or "",
            }
        )

    report_responses = (
        client.batch_chat(
            model=report_model,
            requests=report_requests,
            max_tokens=report_max_tokens,
            poll_interval_seconds=poll_interval_seconds,
        )
        if report_requests
        else []
    )
    reports_by_idx = {
        int(response["custom_id"]): response for response in report_responses
    }

    timestamp = datetime.now()
    output_rows: list[dict] = []
    for idx, row in enumerate(rows):
        summary_response = summaries_by_idx.get(idx, {})
        report_response = reports_by_idx.get(idx, {})
        output_rows.append(
            {
                "generation_id": row["generation_id"],
                "scenario": row["scenario"],
                "summary": summary_response.get("content", ""),
                "summary_model": summary_model,
                "summary_prompt_tokens": summary_response.get("prompt_tokens"),
                "summary_completion_tokens": summary_response.get(
                    "completion_tokens"
                ),
                "summary_total_tokens": summary_response.get("total_tokens"),
                "summary_mistral_batch_error": _json_or_empty(
                    summary_response.get("error")
                ),
                "summary_mistral_batch_raw": _json_or_empty(
                    summary_response.get("raw")
                ),
                "report": report_response.get("content", ""),
                "model": report_model,
                "timestamp": timestamp,
                "prompt_tokens": report_response.get("prompt_tokens"),
                "completion_tokens": report_response.get("completion_tokens"),
                "total_tokens": report_response.get("total_tokens"),
                "mistral_batch_error": _json_or_empty(
                    report_response.get("error")
                ),
                "mistral_batch_raw": _json_or_empty(report_response.get("raw")),
            }
        )

    out_df = pl.DataFrame(output_rows)
    output_path = output_dir / (
        f"aphp_mistral_batch_two_stage_reports_{out_df.height}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    out_df.write_parquet(output_path)
    return out_df
