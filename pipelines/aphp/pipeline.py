"""AP-HP pipeline — ATIH PMSI sampling → clinical scenario → LLM report.

This module implements the AP-HP-specific logic for generating synthetic
medical reports from ATIH PMSI data. It inherits from the common
:class:`~pipelines.pipeline.BasePipeline` and overrides the specific methods
as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

import polars as pl

from core.clients import AnthropicClient, MistralClient, OllamaClient
from pipelines.aphp.report import (
    generate_aphp_report,
    generate_aphp_reports_mistral_batch,
    generate_aphp_reports_mistral_batch_two_stage,
)
from pipelines.pipeline import BasePipeline
from pipelines.report import generate_reports


class APHPPipeline(BasePipeline):
    """AP-HP pipeline — report generation only.

    The input DataFrame is expected to come from fictomed and to contain:
    generation_id, scenario, user_prompt, system_prompt, prefix, prefix_len.
    """

    name = "aphp"

    @override
    def check_data(self) -> None:
        """Check only the Stream report output directory.
        AP-HP source data and referentials are checked by fictomed.
        """
        output_dir = Path(self.config["data"]["output"])
        output_dir.mkdir(parents=True, exist_ok=True)

    @override
    def load_data(self) -> dict[str, pl.LazyFrame]:
        """No data loading in Stream for AP-HP.
        Scenario generation is handled by fictomed.
        """
        return {}

    @override
    def get_fictive(
        self,
        data: dict[str, pl.LazyFrame],
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Not used for AP-HP in Stream.
        fictomed.generate("aphp", ...) is called by runner.py.
        """
        raise NotImplementedError(
            "AP-HP scenario generation is handled by fictomed, not Stream."
        )

    @override
    def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
        """Not used for AP-HP in Stream.
        The DataFrame returned by fictomed already contains scenario prompts.
        """
        return df

    @override
    def get_report(
        self,
        df: pl.DataFrame,
        client: AnthropicClient | MistralClient | OllamaClient,
        model: str,
        batch_size: int = 1000,
    ) -> pl.DataFrame:
        """Generate AP-HP reports.

        Default mode is direct and works with Ollama, Claude and Mistral.
        The optional mistral_batch mode reproduces the AP-HP historical Mistral batch method.
        """
        output_dir = Path(self.config["data"]["output"])
        output_dir.mkdir(parents=True, exist_ok=True)

        generation_cfg = self.config.get("generation", {})
        mode = generation_cfg.get("mode", "direct")
        workflow = generation_cfg.get("workflow", "one_stage")

        if workflow not in {"one_stage", "two_stage"}:
            raise ValueError(
                "generation.workflow doit valoir 'one_stage' ou 'two_stage'."
            )

        if workflow == "two_stage" and mode != "mistral_batch":
            raise ValueError(
                "generation.workflow='two_stage' requires "
                "generation.mode='mistral_batch'."
            )

        if mode == "mistral_batch":
            if not isinstance(client, MistralClient):
                raise TypeError(
                    "generation.mode='mistral_batch' requires --client mistral."
                )

            if workflow == "two_stage":
                two_stage_cfg = generation_cfg.get("two_stage", {})
                summary_cfg = two_stage_cfg.get("summary", {})
                report_cfg = two_stage_cfg.get("report", {})
                return generate_aphp_reports_mistral_batch_two_stage(
                    df,
                    client,
                    summary_model=summary_cfg.get("model", model),
                    report_model=report_cfg.get("model", model),
                    output_dir=output_dir,
                    summary_max_tokens=summary_cfg.get("max_tokens", 8_000),
                    report_max_tokens=report_cfg.get("max_tokens", 128_000),
                    poll_interval_seconds=generation_cfg.get(
                        "poll_interval_seconds", 1
                    ),
                )

            one_stage_cfg = generation_cfg.get("one_stage", {})
            return generate_aphp_reports_mistral_batch(
                df,
                client,
                one_stage_cfg.get("model", model),
                output_dir=output_dir,
                max_tokens=one_stage_cfg.get(
                    "max_tokens", generation_cfg.get("max_tokens", 128_000)
                ),
                poll_interval_seconds=generation_cfg.get("poll_interval_seconds", 1),
            )

        return generate_reports(
            df,
            client,
            model,
            batch_size=batch_size,
            output_dir=output_dir,
            generate_fn=generate_aphp_report,
            system_prompt="",
        )
