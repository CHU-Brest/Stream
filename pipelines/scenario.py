"""Scenario transformation — common primitives for formatting clinical scenarios.

This module provides shared building blocks for transforming fictitious
hospital stays into text scenarios ready for LLM prompting. It is used
by both the Brest and AP-HP pipelines to ensure consistent formatting
logic and reduce code duplication.

Responsibilities:
- Transform fictitious stays into text scenarios for LLM prompting.
- Ensure consistent formatting logic across pipelines.
- Reduce code duplication between pipelines.

Public API:
- :func:`format_scenarios`: Format scenarios for a given pipeline.
"""

from __future__ import annotations

from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Common scenario-transformation helpers
# ---------------------------------------------------------------------------


def format_scenarios(
    df: pl.DataFrame,
    pipeline_type: str = "brest",
    **kwargs: Any,
) -> pl.DataFrame:
    """Format fictitious stays as text scenarios for the LLM.

    Parameters
    ----------
    df:
        DataFrame of fictitious stays (output of ``generate_fictive_stays``).
    pipeline_type:
        Either "brest" or "aphp" to select the appropriate formatting logic.
    **kwargs:
        Additional pipeline-specific arguments (e.g., ``cancer_codes``,
        ``atih_rules`` for AP-HP).

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an added ``scenario`` column containing the
        formatted text prompt.
    """
    if pipeline_type == "brest":
        return _format_brest_scenario(df)
    elif pipeline_type == "aphp":
        return _format_aphp_scenario(df, **kwargs)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


def _format_brest_scenario(df: pl.DataFrame) -> pl.DataFrame:
    """Brest-specific scenario formatting (simple concatenation)."""
    das_str = df["DAS"].list.join(", ")
    das_str = pl.when(das_str == "").then(pl.lit("Aucun")).otherwise(das_str)

    ccam_str = df["CCAM"].list.join(", ")
    ccam_str = pl.when(ccam_str == "").then(pl.lit("Aucun")).otherwise(ccam_str)

    ghm5_display = pl.format("{}{{}} ({{}})", pl.col("GHM5"), pl.col("GHM5_CODE"))
    dp_display = pl.format("{}{{}} ({{}})", pl.col("DP"), pl.col("DP_CODE"))

    scenario = pl.concat_str(
        [
            pl.format("Patient : {{}}, {{}}.", pl.col("SEXE"), pl.col("AGE")),
            pl.format("\nGHM : {{}}.", ghm5_display),
            pl.format("\nDiagnostic principal : {{}}.", dp_display),
            pl.format("\nActes CCAM : {{}}.", ccam_str),
            pl.format("\nDiagnostics associés : {{}}.", das_str),
            pl.format("\nDurée de séjour : {{}} jours.", pl.col("DMS").cast(pl.Utf8)),
        ],
    )

    return df.with_columns(scenario.alias("scenario"))


def _format_aphp_scenario(
    df: pl.DataFrame,
    cancer_codes: frozenset[str],
    atih_rules: dict[str, dict],
) -> pl.DataFrame:
    """AP-HP-specific scenario formatting (rich clinical prompts)."""
    from pipelines.aphp import prompt

    user_prompts: list[str] = []
    system_prompts: list[str] = []

    for row in df.iter_rows(named=True):
        user_prompts.append(prompt.make_user_prompt(row, cancer_codes, atih_rules))
        system_prompts.append(prompt.load_system_prompt(row["template_name"]))

    return df.with_columns(
        pl.Series("scenario", user_prompts, dtype=pl.Utf8),
        pl.Series("system_prompt", system_prompts, dtype=pl.Utf8),
    )
