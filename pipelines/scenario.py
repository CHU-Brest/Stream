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
    scenario_fn,
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
    return scenario_fn(df, **kwargs)
