"""Sampling and random-generation primitives for the Brest pipeline.

This module contains helper functions for weighted sampling, CCAM fallback,
and DAS drawing used by the Brest pipeline.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _format_display(code: str, ref_map: dict[str, str]) -> str:
    """Format as ``'label (code)'``, or just ``code`` if no label found."""
    lib = ref_map.get(code)
    return f"{lib} ({code})" if lib else code


def _build_dms_lookup(dms_df: pl.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Build a GHM5 → (P25, P50, P75) lookup for triangular sampling."""
    lookup: dict[str, tuple[float, float, float]] = {}
    for row in dms_df.iter_rows(named=True):
        left, mode, right = row["DMS_P25"], row["DMS_P50"], row["DMS_P75"]
        if left == right:
            right = left + 1.0
        mode = max(left, min(mode, right))
        lookup[row["GHM5"]] = (left, mode, right)
    return lookup


def _build_ref_map(ref_df: pl.DataFrame) -> dict[str, str]:
    """Build a ``col[0] → col[1]`` dict from a two-column reference frame."""
    cols = ref_df.columns
    return dict(zip(ref_df[cols[0]].to_list(), ref_df[cols[1]].to_list()))


def _weighted_choice(weights: pl.Series, size: int, replace: bool = True) -> np.ndarray:
    """Weighted random draw via numpy. Weights are auto-normalised."""
    p = weights.to_numpy().astype(np.float64)
    p /= p.sum()
    return np.random.choice(len(p), size=size, replace=replace, p=p)


def _ccam_fallback(
    ccam_df: pl.DataFrame,
    row: dict,
    dp_cat: str,
) -> pl.DataFrame | None:
    """CCAM fallback cascade: GHM5+DP → GHM5+DP-category → GHM5."""
    filters = [
        (pl.col("GHM5") == row["GHM5"]) & (pl.col("DP") == row["DP"]),
        (pl.col("GHM5") == row["GHM5"]) & (pl.col("DP").str.starts_with(dp_cat)),
        pl.col("GHM5") == row["GHM5"],
    ]
    for f in filters:
        candidats = ccam_df.filter(f)
        if not candidats.is_empty() and candidats["P_CCAM"].sum() > 0:
            return candidats
    return None


def _draw_das_from_pools(pools: list[pl.DataFrame], n: int) -> list[str]:
    """Draw DAS codes sequentially with ICD-10 category exclusion and fallback."""
    deduped_pools = [
        pool.group_by("DAS").agg(pl.col("P_DAS").sum()) for pool in pools
    ]

    codes: list[str] = []
    cats: list[str] = []

    for _ in range(n):
        drawn = False
        for pool in deduped_pools:
            filtered = pool
            if codes:
                filtered = filtered.filter(~pl.col("DAS").is_in(codes))
            if cats:
                filtered = filtered.filter(
                    ~pl.col("DAS").str.slice(0, 2).is_in(cats)
                )
            if filtered.is_empty() or filtered["P_DAS"].sum() <= 0:
                continue
            idx = _weighted_choice(filtered["P_DAS"], size=1, replace=False)
            code = filtered[int(idx[0]), "DAS"]
            codes.append(code)
            cats.append(code[:2])
            drawn = True
            break

        if not drawn:
            break

    return codes
