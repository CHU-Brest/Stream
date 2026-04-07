"""Fictive stay generation — common primitives for sampling hospital stays.

This module provides shared building blocks for generating fictitious
hospital stays from PMSI data. It is used by both the Brest and AP-HP
pipelines to ensure consistent sampling logic and reduce code duplication.

Responsibilities:
- Generate fictitious hospital stays from PMSI data (Brest/AP-HP).
- Ensure consistent sampling logic across pipelines.
- Reduce code duplication between pipelines.

Public API:
- :func:`generate_fictive_stays`: Generate fictitious stays for a given pipeline.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Common fictive-generation helpers
# ---------------------------------------------------------------------------


def generate_fictive_stays(
    data: dict[str, pl.LazyFrame],
    n_sejours: int = 1000,
    pipeline_type: str = "brest",
    **kwargs: Any,
) -> pl.DataFrame:
    """Generate fictitious stays via weighted PMSI sampling.

    Parameters
    ----------
    data:
        Dict of LazyFrames returned by the pipeline's ``load_data`` method.
    n_sejours:
        Number of fictitious stays to generate.
    pipeline_type:
        Either "brest" or "aphp" to select the appropriate sampling logic.
    **kwargs:
        Additional pipeline-specific arguments (e.g., ``n_ccam``, ``n_das``
        for Brest; ``seed`` for AP-HP).

    Returns
    -------
    pl.DataFrame
        One row per fictitious stay. Columns depend on the pipeline type.
    """
    if pipeline_type == "brest":
        return _generate_brest_fictive(data, n_sejours, **kwargs)
    elif pipeline_type == "aphp":
        return _generate_aphp_fictive(data, n_sejours, **kwargs)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


def _generate_brest_fictive(
    data: dict[str, pl.LazyFrame],
    n_sejours: int = 1000,
    n_ccam: int = 3,
    n_das: int = 3,
    ghm5_pattern: str | None = None,
) -> pl.DataFrame:
    """Brest-specific fictive generation (weighted PMSI sampling)."""
    from pipelines.brest import constants, sampler

    steps = [
        "Chargement référentiels",
        "Tirage DP",
        "Tirage CCAM",
        "Tirage DAS",
        "Tirage DMS",
        "Résolution libellés",
    ]
    pbar = tqdm(steps, desc="Génération scénarios", unit="étape")

    # 0 — Load reference tables
    pbar.set_description("Chargement référentiels")
    dp_df = data["dp"].collect()
    ccam_df = data["ccam"].collect()
    das_df = data["das"].collect()
    dms_lookup = sampler._build_dms_lookup(data["dms"].collect())
    cim10_map = sampler._build_ref_map(
        data["cim10"]
        .filter(pl.col("en_cours") == 1)
        .select("code", "liblong")
        .collect(),
    )
    ccam_map = sampler._build_ref_map(
        data["ccam_ref"]
        .filter(pl.col("en_cours") == 1)
        .select("code", "liblong")
        .collect(),
    )
    ghm5_map = sampler._build_ref_map(
        data["rghm"]
        .filter(
            (pl.col("champ") == "mco")
            & (pl.col("version") == "v2024")
            & (pl.col("type_code") == "racine")
        )
        .select("code", "lib")
        .collect(),
    )
    if ghm5_pattern is not None:
        dp_df = dp_df.filter(pl.col("GHM5").str.contains(ghm5_pattern))
        if dp_df.is_empty():
            pbar.close()
            raise ValueError(f"Aucun séjour DP ne correspond au pattern GHM5 '{ghm5_pattern}'")
    pbar.update(1)

    # 1 — Weighted primary-diagnosis sampling
    pbar.set_description("Tirage DP")
    dp_df = dp_df.filter(pl.col("DP").is_not_null() & pl.col("GHM5").is_not_null())
    if dp_df.is_empty():
        pbar.close()
        raise ValueError("Aucun séjour valide après exclusion des DP/GHM5 null.")
    indices = sampler._weighted_choice(dp_df["P_DP"], size=n_sejours, replace=True)
    sampled = dp_df[indices].select("AGE", "SEXE", "GHM5", "DP")
    pbar.update(1)

    # 2 — CCAM procedure sampling (surgical GHMs only)
    pbar.set_description("Tirage CCAM")
    actes: list[list[str]] = []
    for row in sampled.iter_rows(named=True):
        if "C" not in row["GHM5"] and "K" not in row["GHM5"]:
            actes.append([])
            continue
        dp_cat = row["DP"][:3]
        candidats = sampler._ccam_fallback(ccam_df, row, dp_cat)
        if candidats is None:
            actes.append([])
            continue
        top = (
            candidats.group_by("CCAM")
            .agg(pl.col("P_CCAM").sum())
            .sort("P_CCAM", descending=True)
            .head(5)
        )
        n_target = np.random.randint(1, n_ccam + 1)
        n = min(n_target, len(top))
        idx = sampler._weighted_choice(top["P_CCAM"], size=n, replace=False)
        actes.append([sampler._format_display(top[int(j), "CCAM"], ccam_map) for j in idx])
    pbar.update(1)

    # 3 — Associated-diagnosis sampling (cascading fallback)
    pbar.set_description("Tirage DAS")
    das_list: list[list[str]] = []
    for row in sampled.iter_rows(named=True):
        n_target = np.random.randint(0, n_das + 1)
        if n_target == 0:
            das_list.append([])
            continue

        ghm5_filter = pl.col("GHM5") == row["GHM5"]
        pools = [
            das_df.filter(
                ghm5_filter
                & (pl.col("AGE") == row["AGE"])
                & (pl.col("SEXE") == row["SEXE"])
                & (pl.col("DP") == row["DP"])
            ),
            das_df.filter(
                ghm5_filter & (pl.col("SEXE") == row["SEXE"]) & (pl.col("DP") == row["DP"])
            ),
            das_df.filter(ghm5_filter & (pl.col("DP") == row["DP"])),
            das_df.filter(
                ghm5_filter & (pl.col("DP").str.starts_with(row["DP"][:3]))
            ),
            das_df.filter(ghm5_filter),
        ]

        codes_drawn = sampler._draw_das_from_pools(pools, n_target)
        das_list.append([sampler._format_display(c, cim10_map) for c in codes_drawn])
    pbar.update(1)

    # 4 — Length-of-stay sampling (triangular distribution)
    pbar.set_description("Tirage DMS")
    dms_list: list[int] = []
    for row in sampled.iter_rows(named=True):
        params = dms_lookup.get(row["GHM5"])
        if params is None:
            dms_list.append(1)
        else:
            left, mode, right = params
            val = np.random.triangular(left, mode, right)
            dms_list.append(max(0, int(round(val))))
    pbar.update(1)

    # 5 — Resolve codes to labels
    pbar.set_description("Résolution libellés")
    result = sampled.with_columns(
        pl.Series("CCAM", actes, dtype=pl.List(pl.Utf8)),
        pl.Series("DAS", das_list, dtype=pl.List(pl.Utf8)),
        pl.Series("DMS", dms_list),
    )
    result = result.with_columns(
        pl.col("GHM5").alias("GHM5_CODE"),
        pl.col("DP").alias("DP_CODE"),
    )
    result = result.with_columns(
        pl.col("GHM5").replace_strict(ghm5_map, default=pl.col("GHM5")),
        pl.col("DP").replace_strict(cim10_map, default=pl.col("DP")),
    )

    generation_ids = [str(uuid.uuid4()) for _ in range(len(result))]
    result = result.with_columns(
        pl.Series("generation_id", generation_ids, dtype=pl.Utf8)
    )

    pbar.update(1)
    pbar.close()

    return result


def _generate_aphp_fictive(
    data: dict[str, pl.LazyFrame],
    n_sejours: int = 10,
    seed: int | None = None,
) -> pl.DataFrame:
    """AP-HP-specific fictive generation (scenario-based sampling)."""
    import random as _random
    from tqdm import tqdm

    from pipelines.aphp import scenario as sc, managment

    rng = _random.Random(seed)
    np_rng = np.random.default_rng(seed)

    steps = ["Contexte", "Profils", "Scénarios"]
    pbar = tqdm(steps, desc="AP-HP génération", unit="étape")

    # -- Build context objects (materialise all code sets)
    pbar.set_description("Construction contexte")
    sc_ctx = sc.build_context(data)
    mg_ctx = managment.build_context(data, sc_ctx.cancer_codes, sc_ctx.chronic_codes)
    pbar.update(1)

    # -- Prepare profiles: join descriptions + LOS stats + specialty
    pbar.set_description("Chargement profils")
    profiles_df = sc_ctx.profiles

    # Join drg_parent_description if not already present
    if "drg_parent_description" not in profiles_df.columns:
        drg_descr = (
            data["drg_parents_groups"]
            .select(["drg_parent_code", "drg_parent_description"])
            .collect()
        )
        profiles_df = profiles_df.join(drg_descr, on="drg_parent_code", how="left")

    # Join LOS stats (los_mean, los_sd) if absent
    missing_los = "los_mean" not in profiles_df.columns or "los_sd" not in profiles_df.columns
    if missing_los:
        los_stats = (
            data["drg_statistics"]
            .select(["drg_parent_code", "los_mean", "los_sd"])
            .collect()
        )
        profiles_df = profiles_df.join(los_stats, on="drg_parent_code", how="left")

    # Join dominant specialty if absent
    if "specialty" not in profiles_df.columns:
        spe = (
            data["specialty"]
            .collect()
            .group_by("drg_parent_code")
            .agg(pl.col("specialty").first())
        )
        profiles_df = profiles_df.join(spe, on="drg_parent_code", how="left")

    # Weighted sampling (weight = nb count)
    if "nb" in profiles_df.columns:
        weights = profiles_df["nb"].cast(pl.Float64).fill_null(1.0).to_numpy().copy()
    else:
        weights = np.ones(len(profiles_df))
    weights /= weights.sum()

    indices = np_rng.choice(len(profiles_df), size=n_sejours, replace=True, p=weights)
    sampled = profiles_df[indices]
    pbar.update(1)

    # -- Build one scenario per sampled profile
    pbar.set_description("Construction scénarios")
    rows: list[dict[str, Any]] = []
    for profile in tqdm(
        sampled.iter_rows(named=True),
        desc="Scénarios",
        unit="séjour",
        total=n_sejours,
        leave=False,
    ):
        sc_dict = sc.build_scenario(sc_ctx, profile, rng=rng, np_rng=np_rng)

        # Management decision (coding rule + situa text + template)
        dec = managment.define_managment_type(sc_dict, mg_ctx, np_rng=np_rng)
        sc_dict["situa"] = dec.situa
        sc_dict["coding_rule"] = dec.coding_rule
        sc_dict["template_name"] = dec.template_name

        sc_dict["generation_id"] = str(uuid.uuid4())
        rows.append(sc_dict)

    pbar.update(1)
    pbar.close()

    return _dicts_to_df(rows)


def _dicts_to_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Convert a list of scenario dicts to a Polars DataFrame.

    Polars requires all series to have the same type. We cast list-valued
    columns (``icd_secondary_code``) to ``pl.List(pl.Utf8)`` explicitly and
    fall back to ``pl.Utf8`` for anything else that isn't already a scalar.
    """
    if not rows:
        return pl.DataFrame()

    columns: dict[str, pl.Series] = {}
    all_keys = list(rows[0].keys())

    for key in all_keys:
        values = [r.get(key) for r in rows]
        # List-valued columns → List(Utf8)
        if any(isinstance(v, list) for v in values):
            columns[key] = pl.Series(
                key,
                [v if isinstance(v, list) else [] for v in values],
                dtype=pl.List(pl.Utf8),
            )
        else:
            try:
                columns[key] = pl.Series(key, values)
            except Exception:
                # Last resort: stringify everything
                columns[key] = pl.Series(key, [str(v) if v is not None else None for v in values], dtype=pl.Utf8)

    return pl.DataFrame(columns)
