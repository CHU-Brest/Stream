import random as _random
import uuid
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

from pipelines.aphp import managment
from pipelines.aphp import scenario as sc


# The _dicts_to_df helper is now in pipelines.fictive.py
def generate_aphp_fictive(
    data: dict[str, pl.LazyFrame],
    n_sejours: int = 10,
    seed: int | None = None,
) -> pl.DataFrame:
    """AP-HP-specific fictive generation (scenario-based sampling)."""
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
    missing_los = (
        "los_mean" not in profiles_df.columns or "los_sd" not in profiles_df.columns
    )
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
                columns[key] = pl.Series(
                    key,
                    [str(v) if v is not None else None for v in values],
                    dtype=pl.Utf8,
                )

    return pl.DataFrame(columns)
