import uuid

import numpy as np
import polars as pl
from tqdm import tqdm

from pipelines.brest.sampler import (
    build_dms_lookup,
    build_ref_map,
    ccam_fallback,
    draw_das_from_pools,
    format_display,
    weighted_choice,
)


def generate_brest_fictive(
    data: dict[str, pl.LazyFrame],
    n_sejours: int = 1000,
    n_ccam: int = 3,
    n_das: int = 3,
    ghm5_pattern: str | None = None,
) -> pl.DataFrame:
    """Brest-specific fictive generation (weighted PMSI sampling)."""

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
    dms_lookup = build_dms_lookup(data["dms"].collect())
    cim10_map = build_ref_map(
        data["cim10"]
        .filter(pl.col("en_cours") == 1)
        .select("code", "liblong")
        .collect(),
    )
    ccam_map = build_ref_map(
        data["ccam_ref"]
        .filter(pl.col("en_cours") == 1)
        .select("code", "liblong")
        .collect(),
    )
    ghm5_map = build_ref_map(
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
            raise ValueError(
                f"Aucun séjour DP ne correspond au pattern GHM5 '{ghm5_pattern}'"
            )
    pbar.update(1)

    # 1 — Weighted primary-diagnosis sampling
    pbar.set_description("Tirage DP")
    dp_df = dp_df.filter(pl.col("DP").is_not_null() & pl.col("GHM5").is_not_null())
    if dp_df.is_empty():
        pbar.close()
        raise ValueError("Aucun séjour valide après exclusion des DP/GHM5 null.")
    indices = weighted_choice(dp_df["P_DP"], size=n_sejours, replace=True)
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
        candidats = ccam_fallback(ccam_df, row, dp_cat)
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
        idx = weighted_choice(top["P_CCAM"], size=n, replace=False)
        actes.append([format_display(top[int(j), "CCAM"], ccam_map) for j in idx])
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
                ghm5_filter
                & (pl.col("SEXE") == row["SEXE"])
                & (pl.col("DP") == row["DP"])
            ),
            das_df.filter(ghm5_filter & (pl.col("DP") == row["DP"])),
            das_df.filter(ghm5_filter & (pl.col("DP").str.starts_with(row["DP"][:3]))),
            das_df.filter(ghm5_filter),
        ]

        codes_drawn = draw_das_from_pools(pools, n_target)
        das_list.append([format_display(c, cim10_map) for c in codes_drawn])
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
