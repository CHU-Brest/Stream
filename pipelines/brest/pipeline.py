"""Brest pipeline — weighted PMSI sampling (SNDS source).

This module implements the Brest-specific logic for generating synthetic
medical reports from SNDS PMSI data. It inherits from the common
:class:`~pipelines.pipeline.BasePipeline` and overrides the specific methods
as needed.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from pipelines.pipeline import BasePipeline


class BrestPipeline(BasePipeline):
    """CHU Brest pipeline — weighted PMSI sampling (SNDS source).

    Source data is extracted via ``liabilities/extract_pmsi_tables_SNDS.sas``
    and placed as CSV files in the ``data.input`` directory configured in
    ``servers.yaml``.
    """

    name = "brest"

    SOURCES = {
        "dp": "PMSI_DP.csv",
        "ccam": "PMSI_CCAM_DP.csv",
        "das": "PMSI_DAS.csv",
        "dms": "PMSI_DMS.csv",
        "rghm": "ALL_CLASSIF_PMSI.csv",
        "cim10": "ALL_CIM10.csv",
        "ccam_ref": "ALL_CCAM.csv",
    }

    # -- Data loading ------------------------------------------------------

    def check_data(self) -> None:
        """Convert PMSI CSV files to Parquet if not already present."""
        input_dir = Path(self.config["data"]["input"])
        self.logger.info("Vérification des données d'entrée pour le pipeline Brest.")

        for name, csv_file in self.SOURCES.items():
            parquet = input_dir / f"{name}.parquet"
            if parquet.exists():
                self.logger.debug("Le fichier Parquet %s existe déjà.", parquet)
                continue
            csv_path = input_dir / csv_file
            if not csv_path.exists():
                raise FileNotFoundError(f"Le fichier CSV {csv_path} est introuvable.")
            self.logger.info("Conversion du fichier CSV %s en Parquet.", csv_path)
            pl.read_csv(
                csv_path,
                separator=";",
                encoding="latin-1",
                infer_schema_length=10000,
            ).write_parquet(parquet)
            self.logger.info("Fichier Parquet %s créé avec succès.", parquet)

        self.logger.info("Les données de génération sont présentes et valides.")

    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Load all PMSI Parquet files as LazyFrames."""
        input_dir = Path(self.config["data"]["input"])
        return {
            name: pl.scan_parquet(input_dir / f"{name}.parquet")
            for name in self.SOURCES
        }

    # -- Fictitious stay generation ----------------------------------------

    def get_fictive(
        self,
        data: dict[str, pl.LazyFrame],
        n_sejours: int = 1000,
        n_ccam: int = 3,
        n_das: int = 3,
        ghm5_pattern: str | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        """Generate fictitious stays via weighted PMSI sampling.

        Returns
        -------
        pl.DataFrame
            Columns: generation_id, AGE, SEXE, GHM5, GHM5_CODE, DP, DP_CODE,
            CCAM (list[str]), DAS (list[str]), DMS (int).
        """
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
        dms_lookup = _build_dms_lookup(data["dms"].collect())
        cim10_map = _build_ref_map(
            data["cim10"]
            .filter(pl.col("en_cours") == 1)
            .select("code", "liblong")
            .collect(),
        )
        ccam_map = _build_ref_map(
            data["ccam_ref"]
            .filter(pl.col("en_cours") == 1)
            .select("code", "liblong")
            .collect(),
        )
        ghm5_map = _build_ref_map(
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
        indices = _weighted_choice(dp_df["P_DP"], size=n_sejours, replace=True)
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
            candidats = _ccam_fallback(ccam_df, row, dp_cat)
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
            idx = _weighted_choice(top["P_CCAM"], size=n, replace=False)
            actes.append([_format_display(top[int(j), "CCAM"], ccam_map) for j in idx])
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

            codes_drawn = _draw_das_from_pools(pools, n_target)
            das_list.append([_format_display(c, cim10_map) for c in codes_drawn])
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

    # -- Scenario formatting -----------------------------------------------

    def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
        """Format fictitious stays as text scenarios for the LLM."""
        das_str = df["DAS"].list.join(", ")
        das_str = pl.when(das_str == "").then(pl.lit("Aucun")).otherwise(das_str)

        ccam_str = df["CCAM"].list.join(", ")
        ccam_str = pl.when(ccam_str == "").then(pl.lit("Aucun")).otherwise(ccam_str)

        ghm5_display = pl.format("{}{} ({})", pl.col("GHM5"), pl.col("GHM5_CODE"))
        dp_display = pl.format("{}{} ({})", pl.col("DP"), pl.col("DP_CODE"))

        scenario = pl.concat_str(
            [
                pl.format("Patient : {}, {}.", pl.col("SEXE"), pl.col("AGE")),
                pl.format("\nGHM : {}.", ghm5_display),
                pl.format("\nDiagnostic principal : {}.", dp_display),
                pl.format("\nActes CCAM : {}.", ccam_str),
                pl.format("\nDiagnostics associés : {}.", das_str),
                pl.format("\nDurée de séjour : {} jours.", pl.col("DMS").cast(pl.Utf8)),
            ],
        )

        return df.with_columns(scenario.alias("scenario"))


# -- Helpers (Brest-specific) ----------------------------------------------


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
