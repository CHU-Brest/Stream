"""Brest pipeline — weighted PMSI sampling (SNDS source).

This module implements the Brest-specific logic for generating synthetic
medical reports from SNDS PMSI data. It inherits from the common
:class:`~pipelines.pipeline.BasePipeline` and overrides the specific methods
as needed.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pipelines.brest import constants
from pipelines.fictive import generate_fictive_stays
from pipelines.pipeline import BasePipeline


class BrestPipeline(BasePipeline):
    """CHU Brest pipeline — weighted PMSI sampling (SNDS source).

    Source data is extracted via ``liabilities/extract_pmsi_tables_SNDS.sas``
    and placed as CSV files in the ``data.input`` directory configured in
    ``servers.yaml``.
    """

    name = "brest"

    SOURCES = constants.SOURCES

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
        return generate_fictive_stays(
            data,
            n_sejours=n_sejours,
            pipeline_type="brest",
            n_ccam=n_ccam,
            n_das=n_das,
            ghm5_pattern=ghm5_pattern,
        )

    # -- Scenario formatting -----------------------------------------------

    def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
        """Format fictitious stays as text scenarios for the LLM."""
        das_str = df["DAS"].list.join(", ")
        das_str = pl.when(das_str == "").then(pl.lit("Aucun")).otherwise(das_str)

        ccam_str = df["CCAM"].list.join(", ")
        ccam_str = pl.when(ccam_str == "").then(pl.lit("Aucun")).otherwise(ccam_str)

        ghm5_display = pl.format("{} ({})", pl.col("GHM5"), pl.col("GHM5_CODE"))
        dp_display = pl.format("{} ({})", pl.col("DP"), pl.col("DP_CODE"))

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
    deduped_pools = [pool.group_by("DAS").agg(pl.col("P_DAS").sum()) for pool in pools]

    codes: list[str] = []
    cats: list[str] = []

    for _ in range(n):
        drawn = False
        for pool in deduped_pools:
            filtered = pool
            if codes:
                filtered = filtered.filter(~pl.col("DAS").is_in(codes))
            if cats:
                filtered = filtered.filter(~pl.col("DAS").str.slice(0, 2).is_in(cats))
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
