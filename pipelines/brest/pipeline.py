"""Brest pipeline — weighted PMSI sampling (SNDS source).

This module implements the Brest-specific logic for generating synthetic
medical reports from SNDS PMSI data. It inherits from the common
:class:`~pipelines.pipeline.BasePipeline` and overrides the specific methods
as needed.
"""

from __future__ import annotations

from typing import override
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from pipelines.brest import constants
from pipelines.brest.sampler import (
    build_dms_lookup,
    build_ref_map,
    ccam_fallback,
    draw_das_from_pools,
    format_display,
    weighted_choice,
)
from pipelines.fictive import generate_fictive_stays
from pipelines.pipeline import REPORT_SCHEMA, BasePipeline, flush_batch


class BrestPipeline(BasePipeline):
    """CHU Brest pipeline — weighted PMSI sampling (SNDS source).

    Source data is extracted via ``liabilities/extract_pmsi_tables_SNDS.sas``
    and placed as CSV files in the ``data.input`` directory configured in
    ``servers.yaml``.
    """

    name = "brest"

    SOURCES = constants.SOURCES

    # -- Data loading ------------------------------------------------------

    @override
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

    @override
    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Load all PMSI Parquet files as LazyFrames."""
        input_dir = Path(self.config["data"]["input"])
        return {
            name: pl.scan_parquet(input_dir / f"{name}.parquet")
            for name in self.SOURCES
        }

    # -- Fictitious stay generation ----------------------------------------

    @override
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
            generate_fn=generate_brest_fictive,
            n_ccam=n_ccam,
            n_das=n_das,
            ghm5_pattern=ghm5_pattern,
        )

    # -- Scenario formatting -----------------------------------------------

    @override
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

    @override
    def get_report(
        self,
        df: pl.DataFrame,
        client: AnthropicClient | MistralClient | OllamaClient,
        model: str,
        batch_size: int = 1000,
    ) -> pl.DataFrame:
        """Generate one medical report per scenario via LLM calls.

        Results are flushed to timestamped Parquet files every
        *batch_size* rows in the configured output directory.

        Returns
        -------
        pl.DataFrame
            Columns: generation_id, scenario, report, model, timestamp.
        """
        if "generation_id" not in df.columns:
            raise ValueError(
                "Le DataFrame doit contenir une colonne 'generation_id'. "
                "Assurez-vous de passer par get_fictive avant get_report."
            )

        system_prompt: str = self.prompt["generate"]["system_prompt"]
        output_dir = Path(self.config["data"]["output"])
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        batch: list[dict] = []

        for df_row in tqdm(
            df.iter_rows(named=True), desc="Génération CRH", unit="crh", total=len(df)
        ):
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": df_row["scenario"]},
                ],
            )
            row = {
                "generation_id": df_row["generation_id"],
                "scenario": df_row["scenario"],
                "report": response["message"]["content"],
                "model": model,
                "timestamp": datetime.now(),
            }
            results.append(row)
            batch.append(row)

            if len(batch) >= batch_size:
                flush_batch(batch, output_dir, batch_size)
                batch = []

        if batch:
            flush_batch(batch, output_dir, len(batch))

        return pl.DataFrame(results, schema=REPORT_SCHEMA)


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
