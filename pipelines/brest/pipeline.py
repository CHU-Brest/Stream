"""Brest pipeline — weighted PMSI sampling (SNDS source).

This module implements the Brest-specific logic for generating synthetic
medical reports from SNDS PMSI data. It inherits from the common
:class:`~pipelines.pipeline.BasePipeline` and overrides the specific methods
as needed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import override

import polars as pl
from tqdm import tqdm

from pipelines.brest import constants
from pipelines.brest.fictive import generate_brest_fictive
from pipelines.brest.scenario import format_brest_scenario
from pipelines.fictive import generate_fictive_stays
from pipelines.scenario import format_scenarios
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
        return format_scenarios(df, scenario_fn=format_brest_scenario)

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
