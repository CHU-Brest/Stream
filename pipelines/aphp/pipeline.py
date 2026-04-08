"""AP-HP pipeline — ATIH PMSI sampling → clinical scenario → LLM report.

This module implements the AP-HP-specific logic for generating synthetic
medical reports from ATIH PMSI data. It inherits from the common
:class:`~pipelines.pipeline.BasePipeline` and overrides the specific methods
as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

import polars as pl

from pipelines.aphp import loader, managment, prompt
from pipelines.aphp import scenario as sc
from pipelines.aphp.fictive import generate_aphp_fictive
from pipelines.fictive import generate_fictive_stays
from pipelines.pipeline import BasePipeline
from pipelines.report import generate_reports
from pipelines.scenario import format_scenarios
from pipelines.aphp.scenario import format_aphp_scenario
from pipelines.aphp.report import generate_aphp_report


class APHPPipeline(BasePipeline):
    """AP-HP pipeline — ATIH/PMSI-based synthetic CRH generation.

    Source data (``scenarios_*.parquet``, ``bn_pmsi_related_diag_*.csv``,
    ``bn_pmsi_procedures_*.csv``) must be placed in the directory set by
    ``config["data"]["input"]`` (see ``config/servers.yaml``).
    """

    name = "aphp"

    # Stash for context objects built during get_fictive, re-used by get_scenario
    _sc_ctx: sc.ScenarioContext | None = None
    _mg_ctx: managment.ManagmentContext | None = None
    _atih_rules: dict[str, dict] | None = None

    # ------------------------------------------------------------------
    # 1 — check_data
    # ------------------------------------------------------------------

    @override
    def check_data(self) -> None:
        """Verify that the PMSI input directory exists and contains expected files."""
        self.logger.info("Vérification des données d'entrée pour le pipeline AP-HP.")
        input_dir = Path(self.config["data"]["input"])
        if not input_dir.is_dir():
            self.logger.error(
                "Le répertoire de données AP-HP est introuvable : %s", input_dir
            )
            raise FileNotFoundError(
                f"Le répertoire de données AP-HP est introuvable : {input_dir}\n"
                "Créez ce répertoire et déposez-y les fichiers PMSI "
                "(scenarios_*.parquet, bn_pmsi_related_diag_*.csv, bn_pmsi_procedures_*.csv)."
            )

        self.logger.info("Chargement des fichiers PMSI depuis %s.", input_dir)
        # Probe each expected pattern — raises FileNotFoundError with a clear message
        loader.load_pmsi(input_dir)
        self.logger.info("Fichiers PMSI chargés avec succès.")

        referentials_dir = Path(self.config["data"]["referentials"])
        if not referentials_dir.is_dir():
            self.logger.error(
                "Le répertoire des référentiels AP-HP est introuvable : %s",
                referentials_dir,
            )
            raise FileNotFoundError(
                f"Le répertoire des référentiels AP-HP est introuvable : {referentials_dir}\n"
                "Configurez 'data.referentials' dans servers.yaml et copiez-y les "
                "fichiers référentiels (Parquet/CSV)."
            )

        self.logger.info("Données AP-HP présentes dans %s.", input_dir)

    # ------------------------------------------------------------------
    # 2 — load_data
    # ------------------------------------------------------------------

    @override
    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Return all referentials + PMSI extracts as ``LazyFrame`` objects."""
        return loader.load_data(
            self.config["data"]["input"],
            self.config["data"]["referentials"],
        )

    # ------------------------------------------------------------------
    # 3 — get_fictive
    # ------------------------------------------------------------------

    @override
    def get_fictive(
        self,
        data: dict[str, pl.LazyFrame],
        n_sejours: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Sample *n_sejours* PMSI profiles and build one scenario dict per stay.

        Parameters
        ----------
        data:
            Dict returned by :meth:`load_data`.
        n_sejours:
            Number of fictitious stays to generate.
        seed:
            Optional integer seed for reproducibility (Python RNG + NumPy).

        Returns
        -------
        pl.DataFrame
            One row per scenario. Contains all clinical fields plus
            ``generation_id``, ``situa``, ``coding_rule``, ``template_name``.
        """
        return generate_fictive_stays(
            data,
            n_sejours=n_sejours,
            generate_fn=generate_aphp_fictive,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # 4 — get_scenario
    # ------------------------------------------------------------------

    @override
    def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add ``scenario`` (user prompt) and ``system_prompt`` columns to *df*.

        Requires :meth:`get_fictive` to have been called first (it stashes the
        context objects on ``self``).
        """
        # Load ATIH rules for scenario formatting
        atih_rules = loader.load_atih_rules()

        # Import here to avoid circular imports
        from pipelines.aphp import scenario as sc

        # Rebuild context to get cancer codes (this could be optimized)
        data = self.load_data()
        sc_ctx = sc.build_context(data)

        return format_scenarios(
            df,
            scenario_fn=format_aphp_scenario,
            cancer_codes=sc_ctx.cancer_codes,
            atih_rules=atih_rules,
        )

    # ------------------------------------------------------------------
    # 5 — get_report (override: per-row system prompt)
    # ------------------------------------------------------------------

    @override
    def get_report(
        self,
        df: pl.DataFrame,
        client: Any,
        model: str,
        batch_size: int = 1000,
    ) -> pl.DataFrame:
        """Generate one CRH per scenario using the row's own system prompt.

        Overrides :meth:`~pipelines.pipeline.BasePipeline.get_report` to read
        ``df_row["system_prompt"]`` instead of a single global system prompt.
        """
        output_dir = Path(self.config["data"]["output"])
        return generate_reports(
            df,
            client,
            model,
            batch_size=batch_size,
            output_dir=output_dir,
            generate_fn=generate_aphp_report,
            system_prompt="",  # just for signature since system from is in the df in the function generate_aphp_report
        )
