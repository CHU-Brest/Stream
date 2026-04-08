"""AP-HP pipeline — ATIH PMSI sampling → clinical scenario → LLM report.

This module implements the AP-HP-specific logic for generating synthetic
medical reports from ATIH PMSI data. It inherits from the common
:class:`~pipelines.pipeline.BasePipeline` and overrides the specific methods
as needed.
"""

from __future__ import annotations

import random as _random
import uuid
from pathlib import Path
from typing import Any, override

import numpy as np
import polars as pl
from tqdm import tqdm

from pipelines.aphp import loader, managment, prompt
from pipelines.aphp import scenario as sc
from pipelines.fictive import generate_fictive_stays
from pipelines.pipeline import BasePipeline
from pipelines.report import generate_reports
from pipelines.scenario import format_scenarios


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
            pipeline_type="aphp",
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
        return generate_reports(
            df,
            client,
            model,
            batch_size=batch_size,
            output_dir=self.config["data"]["output"],
            pipeline_type="aphp",
        )


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
