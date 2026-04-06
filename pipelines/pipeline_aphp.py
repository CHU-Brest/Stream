"""AP-HP pipeline — ATIH PMSI sampling → clinical scenario → LLM report.

Implements the AP-HP method from ``24p11/recode-scenario`` inside the Stream
:class:`~pipelines.base.BasePipeline` contract. The five stages map to:

1. :meth:`check_data`   — verify PMSI input files are present.
2. :meth:`load_data`    — lazy-load referentials + PMSI extracts as LazyFrames.
3. :meth:`get_fictive`  — weighted PMSI sampling → one scenario dict per stay.
4. :meth:`get_scenario` — format each scenario dict into user / system prompts.
5. :meth:`get_report`   — LLM generation loop (overrides base to use a
                          per-row system prompt instead of a global one).
"""

from __future__ import annotations

import random as _random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

from pipelines.base import REPORT_SCHEMA, BasePipeline, _flush_batch
from pipelines.aphp import loader, managment, prompt, scenario as sc


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

    def check_data(self) -> None:
        """Verify that the PMSI input directory exists and contains expected files."""
        input_dir = Path(self.config["data"]["input"])
        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"Le répertoire de données AP-HP est introuvable : {input_dir}\n"
                "Créez ce répertoire et déposez-y les fichiers PMSI "
                "(scenarios_*.parquet, bn_pmsi_related_diag_*.csv, bn_pmsi_procedures_*.csv)."
            )

        # Probe each expected pattern — raises FileNotFoundError with a clear message
        loader.load_pmsi(input_dir)

        self.logger.info("Données AP-HP présentes dans %s.", input_dir)

    # ------------------------------------------------------------------
    # 2 — load_data
    # ------------------------------------------------------------------

    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Return all referentials + PMSI extracts as ``LazyFrame`` objects."""
        return loader.load_data(self.config["data"]["input"])

    # ------------------------------------------------------------------
    # 3 — get_fictive
    # ------------------------------------------------------------------

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
        rng = _random.Random(seed)
        np_rng = np.random.default_rng(seed)

        steps = ["Contexte", "Profils", "Scénarios"]
        pbar = tqdm(steps, desc="AP-HP génération", unit="étape")

        # -- Build context objects (materialise all code sets)
        pbar.set_description("Construction contexte")
        sc_ctx = sc.build_context(data)
        mg_ctx = managment.build_context(data, sc_ctx.cancer_codes, sc_ctx.chronic_codes)
        atih_rules = loader.load_atih_rules()

        # Stash for get_scenario
        self._sc_ctx = sc_ctx
        self._mg_ctx = mg_ctx
        self._atih_rules = atih_rules
        pbar.update(1)

        # -- Prepare profiles: join descriptions + LOS stats + specialty
        pbar.set_description("Chargement profils")
        profiles_df = sc_ctx.profiles  # already collected

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

    # ------------------------------------------------------------------
    # 4 — get_scenario
    # ------------------------------------------------------------------

    def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add ``scenario`` (user prompt) and ``system_prompt`` columns to *df*.

        Requires :meth:`get_fictive` to have been called first (it stashes the
        context objects on ``self``).
        """
        if self._sc_ctx is None or self._atih_rules is None:
            raise RuntimeError(
                "Appelez get_fictive() avant get_scenario() : le contexte n'est pas "
                "encore initialisé."
            )

        cancer_codes = self._sc_ctx.cancer_codes
        atih_rules = self._atih_rules

        user_prompts: list[str] = []
        system_prompts: list[str] = []

        for row in tqdm(
            df.iter_rows(named=True),
            desc="Formatage prompts",
            unit="séjour",
            total=len(df),
        ):
            user_prompts.append(prompt.make_user_prompt(row, cancer_codes, atih_rules))
            system_prompts.append(prompt.load_system_prompt(row["template_name"]))

        return df.with_columns(
            pl.Series("scenario", user_prompts, dtype=pl.Utf8),
            pl.Series("system_prompt", system_prompts, dtype=pl.Utf8),
        )

    # ------------------------------------------------------------------
    # 5 — get_report (override: per-row system prompt)
    # ------------------------------------------------------------------

    def get_report(
        self,
        df: pl.DataFrame,
        client: Any,
        model: str,
        batch_size: int = 1000,
    ) -> pl.DataFrame:
        """Generate one CRH per scenario using the row's own system prompt.

        Overrides :meth:`~pipelines.base.BasePipeline.get_report` to read
        ``df_row["system_prompt"]`` instead of a single global system prompt.
        """
        if "generation_id" not in df.columns:
            raise ValueError(
                "Le DataFrame doit contenir une colonne 'generation_id'. "
                "Assurez-vous de passer par get_fictive avant get_report."
            )
        if "system_prompt" not in df.columns:
            raise ValueError(
                "Le DataFrame doit contenir une colonne 'system_prompt'. "
                "Assurez-vous de passer par get_scenario avant get_report."
            )

        output_dir = Path(self.config["data"]["output"])
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        batch: list[dict] = []

        for df_row in tqdm(
            df.iter_rows(named=True),
            desc="Génération CRH",
            unit="crh",
            total=len(df),
        ):
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": df_row["system_prompt"]},
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
                _flush_batch(batch, output_dir, batch_size)
                batch = []

        if batch:
            _flush_batch(batch, output_dir, len(batch))

        return pl.DataFrame(results, schema=REPORT_SCHEMA)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
