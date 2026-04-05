import os
import uuid
from abc import ABC
from datetime import datetime
from typing import Protocol, runtime_checkable

import httpx
import numpy as np
import polars as pl
from tqdm import tqdm

from core.logging import get_logger


# =============================================================================
# Interface et wrappers LLM
# =============================================================================


@runtime_checkable
class LLMClient(Protocol):
    """Interface commune à tous les clients LLM.

    Chaque wrapper expose ``chat(model, messages, **kwargs) -> dict``
    renvoyant ``{"message": {"role": "assistant", "content": str}}``.
    """

    def chat(self, model: str, messages: list, **kwargs) -> dict: ...


class AnthropicClient:
    """Client Anthropic (Claude) normalisé sur l'interface LLMClient.

    Parameters
    ----------
    api_key : str
        Clé API Anthropic.
    verify : bool | str
        ``True`` (défaut), ``False``, ou chemin vers un bundle CA (.pem).
    """

    def __init__(self, api_key: str, verify: bool | str = True):
        from anthropic import Anthropic

        http_client = httpx.Client(verify=verify)
        self._client = Anthropic(api_key=api_key, http_client=http_client)

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        params = dict(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=chat_messages,
        )
        if system:
            params["system"] = system

        response = self._client.messages.create(**params)  # type: ignore

        return {
            "message": {
                "role": "assistant",
                "content": response.content[0].text,
            }
        }


class MistralClient:
    """Client Mistral normalisé sur l'interface LLMClient.

    Parameters
    ----------
    api_key : str
        Clé API Mistral.
    verify : bool | str
        ``True`` (défaut), ``False``, ou chemin vers un bundle CA (.pem).
    """

    def __init__(self, api_key: str, verify: bool | str = True):
        from mistralai import Mistral

        self._client = Mistral(api_key=api_key)
        if verify is not True:
            self._client.client._client = httpx.Client(verify=verify)

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        response = self._client.chat.complete(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=messages,
        )

        return {
            "message": {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
        }


# =============================================================================
# Pipeline de base
# =============================================================================


class BasePipeline(ABC):
    """Pipeline abstrait de génération de CRH synthétiques.

    Contient toute la logique de chargement, d'échantillonnage et de génération
    LLM. Les pipelines concrets (Brest, AP-HP) héritent de cette classe et
    configurent la source de données et les paramètres spécifiques.

    Parameters
    ----------
    config : dict
        Configuration propre au pipeline (sections ``data.input`` / ``data.output``).
    prompt : dict
        Contenu chargé depuis ``config/prompts.yaml``.
    servers : dict
        Configuration des serveurs LLM (ollama, claude, mistral).
    """

    name: str = "base"

    def __init__(self, config: dict, prompt: dict, servers: dict):
        self.config = config
        self.prompt = prompt
        self.servers = servers
        self.logger = get_logger(self.name)

    # ------------------------------------------------------------------
    # Données
    # ------------------------------------------------------------------

    def check_data(self) -> None:
        """Vérifie et convertit les CSV PMSI en Parquet si nécessaire."""
        input_dir = self.config["data"]["input"]
        sources = {
            "dp": "PMSI_DP.csv",
            "ccam": "PMSI_CCAM_DP.csv",
            "das": "PMSI_DAS.csv",
            "dms": "PMSI_DMS.csv",
            "rghm": "ALL_CLASSIF_PMSI.csv",
            "cim10": "ALL_CIM10.csv",
            "ccam_ref": "ALL_CCAM.csv",
        }

        for name, csv_file in sources.items():
            parquet = os.path.join(input_dir, f"{name}.parquet")
            if not os.path.exists(parquet) or pl.scan_parquet(parquet).collect().is_empty():
                pl.read_csv(
                    os.path.join(input_dir, csv_file),
                    separator=";",
                    encoding="latin-1",
                    infer_schema_length=10000,
                ).write_parquet(parquet)

        self.logger.info("Les données de génération sont présentes et valides.")

    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Charge tous les Parquet PMSI en LazyFrame."""
        input_dir = self.config["data"]["input"]
        names = ["dp", "ccam", "das", "dms", "rghm", "cim10", "ccam_ref"]
        return {name: pl.scan_parquet(os.path.join(input_dir, f"{name}.parquet")) for name in names}

    # ------------------------------------------------------------------
    # Génération de séjours fictifs
    # ------------------------------------------------------------------

    def get_fictive(
        self,
        data: dict[str, pl.LazyFrame],
        n_sejours: int = 1000,
        n_ccam: int = 3,
        n_das: int = 3,
        ghm5_pattern: str | None = None,
    ) -> pl.DataFrame:
        """Génère des séjours fictifs par tirage pondéré sur les référentiels DP, CCAM et DAS.

        Parameters
        ----------
        data : dict[str, pl.LazyFrame]
            Clés attendues : "dp", "ccam", "das", "dms", "cim10", "ccam_ref", "rghm".
        n_sejours : int
            Nombre de séjours à tirer.
        n_ccam : int
            Nombre max d'actes CCAM à tirer par séjour.
        n_das : int
            Nombre max de DAS à tirer par séjour.
        ghm5_pattern : str | None
            Si renseigné, filtre les GHM5 contenant ce pattern (ex. "06C").

        Returns
        -------
        pl.DataFrame
            Colonnes : generation_id, AGE, SEXE, GHM5, GHM5_CODE, DP, DP_CODE,
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

        # --- 0. Chargement des référentiels -----------------------------------
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
            key_col="code",
            val_col="liblong",
        )
        ccam_map = _build_ref_map(
            data["ccam_ref"]
            .filter(pl.col("en_cours") == 1)
            .select("code", "liblong")
            .collect(),
            key_col="code",
            val_col="liblong",
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
            key_col="code",
            val_col="lib",
        )
        if ghm5_pattern is not None:
            dp_df = dp_df.filter(pl.col("GHM5").str.contains(ghm5_pattern))
            if dp_df.is_empty():
                pbar.close()
                raise ValueError(f"Aucun séjour DP ne correspond au pattern GHM5 '{ghm5_pattern}'")
        pbar.update(1)

        # --- 1. Tirage pondéré des séjours DP --------------------------------
        pbar.set_description("Tirage DP")
        dp_df = dp_df.filter(pl.col("DP").is_not_null() & pl.col("GHM5").is_not_null())
        if dp_df.is_empty():
            pbar.close()
            raise ValueError("Aucun séjour valide après exclusion des DP/GHM5 null.")
        indices = _weighted_choice(dp_df["P_DP"], size=n_sejours, replace=True)
        sampled = dp_df[indices].select("AGE", "SEXE", "GHM5", "DP")
        pbar.update(1)

        # --- 2. Tirage CCAM (GHM chirurgicaux uniquement) --------------------
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

        # --- 3. Tirage DAS (cascade GHM5+AGE+SEXE+DP → GHM5) ---------------
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

        # --- 4. Tirage DMS (triangulaire P25/P50/P75) ------------------------
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

        # --- 5. Remplacement des codes par les libellés ----------------------
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

    # ------------------------------------------------------------------
    # Génération de CRH via LLM
    # ------------------------------------------------------------------

    def get_report(
        self,
        df: pl.DataFrame,
        client,
        model: str,
        temp_path: str = "temp.parquet",
        batch_size: int = 1000,
    ) -> pl.DataFrame:
        """Génère un CRH par appel LLM pour chaque scénario.

        Parameters
        ----------
        df : pl.DataFrame
            Sortie de get_scenario (doit contenir "scenario" et "generation_id").
        client
            Client LLM exposant ``client.chat(model, messages, **kwargs)``.
        model : str
            Nom du modèle à utiliser.
        temp_path : str
            Chemin du fichier Parquet temporaire (buffer).
        batch_size : int
            Nombre de lignes avant flush vers un fichier final horodaté.

        Returns
        -------
        pl.DataFrame
            Colonnes : generation_id, scenario, report, model, timestamp.
        """
        if "generation_id" not in df.columns:
            raise ValueError(
                "Le DataFrame doit contenir une colonne 'generation_id'. "
                "Assurez-vous de passer par get_fictive avant get_report."
            )

        system_prompt = self.prompt["generate"]["system_prompt"]
        output_dir = self.config["data"]["output"]
        expected_schema = {
            "generation_id": pl.Utf8,
            "scenario": pl.Utf8,
            "report": pl.Utf8,
            "model": pl.Utf8,
            "timestamp": pl.Datetime,
        }

        temp_df = _load_temp(temp_path, expected_schema)
        results: list[dict] = []

        for df_row in tqdm(df.iter_rows(named=True), desc="Génération CRH", unit="crh", total=len(df)):
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

            new_row = pl.DataFrame([row], schema=expected_schema)
            temp_df = pl.concat([temp_df, new_row])
            temp_df.write_parquet(temp_path)

            if len(temp_df) >= batch_size:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_path = os.path.join(
                    output_dir, f"medical_reports_{batch_size}_{ts}.parquet"
                )
                temp_df.write_parquet(final_path)
                temp_df = pl.DataFrame(schema=expected_schema)
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        return pl.DataFrame(results, schema=expected_schema)

    # ------------------------------------------------------------------
    # Sélection du client LLM
    # ------------------------------------------------------------------

    def get_client(self, client_type: str = "ollama") -> tuple:
        """Instancie et retourne le client LLM ainsi que le nom du modèle.

        Parameters
        ----------
        client_type : str
            "ollama", "claude" ou "mistral".

        Returns
        -------
        tuple[client, str]
            (client, model_name)
        """
        if client_type == "ollama":
            from ollama import Client

            cfg = self.servers["ollama"]
            return Client(cfg["host"]), cfg["model"]

        if client_type == "claude":
            cfg = self.servers["claude"]
            return AnthropicClient(
                api_key=cfg["api_key"],
                verify=cfg.get("verify", True),
            ), cfg["model"]

        if client_type == "mistral":
            cfg = self.servers["mistral"]
            return MistralClient(
                api_key=cfg["api_key"],
                verify=cfg.get("verify", True),
            ), cfg["model"]

        raise ValueError(f"Type de client inconnu : '{client_type}'. Valeurs acceptées : ollama, claude, mistral.")


# =============================================================================
# Helpers internes
# =============================================================================


def _load_temp(temp_path: str, expected_schema: dict) -> pl.DataFrame:
    """Charge le fichier temp Parquet s'il existe et que le schéma correspond."""
    if not os.path.exists(temp_path):
        return pl.DataFrame(schema=expected_schema)
    try:
        temp_df = pl.read_parquet(temp_path)
        if dict(zip(temp_df.columns, temp_df.dtypes)) == expected_schema:
            return temp_df
    except Exception:
        pass
    return pl.DataFrame(schema=expected_schema)


def _format_display(code: str, ref_map: dict[str, str]) -> str:
    """Formate 'libellé (code)' ou juste 'code' si pas de libellé."""
    lib = ref_map.get(code)
    return f"{lib} ({code})" if lib else code


def _build_dms_lookup(dms_df: pl.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Construit un dict GHM5 → (P25, P50, P75) pour le tirage triangulaire."""
    lookup: dict[str, tuple[float, float, float]] = {}
    for row in dms_df.iter_rows(named=True):
        left, mode, right = row["DMS_P25"], row["DMS_P50"], row["DMS_P75"]
        if left == right:
            right = left + 1.0
        mode = max(left, min(mode, right))
        lookup[row["GHM5"]] = (left, mode, right)
    return lookup


def _build_ref_map(ref_df: pl.DataFrame, key_col: str, val_col: str) -> dict[str, str]:
    """Construit un dict key_col → val_col depuis un référentiel."""
    return dict(zip(ref_df[key_col].to_list(), ref_df[val_col].to_list()))


def _weighted_choice(weights: pl.Series, size: int, replace: bool = True) -> np.ndarray:
    """Tirage pondéré via numpy. Normalise les poids automatiquement."""
    p = weights.to_numpy().astype(np.float64)
    p /= p.sum()
    return np.random.choice(len(p), size=size, replace=replace, p=p)


def _ccam_fallback(
    ccam_df: pl.DataFrame,
    row: dict,
    dp_cat: str,
) -> pl.DataFrame | None:
    """Fallback CCAM : GHM5+DP → GHM5+catégorie DP → GHM5."""
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
    """Tirage séquentiel de DAS avec exclusion catégorielle [:2] et fallback."""
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
