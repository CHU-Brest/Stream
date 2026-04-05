import os
from abc import ABC, abstractmethod
from datetime import datetime

import httpx
import polars as pl
from tqdm import tqdm

from core.logging import get_logger


# =============================================================================
# Wrappers LLM
# =============================================================================


class AnthropicClient:
    """Client Anthropic (Claude) normalisé sur l'interface chat().

    Parameters
    ----------
    api_key : str
        Clé API Anthropic.
    verify : bool | str
        ``True`` (défaut), ``False``, ou chemin vers un bundle CA (.pem).
    """

    def __init__(self, api_key: str, verify: bool | str = True):
        from anthropic import Anthropic

        self._client = Anthropic(
            api_key=api_key, http_client=httpx.Client(verify=verify)
        )

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
    """Client Mistral normalisé sur l'interface chat().

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

REPORT_SCHEMA = {
    "generation_id": pl.Utf8,
    "scenario": pl.Utf8,
    "report": pl.Utf8,
    "model": pl.Utf8,
    "timestamp": pl.Datetime,
}


class BasePipeline(ABC):
    """Socle commun des pipelines de génération de CRH synthétiques.

    Fournit les briques partagées (wrappers LLM, boucle de génération,
    persistence Parquet) et définit l'interface que chaque pipeline concret
    doit implémenter.
    """

    name: str = "base"

    def __init__(self, config: dict, prompt: dict, servers: dict):
        self.config = config
        self.prompt = prompt
        self.servers = servers
        self.logger = get_logger(self.name)

    # ------------------------------------------------------------------
    # Méthodes abstraites — à implémenter par chaque pipeline
    # ------------------------------------------------------------------

    @abstractmethod
    def check_data(self) -> None:
        """Vérifie la présence des données sources et les prépare si nécessaire."""

    @abstractmethod
    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Charge les données préparées en LazyFrame."""

    @abstractmethod
    def get_fictive(self, data: dict[str, pl.LazyFrame], **kwargs) -> pl.DataFrame:
        """Génère des séjours fictifs à partir des données chargées."""

    @abstractmethod
    def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transforme les séjours fictifs en scénarios textuels pour le LLM."""

    # ------------------------------------------------------------------
    # Génération de CRH via LLM (partagé)
    # ------------------------------------------------------------------

    def get_report(
        self,
        df: pl.DataFrame,
        client,
        model: str,
        batch_size: int = 1000,
    ) -> pl.DataFrame:
        """Génère un CRH par appel LLM pour chaque scénario.

        Les résultats sont écrits en Parquet par lots de ``batch_size`` dans
        le répertoire de sortie configuré.

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
        os.makedirs(output_dir, exist_ok=True)

        results: list[dict] = []
        batch: list[dict] = []

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
            batch.append(row)

            if len(batch) >= batch_size:
                _flush_batch(batch, output_dir, batch_size)
                batch = []

        if batch:
            _flush_batch(batch, output_dir, len(batch))

        return pl.DataFrame(results, schema=REPORT_SCHEMA)

    # ------------------------------------------------------------------
    # Sélection du client LLM (partagé)
    # ------------------------------------------------------------------

    def get_client(self, client_type: str = "ollama") -> tuple:
        """Instancie le client LLM et retourne ``(client, model_name)``."""
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

        raise ValueError(
            f"Type de client inconnu : '{client_type}'. "
            "Valeurs acceptées : ollama, claude, mistral."
        )


# =============================================================================
# Helpers partagés
# =============================================================================


def _flush_batch(batch: list[dict], output_dir: str, count: int) -> None:
    """Écrit un lot de résultats dans un fichier Parquet horodaté."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"medical_reports_{count}_{ts}.parquet")
    pl.DataFrame(batch, schema=REPORT_SCHEMA).write_parquet(path)
