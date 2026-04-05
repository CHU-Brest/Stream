from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import httpx
import polars as pl
from tqdm import tqdm

from core.logger import get_logger

REPORT_SCHEMA = {
    "generation_id": pl.Utf8,
    "scenario": pl.Utf8,
    "report": pl.Utf8,
    "model": pl.Utf8,
    "timestamp": pl.Datetime,
}


class AnthropicClient:
    """Anthropic (Claude) client normalised to the ``chat()`` interface.

    Parameters
    ----------
    api_key:
        Anthropic API key.
    verify:
        ``True`` (default), ``False``, or path to a CA bundle (``.pem``).
    """

    def __init__(self, api_key: str, verify: bool | str = True) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(
            api_key=api_key, http_client=httpx.Client(verify=verify)
        )

    def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        system: str | None = None
        chat_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        params: dict = dict(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=chat_messages,
        )
        if system:
            params["system"] = system

        response = self._client.messages.create(**params)  # type: ignore[arg-type]

        return {
            "message": {
                "role": "assistant",
                "content": response.content[0].text,
            }
        }


class MistralClient:
    """Mistral client normalised to the ``chat()`` interface.

    Parameters
    ----------
    api_key:
        Mistral API key.
    verify:
        ``True`` (default), ``False``, or path to a CA bundle (``.pem``).
    """

    def __init__(self, api_key: str, verify: bool | str = True) -> None:
        from mistralai import Mistral

        self._client = Mistral(api_key=api_key)
        if verify is not True:
            self._client.client._client = httpx.Client(verify=verify)

    def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
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


class BasePipeline(ABC):
    """Base class for synthetic medical-report generation pipelines.

    Provides shared building blocks (LLM wrappers, generation loop, Parquet
    persistence) and defines the interface that each concrete pipeline must
    implement.
    """

    name: str = "base"

    def __init__(self, config: dict, prompt: dict, servers: dict) -> None:
        self.config = config
        self.prompt = prompt
        self.servers = servers
        self.logger = get_logger(self.name)

    # -- Abstract interface ------------------------------------------------

    @abstractmethod
    def check_data(self) -> None:
        """Verify that source data is present and prepare it if needed."""

    @abstractmethod
    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Load prepared data as LazyFrames."""

    @abstractmethod
    def get_fictive(self, data: dict[str, pl.LazyFrame], **kwargs) -> pl.DataFrame:
        """Generate fictitious hospital stays from loaded data."""

    @abstractmethod
    def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform fictitious stays into text scenarios for the LLM."""

    # -- LLM report generation (shared) ------------------------------------

    def get_report(
        self,
        df: pl.DataFrame,
        client: AnthropicClient | MistralClient,
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

    # -- LLM client factory (shared) ---------------------------------------

    def get_client(self, client_type: str = "ollama") -> tuple[AnthropicClient | MistralClient, str]:
        """Instantiate an LLM client and return ``(client, model_name)``."""
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


def _flush_batch(batch: list[dict], output_dir: Path, count: int) -> None:
    """Write a batch of results to a timestamped Parquet file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"medical_reports_{count}_{ts}.parquet"
    pl.DataFrame(batch, schema=REPORT_SCHEMA).write_parquet(path)
