from __future__ import annotations

from pathlib import Path

from core.config import load_config
from core.clients import get_client
from pipelines.aphp.pipeline import APHPPipeline
from pipelines.brest.pipeline import BrestPipeline

PIPELINES = {
    "brest": BrestPipeline,
    "aphp": APHPPipeline,
}

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def run(
    pipeline_name: str,
    client_type: str = "ollama",
    n_sejours: int = 1000,
    n_ccam: int = 1,
    n_das: int = 5,
    ghm5_pattern: str | None = None,
    batch_size: int = 1000,
) -> None:
    """Orchestrate an end-to-end synthetic medical-report generation run."""
    if pipeline_name not in PIPELINES:
        raise ValueError(
            f"Pipeline inconnu : '{pipeline_name}'. "
            f"Valeurs acceptées : {list(PIPELINES.keys())}"
        )

    config = load_config(CONFIG_DIR / "servers.yaml")
    prompt = load_config(CONFIG_DIR / "prompts.yaml")

    pipeline = PIPELINES[pipeline_name](
        config=config["pipelines"][pipeline_name],
        prompt=prompt,
        servers=config["servers"],
    )

    pipeline.check_data()
    data = pipeline.load_data()

    df = pipeline.get_fictive(
        data,
        n_sejours=n_sejours,
        n_ccam=n_ccam,
        n_das=n_das,
        ghm5_pattern=ghm5_pattern,
    )
    df = pipeline.get_scenario(df)

    client, model = get_client(pipeline.servers, client_type)
    pipeline.get_report(df, client, model, batch_size=batch_size)
