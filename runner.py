from core.config import load_config
from pipelines.pipeline_aphp import APHPPipeline
from pipelines.pipeline_brest import BrestPipeline

PIPELINES = {
    "brest": BrestPipeline,
    "aphp": APHPPipeline,
}


def run(
    pipeline_name: str,
    client_type: str = "ollama",
    n_sejours: int = 1000,
    n_ccam: int = 1,
    n_das: int = 5,
    ghm5_pattern: str | None = None,
    batch_size: int = 1000,
) -> None:
    """Orchestre le pipeline de génération de CRH synthétiques de bout en bout."""
    if pipeline_name not in PIPELINES:
        raise ValueError(
            f"Pipeline inconnu : '{pipeline_name}'. "
            f"Valeurs acceptées : {list(PIPELINES.keys())}"
        )

    config = load_config("config/servers.yaml")
    prompt = load_config("config/prompts.yaml")

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

    client, model = pipeline.get_client(client_type)
    pipeline.get_report(df, client, model, batch_size=batch_size)
