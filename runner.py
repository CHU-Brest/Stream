from core.config import load_config
from core.text import get_scenario
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
    """Orchestre le pipeline de génération de CRH synthétiques de bout en bout.

    Parameters
    ----------
    pipeline_name : str
        Identifiant du pipeline : "brest" ou "aphp".
    client_type : str
        LLM à utiliser : "ollama", "claude" ou "mistral".
    n_sejours : int
        Nombre de séjours fictifs à générer.
    n_ccam : int
        Nombre max d'actes CCAM par séjour.
    n_das : int
        Nombre max de diagnostics associés par séjour.
    ghm5_pattern : str | None
        Filtre optionnel sur les codes GHM5 (ex. "06C").
    batch_size : int
        Nombre de CRH générés avant flush vers un fichier Parquet final.
    """
    if pipeline_name not in PIPELINES:
        raise ValueError(
            f"Pipeline inconnu : '{pipeline_name}'. "
            f"Valeurs acceptées : {list(PIPELINES.keys())}"
        )

    config = load_config("config/servers.yaml")
    prompt = load_config("config/prompts.yaml")

    pipeline_config = config["pipelines"][pipeline_name]
    servers = config["servers"]

    pipeline = PIPELINES[pipeline_name](
        config=pipeline_config,
        prompt=prompt,
        servers=servers,
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
    df = get_scenario(df)

    client, model = pipeline.get_client(client_type)
    pipeline.get_report(df, client, model, batch_size=batch_size)
