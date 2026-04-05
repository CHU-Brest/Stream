import polars as pl

from pipelines.base import BasePipeline


class APHPPipeline(BasePipeline):
    """Pipeline AP-HP — à implémenter.

    Ce pipeline implémentera la méthode AP-HP de génération de CRH
    synthétiques (source ATIH / recode-scenario).
    """

    name = "aphp"

    def check_data(self) -> None:
        raise NotImplementedError("Le pipeline AP-HP n'est pas encore implémenté.")

    def load_data(self) -> dict[str, pl.LazyFrame]:
        raise NotImplementedError("Le pipeline AP-HP n'est pas encore implémenté.")

    def get_fictive(self, data: dict[str, pl.LazyFrame], **kwargs) -> pl.DataFrame:
        raise NotImplementedError("Le pipeline AP-HP n'est pas encore implémenté.")

    def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError("Le pipeline AP-HP n'est pas encore implémenté.")
