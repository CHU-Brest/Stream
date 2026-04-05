from pipelines.base import BasePipeline


class APHPPipeline(BasePipeline):
    """Pipeline AP-HP — source de données ATIH (SAS BD).

    Les données PMSI sont extraites via ``liabilities/extract_pmsi_tables_ATIH.sas``
    et déposées dans le répertoire ``data.input`` configuré dans servers.yaml.
    """

    name = "aphp"
