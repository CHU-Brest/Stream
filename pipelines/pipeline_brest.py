from pipelines.base import BasePipeline


class BrestPipeline(BasePipeline):
    """Pipeline CHU Brest — source de données SNDS (Oracle).

    Les données PMSI sont extraites via ``liabilities/extract_pmsi_tables_SNDS.sas``
    et déposées dans le répertoire ``data.input`` configuré dans servers.yaml.
    """

    name = "brest"
