import polars as pl


def format_brest_scenario(df: pl.DataFrame) -> pl.DataFrame:
    """Format fictitious stays as text scenarios for the LLM."""
    das_str = df["DAS"].list.join(", ")
    das_str = pl.when(das_str == "").then(pl.lit("Aucun")).otherwise(das_str)

    ccam_str = df["CCAM"].list.join(", ")
    ccam_str = pl.when(ccam_str == "").then(pl.lit("Aucun")).otherwise(ccam_str)

    ghm5_display = pl.format("{} ({})", pl.col("GHM5"), pl.col("GHM5_CODE"))
    dp_display = pl.format("{} ({})", pl.col("DP"), pl.col("DP_CODE"))

    scenario = pl.concat_str(
        [
            pl.format("Patient : {}, {}.", pl.col("SEXE"), pl.col("AGE")),
            pl.format("\nGHM : {}.", ghm5_display),
            pl.format("\nDiagnostic principal : {}.", dp_display),
            pl.format("\nActes CCAM : {}.", ccam_str),
            pl.format("\nDiagnostics associés : {}.", das_str),
            pl.format("\nDurée de séjour : {} jours.", pl.col("DMS").cast(pl.Utf8)),
        ],
    )

    return df.with_columns(scenario.alias("scenario"))
