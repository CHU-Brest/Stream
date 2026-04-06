"""Lazy loaders for AP-HP referentials and PMSI input data.

Exposes a single :func:`load_data` function that returns a dictionary of
``polars.LazyFrame`` objects covering everything the AP-HP scenario pipeline
needs:

- **Referentials** (versioned in this repo, see ``pipelines/aphp/referentials``):
  ICD-10 official codes, synonyms, GHM stats, French names, ATIH cancer
  methodology tables, specialised ICD code lists, etc.
- **PMSI input data** (user-provided, lives in ``self.config["data"]["input"]``):
  classification profiles, secondary diagnoses, procedures.

Column names are normalised to the conventions used by AP-HP's
``recode-scenario`` (``drg_parent_code``, ``icd_primary_code``,
``case_management_type``, …) so downstream modules can be a 1:1 port of
``utils_v2.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
REFERENTIALS_DIR = PACKAGE_DIR / "referentials"
TEMPLATES_DIR = PACKAGE_DIR / "templates"

# ---------------------------------------------------------------------------
# Referentials — small helpers
# ---------------------------------------------------------------------------

# Code-list referentials: single-column ``code`` extracted from a ``code;descr``
# CSV. The downstream code only consumes the codes themselves.
_CODE_LIST_FILES: dict[str, str] = {
    "icd_codes_chronic_attack": "icd_codes_chronic_attack.csv",
    "procedure_botulic_toxin": "procedure_botulic_toxine.csv",
    "icd_codes_prophylactic_intervention": "icd_codes_prophylactic_intervention.csv",
    "attention_artificial_openings_external_prosthetic_device": "attention_artificial_openings_external_prosthetic_device.csv",
    "icd_codes_iron_deficiency_anemia": "icd_codes_iron_deficiency_anemia.csv",
    "icd_codes_sessions": "icd_codes_sessions.csv",
    "icd_codes_diabetes_chronic": "icd_codes_diabetes_chronic.csv",
    "icd_codes_spontaneous_vertex_delivery": "icd_codes_spontaneous_vertex_delivery.csv",
    "icd_codes_liveborn_infants": "icd_codes_liveborn_infants.csv",
    "icd_codes_medical_abortion": "icd_codes_medical_abortion.csv",
    "icd_codes_legal_abortion": "icd_codes_legal_abortion.csv",
    "icd_codes_supervision": "icd_codes_supervision.csv",
    "icd_codes_supervision_chronic_disease": "icd_codes_supervision_chronic_disease.csv",
    "icd_codes_surgical_followup": "icd_codes_surgical_followup.csv",
    "icd_codes_supervision_pregnancy": "icd_codes_supervision_pregnancy.csv",
    "icd_codes_supervision_post_partum": "icd_codes_supervision_post_partum.csv",
    "icd_codes_cardic_vascular_implants": "icd_codes_cardic_vascular_implants.csv",
    "icd_codes_overnight_study": "icd_codes_overnight_study.csv",
    "icd_codes_sensitization_tests": "icd_codes_sensitization_tests.csv",
    "icd_codes_preoperative_assessment": "icd_codes_preoperative_assessment.csv",
    "icd_codes_family_history": "icd_codes_family_history.csv",
    "icd_codes_personnel_history": "icd_codes_personnel_history.csv",
}


def _scan_code_list(filename: str) -> pl.LazyFrame:
    """Read a ``code;description`` CSV and return only the ``code`` column."""
    return pl.scan_csv(REFERENTIALS_DIR / filename, separator=";").select("code")


# ---------------------------------------------------------------------------
# Referentials — proper tables
# ---------------------------------------------------------------------------


def _scan_official_icd() -> pl.LazyFrame:
    """Official ICD-10 dictionary from ATIH 2025.

    The original ``LIBCIM10MULTI.TXT`` is pipe-delimited and latin-1; it is
    transcoded to UTF-8 Parquet by ``convert_referentials.py``. Mirrors
    ``generate_scenario.load_offical_icd`` (utils_v2.py:306).
    """
    return pl.scan_parquet(
        REFERENTIALS_DIR / "CIM_ATIH_2025" / "LIBCIM10MULTI.parquet"
    )


def _scan_icd_synonyms() -> pl.LazyFrame:
    """Synonyms / alternative phrasings for ICD-10 codes (~16 MB)."""
    return (
        pl.scan_csv(REFERENTIALS_DIR / "cim_synonymes.csv")
        .drop_nulls()
        .rename({"dictionary_keys": "icd_code_description", "code": "icd_code"})
    )


def _scan_drg_statistics() -> pl.LazyFrame:
    """Mean / SD length of stay per GHM root (``drg_parent_code``)."""
    return pl.scan_parquet(REFERENTIALS_DIR / "stat_racines.parquet").rename(
        {"racine": "drg_parent_code", "dms": "los_mean", "dsd": "los_sd"}
    )


def _scan_drg_parents_groups() -> pl.LazyFrame:
    """GHM root descriptions and groupings (2024 ATIH classification)."""
    return pl.scan_parquet(
        REFERENTIALS_DIR / "ghm_rghm_regroupement_2024.parquet"
    ).rename({"racine": "drg_parent_code", "libelle_racine": "drg_parent_description"})


def _scan_chronic() -> pl.LazyFrame:
    """Chronic-disease classification (code, chronic flag, libelle)."""
    return pl.scan_parquet(REFERENTIALS_DIR / "Affections chroniques.parquet")


def _scan_complications() -> pl.LazyFrame:
    """ICD codes flagged as acute complications (CMA list)."""
    return pl.scan_csv(REFERENTIALS_DIR / "cma.csv").drop_nulls()


def _scan_names() -> pl.LazyFrame:
    """French first/last names with gender (``sexe`` ∈ {1, 2})."""
    return pl.scan_csv(
        REFERENTIALS_DIR / "prenoms_nom_sexe.csv", separator=";"
    ).drop_nulls()


def _scan_hospitals() -> pl.LazyFrame:
    """One-column list of CHU hospital names."""
    return pl.scan_csv(
        REFERENTIALS_DIR / "chu", has_header=False, new_columns=["hospital"]
    )


def _scan_specialty() -> pl.LazyFrame:
    """Specialty ↔ DRG mapping with allocation ratios (AP-HP local)."""
    return pl.scan_parquet(
        REFERENTIALS_DIR / "dictionnaire_spe_racine.parquet"
    ).rename(
        {
            "racine": "drg_parent_code",
            "lib_spe_uma": "specialty",
            "ratio_spe_racine": "ratio",
        }
    )


def _scan_procedure_official() -> pl.LazyFrame:
    """Official CCAM procedure dictionary (2024)."""
    return pl.scan_parquet(REFERENTIALS_DIR / "ccam_actes_2024.parquet").rename(
        {"code": "procedure", "libelle_long": "procedure_description"}
    )


def _scan_cancer_codes() -> pl.LazyFrame:
    """ICD-10 codes flagged as cancer by the 2014 ATIH DIM methodology."""
    return pl.scan_parquet(
        REFERENTIALS_DIR
        / "REFERENTIEL_METHODE_DIM_CANCER_20140411__CODES_CIM-10_CANCER.parquet"
    )


def _scan_cancer_treatment() -> pl.LazyFrame:
    """Cancer treatment recommendations (per primary site / histology)."""
    return pl.scan_parquet(
        REFERENTIALS_DIR / "Tableau récapitulatif traitement cancer__Feuille_1.parquet"
    ).rename(
        {
            "Code CIM": "icd_parent_code",
            "Localisation": "primary_site",
            "Type Histologique": "histological_type",
            "Stade": "stage",
            "Marqueurs Tumoraux": "biomarkers",
            "Traitement": "treatment_recommandation",
            "Protocole de Chimiothérapie": "chemotherapy_regimen",
        }
    )


def _scan_icd_categ_weight() -> pl.LazyFrame:
    """Per-ICD-category sampling weights (used by load_icd_categ_weight)."""
    return pl.scan_csv(
        REFERENTIALS_DIR / "ponderation_code_categ.csv",
        separator=";",
        decimal_comma=True,
    ).rename({"diag": "icd_code", "ponderation": "weight"})


def _scan_exclusions() -> pl.LazyFrame:
    """GHM groups excluded from sampling (transplants, dialysis, …)."""
    return pl.scan_csv(REFERENTIALS_DIR / "exclusions")


# ---------------------------------------------------------------------------
# ATIH coding rules (YAML, not tabular)
# ---------------------------------------------------------------------------


def load_atih_rules() -> dict[str, dict]:
    """Parse ``templates/regles_atih.yml`` into ``{rule_id: {texte, criteres}}``.

    Mirrors the YAML-loading block in ``generate_scenario.__init__``
    (utils_v2.py:295–301).
    """
    with (TEMPLATES_DIR / "regles_atih.yml").open("r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    return {
        d["id"]: {
            "texte": d["clinical_coding_scenario"],
            "criteres": d["classification_profile_criteria"],
        }
        for d in rules["regles"]
    }


# ---------------------------------------------------------------------------
# PMSI input data (user-provided, in self.config["data"]["input"])
# ---------------------------------------------------------------------------

# Default file *patterns* for the three PMSI inputs. Real-world filenames are
# date-stamped (e.g. ``bn_pmsi_related_diag_20250818.csv``); we glob for the
# most recent match so users can drop new extracts in without touching config.
_PMSI_PATTERNS: dict[str, tuple[str, ...]] = {
    "profiles": ("scenarios_*.parquet", "scenarios_*"),
    "secondary_icd": ("bn_pmsi_related_diag_*.csv",),
    "procedures": ("bn_pmsi_procedures_*.csv",),
}


def _resolve_pmsi_file(input_dir: Path, key: str) -> Path:
    """Pick the most recent file matching one of the patterns for *key*."""
    patterns = _PMSI_PATTERNS[key]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(input_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Aucun fichier PMSI trouvé pour '{key}' dans {input_dir} "
            f"(patterns essayés : {', '.join(patterns)})."
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


# Column rename for the classification profile parquet (matches the
# ``col_names`` dict passed to ``load_classification_profile`` in the notebook).
_PROFILE_RENAME: dict[str, str] = {
    "racine": "drg_parent_code",
    "diagnostic_associes": "icd_secondary_code",
    "diag2": "icd_primary_code",
    "mdp": "case_management_type",
    "n": "nb",
    "mode_entree": "admission_mode",
    "mode_sortie": "discharge_disposition",
    "mode_hospit": "admission_type",
    "duree": "los",
    "agean": "age2",
    "nbda": "nb_associated",
}

# Column rename for the secondary-ICD and procedures CSVs.
_SECONDARY_RENAME: dict[str, str] = {
    "racine": "drg_parent_code",
    "das": "icd_secondary_code",
    "diag": "icd_primary_code",
    "categ_cim": "icd_primary_parent_code",
    "mdp": "case_management_type",
    "nb_situations": "nb",
    "acte": "procedure",
    "mode_entree": "admission_mode",
    "mode_sortie": "discharge_disposition",
    "mode_hospit": "admission_type",
}


def _safe_rename(lf: pl.LazyFrame, mapping: dict[str, str]) -> pl.LazyFrame:
    """Rename only the columns of *lf* that actually exist in *mapping*."""
    existing = set(lf.collect_schema().names())
    return lf.rename({k: v for k, v in mapping.items() if k in existing})


def _scan_profiles(input_dir: Path) -> pl.LazyFrame:
    """Scan the main PMSI classification profile (Parquet)."""
    path = _resolve_pmsi_file(input_dir, "profiles")
    return _safe_rename(pl.scan_parquet(path), _PROFILE_RENAME)


def _scan_secondary_icd(input_dir: Path) -> pl.LazyFrame:
    """Scan the secondary-diagnoses CSV from PMSI."""
    path = _resolve_pmsi_file(input_dir, "secondary_icd")
    return _safe_rename(pl.scan_csv(path, separator=";"), _SECONDARY_RENAME)


def _scan_procedures(input_dir: Path) -> pl.LazyFrame:
    """Scan the procedures CSV from PMSI."""
    path = _resolve_pmsi_file(input_dir, "procedures")
    return _safe_rename(pl.scan_csv(path, separator=";"), _SECONDARY_RENAME)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_referentials() -> dict[str, pl.LazyFrame]:
    """Return all versioned referentials as ``LazyFrame`` objects.

    The keys mirror the attribute names used by ``generate_scenario`` so the
    downstream port stays readable.
    """
    data: dict[str, pl.LazyFrame] = {
        "icd_official": _scan_official_icd(),
        "icd_synonyms": _scan_icd_synonyms(),
        "icd_categ_weight": _scan_icd_categ_weight(),
        "drg_statistics": _scan_drg_statistics(),
        "drg_parents_groups": _scan_drg_parents_groups(),
        "chronic": _scan_chronic(),
        "complications": _scan_complications(),
        "names": _scan_names(),
        "hospitals": _scan_hospitals(),
        "specialty": _scan_specialty(),
        "procedure_official": _scan_procedure_official(),
        "cancer_codes": _scan_cancer_codes(),
        "cancer_treatment": _scan_cancer_treatment(),
        "exclusions": _scan_exclusions(),
    }
    for key, filename in _CODE_LIST_FILES.items():
        data[key] = _scan_code_list(filename)
    return data


def load_pmsi(input_dir: str | Path) -> dict[str, pl.LazyFrame]:
    """Return the three PMSI input tables as ``LazyFrame`` objects.

    Parameters
    ----------
    input_dir:
        Directory containing the user-provided PMSI extracts. The most recent
        file matching each known pattern (``scenarios_*``,
        ``bn_pmsi_related_diag_*.csv``, ``bn_pmsi_procedures_*.csv``) is picked.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Le répertoire d'entrée PMSI n'existe pas : {input_dir}"
        )
    return {
        "profiles": _scan_profiles(input_dir),
        "secondary_icd": _scan_secondary_icd(input_dir),
        "procedures": _scan_procedures(input_dir),
    }


def load_data(input_dir: str | Path) -> dict[str, pl.LazyFrame]:
    """Merge :func:`load_referentials` and :func:`load_pmsi` into one dict.

    This is the dict that ``APHPPipeline.load_data()`` will return to fulfill
    the ``BasePipeline`` contract.
    """
    return {**load_referentials(), **load_pmsi(input_dir)}


def referential_paths() -> dict[str, Any]:
    """Expose key on-disk paths for use by ``check_data()``."""
    return {
        "referentials_dir": REFERENTIALS_DIR,
        "templates_dir": TEMPLATES_DIR,
        "atih_rules": TEMPLATES_DIR / "regles_atih.yml",
    }
