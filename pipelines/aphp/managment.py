"""Care-management classification — port of ``define_text_managment_type``.

This module is the 1:1 port of ``generate_scenario.define_text_managment_type``
(``utils_v2.py:653–934``). Given a clinical scenario already populated by
:mod:`pipelines.aphp.scenario`, it decides:

- the **coding rule** from the ATIH ICD-10 hierarchy methodology (e.g.
  ``T1``, ``D3-2``, ``S1-Chronic``…),
- a French ``situa`` text describing the hospitalisation context (added to
  the user prompt), and
- the **template name** of the system prompt to load (``medical_inpatient.txt``,
  ``surgery_outpatient.txt``, ``delivery_inpatient_csection_urg.txt``, …).

The original is a long ``if/elif`` cascade of conditional medical coding
rules. We deliberately keep the same control flow — even small re-orderings
could change which rule fires for a borderline case. The only change is
that all the ``self.icd_codes_*`` / ``self.drg_parent_code_*`` attributes
are gathered upfront in a :class:`ManagmentContext` so the function stays
pure (no global state, takes the case dict and the context).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from . import constants as C

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _collect_codes(lf: pl.LazyFrame) -> frozenset[str]:
    """Collect a single ``code`` column from a code-list LazyFrame."""
    return frozenset(lf.collect()["code"].drop_nulls().to_list())


@dataclass(frozen=True)
class ManagmentContext:
    """Frozen sets of codes consulted by :func:`define_managment_type`.

    Built once per pipeline run via :func:`build_context` from the dict
    returned by :func:`pipelines.aphp.loader.load_data` plus the cancer /
    chronic sets already computed by :class:`pipelines.aphp.scenario.ScenarioContext`.
    """

    # ICD code lists (from ``referentials/icd_codes_*.csv``)
    icd_codes_overnight_study: frozenset[str]
    icd_codes_sensitization_tests: frozenset[str]
    icd_codes_legal_abortion: frozenset[str]
    icd_codes_medical_abortion: frozenset[str]
    icd_codes_supervision_chronic_disease: frozenset[str]
    icd_codes_diabetes_chronic: frozenset[str]

    # Procedure code list (from ``referentials/procedure_botulic_toxine.csv``)
    procedure_botulic_toxin: frozenset[str]

    # Cancer / chronic sets — computed by ScenarioContext, passed in here
    icd_codes_cancer: frozenset[str]
    icd_codes_chronic: frozenset[str]


def build_context(
    data: dict[str, pl.LazyFrame],
    cancer_codes: frozenset[str],
    chronic_codes: frozenset[str],
) -> ManagmentContext:
    """Resolve all code-list LazyFrames the management classifier needs."""
    return ManagmentContext(
        icd_codes_overnight_study=_collect_codes(data["icd_codes_overnight_study"]),
        icd_codes_sensitization_tests=_collect_codes(data["icd_codes_sensitization_tests"]),
        icd_codes_legal_abortion=_collect_codes(data["icd_codes_legal_abortion"]),
        icd_codes_medical_abortion=_collect_codes(data["icd_codes_medical_abortion"]),
        icd_codes_supervision_chronic_disease=_collect_codes(
            data["icd_codes_supervision_chronic_disease"]
        ),
        icd_codes_diabetes_chronic=_collect_codes(data["icd_codes_diabetes_chronic"]),
        procedure_botulic_toxin=_collect_codes(data["procedure_botulic_toxin"]),
        icd_codes_cancer=cancer_codes,
        icd_codes_chronic=chronic_codes,
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ManagmentDecision:
    """Tuple-like result of :func:`define_managment_type`."""

    situa: str
    coding_rule: str
    template_name: str


# ---------------------------------------------------------------------------
# The classifier — 1:1 port of utils_v2.py:653–934
# ---------------------------------------------------------------------------


def define_managment_type(
    case: dict[str, Any],
    ctx: ManagmentContext,
    *,
    np_rng: np.random.Generator | None = None,
) -> ManagmentDecision:
    """Decide ``(situa, coding_rule, template_name)`` for one *case*.

    *case* is a scenario dict produced by :func:`pipelines.aphp.scenario.build_scenario`.
    The function mirrors the AP-HP cascade exactly, including ordering and
    edge cases — see ``utils_v2.py:653`` for the original. Random branches
    (delivery, chronic-DP) use *np_rng* if given for reproducibility.
    """
    np_rng = np_rng or np.random.default_rng()

    # The clinical situation as a French text snippet appended to the prompt
    situa = ""
    # The ATIH coding rule identifier (e.g. "T1", "D3-2", "S1-Chronic")
    coding_rule = ""
    # System-prompt template filename, default = generic medical inpatient
    template_name = "medical_inpatient.txt"

    # Hospitalisation type text + suffix used in template names
    if case.get("admission_type") == "Outpatient":
        text_admission_type = " en hospitalisation ambulatoire"
        ind_template = "out"
    else:
        text_admission_type = "en hospialisation complète"
        ind_template = "in"

    # Onco suffix used by some templates when the primary code is a cancer
    if case.get("icd_primary_code") in ctx.icd_codes_cancer:
        ind_template_onco = "_onco"
    else:
        ind_template_onco = ""

    icd_primary_code = case.get("icd_primary_code") or ""
    drg_parent_code = case.get("drg_parent_code") or ""
    case_management_type = case.get("case_management_type")
    procedure = case.get("procedure")
    histological_type = case.get("histological_type")
    drg_parent_description = (case.get("drg_parent_description") or "").lower()
    text_procedure = (case.get("text_procedure") or "").lower()
    icd_primary_description = case.get("icd_primary_description") or ""

    # FIRST : SIMPLE CASES — situations not depending on the primary diagnosis
    # ------------------------------------------------------------------------

    # Cancer driven entirely by treatment recommendations (handled outside the
    # cancer/non-cancer cascade because all of the scenario already comes from
    # the treatment table — see utils_v2.py:696)
    if histological_type is not None and drg_parent_code[2:3] not in ("C", "K"):
        situa = "Hospitalisation pour prise en charge du cancer"
        coding_rule = "other"

    # Règle D3-1 : hospitalisation pour exploration nocturne ou apparentée
    elif case_management_type in ctx.icd_codes_overnight_study:
        coding_rule = "D3-1"
        # Note: the original sets the *outpatient* template here (utils_v2.py:705)
        template_name = "medical_outpatient.txt"
        situa = "Prise en charge pour exploration nocturne ou apparentée telle"

    # Règle D3-2 : séjour programmé pour tests allergologiques
    elif case_management_type in ctx.icd_codes_sensitization_tests:
        coding_rule = "D3-2"
        template_name = "medical_outpatient.txt"
        situa = (
            "Prise en charge en hospitalistion de jour pour réalisation de "
            "test de réactivité allergiques"
        )

    # Règle T1 : traitements répétitifs — transfusions
    elif drg_parent_code in C.DRG_TRANSFUSION:
        coding_rule = "T1"
        situa = "Prise en charge pour " + drg_parent_description
        template_name = "medical_outpatient.txt"

    # Règle T1 : traitements répétitifs — aphérèse
    elif drg_parent_code in C.DRG_APHERESIS:
        coding_rule = "T1"
        situa = "Prise en charge pour " + drg_parent_description
        template_name = "medical_outpatient.txt"

    # Règle T1 dialyse : exclu pour l'instant
    # Règle T1 chimio : géré dans la branche cancer plus bas

    # Règle T2 : exception à T1 — ascite
    elif icd_primary_code in C.ICD_ASCITES:
        coding_rule = "T2-R18"
        situa = "Prise en charge pour ponction d'ascite  " + text_admission_type
        template_name = "medical_" + ind_template + "patient.txt"

    # Règle T2 : exception à T1 — épanchement pleural
    elif icd_primary_code in C.ICD_PLEURAL_EFFUSION:
        coding_rule = "T2-J9"
        situa = "Prise en charge pour ponction pleurale " + text_admission_type
        template_name = "medical_" + ind_template + "patient.txt"

    # Règle T2 : exception à T1 — toxine botulique (ambulatoire)
    elif (
        procedure in ctx.procedure_botulic_toxin
        and case.get("admission_type") == "Outpatient"
    ):
        coding_rule = "T2-Toxine"
        situa = (
            "Prise en charge en hospitalisation ambulatoire pour injection "
            "de toxine botulique"
        )
        template_name = "medical_" + ind_template + "patient.txt"

    # Règle T2 : exception à T1 — douleur chronique rebelle
    elif icd_primary_code in C.ICD_T2_CHRONIC_INTRACTABLE_PAIN:
        coding_rule = "T2-R52"
        situa = "Prise en charge d'une douleur chronique rebelle " + text_admission_type
        template_name = "medical_" + ind_template + "patient.txt"

    # Règle T3 : traitement chirurgical unique (DRG type C, hors accouchement)
    elif drg_parent_code[2:3] == "C" and drg_parent_code not in C.DRG_DELIVERY:
        coding_rule = "T3"
        situa = (
            "Prise en charge "
            + text_admission_type
            + " pour "
            + drg_parent_description
        )
        template_name = "surgery_" + ind_template + "patient.txt"

    # Règle T4 : chirurgie esthétique
    elif case_management_type in C.ICD_COSMETIC_SURGERY:
        coding_rule = "T4"
        situa = (
            "Prise en charge " + text_admission_type + " pour " + text_procedure
        )
        template_name = "surgery_" + ind_template + "patient.txt"

    # Règle T5 : chirurgie plastique non esthétique
    elif case_management_type in C.ICD_PLASTIC_SURGERY:
        coding_rule = "T5"
        situa = (
            "Prise en charge " + text_admission_type + " pour " + text_procedure
        )
        template_name = "surgery_" + ind_template + "patient.txt"

    # Règle T6 : intervention de confort
    elif case_management_type in C.ICD_COMFORT_INTERVENTION:
        coding_rule = "T6"
        situa = (
            "Prise en charge " + text_admission_type + " pour " + text_procedure
        )
        template_name = "surgery_" + ind_template + "patient.txt"

    # Règle T7 : soins de stomies
    elif drg_parent_code in C.DRG_STOMIES:
        coding_rule = "T7"
        situa = (
            "Prise en charge "
            + text_admission_type
            + " pour "
            + drg_parent_description
        )
        template_name = "medical_" + ind_template + "patient.txt"

    # Règle T8 : actes thérapeutiques par voie endoscopique / endovasculaire
    # Note: in the original, this is gated only on icd_primary_code == "C186"
    # (the broader DRG-K check is commented out — see utils_v2.py:790).
    elif icd_primary_code == "C186":
        coding_rule = "T8"
        situa = (
            "Prise en charge "
            + text_admission_type
            + " pour "
            + drg_parent_description
        )
        template_name = "medical_" + ind_template + "patient.txt"

    # Règle T11 : soins palliatifs
    elif drg_parent_code in C.DRG_PALLIATIVE_CARE:
        coding_rule = "T11"
        situa = (
            "Prise en charge "
            + text_admission_type
            + " pour soins palliatifs"
        )
        template_name = "medical_" + ind_template + "patient.txt"

    # Règle IVG : interruption volontaire de grossesse
    elif case_management_type in ctx.icd_codes_legal_abortion:
        coding_rule = "Legal_Abortion"
        situa = "Prise en charge pour interruption volontaire de grossesse"
        template_name = "medical_" + ind_template + "patient.txt"

    # Règle IMG : interruption médicale de grossesse
    elif case_management_type in ctx.icd_codes_medical_abortion:
        coding_rule = "Medical_Abortion"
        situa = "Prise en charge pour interruption médicale de grossesse"
        template_name = "medical_" + ind_template + "patient.txt"

    # Règle T12 : accouchements
    elif drg_parent_code in C.DRG_DELIVERY:
        coding_rule = "T12"

        # 85% via emergencies (default), 15% preceded by an inpatient admission
        option_delivery = np_rng.choice(2, p=[0.85, 0.15])
        suffix_temp_delivery = "_urg" if option_delivery == 1 else "_hospit"

        if procedure in C.PROC_CSECTION:
            situa = "Prise en charge pour accouchement par césarienne"
            template_name = (
                "delivery_inpatient_csection" + suffix_temp_delivery + ".txt"
            )
        else:
            situa = "Prise en charge pour accouchement par voie basse"
            template_name = "delivery_inpatient" + suffix_temp_delivery + ".txt"

    # Décès intra-hospitalier
    elif (drg_parent_code in C.DRG_DECEASED) or (
        case.get("discharge_disposition") == "DECES"
    ):
        situa = "Hospitalisation au cours de laquelle le patient est décédé"
        # NB: original leaves coding_rule and template_name at defaults here
        # (utils_v2.py:838–841) and assigns a stray ``code = 10`` we drop.

    # SECOND : situations dépendant du diagnostic principal
    # ------------------------------------------------------------------------

    # Cancer / chronic primary diagnoses
    elif icd_primary_code in ctx.icd_codes_chronic:
        # Règle T1 — chimiothérapie répétitive
        if drg_parent_code in C.DRG_CHIMIO:
            coding_rule = "T1"
            situa = (
                "Prise en charge "
                + text_admission_type
                + "pour cure de chimiothérapie"
            )
            template_name = "medical_" + ind_template + "patient_onco.txt"

            chemo = case.get("chemotherapy_regimen")
            if chemo is not None and not isinstance(chemo, float):
                situa += ". Le protocole actuellement suivi est : " + chemo

        # Règle T1 — administration de traitement médicamenteux nécessitant hospit
        # Note: original uses ``if`` (not ``elif``) here (utils_v2.py:868),
        # so this branch can override the chimio one above. Kept faithfully.
        if case_management_type in ["Z512"]:
            coding_rule = "T1"
            situa = (
                "Prise en charge "
                + text_admission_type
                + "pour administration d'un traitement médicamenteux nécessitant"
                + " une hospitalisation"
            )
            template_name = "medical_" + ind_template + "patient.txt"

        # Règle T1 — radiothérapie répétitive
        elif drg_parent_code in C.DRG_RADIO:
            coding_rule = "T1"
            situa = (
                "Prise en charge "
                + text_admission_type
                + " pour réalisation du traitement par radiothérapie"
            )
            template_name = "medical_" + ind_template + "patient_onco.txt"

        # Règles D1 / D5 / D9 — chronic disease as primary diagnosis
        elif case_management_type == "DP":
            option = np_rng.choice(4, p=[0.4, 0.2, 0.2, 0.2])
            if option == 0:
                coding_rule = "D1"
                situa = (
                    "Première hospitalisation "
                    + text_admission_type
                    + " pour découverte de "
                    + icd_primary_description
                )
                template_name = (
                    "medical_" + ind_template + "patient" + ind_template_onco + ".txt"
                )

            elif option == 1:
                coding_rule = "D9"
                situa = (
                    "Hospitalisation "
                    + text_admission_type
                    + " pour bilan initial pré-trérapeutique de "
                    + icd_primary_description
                )
                template_name = (
                    "medical_" + ind_template + "patient" + ind_template_onco + ".txt"
                )

            # NB: original re-tests ``option == 1`` here (utils_v2.py:892), making
            # this branch unreachable. Faithful 1:1 port preserves it so any
            # future fix shows up as a deliberate diff.
            elif option == 1:
                coding_rule = "D9"
                situa = (
                    "Hospitalisation "
                    + text_admission_type
                    + " pour mise en route du traitement de "
                    + icd_primary_description
                )
                template_name = (
                    "medical_" + ind_template + "patient" + ind_template_onco + ".txt"
                )
            else:
                if icd_primary_code in ctx.icd_codes_cancer:
                    coding_rule = "D5"
                    situa = (
                        "Hospitalisation "
                        + text_admission_type
                        + " pour rechutte après traitement de  "
                        + icd_primary_description
                    )
                    template_name = (
                        "medical_" + ind_template + "patient" + ind_template_onco + ".txt"
                    )

                elif icd_primary_code in ctx.icd_codes_diabetes_chronic:
                    coding_rule = "D5"
                    situa = (
                        "Hospitalisation "
                        + text_admission_type
                        + " pour changement de stratégie thérapeutique  "
                        + icd_primary_description
                    )
                    template_name = (
                        "medical_" + ind_template + "patient" + ind_template_onco + ".txt"
                    )

                elif icd_primary_code[0:3] not in ["E05", "J45", "K85"]:
                    coding_rule = "D5"
                    situa = (
                        "Hospitalisation "
                        + text_admission_type
                        + " pour poussée aigue de la maladie  "
                        + icd_primary_description
                    )
                    template_name = (
                        "medical_" + ind_template + "patient" + ind_template_onco + ".txt"
                    )

        # Règle S1-Chronic — surveillance d'une maladie chronique
        elif case_management_type in ctx.icd_codes_supervision_chronic_disease:
            coding_rule = "S1-Chronic"
            situa = (
                "Surveillance "
                + text_admission_type
                + " de "
                + icd_primary_description
            )
            template_name = (
                "medical_" + ind_template + "patient" + ind_template_onco + ".txt"
            )

    # Default — acute medical pathology
    else:
        if case_management_type == "DP":
            situa = (
                "Pour prise en charge diagnostique et thérapeutique du "
                "diagnotic principal " + text_admission_type
            )
        else:
            situa = (
                "Pour prise en charge "
                + text_admission_type
                + " pour "
                + (case.get("case_management_type_description") or "")
            )
        coding_rule = "other"
        template_name = (
            "medical_" + ind_template + "patient" + ind_template_onco + ".txt"
        )

    return ManagmentDecision(
        situa=situa, coding_rule=coding_rule, template_name=template_name
    )
