"""Hardcoded ICD-10 / CCAM / GHM code lists used by the AP-HP pipeline.

These are the literal lists from ``recode-scenario/utils_v2.py``
(``generate_scenario.__init__``, lines 179–216). They are clinical
classifications maintained by AP-HP and do not change frequently, so
inlining them keeps the code self-documenting.
"""

from __future__ import annotations

# ---- Cancer metastasis
ICD_CANCER_META_LN: list[str] = [
    "C770", "C771", "C772", "C773", "C774", "C775", "C778", "C779",
]
ICD_CANCER_META: list[str] = [
    "C780", "C781", "C782", "C783", "C784", "C785", "C786", "C787", "C788",
    "C790", "C791", "C792", "C793", "C794", "C795", "C796", "C797", "C798",
]

# ---- Treatment / contact codes
ICD_CONTACT_TT_REP: list[str] = ["Z491", "Z511", "Z512", "Z5101", "Z513", "Z516"]
ICD_CHIMIO_NON_TUM: list[str] = ["Z512"]

# ---- Cancer-treatment GHM groups
DRG_CHIMIO: list[str] = ["28Z07", "17M05", "17M06"]
DRG_RADIO: list[str] = [
    "17K04", "17K05", "17K08", "17K09",
    "28Z10", "28Z11", "28Z18", "28Z19", "28Z20", "28Z21", "28Z22", "28Z23",
    "28Z24", "28Z25",
]

# ---- Comfort / cosmetic / prophylactic
ICD_T2_CHRONIC_INTRACTABLE_PAIN: list[str] = ["R5210", "R5218"]
ICD_ASCITES: list[str] = ["R18"]
ICD_PLEURAL_EFFUSION: list[str] = ["J90", "J91", "J940", "J941", "J942", "J948", "J949"]
ICD_COSMETIC_SURGERY: list[str] = ["Z410", "Z411"]
ICD_COMFORT_INTERVENTION: list[str] = ["Z4180"]
ICD_PLASTIC_SURGERY: list[str] = [
    "Z420", "Z421", "Z423", "Z424", "Z425", "Z426", "Z427", "Z428", "Z429",
]
ICD_PROPHYLACTIC_INTERVENTION: list[str] = ["Z400", "Z401", "Z408"]

# ---- Special-context GHM groups
DRG_GREFFE: list[str] = ["27Z02", "27Z03", "27Z04"]
DRG_TRANSFUSION: list[str] = ["28Z14"]
DRG_PALLIATIVE_CARE: list[str] = ["23Z02"]
DRG_STOMIES: list[str] = ["06M17"]
DRG_APHERESIS: list[str] = ["28Z16"]
DRG_DECEASED: list[str] = ["04M24"]
DRG_BILAN: list[str] = ["23M03"]

# ---- Delivery
DRG_VAGINAL_DELIVERY: list[str] = [
    "14C03", "14Z09", "14Z10", "14Z11", "14Z12", "14Z13", "14Z14",
]
DRG_CSECTION: list[str] = ["14C06", "14C07", "14C08"]
DRG_DELIVERY: list[str] = DRG_VAGINAL_DELIVERY + DRG_CSECTION

PROC_VAGINAL_DELIVERY: list[str] = [
    "JQGD001", "JQGD002", "JQGD003", "JQGD004", "JQGD005",
    "JQGD007", "JQGD008", "JQGD010", "JQGD012", "JQGD013",
]
PROC_CSECTION: list[str] = ["JQGA002", "JQGA003", "JQGA004", "JQGA005"]

# ---- Exclusions
ICD_EXCLUSIONS: list[str] = [
    "Z40", "Z08", "Z09", "Z48", "Z71", "Z48", "Z34", "Z35", "Z39",
    "Z94", "Z95", "Z94", "Z96", "Z3908", "Z762",
]
EXCLUSION_SPECIALTY: list[str] = [
    "THERAPIE TRANSFUSION",
    "PSYCHIATRIE INFANTO-JUVENILE NON SECTORISE",
    "PHYSIOLOGIE",
    "PHYSIOLOGIE PEDIATRIQUE",
]
