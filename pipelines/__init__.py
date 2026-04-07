"""Stream pipelines — synthetic medical-report generation.

This package provides two concrete pipelines (Brest and AP-HP) and a
set of shared modules for generating synthetic medical reports from
PMSI data.

Core modules:
- :mod:`pipelines.fictive`: Generate fictitious hospital stays from PMSI data.
- :mod:`pipelines.scenario`: Transform stays into text scenarios for LLMs.
- :mod:`pipelines.report`: Generate CRH reports via LLM calls.

Concrete pipelines:
- :mod:`pipelines.brest`: Brest pipeline (weighted PMSI sampling).
- :mod:`pipelines.aphp`: AP-HP pipeline (scenario-based sampling).

Base classes:
- :mod:`pipelines.pipeline`: Common pipeline infrastructure.
"""

from core.clients import AnthropicClient, MistralClient
from pipelines.fictive import generate_fictive_stays
from pipelines.report import generate_reports
from pipelines.scenario import format_scenarios

__all__ = [
    "generate_fictive_stays",
    "format_scenarios",
    "generate_reports",
    "BasePipeline",
    "AnthropicClient",
    "MistralClient",
]
