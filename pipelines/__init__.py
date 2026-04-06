"""Stream pipelines — synthetic medical-report generation.

This package exposes the two concrete pipelines (Brest and AP-HP) and the
common infrastructure they share.
"""

from pipelines.brest.pipeline import BrestPipeline
from pipelines.aphp.pipeline import APHPPipeline
from pipelines.pipeline import BasePipeline, AnthropicClient, MistralClient

__all__ = ["BrestPipeline", "APHPPipeline", "BasePipeline", "AnthropicClient", "MistralClient"]
