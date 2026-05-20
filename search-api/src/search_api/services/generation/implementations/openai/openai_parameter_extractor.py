"""
OpenAI Parameter Extractor Implementation
Supports both parallel (default) and sequential parameter extraction for optimal performance.
"""
import logging
import os
from typing import Dict, List, Any

from search_api.services.generation.implementations.base_parameter_extractor import BaseParameterExtractor

logger = logging.getLogger(__name__)

class OpenAIParameterExtractor(BaseParameterExtractor):
    """OpenAI implementation of parameter extractor."""

    def __init__(self, client):
        super().__init__(client)
        # Allow a separate, faster deployment for extraction (e.g. gpt-4o-mini).
        # Falls back to the main deployment if not set.
        self.extraction_deployment = (
            os.environ.get("AZURE_OPENAI_EXTRACTION_DEPLOYMENT") or
            client.deployment_name
        )
        logger.info(f"Parameter extractor using deployment: {self.extraction_deployment}")

    def _make_llm_call(self, messages: List[Dict], temperature: float = 0.1) -> Dict[str, Any]:
        """Make OpenAI chat completion call using the extraction-specific deployment."""
        return self.client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=1000,
            model=self.extraction_deployment
        )