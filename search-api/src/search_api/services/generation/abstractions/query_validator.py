"""Abstract base class for query validators."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class QueryValidator(ABC):
    """Abstract base class for query validation services."""

    @abstractmethod
    def validate_query_relevance(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate if a query is relevant to the system's scope.

        Args:
            query: The user's search query to validate.
            context: Optional additional context for validation.

        Returns:
            Dict containing validation results with keys:
            - is_relevant: Boolean indicating if query is relevant
            - confidence: Confidence score (0.0 to 1.0)
            - reasoning: List of reasons for the decision
            - recommendation: Recommendation for how to proceed
            - suggested_response: Optional response for irrelevant queries
            - query_type: Type of query ("generic_informational", "specific_search", "ambiguous", "broad_category_search")
            - generic_response: AI-generated response for generic informational queries
            - category_filter: For broad_category_search, the category to filter by (e.g., "mining", "lng", "pipeline")

        Raises:
            Exception: If validation fails.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass