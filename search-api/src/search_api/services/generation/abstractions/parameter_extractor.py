"""Abstract base class for parameter extractors."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ParameterExtractor(ABC):
    """Abstract base class for LLM parameter extractors."""
    
    @abstractmethod
    def extract_parameters(self, query: str, available_projects: Optional[Dict] = None,
                         available_document_types: Optional[Dict] = None,
                         available_strategies: Optional[Dict] = None,
                         supplied_project_ids: Optional[list] = None,
                         supplied_document_type_ids: Optional[list] = None,
                         supplied_search_strategy: Optional[str] = None) -> Dict[str, Any]:
        """Extract search parameters from a query.
        
        Args:
            query: User's search query
            available_projects: Dict of available projects {name: id}
            available_document_types: Dict of available document types
            available_strategies: Dict of available search strategies
            supplied_project_ids: Already provided project IDs (skip LLM extraction if provided)
            supplied_document_type_ids: Already provided document type IDs (skip LLM extraction if provided)
            supplied_search_strategy: Already provided search strategy (skip LLM extraction if provided)
            
        Returns:
            Dict containing extracted parameters:
            {
                'project_ids': List[str],
                'document_type_ids': List[str], 
                'search_strategy': str,
                'semantic_query': str,
                'relevance': str,
                'reasoning': str,
                'confidence': float,
                'provider': str,
                'model': str
            }
        """
        pass