"""Utility Tool Implementations for Agentic Search.

This module provides utility tools for processing, filtering, and transforming
search results. These tools perform in-memory operations and are typically fast.

Tools:
    - extract_entities: Extract named entities from text
    - filter_by_relevance: Filter results by relevance score
    - deduplicate_results: Remove duplicate items from results
    - merge_results: Merge multiple result sets
    - summarize_results: Generate summary statistics

Usage:
    >>> from search_api.services.search_handlers.agent.tools.utility_tools import (
    ...     register_utility_tools,
    ...     create_extract_entities_tool,
    ... )
    >>>
    >>> # Register all utility tools
    >>> from search_api.services.search_handlers.agent.tools import ToolRegistry
    >>> registry = ToolRegistry()
    >>> registered = register_utility_tools(registry)
    >>> print(f"Registered: {registered}")
"""

import logging
import re
from typing import Dict, Any, List, Optional, Set

from .tool_registry import (
    create_tool,
    Tool,
    ToolRegistry,
    ToolCategory,
    CostEstimate,
    ParameterType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXTRACT ENTITIES TOOL
# =============================================================================

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text using pattern matching.

    Identifies projects, locations, dates, document types, and organizations
    using regex patterns and keyword matching.

    Args:
        text: Text to analyze and extract entities from

    Returns:
        Dictionary with entity types as keys and lists of found entities as values
    """
    entities = {
        "projects": [],
        "locations": [],
        "dates": [],
        "document_types": [],
        "organizations": [],
        "keywords": []
    }

    # Project patterns
    # Look for "X Project", "X Mine", "X Facility", etc.
    project_patterns = [
        r"([A-Z][A-Za-z\s&\-]+(?:Project|Mine|Facility|Plant|Pipeline|Terminal|LNG))",
        r"(?:Project|Mine|Facility|Plant)\s+([A-Z][A-Za-z\s&\-]+)"
    ]

    for pattern in project_patterns:
        matches = re.findall(pattern, text)
        entities["projects"].extend([m.strip() for m in matches if len(m.strip()) > 3])

    # Location patterns - Canadian cities, regions, provinces
    location_keywords = [
        # BC Cities
        "Vancouver", "Victoria", "Surrey", "Burnaby", "Richmond",
        "Kelowna", "Prince George", "Kamloops", "Nanaimo", "Abbotsford",
        # BC Regions
        "Peace River", "Fort St. John", "Fort Nelson", "Dawson Creek",
        "Lower Mainland", "Northern BC", "Vancouver Island", "Okanagan",
        "Cariboo", "Kootenay", "Fraser Valley", "Thompson",
        # Provinces
        "British Columbia", "BC", "Alberta", "AB",
        # Geographic features
        "Columbia River", "Fraser River", "Skeena"
    ]

    for location in location_keywords:
        if location.lower() in text.lower():
            entities["locations"].append(location)

    # Date patterns
    date_patterns = [
        r"\b(20\d{2})\b",  # Year 2000-2099
        r"\b(20\d{2}-\d{2}-\d{2})\b",  # ISO date
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2})\b",  # Month Year
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2})\b"  # Full date
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["dates"].extend(matches)

    # Document type patterns
    doc_type_keywords = [
        "assessment report", "environmental assessment", "EA report",
        "certificate", "EAC", "environmental certificate",
        "correspondence", "letter", "memo",
        "meeting notes", "meeting minutes",
        "public comment", "submission",
        "technical report", "technical memo",
        "environmental impact statement", "EIS",
        "application", "review", "decision"
    ]

    for doc_type in doc_type_keywords:
        if doc_type.lower() in text.lower():
            entities["document_types"].append(doc_type)

    # Organization patterns
    org_patterns = [
        r"([A-Z][A-Za-z\s&]+(?:Limited|Ltd|Inc|Corp|Corporation|Company|Co\.))",
        r"(First Nation[s]?\s+[A-Z][A-Za-z\s]+)",
        r"([A-Z][A-Za-z\s]+(?:Band|Council|Authority|Ministry|Department))"
    ]

    for pattern in org_patterns:
        matches = re.findall(pattern, text)
        entities["organizations"].extend([m.strip() for m in matches if len(m.strip()) > 3])

    # Topic/keyword extraction
    topic_keywords = [
        "water quality", "air quality", "noise", "traffic",
        "wildlife", "habitat", "fish", "salmon", "caribou",
        "First Nations", "Indigenous", "consultation", "engagement",
        "environment", "environmental", "impact", "assessment",
        "mining", "mine", "LNG", "pipeline", "oil", "gas",
        "permit", "approval", "compliance", "mitigation"
    ]

    for keyword in topic_keywords:
        if keyword.lower() in text.lower():
            entities["keywords"].append(keyword)

    # Deduplicate all entity lists
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities


def create_extract_entities_tool() -> Tool:
    """Create the extract_entities tool.

    Returns:
        Configured Tool instance for entity extraction
    """
    return (
        create_tool("extract_entities")

        .description(
            "Extract named entities from text including project names, locations, dates, "
            "document types, organizations, and topic keywords. Uses pattern matching. "
            "Best for: understanding query intent, identifying key entities, preprocessing queries."
        )
        .category(ToolCategory.UTILITY)
        .cost(CostEstimate.VERY_LOW)  # Pure computation, no external calls
        .returns(
            "Dictionary with entity types as keys (projects, locations, dates, "
            "document_types, organizations, keywords) and lists of found entities"
        )

        .parameter(
            "text",
            ParameterType.STRING,
            "Text to analyze and extract entities from. "
            "Can be a user query, document content, or any text.",
            required=True,
            examples=[
                "What are the impacts of the Red Mountain Mining Project in Northern BC?",
                "Air Liquide Corporation assessment report from 2023",
                "First Nations consultation for the LNG facility near Fort Nelson"
            ]
        )

        .implementation(extract_entities)

        .example(
            input={
                "text": "What are the water quality impacts from the Copper Creek Mine Project in Northern BC during 2023?"
            },
            output={
                "projects": ["Copper Creek Mine Project"],
                "locations": ["Northern BC"],
                "dates": ["2023"],
                "document_types": [],
                "organizations": [],
                "keywords": ["water quality", "mining", "mine", "impact"]
            }
        )

        .example(
            input={
                "text": "Air Liquide Corporation environmental assessment report correspondence"
            },
            output={
                "projects": [],
                "locations": [],
                "dates": [],
                "document_types": ["environmental assessment", "assessment report", "correspondence"],
                "organizations": ["Air Liquide Corporation"],
                "keywords": ["environmental", "assessment"]
            }
        )

        .metadata(
            version="1.0.0",
            tags=["utility", "nlp", "extraction", "fast"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# FILTER BY RELEVANCE TOOL
# =============================================================================

def filter_by_relevance(
    results: List[Dict[str, Any]],
    min_score: float = 0.5,
    score_field: str = "similarity_score"
) -> Dict[str, Any]:
    """Filter results by relevance/similarity score.

    Args:
        results: List of result dictionaries to filter
        min_score: Minimum score threshold (0.0 to 1.0)
        score_field: Field name containing the score

    Returns:
        Dictionary with filtered results and statistics
    """
    if not results:
        return {
            "results": [],
            "total_input": 0,
            "total_output": 0,
            "filtered_count": 0,
            "min_score_used": min_score
        }

    original_count = len(results)
    filtered = []

    for item in results:
        score = item.get(score_field)

        # Handle various score formats
        if score is None:
            # Try alternative field names
            score = item.get("score", item.get("relevance", item.get("confidence")))

        if score is not None:
            try:
                score = float(score)
                if score >= min_score:
                    filtered.append(item)
            except (ValueError, TypeError):
                # If score can't be parsed, include the item
                filtered.append(item)
        else:
            # No score field, include by default
            filtered.append(item)

    return {
        "results": filtered,
        "total_input": original_count,
        "total_output": len(filtered),
        "filtered_count": original_count - len(filtered),
        "min_score_used": min_score,
        "retention_rate": len(filtered) / original_count if original_count > 0 else 0
    }


def create_filter_by_relevance_tool() -> Tool:
    """Create the filter_by_relevance tool.

    Returns:
        Configured Tool instance for filtering by relevance
    """
    return (
        create_tool("filter_by_relevance")

        .description(
            "Filter a list of results by their relevance/similarity score. "
            "Removes items below the minimum score threshold. "
            "Best for: cleaning up search results, removing low-quality matches, focusing on relevant content."
        )
        .category(ToolCategory.UTILITY)
        .cost(CostEstimate.VERY_LOW)
        .returns(
            "Dictionary with 'results' (filtered list), counts, and retention rate"
        )

        .parameter(
            "results",
            ParameterType.LIST,
            "List of result dictionaries to filter. "
            "Each item should have a score field.",
            required=True
        )

        .parameter(
            "min_score",
            ParameterType.FLOAT,
            "Minimum score threshold. Items below this score are removed.",
            required=False,
            default=0.5,
            constraints={"min": 0.0, "max": 1.0}
        )

        .parameter(
            "score_field",
            ParameterType.STRING,
            "Name of the field containing the score.",
            required=False,
            default="similarity_score",
            examples=["similarity_score", "score", "relevance", "confidence"]
        )

        .implementation(filter_by_relevance)

        .example(
            input={
                "results": [
                    {"id": 1, "similarity_score": 0.9},
                    {"id": 2, "similarity_score": 0.3},
                    {"id": 3, "similarity_score": 0.7}
                ],
                "min_score": 0.5
            },
            output={
                "results": [
                    {"id": 1, "similarity_score": 0.9},
                    {"id": 3, "similarity_score": 0.7}
                ],
                "total_input": 3,
                "total_output": 2,
                "filtered_count": 1,
                "retention_rate": 0.67
            }
        )

        .metadata(
            version="1.0.0",
            tags=["utility", "filter", "relevance", "fast"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# DEDUPLICATE RESULTS TOOL
# =============================================================================

def deduplicate_results(
    results: List[Dict[str, Any]],
    key: str = "document_id",
    keep: str = "first"
) -> Dict[str, Any]:
    """Remove duplicate items from results list.

    Args:
        results: List of result dictionaries to deduplicate
        key: Dictionary key to use for deduplication
        keep: Which duplicate to keep: 'first' or 'highest_score'

    Returns:
        Dictionary with deduplicated results and statistics
    """
    if not results:
        return {
            "results": [],
            "total_input": 0,
            "total_output": 0,
            "duplicates_removed": 0
        }

    original_count = len(results)
    seen: Set[Any] = set()
    unique = []

    # If keeping highest score, sort by score first
    if keep == "highest_score":
        # Sort by score descending (highest first)
        def get_score(item):
            score = item.get("similarity_score", item.get("score", item.get("relevance", 0)))
            try:
                return float(score) if score is not None else 0
            except (ValueError, TypeError):
                return 0
        results = sorted(results, key=get_score, reverse=True)

    for item in results:
        id_value = item.get(key)

        if id_value is None:
            # If no key found, try alternative keys
            id_value = item.get("id", item.get("_id", item.get("chunk_id")))

        if id_value is None:
            # If still no key, always include (can't dedupe without ID)
            unique.append(item)
            continue

        # Make hashable
        if isinstance(id_value, list):
            id_value = tuple(id_value)

        if id_value not in seen:
            seen.add(id_value)
            unique.append(item)

    return {
        "results": unique,
        "total_input": original_count,
        "total_output": len(unique),
        "duplicates_removed": original_count - len(unique),
        "key_used": key,
        "keep_strategy": keep
    }


def create_deduplicate_results_tool() -> Tool:
    """Create the deduplicate_results tool.

    Returns:
        Configured Tool instance for deduplication
    """
    return (
        create_tool("deduplicate_results")

        .description(
            "Remove duplicate items from a list of results based on a key field. "
            "Can preserve first occurrence or highest-scoring duplicate. "
            "Best for: cleaning up merged search results, removing duplicate documents."
        )
        .category(ToolCategory.UTILITY)
        .cost(CostEstimate.VERY_LOW)
        .returns(
            "Dictionary with 'results' (deduplicated list), counts, and metadata"
        )

        .parameter(
            "results",
            ParameterType.LIST,
            "List of result dictionaries to deduplicate.",
            required=True
        )

        .parameter(
            "key",
            ParameterType.STRING,
            "Dictionary key to use for deduplication. "
            "Items with the same value for this key are considered duplicates.",
            required=False,
            default="document_id",
            examples=["document_id", "chunk_id", "id", "_id"]
        )

        .parameter(
            "keep",
            ParameterType.STRING,
            "Which duplicate to keep: 'first' (first occurrence) or "
            "'highest_score' (duplicate with highest similarity score).",
            required=False,
            default="first",
            constraints={"enum": ["first", "highest_score"]}
        )

        .implementation(deduplicate_results)

        .example(
            input={
                "results": [
                    {"document_id": "doc1", "name": "Report A", "score": 0.9},
                    {"document_id": "doc2", "name": "Report B", "score": 0.7},
                    {"document_id": "doc1", "name": "Report A (duplicate)", "score": 0.8}
                ],
                "key": "document_id",
                "keep": "highest_score"
            },
            output={
                "results": [
                    {"document_id": "doc1", "name": "Report A", "score": 0.9},
                    {"document_id": "doc2", "name": "Report B", "score": 0.7}
                ],
                "total_input": 3,
                "total_output": 2,
                "duplicates_removed": 1
            }
        )

        .metadata(
            version="1.0.0",
            tags=["utility", "dedup", "merge", "fast"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# MERGE RESULTS TOOL
# =============================================================================

def merge_results(
    result_sets: List[List[Dict[str, Any]]],
    deduplicate: bool = True,
    key: str = "document_id",
    sort_by_score: bool = True
) -> Dict[str, Any]:
    """Merge multiple result sets into one.

    Args:
        result_sets: List of result lists to merge
        deduplicate: Whether to remove duplicates after merging
        key: Key to use for deduplication
        sort_by_score: Whether to sort by score after merging

    Returns:
        Dictionary with merged results and statistics
    """
    # Flatten all result sets
    merged = []
    for result_set in result_sets:
        if isinstance(result_set, list):
            merged.extend(result_set)

    total_before_dedup = len(merged)

    # Deduplicate if requested
    if deduplicate and merged:
        dedup_result = deduplicate_results(merged, key=key, keep="highest_score")
        merged = dedup_result["results"]

    # Sort by score if requested
    if sort_by_score and merged:
        def get_score(item):
            score = item.get("similarity_score", item.get("score", item.get("relevance", 0)))
            try:
                return float(score) if score is not None else 0
            except (ValueError, TypeError):
                return 0
        merged = sorted(merged, key=get_score, reverse=True)

    return {
        "results": merged,
        "total_input_sets": len(result_sets),
        "total_items_before_merge": sum(len(rs) for rs in result_sets if isinstance(rs, list)),
        "total_after_merge": total_before_dedup,
        "total_output": len(merged),
        "duplicates_removed": total_before_dedup - len(merged) if deduplicate else 0,
        "sorted": sort_by_score
    }


def create_merge_results_tool() -> Tool:
    """Create the merge_results tool.

    Returns:
        Configured Tool instance for merging results
    """
    return (
        create_tool("merge_results")

        .description(
            "Merge multiple result sets into a single list. "
            "Optionally deduplicates and sorts by score. "
            "Best for: combining results from multiple searches, consolidating findings."
        )
        .category(ToolCategory.UTILITY)
        .cost(CostEstimate.VERY_LOW)
        .returns(
            "Dictionary with 'results' (merged list), statistics, and metadata"
        )

        .parameter(
            "result_sets",
            ParameterType.LIST,
            "List of result lists to merge together.",
            required=True
        )

        .parameter(
            "deduplicate",
            ParameterType.BOOLEAN,
            "Whether to remove duplicates after merging.",
            required=False,
            default=True
        )

        .parameter(
            "key",
            ParameterType.STRING,
            "Key to use for deduplication if enabled.",
            required=False,
            default="document_id"
        )

        .parameter(
            "sort_by_score",
            ParameterType.BOOLEAN,
            "Whether to sort results by score (highest first).",
            required=False,
            default=True
        )

        .implementation(merge_results)

        .example(
            input={
                "result_sets": [
                    [{"document_id": "doc1", "score": 0.9}],
                    [{"document_id": "doc2", "score": 0.7}, {"document_id": "doc1", "score": 0.8}]
                ],
                "deduplicate": True,
                "sort_by_score": True
            },
            output={
                "results": [
                    {"document_id": "doc1", "score": 0.9},
                    {"document_id": "doc2", "score": 0.7}
                ],
                "total_input_sets": 2,
                "total_output": 2,
                "duplicates_removed": 1
            }
        )

        .metadata(
            version="1.0.0",
            tags=["utility", "merge", "consolidate"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# SUMMARIZE RESULTS TOOL
# =============================================================================

def summarize_results(
    results: List[Dict[str, Any]],
    include_score_stats: bool = True
) -> Dict[str, Any]:
    """Generate summary statistics for a result set.

    Args:
        results: List of result dictionaries to summarize
        include_score_stats: Whether to include score statistics

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {
            "total_results": 0,
            "has_results": False,
            "summary": "No results found."
        }

    # Basic counts
    total = len(results)

    # Extract unique values for various fields
    unique_projects = set()
    unique_doc_types = set()
    unique_documents = set()

    scores = []

    for item in results:
        # Project info
        proj_id = item.get("project_id", item.get("projectId"))
        if proj_id:
            unique_projects.add(proj_id)

        proj_name = item.get("project_name", item.get("projectName"))
        if proj_name:
            unique_projects.add(proj_name)

        # Document type
        doc_type = item.get("document_type", item.get("documentType", item.get("document_type_id")))
        if doc_type:
            unique_doc_types.add(doc_type)

        # Document ID
        doc_id = item.get("document_id", item.get("documentId", item.get("id")))
        if doc_id:
            unique_documents.add(doc_id)

        # Score
        score = item.get("similarity_score", item.get("score", item.get("relevance")))
        if score is not None:
            try:
                scores.append(float(score))
            except (ValueError, TypeError):
                pass

    # Build summary
    summary = {
        "total_results": total,
        "has_results": True,
        "unique_projects": len(unique_projects),
        "unique_document_types": len(unique_doc_types),
        "unique_documents": len(unique_documents),
        "projects": list(unique_projects)[:10],  # Limit for readability
        "document_types": list(unique_doc_types)
    }

    # Score statistics
    if include_score_stats and scores:
        summary["score_stats"] = {
            "count": len(scores),
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores),
            "median": sorted(scores)[len(scores) // 2]
        }

    # Generate text summary
    summary_text = f"Found {total} results"
    if unique_projects:
        summary_text += f" from {len(unique_projects)} project(s)"
    if unique_doc_types:
        summary_text += f", {len(unique_doc_types)} document type(s)"
    if scores:
        summary_text += f". Scores range from {min(scores):.2f} to {max(scores):.2f}"

    summary["summary"] = summary_text

    return summary


def create_summarize_results_tool() -> Tool:
    """Create the summarize_results tool.

    Returns:
        Configured Tool instance for summarizing results
    """
    return (
        create_tool("summarize_results")

        .description(
            "Generate summary statistics for a result set including counts, "
            "unique values, and score distribution. "
            "Best for: understanding result composition, quality assessment, reporting."
        )
        .category(ToolCategory.UTILITY)
        .cost(CostEstimate.VERY_LOW)
        .returns(
            "Dictionary with counts, unique values, score statistics, and text summary"
        )

        .parameter(
            "results",
            ParameterType.LIST,
            "List of result dictionaries to summarize.",
            required=True
        )

        .parameter(
            "include_score_stats",
            ParameterType.BOOLEAN,
            "Whether to include score distribution statistics.",
            required=False,
            default=True
        )

        .implementation(summarize_results)

        .example(
            input={
                "results": [
                    {"document_id": "doc1", "project_name": "Project A", "score": 0.9},
                    {"document_id": "doc2", "project_name": "Project A", "score": 0.7},
                    {"document_id": "doc3", "project_name": "Project B", "score": 0.8}
                ],
                "include_score_stats": True
            },
            output={
                "total_results": 3,
                "has_results": True,
                "unique_projects": 2,
                "unique_documents": 3,
                "score_stats": {
                    "min": 0.7,
                    "max": 0.9,
                    "avg": 0.8
                },
                "summary": "Found 3 results from 2 project(s). Scores range from 0.70 to 0.90"
            }
        )

        .metadata(
            version="1.0.0",
            tags=["utility", "summary", "statistics"],
            idempotent=True
        )

        .build()
    )


# =============================================================================
# REGISTRATION FUNCTIONS
# =============================================================================

def register_utility_tools(registry: ToolRegistry) -> List[str]:
    """Register all utility tools with the registry.

    Args:
        registry: ToolRegistry instance to register tools with

    Returns:
        List of registered tool names
    """
    tools = [
        create_extract_entities_tool(),
        create_filter_by_relevance_tool(),
        create_deduplicate_results_tool(),
        create_merge_results_tool(),
        create_summarize_results_tool(),
    ]

    registered = []
    for tool in tools:
        try:
            registry.register(tool)
            registered.append(tool.name)
            logger.info(f"Registered utility tool: {tool.name}")
        except ValueError as e:
            logger.warning(f"Failed to register tool {tool.name}: {e}")

    logger.info(f"Registered {len(registered)} utility tools: {', '.join(registered)}")
    return registered


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Tool creation functions
    "create_extract_entities_tool",
    "create_filter_by_relevance_tool",
    "create_deduplicate_results_tool",
    "create_merge_results_tool",
    "create_summarize_results_tool",

    # Implementation functions
    "extract_entities",
    "filter_by_relevance",
    "deduplicate_results",
    "merge_results",
    "summarize_results",

    # Registration
    "register_utility_tools",
]


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    """Test utility tools when running module directly."""
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Utility Tools Module")
    print("=" * 50)

    # Create tools and show info
    tools = [
        create_extract_entities_tool(),
        create_filter_by_relevance_tool(),
        create_deduplicate_results_tool(),
        create_merge_results_tool(),
        create_summarize_results_tool(),
    ]

    for tool in tools:
        print(f"\nTool: {tool.name}")
        print(f"  Category: {tool.category.value}")
        print(f"  Cost: {tool.cost_estimate.value}")
        print(f"  Parameters: {len(tool.parameters)}")

    # Test entity extraction
    print("\n" + "=" * 50)
    print("Testing Entity Extraction")

    test_text = "What are the water quality impacts from the Red Mountain Mining Project in Northern BC during 2023?"
    entities = extract_entities(test_text)
    print(f"\nInput: {test_text}")
    print(f"Entities: {entities}")

    # Test deduplication
    print("\n" + "=" * 50)
    print("Testing Deduplication")

    test_results = [
        {"document_id": "doc1", "name": "A", "score": 0.9},
        {"document_id": "doc2", "name": "B", "score": 0.7},
        {"document_id": "doc1", "name": "A dup", "score": 0.8}
    ]
    dedup_result = deduplicate_results(test_results, keep="highest_score")
    print(f"\nInput: {test_results}")
    print(f"Output: {dedup_result}")

    # Test registry
    print("\n" + "=" * 50)
    print("Testing Registry Registration")

    from .tool_registry import ToolRegistry

    registry = ToolRegistry()
    registered = register_utility_tools(registry)

    print(f"\nRegistered tools: {registered}")
    print(f"Registry statistics: {registry.get_statistics()}")

    sys.exit(0)
