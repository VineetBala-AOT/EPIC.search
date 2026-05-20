"""Agent-based query processing with tool execution capabilities."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from flask import current_app
from search_api.clients.vector_search_client import VectorSearchClient

logger = logging.getLogger(__name__)


class VectorSearchAgent:
    """Agent that can execute complex queries using available VectorSearchClient tools."""
    
    def __init__(self, llm_client=None, user_location: Optional[Dict[str, Any]] = None, 
                 project_ids: Optional[List[str]] = None, document_type_ids: Optional[List[str]] = None,
                 search_strategy: Optional[str] = None, ranking: Optional[Dict[str, Any]] = None,
                 location: Optional[Dict[str, Any]] = None, project_status: Optional[str] = None, 
                 years: Optional[List[int]] = None, parallel_searches_enabled: bool = True,
                 max_parallel_workers: int = 4):
        """Initialize the agent with available tools and optional LLM client.
        
        Args:
            llm_client: Required LLM client for intelligent planning
            user_location: Optional user location data for location-aware queries
            project_ids: Optional user-provided project IDs to use in search calls
            document_type_ids: Optional user-provided document type IDs to use in search calls
            search_strategy: Optional user-provided search strategy to use in search calls
            ranking: Optional user-provided ranking configuration to use in search calls
            location: Optional location parameter (user-provided takes precedence)
            project_status: Optional project status parameter (user-provided takes precedence)
            years: Optional years parameter (user-provided takes precedence)
        """
        self.available_tools = self._get_available_tools()
        self.llm_client = llm_client
        self.user_location = user_location
        self.user_project_ids = project_ids
        self.user_document_type_ids = document_type_ids
        self.user_search_strategy = search_strategy
        self.user_ranking = ranking
        self.location = location
        self.user_project_status = project_status
        self.user_years = years
        self.parallel_searches_enabled = parallel_searches_enabled
        self.max_parallel_workers = max_parallel_workers
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools from VectorSearchClient.
        
        Returns:
            List of tool definitions with names, descriptions, and parameters
        """
        tools = [
            {
                "name": "validate_query_relevance",
                "description": "Validate if the query is relevant to Environmental Assessment Office (EAO) scope - ALWAYS use this as the first step",
                "parameters": {
                    "query": "the user query to validate (required)"
                },
                "returns": "validation result with is_relevant boolean, confidence score, reasoning, and recommendation"
            },
            {
                "name": "search",
                "description": "Perform vector similarity search with various strategies and filtering",
                "parameters": {
                    "query": "search query text (required)",
                    "project_ids": "list of project IDs to search in - use ALL matching IDs when multiple projects are relevant (optional list)",
                    "document_type_ids": "list of document type IDs to filter - use ALL matching IDs when multiple document types are relevant (optional list)",
                    "location": "location filter - city, region, or area name (optional string)",
                    "user_location": "user location context with city, region, latitude, longitude (optional dict)",
                    "project_status": "project status filter - 'active', 'completed', 'recent', etc. (optional string)",
                    "years": "list of years to filter by - [2023, 2024] for recent documents (optional list of integers)",
                    "search_strategy": "search strategy to use (optional string)",
                    "ranking": "ranking configuration with minScore and topN (optional dict)"
                },
                "returns": "search results with documents and similarity scores"
            },
            {
                "name": "get_projects_list",
                "description": "Get all available projects in the system",
                "parameters": {},
                "returns": "list of projects with IDs and names"
            },
            {
                "name": "get_document_types", 
                "description": "Get all available document types",
                "parameters": {},
                "returns": "dictionary of document types with metadata"
            },
            {
                "name": "get_search_strategies",
                "description": "Get available search strategies and their capabilities", 
                "parameters": {},
                "returns": "dictionary of search strategies with descriptions"
            },
            {
                "name": "validate_chunks_relevance",
                "description": "Filter search results using LLM to keep only chunks relevant to the original query - use after each search step",
                "parameters": {
                    "query": "the original user query for relevance checking (required)",
                    "search_results": "the search results to filter for relevance (required list)",
                    "step_name": "name of the search step being validated for context (required string)"
                },
                "returns": "filtered search results containing only relevant chunks with filtering statistics"
            },
            {
                "name": "verify_reduce",
                "description": "Collect and combine all validated chunks from filter steps into final verified dataset - use after all filter steps complete",
                "parameters": {
                    "filter_steps": "list of filter step names to collect validated chunks from (required list)"
                },
                "returns": "combined verified chunks from all validation steps with counts and metadata"
            },
            {
                "name": "consolidate_results",
                "description": "Merge and deduplicate results from multiple search executions - use after completing all searches",
                "parameters": {
                    "merge_strategy": "how to handle duplicate documents: 'deduplicate' (default), 'preserve_all', 'highest_score' (optional string)"
                },
                "returns": "consolidated documents and chunks with deduplication metrics and total counts"
            },
            {
                "name": "summarize_results",
                "description": "Generate comprehensive AI summary of consolidated search results - use as final step after consolidation",
                "parameters": {
                    "query": "original search query for context (required)",
                    "include_metadata": "include search metadata in summary context (optional boolean, default: true)"
                },
                "returns": "AI-generated summary of all consolidated search results with confidence score and method details"
            }
        ]
        return tools
    
    def create_execution_plan(self, query: str, reason: str) -> List[Dict[str, Any]]:
        """Create an intelligent execution plan using LLM reasoning.
        
        Args:
            query: The user query
            reason: Why it was classified as agent-required
            
        Returns:
            List of execution steps with tools and parameters
        """
        # Enhance query with location and temporal keywords BEFORE LLM planning
        # This ensures the LLM sees the enhanced query and can make better decisions
        enhanced_query = self._enhance_query_with_keywords(query)
        
        # Log enhancement if it occurred
        if enhanced_query != query:
            logger.info(f"🔍 PRE-PLANNING STUFFING: Enhanced query from '{query}' to '{enhanced_query}'")
        
        return self._create_llm_execution_plan(enhanced_query, reason)
    
    def _create_llm_execution_plan(self, query: str, reason: str) -> List[Dict[str, Any]]:
        """Create an execution plan using LLM reasoning.
        
        Args:
            query: The user query
            reason: Why it was classified as agent-required
            
        Returns:
            List of execution steps
        """
               
        # Get search execution parameters from environment variables
        min_searches = int(os.getenv("AGENT_MIN_SEARCHES", "1"))
        max_searches = int(os.getenv("AGENT_MAX_SEARCHES", "3"))
        
        # Ensure min is not greater than max
        if min_searches > max_searches:
            logger.warning(f"🤖 CONFIG: AGENT_MIN_SEARCHES ({min_searches}) > AGENT_MAX_SEARCHES ({max_searches}), adjusting min to {max_searches}")
            min_searches = max_searches
        
        # Calculate total steps (validation + optional list steps + search steps + validation steps + verify_reduce + consolidation + summarization)
        min_total_steps = 1 + min_searches + min_searches + 3  # 1 validation + searches + validations + verify_reduce + consolidation + summarization  
        max_total_steps = 3 + max_searches + max_searches + 3  # 1 validation + 2 optional list steps + searches + validations + verify_reduce + consolidation + summarization
        
        logger.info(f"🤖 CONFIG: Using search execution range: {min_searches}-{max_searches} searches + validations ({min_total_steps}-{max_total_steps} total steps)")
        
        # Try to get available projects and document types from previous runs (if available)
        available_projects_context = ""
        available_document_types_context = ""
        
        # TODO: This would be populated from execution context in a re-planning scenario
        # For now, the LLM works with placeholders and context is resolved during parameter enhancement
        
        planning_prompt = f"""Create an execution plan for: "{query}"

Available tools: validate_query_relevance, get_projects_list, get_document_types, search, validate_chunks_relevance, verify_reduce, consolidate_results, summarize_results

{available_projects_context}
{available_document_types_context}

ALWAYS create {min_total_steps}-{max_total_steps} steps:
1. validate_query_relevance (always first)
2. get_projects_list (if projects mentioned)  
3. get_document_types (if doc types mentioned)
4+. search ({min_searches}-{max_searches} searches with specific parameters)
THEN. filter_search_* steps (validate_chunks_relevance tool for each search to filter results)
THEN. verify_reduce (collect all validated chunks from filter steps)
2ND LAST. consolidate_results (merge and deduplicate all verified results)
LAST. summarize_results (generate AI summary of consolidated results)

SEARCH EXECUTION STRATEGY:
- You MUST execute between {min_searches} and {max_searches} search operations
- Break complex queries into {min_searches}-{max_searches} focused, semantic sub-queries
- Each search should target a specific aspect or variation of the user's question
- Use different parameter combinations to ensure comprehensive coverage
- Consider different semantic angles, synonyms, and related concepts

QUERY DECOMPOSITION GUIDELINES:
- For multi-faceted queries: Break into separate searches for each aspect
- For broad topics: Use different keyword variations and semantic approaches
- For complex questions: Create searches that build upon each other
- For entity-specific queries: Separate searches for different entities mentioned
- For temporal queries: Use different time ranges or temporal perspectives

IMPORTANT RULES:
- Focus on creating diverse, comprehensive search strategies with {min_searches}-{max_searches} semantic variations
- Break complex queries into multiple focused searches that cover different aspects
- Use the new structured parameters (location, project_status, years) instead of keyword stuffing
- NOTE: Project IDs and document type IDs are automatically selected by the parameter extractor based on query intent

NEW SEARCH PARAMETERS AVAILABLE:
- location: Use for "near me", "local", geographic terms (e.g., "Vancouver", "Peace River region")
- project_status: Use for temporal context ("recent", "active", "completed", "historical")  
- years: List of specific years [2023, 2024] for temporal filtering
- These replace keyword stuffing - use structured parameters instead of adding keywords to query

MULTI-SEARCH STRATEGY EXAMPLES:

Example 1 - "Project letters about First Nations consultation" (needs {min_searches}-{max_searches} searches + validations):
[
  {{"step_name": "validate_query", "tool": "validate_query_relevance", "parameters": {{}}, "reasoning": "Check query relevance"}},
  {{"step_name": "search_consultation_letters", "tool": "search", "parameters": {{"query": "First Nations consultation correspondence letters"}}, "reasoning": "Search for consultation correspondence"}},
  {{"step_name": "search_consultation_meetings", "tool": "search", "parameters": {{"query": "First Nations consultation engagement meetings"}}, "reasoning": "Search for consultation activities"}},
  {{"step_name": "search_consultation_impacts", "tool": "search", "parameters": {{"query": "First Nations impacts concerns issues"}}, "reasoning": "Search for consultation-related impacts and concerns"}},
  {{"step_name": "filter_search_consultation_letters", "tool": "validate_chunks_relevance", "parameters": {{"search_results": "results_from_search_consultation_letters", "step_name": "search_consultation_letters"}}, "reasoning": "Filter search_consultation_letters results for relevance"}},
  {{"step_name": "filter_search_consultation_meetings", "tool": "validate_chunks_relevance", "parameters": {{"search_results": "results_from_search_consultation_meetings", "step_name": "search_consultation_meetings"}}, "reasoning": "Filter search_consultation_meetings results for relevance"}},
  {{"step_name": "filter_search_consultation_impacts", "tool": "validate_chunks_relevance", "parameters": {{"search_results": "results_from_search_consultation_impacts", "step_name": "search_consultation_impacts"}}, "reasoning": "Filter search_consultation_impacts results for relevance"}},
  {{"step_name": "verify_reduce", "tool": "verify_reduce", "parameters": {{"filter_steps": ["filter_search_consultation_letters", "filter_search_consultation_meetings", "filter_search_consultation_impacts"]}}, "reasoning": "Collect all validated chunks from filter steps"}},
  {{"step_name": "consolidate_results", "tool": "consolidate_results", "parameters": {{"merge_strategy": "deduplicate"}}, "reasoning": "Merge and deduplicate all verified results"}},
  {{"step_name": "summarize_results", "tool": "summarize_results", "parameters": {{"include_metadata": true}}, "reasoning": "Generate comprehensive summary of all findings"}}
]

Example 2 - "who is the main proponent for the Air Liquide project?" (needs {min_searches}-{max_searches} searches + validations):
[
  {{"step_name": "validate_query", "tool": "validate_query_relevance", "parameters": {{}}, "reasoning": "Check query relevance"}},
  {{"step_name": "search_proponent_info", "tool": "search", "parameters": {{"query": "Air Liquide proponent applicant company organization"}}, "reasoning": "Search for proponent information"}},
  {{"step_name": "search_project_details", "tool": "search", "parameters": {{"query": "Air Liquide project description application details"}}, "reasoning": "Search for project details and applicant info"}},
  {{"step_name": "search_application_info", "tool": "search", "parameters": {{"query": "Air Liquide application submitted by company proponent"}}, "reasoning": "Search for application and submission information"}},
  {{"step_name": "filter_search_proponent_info", "tool": "validate_chunks_relevance", "parameters": {{"search_results": "results_from_search_proponent_info", "step_name": "search_proponent_info"}}, "reasoning": "Filter search_proponent_info results for relevance"}},
  {{"step_name": "filter_search_project_details", "tool": "validate_chunks_relevance", "parameters": {{"search_results": "results_from_search_project_details", "step_name": "search_project_details"}}, "reasoning": "Filter search_project_details results for relevance"}},
  {{"step_name": "filter_search_application_info", "tool": "validate_chunks_relevance", "parameters": {{"search_results": "results_from_search_application_info", "step_name": "search_application_info"}}, "reasoning": "Filter search_application_info results for relevance"}},
  {{"step_name": "verify_reduce", "tool": "verify_reduce", "parameters": {{"filter_steps": ["filter_search_proponent_info", "filter_search_project_details", "filter_search_application_info"]}}, "reasoning": "Collect all validated chunks from filter steps"}},
  {{"step_name": "consolidate_results", "tool": "consolidate_results", "parameters": {{"merge_strategy": "deduplicate"}}, "reasoning": "Merge and deduplicate all verified results"}},
  {{"step_name": "summarize_results", "tool": "summarize_results", "parameters": {{"include_metadata": true}}, "reasoning": "Generate comprehensive summary of all findings"}}
]

Example 3 - "water quality impacts from mining projects" (needs {min_searches}-{max_searches} searches + validations):
[
  {{"step_name": "validate_query", "tool": "validate_query_relevance", "parameters": {{}}, "reasoning": "Check query relevance"}},
  {{"step_name": "search_water_quality_mining", "tool": "search", "parameters": {{"query": "water quality mining contamination pollution", "project_status": "active"}}, "reasoning": "Search for water quality impacts from mining"}},
  {{"step_name": "search_aquatic_effects", "tool": "search", "parameters": {{"query": "aquatic ecosystems fish habitat mining effects", "years": [2020, 2021, 2022, 2023, 2024]}}, "reasoning": "Search for aquatic ecosystem impacts"}},
  {{"step_name": "search_groundwater_surface", "tool": "search", "parameters": {{"query": "groundwater surface water quality monitoring mining"}}, "reasoning": "Search for groundwater and surface water monitoring"}},
  {{"step_name": "filter_search_water_quality_mining", "tool": "validate_chunks_relevance", "parameters": {{"search_results": "results_from_search_water_quality_mining", "step_name": "search_water_quality_mining"}}, "reasoning": "Filter search_water_quality_mining results for relevance"}},
  {{"step_name": "filter_search_aquatic_effects", "tool": "validate_chunks_relevance", "parameters": {{"search_results": "results_from_search_aquatic_effects", "step_name": "search_aquatic_effects"}}, "reasoning": "Filter search_aquatic_effects results for relevance"}},
  {{"step_name": "filter_search_groundwater_surface", "tool": "validate_chunks_relevance", "parameters": {{"search_results": "results_from_search_groundwater_surface", "step_name": "search_groundwater_surface"}}, "reasoning": "Filter search_groundwater_surface results for relevance"}},
  {{"step_name": "verify_reduce", "tool": "verify_reduce", "parameters": {{"filter_steps": ["filter_search_water_quality_mining", "filter_search_aquatic_effects", "filter_search_groundwater_surface"]}}, "reasoning": "Collect all validated chunks from filter steps"}},
  {{"step_name": "consolidate_results", "tool": "consolidate_results", "parameters": {{"merge_strategy": "deduplicate"}}, "reasoning": "Merge and deduplicate all verified results"}},
  {{"step_name": "summarize_results", "tool": "summarize_results", "parameters": {{"include_metadata": true}}, "reasoning": "Generate comprehensive summary of all findings"}}
]

CRITICAL: Focus on execution strategy, not parameter selection:
- Project IDs and document type IDs are automatically selected by the parameter extractor
- Create diverse search queries that cover different semantic aspects of the user's question
- Use structured parameters (location, project_status, years) when relevant to the query context

SEARCH VARIATION STRATEGIES:
- Semantic variations: Use synonyms and related terms across searches
- Scope variations: Broad vs specific searches for comprehensive coverage
- Temporal variations: Different time periods or project statuses
- Entity variations: Different entities, stakeholders, or geographic areas
- Document type variations: Different document types for the same topic
- Impact variations: Different types of impacts or concerns for the same project"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert query planning agent. CRITICAL: Return ONLY valid JSON array. NO comments like /* */, NO placeholder text, NO markdown, NO explanations. Only use the available tools provided in the prompt. NO analysis steps. Raw JSON only."},
                {"role": "user", "content": planning_prompt}
            ]
            
            # Calculate max_tokens based on search count - more searches need more tokens
            # Account for additional verification and validation steps
            estimated_tokens = 300 + (max_searches * 180)  # Increased base + per-step tokens
            max_tokens = max(1200, min(estimated_tokens, 3500))  # Between 1200-3500 tokens (increased limits)
            
            logger.info(f"🤖 AGENT: Using max_tokens={max_tokens} for {max_searches} potential searches")
            response = self.llm_client.chat_completion(messages, temperature=0.1, max_tokens=max_tokens)
            
            if response and "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"].strip()
                logger.info(f"🤖 AGENT: Raw LLM response content: {content[:500]}...")
                
                # Check if response was truncated (common signs of truncation)
                choice = response["choices"][0]
                finish_reason = choice.get("finish_reason", "unknown")
                usage = response.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                
                if finish_reason == "length":
                    logger.warning(f"🤖 AGENT: Response was truncated due to token limit. Used {completion_tokens}/{max_tokens} tokens")
                elif not content.rstrip().endswith((']', '}')):
                    logger.warning(f"🤖 AGENT: Response appears truncated - doesn't end with closing bracket/brace. Used {completion_tokens}/{max_tokens} tokens")
                else:
                    logger.info(f"🤖 AGENT: Response complete. Used {completion_tokens}/{max_tokens} tokens")
                
                # Clean the content - remove markdown blocks and extra whitespace
                content_clean = self._clean_json_response(content)
                
                # Apply common JSON fixes
                content_clean = self._fix_common_json_issues(content_clean)
                
                logger.info(f"🤖 AGENT: Cleaned content for JSON parsing: {content_clean[:200]}...")
                
                try:
                    execution_plan = json.loads(content_clean)
                    if isinstance(execution_plan, list) and len(execution_plan) > 0:
                        # Validate that each step has required fields
                        for i, step in enumerate(execution_plan):
                            required_fields = ["step_name", "tool", "parameters", "reasoning"]
                            missing_fields = [field for field in required_fields if field not in step]
                            if missing_fields:
                                logger.warning(f"🤖 AGENT: Step {i+1} missing required fields: {missing_fields}")
                        
                        logger.info(f"🤖 AGENT: LLM created execution plan with {len(execution_plan)} steps")
                        return execution_plan
                    else:
                        logger.error("🤖 AGENT: LLM planning failed - execution plan is not a valid list")
                        return []
                except json.JSONDecodeError as json_err:
                    logger.error(f"🤖 AGENT: JSON parsing failed: {json_err}")
                    logger.error(f"🤖 AGENT: JSON error details - Line: {json_err.lineno}, Column: {json_err.colno}")
                    
                    # Log the problematic area around the error line
                    lines = content_clean.split('\n')
                    error_line = json_err.lineno - 1  # Convert to 0-based index
                    start_line = max(0, error_line - 3)  # Show 3 lines before
                    end_line = min(len(lines), error_line + 4)  # Show 3 lines after
                    
                    logger.error(f"🤖 AGENT: Content around error (lines {start_line+1}-{end_line}):")
                    for i in range(start_line, end_line):
                        prefix = ">>> " if i == error_line else "    "
                        logger.error(f"{prefix}Line {i+1}: {lines[i] if i < len(lines) else '(EOF)'}")
                    
                    logger.error(f"🤖 AGENT: Full content length: {len(content_clean)} chars, {len(lines)} lines")
                    
                    # Try additional fixes for common JSON issues
                    logger.info("🤖 AGENT: Attempting additional JSON fixes...")
                    try:
                        # Try more aggressive comma fixing
                        content_fixed = self._aggressive_json_fix(content_clean)
                        logger.info(f"🤖 AGENT: Trying fixed JSON (first 300 chars): {content_fixed[:300]}...")
                        logger.info(f"🤖 AGENT: Fixed JSON (last 300 chars): ...{content_fixed[-300:]}")
                        
                        execution_plan = json.loads(content_fixed)
                        if isinstance(execution_plan, list) and len(execution_plan) > 0:
                            logger.info(f"🤖 AGENT: JSON fix successful! Created execution plan with {len(execution_plan)} steps")
                            return execution_plan
                    except json.JSONDecodeError as retry_err:
                        logger.error(f"🤖 AGENT: Retry JSON parsing also failed: {retry_err}")
                        
                        # Show the specific area around the retry error too
                        retry_lines = content_fixed.split('\n')
                        retry_error_line = retry_err.lineno - 1
                        retry_start = max(0, retry_error_line - 2)
                        retry_end = min(len(retry_lines), retry_error_line + 3)
                        
                        logger.error(f"🤖 AGENT: Retry error around line {retry_err.lineno}:")
                        for i in range(retry_start, retry_end):
                            prefix = ">>> " if i == retry_error_line else "    "
                            logger.error(f"{prefix}Line {i+1}: {retry_lines[i] if i < len(retry_lines) else '(EOF)'}")
                        
                        # Try one more ultra-specific fix for this exact pattern
                        logger.info("🤖 AGENT: Trying ultra-specific JSON fix...")
                        try:
                            content_ultra_fixed = self._ultra_specific_json_fix(content_fixed, retry_err.lineno)
                            execution_plan = json.loads(content_ultra_fixed)
                            if isinstance(execution_plan, list) and len(execution_plan) > 0:
                                logger.info(f"🤖 AGENT: Ultra-specific fix successful! Created execution plan with {len(execution_plan)} steps")
                                return execution_plan
                        except Exception as ultra_err:
                            logger.error(f"🤖 AGENT: Ultra-specific fix also failed: {ultra_err}")
                            
                            # Final fallback: try to salvage valid JSON objects from the content
                            logger.info("🤖 AGENT: Attempting to salvage valid JSON objects from content...")
                            salvaged_plan = self._salvage_json_objects(content_ultra_fixed)
                            if salvaged_plan:
                                logger.info(f"🤖 AGENT: Salvaged {len(salvaged_plan)} valid execution steps")
                                return salvaged_plan
                    except Exception as retry_err:
                        logger.error(f"🤖 AGENT: Retry attempt failed with error: {retry_err}")
                    
                    return []
            else:
                logger.error("🤖 AGENT: LLM planning failed - no valid execution plan generated")
                return []
                
        except Exception as e:
            logger.error(f"🤖 AGENT: Error in LLM planning: {e}")
            logger.error(f"🤖 AGENT: Exception type: {type(e)}")
            logger.error(f"🤖 AGENT: Exception traceback:", exc_info=True)
            return []
    
    def _clean_json_response(self, content: str) -> str:
        """Clean LLM response content to extract valid JSON.
        
        Args:
            content: Raw LLM response content
            
        Returns:
            Cleaned JSON string
        """
        # Remove common markdown code block markers
        content = content.replace("```json", "").replace("```", "").strip()
        
        # Remove any leading/trailing markdown blocks
        while content.startswith('```'):
            newline_pos = content.find('\n')
            if newline_pos != -1:
                content = content[newline_pos + 1:].strip()
            else:
                content = content[3:].strip()
                
        while content.endswith('```'):
            content = content[:-3].strip()
        
        # Remove any explanatory text before JSON
        lines = content.split('\n')
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('[') or line.strip().startswith('{'):
                json_start = i
                break
        
        if json_start >= 0:
            content = '\n'.join(lines[json_start:])
        
        # Remove any explanatory text after JSON
        # Find the last ] or } that might end the JSON
        last_bracket = max(content.rfind(']'), content.rfind('}'))
        if last_bracket != -1:
            content = content[:last_bracket + 1]
        
        return content.strip()
    
    def _fix_common_json_issues(self, content: str) -> str:
        """Fix common JSON syntax issues that LLMs often create.
        
        Args:
            content: JSON content to fix
            
        Returns:
            Fixed JSON content
        """
        import re
        
        # Fix missing commas between objects - look for } followed by " on next line
        # Pattern: }\n    "key" should become },\n    "key"
        content = re.sub(r'(\})\s*\n(\s*")', r'\1,\n\2', content)
        
        # Fix missing commas between array elements - look for } followed by { on next line  
        # Pattern: }\n  { should become },\n  {
        content = re.sub(r'(\})\s*\n(\s*\{)', r'\1,\n\2', content)
        
        # Fix missing commas at end of array elements - look for } followed by ] with possible whitespace
        # But only if there's no comma already
        # Pattern: }\n  ] should stay as }\n  ] (no comma needed before closing bracket)
        # Pattern: }\n something_else should become },\n something_else
        
        # More aggressive fix: add comma after } if it's not followed by ] or , or }
        content = re.sub(r'(\})\s*\n(?!\s*[\]\},])', r'\1,\n', content)
        
        # Fix missing commas after array elements containing values
        # Pattern: ]\n  { should become ],\n  {
        content = re.sub(r'(\])\s*\n(\s*\{)', r'\1,\n\2', content)
        
        # Fix missing commas within objects - look for " followed by " on next line
        # Pattern: "value"\n    "key" should become "value",\n    "key"
        content = re.sub(r'(")\s*\n(\s*")', r'\1,\n\2', content)
        
        # Fix missing commas after } within objects - look for } followed by " on next line (within object)
        # This is more specific than the first rule
        content = re.sub(r'(\})\s*\n(\s*"[^"]+"\s*:)', r'\1,\n\2', content)
        
        # Remove trailing commas before closing brackets/braces
        content = re.sub(r',(\s*[\]\}])', r'\1', content)
        
        return content
    
    def _aggressive_json_fix(self, content: str) -> str:
        """Apply more aggressive JSON fixes for severely malformed JSON.
        
        Args:
            content: JSON content that failed initial parsing
            
        Returns:
            More aggressively fixed JSON content
        """
        import re
        
        # Start with the basic fixes
        content = self._fix_common_json_issues(content)
        
        # Add missing commas between consecutive objects more aggressively
        # Look for } followed by anything that starts with { after whitespace
        content = re.sub(r'(\})\s*\n\s*(\{)', r'\1,\n  \2', content)
        
        # Fix the specific error pattern: line ending with } but not },
        # when it should have a comma (not the last item in array/object)
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            fixed_lines.append(line)
            
            # Check if this line ends with } (but not },) and needs a comma
            if stripped.endswith('}') and not stripped.endswith('},') and not stripped.endswith('}}'):
                # Look ahead to see if we need a comma
                has_more_content = False
                for j in range(i + 1, len(lines)):
                    next_stripped = lines[j].strip()
                    if next_stripped:  # Non-empty line found
                        # If next content starts with { or ", we need a comma
                        if next_stripped.startswith(('{', '"')):
                            has_more_content = True
                        break
                
                # If there's more content coming, add comma
                if has_more_content:
                    # Replace the last } with },
                    fixed_lines[-1] = line.rstrip().rstrip('}') + '},'
        
        # Additional fix: ensure the last object in an array doesn't have a trailing comma
        # Look for pattern: },\n  ] and change to }\n  ]
        content = '\n'.join(fixed_lines)
        content = re.sub(r'},(\s*\])', r'}\1', content)
        
        # Check if JSON is truncated and needs completion
        if not content.strip().endswith((']', '}')):
            lines = content.split('\n')
            
            # Count opening vs closing brackets/braces to see if we need to close
            open_brackets = content.count('[')
            close_brackets = content.count(']')
            open_braces = content.count('{')
            close_braces = content.count('}')
            
            if open_brackets > close_brackets:
                # Missing closing array brackets
                for _ in range(open_brackets - close_brackets):
                    lines.append(']')
                logger.info(f"🤖 AGENT: Aggressive fix added {open_brackets - close_brackets} missing closing array brackets")
            
            if open_braces > close_braces:
                # Missing closing object braces
                for _ in range(open_braces - close_braces):
                    lines.append('}')
                logger.info(f"🤖 AGENT: Aggressive fix added {open_braces - close_braces} missing closing object braces")
            
            content = '\n'.join(lines)
        
        return content
    
    def _ultra_specific_json_fix(self, content: str, error_line: int) -> str:
        """Apply ultra-specific fix for the exact line that's failing.
        
        Args:
            content: JSON content that failed parsing
            error_line: The line number where parsing failed (1-based)
            
        Returns:
            JSON with targeted fix applied
        """
        lines = content.split('\n')
        
        # Convert to 0-based index
        error_line_idx = error_line - 1
        
        if 0 <= error_line_idx < len(lines):
            problem_line = lines[error_line_idx]
            
            # If the problem line ends with just } and column 6 suggests missing comma
            # Try adding comma at the end
            if problem_line.strip().endswith('}') and not problem_line.strip().endswith('},'):
                # Check if this is likely the last item in an array (followed by ])
                is_last_item = False
                has_closing_bracket = False
                
                for i in range(error_line_idx + 1, len(lines)):
                    next_line = lines[i].strip()
                    if next_line:
                        if next_line.startswith(']'):
                            is_last_item = True
                            has_closing_bracket = True
                        break
                
                if not is_last_item:
                    # Add comma to this line
                    lines[error_line_idx] = problem_line.rstrip() + ','
                    logger.info(f"🤖 AGENT: Added comma to line {error_line}: '{lines[error_line_idx].strip()}'")
                elif not has_closing_bracket:
                    # This is the last item but there's no closing bracket - add it
                    lines.append('  }')
                    lines.append(']')
                    logger.info(f"🤖 AGENT: Added missing closing brackets after line {error_line}")
        
        # Check if JSON looks truncated and needs completion
        content_joined = '\n'.join(lines)
        if not content_joined.strip().endswith(']'):
            # Count opening vs closing brackets/braces to see if we need to close
            open_brackets = content_joined.count('[')
            close_brackets = content_joined.count(']')
            open_braces = content_joined.count('{')
            close_braces = content_joined.count('}')
            
            if open_brackets > close_brackets:
                # Missing closing array brackets
                for _ in range(open_brackets - close_brackets):
                    lines.append(']')
                logger.info(f"🤖 AGENT: Added {open_brackets - close_brackets} missing closing array brackets")
            
            if open_braces > close_braces:
                # Missing closing object braces
                for _ in range(open_braces - close_braces):
                    lines.append('}')
                logger.info(f"🤖 AGENT: Added {open_braces - close_braces} missing closing object braces")
        
        return '\n'.join(lines)
    
    def _salvage_json_objects(self, content: str) -> List[Dict[str, Any]]:
        """Extract valid JSON objects from malformed content as a last resort.
        
        Args:
            content: Malformed JSON content
            
        Returns:
            List of valid JSON objects that could be extracted
        """
        import re
        
        salvaged_objects = []
        
        try:
            # First try to complete the truncated JSON before salvaging
            # Look for the last incomplete object that might just need closing
            lines = content.split('\n')
            
            # Check if we have an incomplete last object
            in_object = False
            object_depth = 0
            last_complete_line = -1
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if '{' in stripped:
                    object_depth += stripped.count('{')
                    in_object = True
                if '}' in stripped:
                    object_depth -= stripped.count('}')
                    if object_depth == 0 and in_object:
                        last_complete_line = i
                        in_object = False
            
            # If we're still in an object at the end, try to complete it
            if in_object and object_depth > 0:
                completed_content = content
                # Add missing closing braces and array bracket
                for _ in range(object_depth):
                    completed_content += "\n    }"
                if not completed_content.rstrip().endswith(']'):
                    completed_content += "\n]"
                
                logger.info(f"🤖 AGENT: Attempting to complete truncated JSON...")
                try:
                    completed_plan = json.loads(completed_content)
                    if isinstance(completed_plan, list) and completed_plan:
                        logger.info(f"🤖 AGENT: Successfully completed truncated JSON! Recovered {len(completed_plan)} steps")
                        return completed_plan
                except:
                    logger.warning(f"🤖 AGENT: Could not complete truncated JSON, falling back to object salvage")
            
            # Fallback to individual object salvaging
            # Look for individual JSON objects in the content
            # Match patterns like { "step_name": "...", ... } (more flexible pattern)
            object_pattern = r'\{\s*"step_name"\s*:\s*"[^"]*"[^{}]*(?:\{[^{}]*\}[^{}]*)?\}'
            matches = re.finditer(object_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    obj_str = match.group()
                    # Try to parse this individual object
                    obj = json.loads(obj_str)
                    
                    # Validate it has required fields
                    if "step_name" in obj and "tool" in obj:
                        salvaged_objects.append(obj)
                        logger.info(f"🤖 AGENT: Salvaged valid object: {obj.get('step_name', 'unknown')}")
                except:
                    continue
                    
            # If no objects found with strict pattern, try looser patterns
            if not salvaged_objects:
                logger.info(f"🤖 AGENT: Trying looser salvage patterns...")
                # Split by lines and try to reconstruct objects manually
                current_obj = ""
                brace_count = 0
                
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('{') or (brace_count > 0):
                        current_obj += line + "\n"
                        brace_count += stripped.count('{') - stripped.count('}')
                        
                        if brace_count == 0 and current_obj.strip():
                            try:
                                obj = json.loads(current_obj.strip())
                                if isinstance(obj, dict) and "step_name" in obj and "tool" in obj:
                                    salvaged_objects.append(obj)
                                    logger.info(f"🤖 AGENT: Loosely salvaged object: {obj.get('step_name', 'unknown')}")
                            except:
                                pass
                            current_obj = ""
                            
        except Exception as e:
            logger.error(f"🤖 AGENT: JSON salvage operation failed: {e}")
        
        return salvaged_objects

    def _group_execution_steps(self, execution_plan: List[Dict[str, Any]]) -> List[List[int]]:
        """Group consecutive parallelizable steps together for parallel execution.
        
        Args:
            execution_plan: List of execution steps
            
        Returns:
            List of groups, where each group is a list of step indices.
            Non-parallelizable steps are in individual groups, parallelizable steps are grouped together.
            Search steps and validate_chunks_relevance steps can be parallelized.
        """
        groups = []
        search_steps = []
        validation_steps = []
        
        for i, step in enumerate(execution_plan):
            tool_name = step.get("tool")
            
            if tool_name == "search":
                search_steps.append(i)
            elif tool_name == "validate_chunks_relevance":
                validation_steps.append(i)
            else:
                # Before adding non-parallelizable step, close any open parallel groups
                if search_steps:
                    groups.append(search_steps)
                    search_steps = []
                if validation_steps:
                    groups.append(validation_steps)
                    validation_steps = []
                    
                # Add non-parallelizable step as individual group
                groups.append([i])
        
        # Close any remaining parallel groups at the end
        if search_steps:
            groups.append(search_steps)
        if validation_steps:
            groups.append(validation_steps)
            
        return groups

    def _execute_search_step_with_context(self, step: Dict[str, Any], step_index: int, context: Dict[str, Any], app_instance=None) -> Dict[str, Any]:
        """Execute a search step with Flask application context preserved.
        
        This wrapper ensures that Flask's application context is available
        in worker threads during parallel execution.
        
        Args:
            step: The search step to execute
            step_index: Index of the step in the execution plan
            context: Current execution context (read-only for parallel execution)
            app_instance: Flask app instance (passed from main thread)
            
        Returns:
            Dictionary containing step execution result with step_index for ordering
        """
        try:
            if app_instance is None:
                # Fallback: try to get current app (might fail in worker thread)
                app_instance = current_app._get_current_object()
            
            # Push application context in the worker thread
            with app_instance.app_context():
                try:
                    logger.info(f"🔍 PARALLEL WORKER: Executing step {step_index + 1} with Flask app context")
                    return self._execute_search_step(step, step_index, context)
                except Exception as e:
                    logger.error(f"🔍 PARALLEL WORKER: Error in step {step_index + 1}: {e}")
                    return {
                        "step_index": step_index,
                        "step": step.get("step_name", "unknown"),
                        "tool": step.get("tool", "unknown"),
                        "parameters": step.get("parameters", {}),
                        "original_parameters": step.get("parameters", {}),
                        "reasoning": step.get("reasoning", ""),
                        "result": {"success": False, "error": str(e)}
                    }
            
        except Exception as e:
            logger.error(f"🔍 PARALLEL CONTEXT: Failed to create Flask context for step {step_index + 1}: {e}")
            # Fallback result for context creation failures
            return {
                "step_index": step_index,
                "step": step.get("step_name", "unknown"),
                "tool": step.get("tool", "unknown"),
                "parameters": step.get("parameters", {}),
                "original_parameters": step.get("parameters", {}),
                "reasoning": step.get("reasoning", ""),
                "result": {"success": False, "error": f"Flask context error: {str(e)}"}
            }

    def _execute_search_step(self, step: Dict[str, Any], step_index: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single search step. Used for parallel execution.
        
        Args:
            step: The search step to execute
            step_index: Index of the step in the execution plan
            context: Current execution context (read-only for parallel execution)
            
        Returns:
            Dictionary containing step execution result with step_index for ordering
        """
        try:
            logger.info(f"🔍 PARALLEL: Executing search step {step_index + 1}: {step['step_name']}")
            
            # Check if this step should be skipped
            if self._should_skip_step(step, context):
                logger.info(f"🔍 PARALLEL: Skipping redundant step {step_index + 1}: {step['step_name']}")
                return {
                    "step_index": step_index,
                    "step": step["step_name"],
                    "tool": step["tool"], 
                    "parameters": step["parameters"],
                    "original_parameters": step["parameters"],
                    "reasoning": step.get("reasoning", ""),
                    "result": {"success": True, "result": "Skipped - information already available", "skipped": True}
                }
            
            # Enhance parameters based on current context
            enhanced_parameters = self._enhance_step_parameters(step["parameters"], context, step["tool"])
            
            # Execute the search
            tool_result = self.execute_tool(step["tool"], enhanced_parameters, context)
            
            # For validation steps, replace search_results parameters with LLM input chunks if available
            display_parameters = enhanced_parameters.copy()
            if step["tool"] == "validate_chunks_relevance" and isinstance(tool_result, dict) and tool_result.get("success"):
                result_data = tool_result.get("result", {})
                if "llm_input_chunks" in result_data:
                    display_parameters["search_results"] = result_data["llm_input_chunks"]
                    # Remove the full search results from display to avoid confusion
                    display_parameters.pop("_resolved_search_results", None)
                    display_parameters["_validation_note"] = "search_results shows structured chunks sent to LLM"
            
            return {
                "step_index": step_index,
                "step": step["step_name"],
                "tool": step["tool"],
                "parameters": display_parameters,
                "original_parameters": step["parameters"],
                "reasoning": step.get("reasoning", ""),
                "result": tool_result
            }
            
        except Exception as e:
            logger.error(f"🔍 PARALLEL: Error executing search step {step_index + 1}: {e}")
            return {
                "step_index": step_index,
                "step": step["step_name"],
                "tool": step["tool"],
                "parameters": step.get("parameters", {}),
                "original_parameters": step.get("parameters", {}),
                "reasoning": step.get("reasoning", ""),
                "result": {"success": False, "error": str(e)}
            }

    def _should_skip_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Determine if a step should be skipped based on available context.
        
        Args:
            step: The execution step to evaluate
            context: Current execution context with discovered information
            
        Returns:
            True if step should be skipped, False otherwise
        """
        if step is None:
            logger.error("🤖 SKIP CHECK: Step is None")
            return False
            
        if context is None:
            logger.error("🤖 SKIP CHECK: Context is None")
            return False
            
        tool_name = step.get("tool", "")
        step_name = step.get("step_name", "").lower()
        parameters = step.get("parameters", {})
        if parameters is None:
            logger.error("🤖 SKIP CHECK: Parameters is None")
            return False
        query = parameters.get("query", "").lower()
        
        # Skip search steps that are looking for project IDs when we already have them
        if tool_name == "search" and context["discovered_project_ids"]:
            # Check if this search is looking for project identification
            if any(term in step_name for term in ["project id", "identify project", "find project"]):
                logger.info(f"🤖 SKIP: Already have project IDs: {context['discovered_project_ids']}")
                return True
            
            # Check if query is trying to find project info we already have
            if any(term in query for term in ["project id", "project name", "identify project"]):
                # But only skip if we're looking for a specific project we already found
                for project_name, project_id in context["project_name_to_id_mapping"].items():
                    if project_name in query:
                        logger.info(f"🤖 SKIP: Already have project_id for '{project_name}': {project_id}")
                        return True
        
        # Skip document type searches when we already have the mappings
        if tool_name == "search" and context["discovered_document_type_ids"]:
            if any(term in step_name for term in ["document type", "find document", "identify document"]):
                logger.info(f"🤖 SKIP: Already have document types: {context['discovered_document_type_ids']}")
                return True
        
        # Skip get_projects_list if user already provided project IDs or we already have results
        if tool_name == "get_projects_list":
            if self.user_project_ids is not None:
                logger.info(f"🤖 SKIP: User provided project_ids, no need to fetch project list: {self.user_project_ids}")
                return True
            elif context["project_name_to_id_mapping"]:
                logger.info("🤖 SKIP: Already have project list")
                return True
            
        # Skip get_document_types if user already provided document type IDs or we already have results
        if tool_name == "get_document_types":
            if self.user_document_type_ids is not None:
                logger.info(f"🤖 SKIP: User provided document_type_ids, no need to fetch document types: {self.user_document_type_ids}")
                return True
            elif context["document_type_name_to_id_mapping"]:
                logger.info("🤖 SKIP: Already have document types list")
                return True
        
        return False

    def _filter_relevant_chunks_with_llm(self, query: str, search_results: List[Dict], step_name: str) -> List[Dict]:
        """Filter search results using LLM to keep only relevant chunks.
        
        Args:
            query: Original user query
            search_results: List of document chunks from search
            step_name: Name of the search step for context
            
        Returns:
            Filtered list of relevant chunks
        """
        if not search_results:
            logger.info(f"🔍 LLM FILTERING: No search results to validate for step '{step_name}' - returning empty list")
            empty_validation = {
                "step": step_name,
                "validation_metrics": {"total_received": 0, "total_valid": 0, "total_invalid": 0},
                "valid_chunk_ids": [],
                "chunks_kept": 0
            }
            return [], empty_validation
        
        # Prepare structured chunks for LLM evaluation
        llm_chunks = []
        chunk_id_to_original = {}
        
        for i, chunk in enumerate(search_results[:10]):  # Limit to first 10 chunks for efficiency
            # Create step-specific chunk ID for better tracking across multiple validation steps
            step_short = step_name.replace("search_", "").replace("filter_", "")[:10]  # Shorten step name
            chunk_id = f"{step_short}_chunk_{i+1}"
            chunk_text = ""
            
            if isinstance(chunk, dict):
                # Extract text content from chunk
                if "content" in chunk:
                    chunk_text = chunk.get("content", "")
                elif "text" in chunk:
                    chunk_text = chunk.get("text", "")
                elif "snippet" in chunk:
                    chunk_text = chunk.get("snippet", "")
                else:
                    chunk_text = str(chunk)
            else:
                chunk_text = str(chunk)
            
            # Create structured chunk for LLM
            llm_chunk = {
                "id": chunk_id,
                "content": chunk_text  # Send full content for better validation
            }
            
            llm_chunks.append(llm_chunk)
            chunk_id_to_original[chunk_id] = chunk
        
        total_chars_sent = sum(len(chunk["content"]) for chunk in llm_chunks)
        logger.info(f"🔍 LLM FILTERING: Sending {len(llm_chunks)} structured chunks ({total_chars_sent} chars) to LLM for relevance check")
        
        relevance_prompt = f"""
Query: "{query}"
Step Context: {step_name}

Evaluate these document chunks for relevance to the query. Return a JSON response with the IDs of chunks that are directly relevant.

Chunks to evaluate:
{json.dumps(llm_chunks, indent=2)}

Instructions:
- Only include chunks that directly answer or relate to the query
- Exclude chunks that are tangentially related or off-topic  
- Be strict - better to filter out borderline cases

Return your response as JSON in this exact format:
{{
    "valid_chunk_ids": ["chunk_1", "chunk_3", "chunk_5"],
    "metrics": {{
        "total_received": {len(llm_chunks)},
        "total_valid": <count_of_valid_chunks>,
        "total_invalid": <count_of_invalid_chunks>
    }}
}}

Response:"""

        # DEBUG: Show exactly what's being sent to the LLM
        logger.info(f"🔍 LLM VALIDATION DEBUG: Starting LLM validation for step '{step_name}' with {len(llm_chunks)} chunks")
        logger.info(f"🔍 LLM VALIDATION PROMPT: Full prompt being sent to LLM:")
        logger.info(f"🔍 LLM VALIDATION PROMPT: {relevance_prompt}")
        logger.info(f"🔍 LLM VALIDATION PROMPT: --- END OF LLM PROMPT ---")

        try:
            if self.llm_client:
                logger.info(f"🔍 LLM VALIDATION DEBUG: Calling LLM client for validation...")
                messages = [{"role": "user", "content": relevance_prompt}]
                response = self.llm_client.chat_completion(messages, temperature=0.1, max_tokens=1000)
                logger.info(f"🔍 LLM VALIDATION DEBUG: LLM response received: {response[:200]}...")
                
                # Parse LLM response to get relevant indices
                import re
                
                # Try to extract JSON response with structured format
                json_match = re.search(r'\{[^}]*"valid_chunk_ids"[^}]*\}', response, re.DOTALL)
                if json_match:
                    try:
                        validation_result = json.loads(json_match.group())
                        if validation_result is None:
                            logger.warning(f"🤖 LLM FILTER: JSON parsing returned None for step '{step_name}'")
                            validation_result = {}
                        valid_chunk_ids = validation_result.get("valid_chunk_ids", [])
                        metrics = validation_result.get("metrics", {})
                        
                        # Map valid chunk IDs back to original chunks
                        filtered_results = []
                        removed_chunks = []
                        
                        for chunk_id in valid_chunk_ids:
                            if chunk_id in chunk_id_to_original:
                                filtered_results.append(chunk_id_to_original[chunk_id])
                        
                        # Track removed chunks
                        for chunk_id, original_chunk in chunk_id_to_original.items():
                            if chunk_id not in valid_chunk_ids:
                                removed_chunks.append(original_chunk)
                        
                        # Log structured validation response
                        validation_summary = {
                            "step": step_name,
                            "validation_metrics": metrics,
                            "valid_chunk_ids": valid_chunk_ids,
                            "chunks_kept": len(filtered_results),
                            "removed_chunks": removed_chunks
                        }
                        logger.info(f"🤖 LLM VALIDATION RESULT: {json.dumps(validation_summary, indent=2)}")
                        
                        # Return both filtered results and validation response
                        return filtered_results, validation_summary
                    except json.JSONDecodeError as e:
                        logger.warning(f"🤖 LLM FILTER: Failed to parse JSON response: {e}")
                else:
                    # Fallback: try old format for compatibility
                    json_match = re.search(r'\[[\d,\s]*\]', response)
                    if json_match:
                        relevant_indices = json.loads(json_match.group())
                        if isinstance(relevant_indices, list):
                            logger.warning(f"🤖 LLM FILTER: Using deprecated index-based format")
                            
                            # Calculate removed chunks for fallback format
                            kept_results = search_results[:len(relevant_indices)]
                            removed_results = search_results[len(relevant_indices):]
                            
                            fallback_validation = {
                                "step": step_name,
                                "validation_metrics": {"total_received": len(search_results), "total_valid": len(relevant_indices), "total_invalid": len(removed_results)},
                                "valid_chunk_ids": [f"chunk_{i+1}" for i in relevant_indices],
                                "chunks_kept": len(relevant_indices),
                                "removed_chunks": removed_results
                            }
                            return kept_results, fallback_validation
            else:
                logger.warning(f"🔍 LLM VALIDATION DEBUG: No LLM client available for validation - returning all results")
                
        except Exception as e:
            logger.warning(f"🤖 LLM FILTER: Failed to filter chunks with LLM: {e}, returning original results")
        
        # Fallback: return original results if LLM filtering fails
        fallback_validation = {
            "step": step_name,
            "validation_metrics": {"total_received": len(search_results), "total_valid": len(search_results), "total_invalid": 0},
            "valid_chunk_ids": [],
            "chunks_kept": len(search_results),
            "error": "LLM validation failed"
        }
        return search_results, fallback_validation
    
    def _format_tools_for_llm(self) -> str:
        """Format tools list for LLM context with constraints."""
        tools_text = "AVAILABLE TOOLS AND CONSTRAINTS:\n\n"
        
        for i, tool in enumerate(self.available_tools, 1):
            tools_text += f"{i}. {tool['name']}\n"
            tools_text += f"   Description: {tool['description']}\n"
            tools_text += f"   Parameters: {tool['parameters']}\n"
            tools_text += f"   Returns: {tool['returns']}\n\n"
        
        tools_text += """
CRITICAL CONSTRAINTS:
- NO temporal/date filtering is available - documents cannot be filtered by date
- NO geographic/location filtering is available - documents cannot be filtered by location fields
- project_ids must be actual project ID lists (not placeholder strings) - include ALL matching project IDs
- document_type_ids must be actual document type ID lists (not placeholder strings) - include ALL matching document type IDs
- search() parameters: query (required), project_ids (optional list), document_type_ids (optional list), search_strategy (optional string)
- When multiple projects or document types are detected, use ALL relevant IDs in the respective lists
- NEW: Use structured location, project_status, and years parameters instead of keyword stuffing
- Always use concrete values, never placeholder text like "list of project IDs"

NEW STRUCTURED FILTERING APPROACH:
- USE location parameter: For "near me", "local", geographic terms → location: "Vancouver" or "British Columbia"
- USE project_status parameter: For temporal context → project_status: "recent", "active", "completed", "historical"  
- USE years parameter: For date filtering → years: [2023, 2024, 2025] or [2015, 2016, 2017]
- This replaces keyword stuffing with proper database-level filtering
- Clean queries with structured parameters for better search results

TEMPORAL QUERY STRATEGY (NEW):
- For "recent" queries: years: [2023, 2024, 2025], project_status: "recent"
- For "before 2020" queries: years: [2015, 2016, 2017, 2018, 2019], project_status: "historical"  
- For specific years mentioned: Extract and use in years parameter

LOCATION QUERY STRATEGY (NEW):
- For "near me": location: user's city or "British Columbia" as default
- For specific locations: location: "Vancouver", "Peace River region", etc.
- For regional queries: location: "Lower Mainland", "Vancouver Island", "Northern BC"

LOCATION QUERY STRATEGY:
- For "near me" queries, use semantic search with BC location keywords like "Victoria Vancouver Island British Columbia BC"
- For specific locations mentioned (e.g., "near Langford"), keep the original location terms
- For regional queries, add broader geographic terms like "British Columbia BC Lower Mainland Vancouver Island"
- Combine location keywords with the main query terms
- Acknowledge limitations and suggest proper geographic filtering tools when relevant
"""
        return tools_text
    
    def _get_user_location_context(self) -> Optional[Dict[str, Any]]:
        """Get user location from stored user_location attribute.
        
        This method now uses the user_location passed from the request body
        instead of extracting from headers to avoid CORS issues.
        
        Returns:
            Dict with location context or None if not available
        """
        if self.user_location:
            logger.info(f"🌍 LOCATION: Using location from request body: {self.user_location}")
            return self.user_location
        
        logger.debug("🌍 LOCATION: No user location available")
        return None

    def _enhance_query_with_location_keywords(self, query: str) -> str:
        """Enhance query with location keywords if it contains location-related terms.
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced query with location keywords, or original query if no enhancement needed
        """
        query_lower = query.lower()
        
        # Check if query contains location-related terms
        location_terms = ["local", "near me", "nearby", "close to", "in my area", "regional", "area"]
        has_location_terms = any(term in query_lower for term in location_terms)
        
        if not has_location_terms:
            return query
            
        # Get user location for keyword stuffing
        user_location = self._get_user_location_context()
        location_keywords = []
        
        if user_location:
            # Use provided location for keyword stuffing
            if 'city' in user_location:
                location_keywords.append(user_location['city'])
            if 'region' in user_location:
                location_keywords.append(user_location['region'])
                
            logger.info(f"🌍 LOCATION STUFFING: Using user location: {user_location}")
        
        # Always add BC context since this is EAO (British Columbia Environmental Assessment Office)
        if not any(bc_term in ' '.join(location_keywords).lower() for bc_term in ['bc', 'british columbia']):
            location_keywords.extend(['British Columbia', 'BC'])
        
        if location_keywords:
            enhanced_query = f"{query} {' '.join(location_keywords)}"
            logger.info(f"🌍 LOCATION STUFFING: Enhanced query with keywords: {location_keywords}")
            return enhanced_query
        
        return query

    def _enhance_query_with_temporal_keywords(self, query: str) -> str:
        """Enhance query with temporal keywords if it contains time-related terms.
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced query with temporal keywords, or original query if no enhancement needed
        """
        query_lower = query.lower()
        
        # Check if query contains temporal terms
        temporal_terms = ["before", "after", "recent", "latest", "current", "new", "last", "time", "date", "year"]
        has_temporal_terms = any(term in query_lower for term in temporal_terms)
        
        if not has_temporal_terms:
            return query
            
        temporal_keywords = []
        
        # Add temporal keywords based on query context
        if any(word in query_lower for word in ["before", "prior", "earlier"]):
            temporal_keywords.extend(["2019", "2018", "2017", "before", "2020"])
            logger.info("⏰ TEMPORAL STUFFING: Adding historical date keywords")
        elif any(word in query_lower for word in ["recent", "latest", "current", "new", "last"]):
            temporal_keywords.extend(["2023", "2024", "2025", "recent", "latest"])
            logger.info("⏰ TEMPORAL STUFFING: Adding recent date keywords")
        
        if temporal_keywords:
            enhanced_query = f"{query} {' '.join(temporal_keywords)}"
            logger.info(f"⏰ TEMPORAL STUFFING: Enhanced query with keywords: {temporal_keywords}")
            return enhanced_query
        
        return query

    def _enhance_query_with_keywords(self, query: str) -> str:
        """Enhance query with both location and temporal keywords if applicable.
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced query with relevant keywords
        """
        # Apply temporal enhancement first
        enhanced_query = self._enhance_query_with_temporal_keywords(query)
        
        # Then apply location enhancement
        enhanced_query = self._enhance_query_with_location_keywords(enhanced_query)
        
        return enhanced_query
    
    def _extract_search_parameters_from_query(self, query: str) -> Dict[str, Any]:
        """Extract structured search parameters from query instead of just keyword stuffing.
        
        Args:
            query: Original search query
            
        Returns:
            Dict with structured parameters for location, years, project_status
        """
        query_lower = query.lower()
        parameters = {}
        
        # Extract location parameters
        location_terms = ["near me", "nearby", "close to", "in my area", "local", "regional"]
        if any(term in query_lower for term in location_terms):
            user_location = self._get_user_location_context()
            if user_location:
                if 'city' in user_location:
                    parameters["location"] = user_location['city']
                    logger.info(f"🌍 LOCATION PARAM: Extracted location: {user_location['city']}")
            else:
                # Default to BC context
                parameters["location"] = "British Columbia"
                logger.info(f"🌍 LOCATION PARAM: Default to BC context")
        
        # Extract temporal parameters  
        temporal_terms = ["recent", "latest", "current", "new", "last"]
        historical_terms = ["before", "prior", "earlier", "old", "historical"]
        
        if any(term in query_lower for term in temporal_terms):
            parameters["years"] = [2023, 2024, 2025]
            parameters["project_status"] = "recent"
            logger.info(f"⏰ TEMPORAL PARAM: Extracted years: [2023, 2024, 2025], status: recent")
        elif any(term in query_lower for term in historical_terms):
            parameters["years"] = [2015, 2016, 2017, 2018, 2019, 2020]
            parameters["project_status"] = "historical"
            logger.info(f"⏰ TEMPORAL PARAM: Extracted years: [2015-2020], status: historical")
        
        return parameters
    
    def _generate_search_variations(self, base_query: str, num_variations: int = 3) -> List[str]:
        """Generate semantic variations of the base query for comprehensive search coverage.
        
        Args:
            base_query: The optimized semantic query from parameter extraction
            num_variations: Number of variations to generate (default: 3)
            
        Returns:
            List of search query variations
        """
        variations = [base_query]  # Always include the base query

        # No LLM needed when only the base query is requested
        if num_variations <= 1:
            return variations

        # Generate additional variations based on the base query
        try:
            if self.llm_client:
                prompt = f"""
You are a search optimization specialist for the BC Environmental Assessment Office (EAO).
Generate high-quality semantic variations of the user query to improve document retrieval.

Original Query: "{base_query}"

Generate exactly {num_variations - 1} variations.

Rules:
- Preserve the exact meaning and search intent.
- Rephrase using synonyms, alternate wording, and expanded abbreviations
  (e.g., EA → Environmental Assessment, AIR → Application Information Requirements).
- Use terminology commonly found in EAO documents, such as project descriptions,
  impact assessments, technical reports, environmental management plans, or compliance documents.
- Keep each variation concise (3–10 words).
- **Do not invent or change any project names, geographic locations, species, or entities**.
  Treat all proper nouns in the query as fixed. For example, if the query says
  "mine near Peace," "Peace" is a location and must not be used as a mine name.
- Focus on alternative phrasing, synonyms, and abbreviation expansions **only**.
- Do NOT include explanations, numbering, or extra text.
- Output MUST be a valid JSON array of strings, with no text before or after.

Return JSON only.
"""

                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.chat_completion(messages, temperature=0.3, max_tokens=300)
                
                if response and "choices" in response:
                    content = response["choices"][0]["message"]["content"].strip()
                    
                    # Clean up potential markdown formatting
                    if '```json' in content:
                        start = content.find('[')
                        end = content.rfind(']') + 1
                        if start != -1 and end != 0:
                            content = content[start:end]
                    
                    try:
                        import json
                        llm_variations = json.loads(content)
                        if isinstance(llm_variations, list):
                            variations.extend(llm_variations[:num_variations-1])
                            logger.info(f"🔍 Generated {len(llm_variations)} LLM search variations")
                    except json.JSONDecodeError:
                        logger.warning("🔍 Failed to parse LLM variations, using fallback")
        except Exception as e:
            logger.warning(f"🔍 LLM variation generation failed: {e}")
        
        # Fallback: create simple variations if LLM failed
        if len(variations) < num_variations:
            # Add basic keyword rearrangements
            words = base_query.split()
            if len(words) > 2:
                # Reverse word order
                variations.append(" ".join(reversed(words)))
            if len(words) > 3:
                # Use middle words only
                middle = words[1:-1] if len(words) > 3 else words[1:]
                if middle:
                    variations.append(" ".join(middle))
        
        # Ensure we don't exceed the requested number and remove duplicates
        unique_variations = []
        seen = set()
        for var in variations:
            if var.lower() not in seen:
                unique_variations.append(var)
                seen.add(var.lower())
                if len(unique_variations) >= num_variations:
                    break
        
        logger.info(f"🔍 Final search variations: {unique_variations}")
        return unique_variations
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a specific tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Clean parameters to pass to the tool (without massive context objects)
            context: Optional execution context passed separately (used by verify_reduce to access step results)
            
        Returns:
            Dict with tool execution result and success status
        """
        try:
            # Validate parameters before execution
            validation_error = self._validate_parameters(tool_name, parameters)
            if validation_error:
                logger.error(f"🤖 AGENT: Parameter validation failed for '{tool_name}': {validation_error}")
                return {"success": False, "error": f"Parameter validation failed: {validation_error}"}
            
            # Log parameters - for validation tools show only structured chunks that will be sent to LLM
            if tool_name == "validate_chunks_relevance":
                search_results = parameters.get("_resolved_search_results", [])
                query = parameters.get("query", "")
                
                if isinstance(search_results, list) and search_results:
                    # Create the same structured chunks that will be sent to LLM
                    llm_chunks_preview = []
                    for i, item in enumerate(search_results[:10]):  # Match the limit in _filter_relevant_chunks_with_llm
                        chunk_id = f"chunk_{i+1}"
                        content = ""
                        if isinstance(item, dict) and "content" in item:
                            content = str(item["content"])[:500]  # Truncate only for logging preview
                        else:
                            content = str(item)[:500]
                        
                        llm_chunks_preview.append({
                            "id": chunk_id,
                            "content": content[:100] + "..." if len(content) > 100 else content  # Truncate for logging
                        })
                    
                    logger.info(f"🔍 VALIDATION: Executing validation for query: '{query}' with {len(llm_chunks_preview)} structured chunks")
                    logger.info(f"🔍 VALIDATION: Structured chunks being sent to LLM: {llm_chunks_preview}")
                else:
                    logger.info(f"🔍 VALIDATION: Validating search results for query: '{query}' (no content found)")
            elif tool_name == "verify_reduce":
                # For verify_reduce, only log the filter steps, not the massive execution context
                filter_steps = parameters.get("filter_steps", [])
                logger.info(f"🔗 VERIFY REDUCE: Collecting verified chunks from filter steps: {filter_steps}")
            else:
                logger.info(f"🤖 AGENT: Executing tool '{tool_name}' with parameters: {parameters}")
            
            if tool_name == "search":
                # Get the query (already enhanced in create_execution_plan)
                search_query = parameters.get("query", "")
                
                # Prioritize user-provided parameters over LLM-generated ones
                final_project_ids = self.user_project_ids if self.user_project_ids is not None else parameters.get("project_ids", [])
                final_document_type_ids = self.user_document_type_ids if self.user_document_type_ids is not None else parameters.get("document_type_ids", [])
                final_search_strategy = self.user_search_strategy if self.user_search_strategy is not None else parameters.get("search_strategy", "")
                final_ranking = self.user_ranking if self.user_ranking is not None else parameters.get("ranking", None)
                
                # Log parameter usage
                if self.user_project_ids is not None:
                    logger.info(f"🤖 AGENT: Using user-provided project_ids: {final_project_ids}")
                elif parameters.get("project_ids"):
                    logger.info(f"🤖 AGENT: Using LLM-generated project_ids: {final_project_ids}")
                    
                if self.user_document_type_ids is not None:
                    logger.info(f"🤖 AGENT: Using user-provided document_type_ids: {final_document_type_ids}")
                elif parameters.get("document_type_ids"):
                    logger.info(f"🤖 AGENT: Using LLM-generated document_type_ids: {final_document_type_ids}")
                    
                if self.user_search_strategy is not None:
                    logger.info(f"🤖 AGENT: Using user-provided search_strategy: {final_search_strategy}")
                elif parameters.get("search_strategy"):
                    logger.info(f"🤖 AGENT: Using LLM-generated search_strategy: {final_search_strategy}")
                
                if self.user_ranking is not None:
                    logger.info(f"🤖 AGENT: Using user-provided ranking: {final_ranking}")
                elif parameters.get("ranking"):
                    logger.info(f"🤖 AGENT: Using LLM-generated ranking: {final_ranking}")
                
                # Get the new filtering parameters
                location = parameters.get("location", "")
                project_status = parameters.get("project_status", "")
                years = parameters.get("years", [])
                user_location = parameters.get("user_location", None)
                
                # Log new parameter usage
                if location:
                    logger.info(f"🤖 AGENT: Using location filter: {location}")
                if project_status:
                    logger.info(f"🤖 AGENT: Using project_status filter: {project_status}")
                if years:
                    logger.info(f"🤖 AGENT: Using years filter: {years}")
                
                result = VectorSearchClient.search(
                    query=search_query,
                    project_ids=final_project_ids,
                    document_type_ids=final_document_type_ids,
                    location=location,
                    project_status=project_status,
                    years=years,
                    search_strategy=final_search_strategy,
                    ranking=final_ranking,
                    user_location=user_location
                )
            elif tool_name == "validate_query_relevance":
                # Execute query validation using the validation service
                query = parameters.get("query", "")
                if not query:
                    raise ValueError("Query parameter is required for validation")
                
                logger.info(f"🔍 AGENT: Validating query relevance for: '{query}'")
                
                try:
                    # Import and use the query validator factory
                    from search_api.services.generation.factories import QueryValidatorFactory
                    validator = QueryValidatorFactory.create_validator()
                    validation_result = validator.validate_query_relevance(query)
                    
                    logger.info(f"🔍 AGENT: Query validation completed: {validation_result}")
                    result = validation_result
                    
                except Exception as validation_error:
                    logger.error(f"🔍 AGENT: Query validation failed: {validation_error}")
                    # Return fallback validation allowing the query to proceed
                    result = {
                        "is_relevant": True,
                        "confidence": 0.5,
                        "reasoning": ["Validation service unavailable", "Allowing query to proceed"],
                        "recommendation": "proceed_with_search",
                        "validation_error": str(validation_error)
                    }
            elif tool_name == "get_projects_list":
                result = VectorSearchClient.get_projects_list()
            elif tool_name == "get_document_types":
                result = VectorSearchClient.get_document_types()
            elif tool_name == "get_search_strategies":
                result = VectorSearchClient.get_search_strategies()
            elif tool_name == "verify_reduce":
                # Collect all validated chunks from the specified filter steps
                filter_steps = parameters.get("filter_steps", [])
                
                # Create clean parameters without massive execution context
                temp_parameters = {
                    "filter_steps": filter_steps,
                    "_execution_context": context if context else {}
                }
                
                result = self._collect_verified_chunks(filter_steps, temp_parameters)
            elif tool_name == "consolidate_results":
                # This tool is handled by the execution context - it needs access to all search results
                # We'll implement the actual consolidation logic in the _update_execution_context method
                result = {
                    "message": "Consolidation will be handled by execution context",
                    "consolidation_pending": True
                }
            elif tool_name == "validate_chunks_relevance":
                # This tool uses LLM to validate chunk relevance against the original query
                query = parameters.get("query", "")
                search_results_ref = parameters.get("search_results", "")
                step_name = parameters.get("step_name", "")
                
                if not query:
                    raise ValueError("Query parameter is required for chunk validation")
                if not step_name:
                    raise ValueError("Step name is required for chunk validation")
                
                # The search_results parameter will be a reference like "results_from_search_step"
                # We need to get the actual results from the execution context
                # This will be handled in _enhance_step_parameters method
                search_results = parameters.get("_resolved_search_results", [])
                
                if not search_results:
                    # If no resolved results, return empty validation with structured format
                    result = {
                        "original_count": 0,
                        "filtered_count": 0,
                        "relevant_chunks": [],
                        "step_name": step_name,
                        "validation_completed": True,
                        "message": "No search results found to validate",
                        "llm_input_chunks": [],  # What was sent to LLM (empty)
                        "validation_summary": {
                            "chunks_sent_to_llm": 0,
                            "chunks_validated_as_relevant": 0
                        }
                    }
                else:
                    # Create structured chunks that will be sent to LLM (same logic as in _filter_relevant_chunks_with_llm)
                    llm_input_chunks = []
                    for i, chunk in enumerate(search_results[:10]):  # Match the limit
                        chunk_id = f"chunk_{i+1}"
                        chunk_text = ""
                        if isinstance(chunk, dict):
                            if "content" in chunk:
                                chunk_text = chunk.get("content", "")
                            elif "text" in chunk:
                                chunk_text = chunk.get("text", "")
                            elif "snippet" in chunk:
                                chunk_text = chunk.get("snippet", "")
                            else:
                                chunk_text = str(chunk)
                        else:
                            chunk_text = str(chunk)
                        
                        llm_input_chunks.append({
                            "id": chunk_id,
                            "content": chunk_text  # Send full content for better validation
                        })
                    
                    # Use LLM to filter relevant chunks
                    logger.info(f"🔍 VALIDATION DEBUG: About to call LLM validation for step '{step_name}' with {len(search_results)} search results")
                    relevant_chunks, llm_validation_response = self._filter_relevant_chunks_with_llm(query, search_results, step_name)
                    logger.info(f"🔍 VALIDATION DEBUG: LLM validation completed - {len(relevant_chunks)} relevant chunks returned")
                    
                    result = {
                        "original_count": len(search_results),
                        "filtered_count": len(relevant_chunks),
                        "relevant_chunks": relevant_chunks,
                        "step_name": step_name,
                        "validation_completed": True,
                        "llm_input_chunks": llm_input_chunks,  # What was actually sent to LLM
                        "llm_validation_response": llm_validation_response,  # What the LLM decided
                        "validation_summary": {
                            "chunks_sent_to_llm": len(llm_input_chunks),
                            "chunks_validated_as_relevant": len(relevant_chunks),
                            "llm_decision": llm_validation_response
                        }
                    }
            elif tool_name == "summarize_results":
                # Perform actual summarization
                query = parameters.get("query", "")
                include_metadata = parameters.get("include_metadata", True)
                
                if not query:
                    raise ValueError("Query parameter is required for summarization")
                
                # Get consolidated results from context
                if not context or "consolidated_results" not in context:
                    result = "No consolidated results available for summarization"
                    logger.warning("📝 AGENT SUMMARY: No consolidated results in context")
                else:
                    logger.info("📝 AGENT SUMMARY: Starting summarization of consolidated results...")
                    
                    try:
                        from search_api.services.generation.factories import SummarizerFactory
                        
                        summarizer = SummarizerFactory.create_summarizer()
                        
                        # Combine documents and chunks for summarization
                        consolidated_results = context["consolidated_results"]
                        all_results = consolidated_results.get("documents", []) + consolidated_results.get("document_chunks", [])
                        
                        if all_results:
                            summary_result = summarizer.summarize_search_results(
                                query=query,
                                documents_or_chunks=all_results,
                                search_context={
                                    "context": "Agent consolidation summary",
                                    "search_strategy": "agent_multi_search", 
                                    "total_documents": len(consolidated_results.get("documents", [])),
                                    "total_chunks": len(consolidated_results.get("document_chunks", [])),
                                    "search_executions": getattr(self, 'search_count', 1)
                                }
                            )
                            
                            result = summary_result.get("summary", "Summary generation failed")
                            logger.info(f"📝 AGENT SUMMARY: Generated summary using {summary_result.get('provider', 'unknown')} with confidence {summary_result.get('confidence', 0)}")
                        else:
                            result = "No relevant documents were found for the given query."
                            logger.warning("📝 AGENT SUMMARY: No results to summarize")
                            
                    except Exception as e:
                        logger.error(f"📝 AGENT SUMMARY: Summarization failed: {e}")
                        result = "Summary generation failed due to an error."
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            logger.info(f"🤖 AGENT: Tool '{tool_name}' executed successfully")
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"🤖 AGENT: Error executing tool '{tool_name}': {e}")
            return {"success": False, "error": str(e)}

    def _validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[str]:
        """Validate parameters for a tool before execution.
        
        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        if tool_name == "search":
            # Check for placeholder strings in list parameters
            project_ids = parameters.get("project_ids", [])
            if isinstance(project_ids, str):
                return f"project_ids must be a list, not string: '{project_ids}'"
            
            document_type_ids = parameters.get("document_type_ids", [])
            if isinstance(document_type_ids, str):
                return f"document_type_ids must be a list, not string: '{document_type_ids}'"
            
            # Validate years parameter
            years = parameters.get("years", [])
            if years and not isinstance(years, list):
                return f"years must be a list of integers, not {type(years)}: '{years}'"
            
            if years and not all(isinstance(year, int) and 1900 <= year <= 2100 for year in years):
                return f"years must contain valid year integers (1900-2100): '{years}'"
                
        elif tool_name == "get_document_similarity":
            document_id = parameters.get("document_id", "")
            if not document_id or "ID of" in str(document_id):
                return f"document_id must be a concrete ID, not placeholder: '{document_id}'"
        elif tool_name == "validate_query_relevance":
            query = parameters.get("query", "")
            if not query:
                return "query parameter is required for validation"
        elif tool_name == "validate_chunks_relevance":
            query = parameters.get("query", "")
            # Query will be auto-provided if not specified, so don't require it here
            search_results = parameters.get("search_results", [])
            # Allow empty search_results (e.g., when previous search returned no results)
            if search_results is None:
                return "search_results parameter is required for chunk validation"
            # Allow string references that will be resolved during parameter enhancement
            if isinstance(search_results, str) and not search_results.startswith("results_from_"):
                return "search_results must be a list of documents/chunks or a valid step reference starting with 'results_from_'"
        elif tool_name == "consolidate_results":
            # consolidate_results parameters are optional, no validation needed
            merge_strategy = parameters.get("merge_strategy", "deduplicate")
            if merge_strategy not in ["deduplicate", "preserve_all", "highest_score"]:
                return f"merge_strategy must be one of: 'deduplicate', 'preserve_all', 'highest_score', got: '{merge_strategy}'"
        elif tool_name == "summarize_results":
            query = parameters.get("query", "")
            if not query:
                return "query parameter is required for summarization"
                
        return None  # Validation passed

    def generate_tool_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """Generate tool suggestions using LLM analysis of the query.
        
        Args:
            query: The user query to analyze
            
        Returns:
            List of suggested tools to improve the API
        """
        if not self.llm_client:
            logger.info("🤖 TOOL SUGGESTIONS: No LLM client available, skipping tool suggestions")
            return []
        
        tools_context = self._format_tools_for_llm()
        
        suggestion_prompt = f"""You are an API design expert analyzing a user query to suggest missing tools.

USER QUERY: "{query}"

CURRENT AVAILABLE TOOLS:
{tools_context}

TASK: Analyze the query and suggest up to 3 NEW tools that would improve the API's capabilities for this type of query.

FOCUS AREAS:
- Temporal/date filtering (if query involves time/dates)
- Geographic/location filtering (if query involves location)
- Advanced search capabilities
- Data aggregation and statistics
- Document metadata and classification

For each suggested tool, provide:
1. function_name: Clear, descriptive name
2. description: What the tool does
3. endpoint: Suggested API endpoint
4. justification: Why this tool would help with the query
5. priority: HIGH/MEDIUM/LOW based on impact
6. parameters: Expected input parameters

Respond with a JSON array of tool suggestions:
[
  {{{{
    "function_name": "tool_name",
    "description": "what it does",
    "endpoint": "/api/suggested/endpoint",
    "justification": "why it helps with this query",
    "priority": "HIGH|MEDIUM|LOW",
    "parameters": {{{{
      "param1": "description",
      "param2": "description"
    }}}}
  }}}}
]

JSON Response:"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert API designer. Respond with valid JSON only."},
                {"role": "user", "content": suggestion_prompt}
            ]
            
            response = self.llm_client.chat_completion(messages, temperature=0.3, max_tokens=1000)
            
            if response and "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"].strip()
                content_clean = content.replace("```json", "").replace("```", "").strip()
                
                suggestions = json.loads(content_clean)
                logger.info(f"🤖 TOOL SUGGESTIONS: LLM generated {len(suggestions)} tool suggestions")
                return suggestions
            else:
                logger.warning("🤖 TOOL SUGGESTIONS: LLM suggestion generation failed")
                return []
                
        except Exception as e:
            logger.error(f"🤖 TOOL SUGGESTIONS: Error in LLM tool suggestion: {e}")
            return []

    def _llm_select_parameters(self, parameters: Dict[str, Any], context: Dict[str, Any], selection_type: str) -> Dict[str, Any]:
        """Use LLM to intelligently select project_ids or document_type_ids from available options.
        
        Args:
            parameters: Current search parameters
            context: Execution context with available options
            selection_type: "projects" or "document_types"
            
        Returns:
            Parameters with LLM-selected IDs
        """
        if not self.llm_client:
            logger.warning(f"🤖 LLM SELECTION: No LLM client available for {selection_type} selection")
            return parameters
        
        query = parameters.get("query", "")
        original_query = context.get("original_query", query)
        
        if selection_type == "projects":
            available_context = context.get("available_projects_context", "")
            param_name = "project_ids"
        else:  # document_types
            available_context = context.get("available_document_types_context", "")
            param_name = "document_type_ids"
        
        if not available_context:
            logger.info(f"🤖 LLM SELECTION: No {selection_type} context available")
            return parameters
        
        if selection_type == "document_types":
            selection_prompt = f"""You are selecting relevant document types for a search query.

ORIGINAL USER QUERY: "{original_query}"
SEARCH QUERY: "{query}"

{available_context}

Based on the queries above, select the most relevant document_type_ids from the available options.
Return ONLY a JSON array of the selected IDs, or an empty array if none are specifically relevant.

WHEN TO SELECT DOCUMENT TYPES:
✓ Query explicitly mentions document types: "show me letters", "find reports", "get presentations"
✓ Query asks for specific document format: "correspondence about X", "memos regarding Y"

WHEN TO RETURN EMPTY ARRAY [] (search all document types):
✗ General information queries: "who is the proponent?", "what is the project status?"
✗ Factual questions about projects: "when was X approved?", "where is Y located?"
✗ Broad queries: "tell me about project X", "information on Y"
✗ Process questions: "what are the impacts?", "what consultation occurred?"

DOCUMENT TYPE GROUPING RULES (when document types ARE relevant):
- If query mentions a document type but NO specific act: include ALL variations across acts
- If query mentions BOTH document type AND specific act: only include that specific combination
- Examples: "letters" → all letter types; "Environmental Assessment Act letters" → only EA letters

Response format: ["id1", "id2", "id3"] or []

JSON Response:"""
        else:  # projects
            selection_prompt = f"""You are selecting relevant projects for a search query.

ORIGINAL USER QUERY: "{original_query}"
SEARCH QUERY: "{query}"

{available_context}

Based on the queries above, select the most relevant project_ids from the available options.
Return ONLY a JSON array of the selected IDs, or an empty array if none are specifically relevant.

WHEN TO SELECT SPECIFIC PROJECTS:
✓ Query mentions specific project names: "Air Liquide Liquid Nitrogen Plant Project"
✓ Query asks about a particular project: "for Project X, who is the proponent?"

WHEN TO RETURN EMPTY ARRAY [] (search all projects):
✗ General queries about multiple projects: "what projects are in BC?", "show me mining projects"
✗ Comparative queries: "which projects have the most impacts?"
✗ Broad topic searches: "projects with water concerns", "mining project impacts"
✗ When no specific project is mentioned by name

SELECTION RULES:
- Be very specific - only select projects explicitly mentioned in the query
- If unsure whether a project name matches, include it (better to include than miss)
- For broad queries about project categories or topics, return empty array to search all

Response format: ["id1", "id2"] or []

JSON Response:"""

        try:
            messages = [
                {"role": "system", "content": f"You are an expert at selecting relevant {selection_type} for search queries. Return only a JSON array of IDs."},
                {"role": "user", "content": selection_prompt}
            ]
            
            response = self.llm_client.chat_completion(messages, temperature=0.1, max_tokens=200)
            
            if response and "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"].strip()
                content_clean = content.replace("```json", "").replace("```", "").strip()
                
                try:
                    selected_ids = json.loads(content_clean)
                    if isinstance(selected_ids, list):
                        if selected_ids:
                            # For document types, apply smart grouping to ensure we get all related types
                            if selection_type == "document_types":
                                selected_ids = self._expand_document_type_groups(selected_ids, context, original_query)
                            
                            parameters[param_name] = selected_ids
                            logger.info(f"🤖 LLM SELECTION: Selected {len(selected_ids)} {selection_type} IDs: {selected_ids}")
                        else:
                            logger.info(f"🤖 LLM SELECTION: No specific {selection_type} selected - searching all")
                            # Remove parameter to search all
                            parameters.pop(param_name, None)
                    else:
                        logger.warning(f"🤖 LLM SELECTION: Invalid response format for {selection_type}")
                except json.JSONDecodeError as e:
                    logger.error(f"🤖 LLM SELECTION: JSON parsing failed for {selection_type}: {e}")
            
        except Exception as e:
            logger.error(f"🤖 LLM SELECTION: Error selecting {selection_type}: {e}")
        
        return parameters

    def _expand_document_type_groups(self, selected_ids: List[str], context: Dict[str, Any], query: str) -> List[str]:
        """Expand document type selections to include all related types under different acts.
        
        Args:
            selected_ids: Initially selected document type IDs
            context: Execution context with available document types
            query: Original query to check for specific act mentions
            
        Returns:
            Expanded list of document type IDs including related types
        """
        available_document_types = context.get("available_document_types", [])
        if not available_document_types:
            return selected_ids
        
        # Check if query mentions specific acts - if so, don't expand
        act_keywords = [
            "environmental assessment act", "water act", "mines act", "forest act", 
            "wildlife act", "fisheries act", "species at risk act", "impact assessment act"
        ]
        query_lower = query.lower()
        has_specific_act = any(act in query_lower for act in act_keywords)
        
        if has_specific_act:
            logger.info("🔍 DOC TYPE EXPANSION: Specific act mentioned - not expanding document types")
            return selected_ids
        
        # Build mappings for document type expansion
        id_to_doc_type = {}
        semantic_groups = {}  # Group document types by semantic similarity (aliases, base names)
        
        for doc_type in available_document_types:
            if isinstance(doc_type, dict) and "document_type_id" in doc_type and "document_type_name" in doc_type:
                doc_id = doc_type["document_type_id"]
                doc_name = doc_type["document_type_name"].lower()
                aliases = [alias.lower() for alias in doc_type.get("aliases", [])]
                act = doc_type.get("act", "")
                
                id_to_doc_type[doc_id] = doc_type
                
                # Create semantic keys for this document type (name + aliases)
                semantic_keys = [doc_name] + aliases
                
                # Also extract base name (remove act suffixes if present)
                base_name = doc_name
                if " - " in base_name:
                    base_name = base_name.split(" - ")[0].strip()
                elif " (" in base_name:
                    base_name = base_name.split(" (")[0].strip()
                
                semantic_keys.append(base_name)
                
                # Group by each semantic key
                for key in semantic_keys:
                    if key not in semantic_groups:
                        semantic_groups[key] = []
                    if doc_id not in semantic_groups[key]:
                        semantic_groups[key].append(doc_id)
        
        # Expand selected IDs to include all semantically related types
        expanded_ids = set(selected_ids)
        
        for selected_id in selected_ids:
            if selected_id in id_to_doc_type:
                selected_doc_type = id_to_doc_type[selected_id]
                selected_name = selected_doc_type["document_type_name"]
                selected_aliases = selected_doc_type.get("aliases", [])
                
                # Find all semantic keys for this selected document type
                semantic_keys = [selected_name.lower()] + [alias.lower() for alias in selected_aliases]
                
                # Extract base name
                base_name = selected_name.lower()
                if " - " in base_name:
                    base_name = base_name.split(" - ")[0].strip()
                elif " (" in base_name:
                    base_name = base_name.split(" (")[0].strip()
                semantic_keys.append(base_name)
                
                # Find all related document types by semantic similarity
                related_count = 0
                for key in semantic_keys:
                    if key in semantic_groups:
                        for related_id in semantic_groups[key]:
                            if related_id not in expanded_ids:
                                expanded_ids.add(related_id)
                                related_count += 1
                
                if related_count > 0:
                    logger.info(f"🔍 DOC TYPE EXPANSION: Expanded '{selected_name}' to include {related_count} related types")
        
        expanded_list = list(expanded_ids)
        if len(expanded_list) > len(selected_ids):
            logger.info(f"🔍 DOC TYPE EXPANSION: Expanded from {len(selected_ids)} to {len(expanded_list)} document types")
        
        return expanded_list

    def _enhance_step_parameters(self, original_parameters: Dict[str, Any], context: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Enhance step parameters based on execution context and discovered data.
        
        Args:
            original_parameters: The original parameters from the execution plan
            context: Execution context with discovered IDs and mappings
            tool_name: Name of the tool being executed
            
        Returns:
            Enhanced parameters with discovered IDs where appropriate
        """
        enhanced = original_parameters.copy()
        
        if tool_name == "search":
            # LLM-driven parameter selection when context is available
            query = enhanced.get("query", "").lower()
            original_query = context.get("original_query", query)
            
            # If we have available projects/document types and the LLM used placeholders,
            # let the LLM make intelligent selections from the available options
            # Only trigger LLM selection if the original planning included placeholders
            # Don't trigger if the parameter wasn't included at all (means LLM didn't think it was needed)
            
            if (context.get("available_projects_context") and 
                "project_ids" in enhanced and
                any(isinstance(pid, str) and "obtained" in str(pid).lower() for pid in enhanced.get("project_ids", []))):
                
                enhanced = self._llm_select_parameters(enhanced, context, "projects")
            
            if (context.get("available_document_types_context") and 
                "document_type_ids" in enhanced and
                any(isinstance(dtid, str) and "obtained" in str(dtid).lower() for dtid in enhanced.get("document_type_ids", []))):
                
                enhanced = self._llm_select_parameters(enhanced, context, "document_types")
            
            # Project IDs are now handled via LLM selection or placeholder replacement
            
            # Automatically add discovered document type IDs if query mentions document types we have mappings for
            for doc_type_name, doc_type_id in context["document_type_name_to_id_mapping"].items():
                if doc_type_name in query and not enhanced.get("document_type_ids"):
                    enhanced["document_type_ids"] = [doc_type_id]
                    logger.info(f"🤖 CONTEXT: Auto-added document_type_id {doc_type_id} for query mentioning '{doc_type_name}'")
            
            # If query mentions SPECIFIC document types and we have those types, use them
            if not enhanced.get("document_type_ids") and context["discovered_document_type_ids"]:
                # Only auto-add document type IDs for specific document type requests
                specific_doc_terms = ["letter", "correspondence", "memo", "report", "assessment", "transcript"]
                if any(term in query for term in specific_doc_terms):
                    enhanced["document_type_ids"] = context["discovered_document_type_ids"]
                    logger.info(f"🤖 CONTEXT: Auto-added discovered document_type_ids for specific document type: {enhanced['document_type_ids']}")
                else:
                    logger.info(f"🤖 CONTEXT: Skipping auto-document type IDs - query seems to be asking for broad search across document types")
            
            # Project selection is now handled entirely by LLM via placeholder replacement
            # All available projects are provided in discovered_project_ids for LLM to choose from
                    
        elif tool_name == "validate_chunks_relevance":
            # Provide original query if not specified
            if "query" not in enhanced or not enhanced["query"]:
                enhanced["query"] = context["original_query"]
                logger.info(f"🤖 CONTEXT: Added original query to validation step: {enhanced['query']}")
            
            # Handle reference to search results from previous steps
            search_results_ref = enhanced.get("search_results")
            if search_results_ref and isinstance(search_results_ref, str):
                # The reference should be like "results_from_search_nooaitch_letters"
                # We need to find the actual results in the context
                if search_results_ref in context.get("step_results", {}):
                    search_results = context["step_results"][search_results_ref]
                    if search_results and search_results.get("success") and search_results.get("result"):
                        # Extract the documents/chunks from the search result
                        result_data = search_results["result"]
                        
                        # Handle search results which come as tuple (documents, chunks, api_response)
                        if isinstance(result_data, tuple) and len(result_data) >= 2:
                            documents = result_data[0] if result_data[0] else []
                            chunks = result_data[1] if result_data[1] else []
                            # Combine documents and chunks for validation
                            combined_results = []
                            if isinstance(documents, list):
                                combined_results.extend(documents)
                            if isinstance(chunks, list):
                                combined_results.extend(chunks)
                            enhanced["search_results"] = combined_results
                            enhanced["_resolved_search_results"] = combined_results
                            logger.info(f"🔍 VALIDATION: Resolved '{search_results_ref}' → {len(documents)} docs + {len(chunks)} chunks = {len(combined_results)} total for validation")
                        elif isinstance(result_data, dict) and "documents" in result_data:
                            enhanced["search_results"] = result_data["documents"]
                            enhanced["_resolved_search_results"] = result_data["documents"]
                            logger.info(f"🔍 VALIDATION: Resolved '{search_results_ref}' → {len(result_data['documents'])} documents for validation")
                        elif isinstance(result_data, list):
                            # If it's already a list of documents
                            enhanced["search_results"] = result_data
                            enhanced["_resolved_search_results"] = result_data
                            logger.info(f"🔍 VALIDATION: Resolved '{search_results_ref}' → {len(result_data)} items for validation")
                        else:
                            logger.warning(f"🔍 VALIDATION: Could not extract documents from '{search_results_ref}' - unexpected format: {type(result_data)}")
                            enhanced["search_results"] = []
                            enhanced["_resolved_search_results"] = []
                    else:
                        logger.warning(f"🔍 VALIDATION: Referenced search result '{search_results_ref}' was not successful or has no result")
                        enhanced["search_results"] = []
                        enhanced["_resolved_search_results"] = []
                else:
                    logger.warning(f"🔍 VALIDATION: Search results reference '{search_results_ref}' not found in context")
                    enhanced["search_results"] = []
                    enhanced["_resolved_search_results"] = []
                    
        elif tool_name == "verify_reduce":
            # For verify_reduce, keep parameters clean - don't add massive execution context
            # Context is passed separately to avoid polluting logged parameters
            # This ensures clean JSON logging while still providing tool access to step results
            if context is not None:
                step_results_count = len(context.get("step_results", {}) or {})
                logger.info(f"🔗 VERIFY REDUCE: Execution context available with {step_results_count} step results")
            
        elif tool_name == "summarize_results":
            # Provide original query if not specified
            if "query" not in enhanced or not enhanced["query"]:
                enhanced["query"] = context["original_query"]
                logger.info(f"🤖 CONTEXT: Added original query to summarization step: {enhanced['query']}")
                
        elif tool_name == "validate_query_relevance":
            # Provide original query if not specified
            if "query" not in enhanced or not enhanced["query"]:
                enhanced["query"] = context["original_query"]
                logger.info(f"🤖 CONTEXT: Added original query to validation step: {enhanced['query']}")
            
        if tool_name == "search":
            # Replace placeholder project_ids with discovered ones or resolve from original query
            if "project_ids" in enhanced and isinstance(enhanced["project_ids"], list):
                placeholder_patterns = ["obtained", "project_id_for_", "_id", "previous_step", "discovered", "from_"]
                has_placeholders = any(
                    any(pattern in str(pid).lower() for pattern in placeholder_patterns)
                    for pid in enhanced["project_ids"]
                )
                if has_placeholders and context["discovered_project_ids"]:
                    enhanced["project_ids"] = context["discovered_project_ids"]
                    logger.info(f"🤖 CONTEXT: Replaced placeholder project_ids with discovered: {enhanced['project_ids']}")
                elif has_placeholders:
                    logger.warning(f"🤖 CONTEXT: Found placeholders but no discovered project IDs available")
            
            # Convert project names to IDs if LLM provided names instead of IDs
            if "project_ids" in enhanced and isinstance(enhanced["project_ids"], list):
                converted_ids = []
                needs_conversion = False
                
                for pid in enhanced["project_ids"]:
                    pid_str = str(pid).lower().strip()
                    # Check if this looks like a name rather than an ID (IDs are typically long hex strings)
                    if len(pid_str) < 20 and not pid_str.startswith(('6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f')):
                        # This looks like a name, try to convert it
                        if pid_str in context["project_name_to_id_mapping"]:
                            converted_id = context["project_name_to_id_mapping"][pid_str]
                            converted_ids.append(converted_id)
                            logger.info(f"🤖 CONTEXT: Converted project name '{pid}' to ID: {converted_id}")
                            needs_conversion = True
                        else:
                            # Check partial matches (e.g., "site c" matches "Site C Clean Energy Project")
                            partial_match = None
                            for name, proj_id in context["project_name_to_id_mapping"].items():
                                if pid_str in name or name in pid_str:
                                    partial_match = proj_id
                                    logger.info(f"🤖 CONTEXT: Partial match - converted '{pid}' to ID: {proj_id} (matched with '{name}')")
                                    break
                            
                            if partial_match:
                                converted_ids.append(partial_match)
                                needs_conversion = True
                            else:
                                logger.warning(f"🤖 CONTEXT: Could not convert project name '{pid}' to ID - no mapping found")
                                converted_ids.append(pid)  # Keep original if no conversion found
                    else:
                        # This already looks like an ID, keep it
                        converted_ids.append(pid)
                
                if needs_conversion:
                    enhanced["project_ids"] = converted_ids
                    logger.info(f"🤖 CONTEXT: Final project_ids after conversion: {enhanced['project_ids']}")
            
            # Replace placeholder document_type_ids with resolved ones from original query
            if "document_type_ids" in enhanced and isinstance(enhanced["document_type_ids"], list):
                placeholder_patterns = ["obtained", "type_id", "_id", "previous_step", "discovered", "from_"]
                has_placeholders = any(
                    any(pattern in str(dtid).lower() for pattern in placeholder_patterns)
                    for dtid in enhanced["document_type_ids"]
                )
                if has_placeholders:
                    logger.warning(f"🤖 CONTEXT: Found placeholders in document_type_ids but no auto-discovery - LLM should select from available document types")
            
            # Convert document type names to IDs if LLM provided names instead of IDs
            if "document_type_ids" in enhanced and isinstance(enhanced["document_type_ids"], list):
                converted_ids = []
                needs_conversion = False
                
                for dtid in enhanced["document_type_ids"]:
                    dtid_str = str(dtid).lower().strip()
                    # Check if this looks like a name rather than an ID (IDs are typically long hex strings)
                    if len(dtid_str) < 20 and not dtid_str.startswith(('6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f')):
                        # This looks like a name, try to convert it
                        if dtid_str in context["document_type_name_to_id_mapping"]:
                            converted_id = context["document_type_name_to_id_mapping"][dtid_str]
                            converted_ids.append(converted_id)
                            logger.info(f"🤖 CONTEXT: Converted document type name '{dtid}' to ID: {converted_id}")
                            needs_conversion = True
                        else:
                            # Check partial matches (e.g., "letter" matches "correspondence letter")
                            partial_match = None
                            for name, doc_id in context["document_type_name_to_id_mapping"].items():
                                if dtid_str in name or name in dtid_str:
                                    partial_match = doc_id
                                    logger.info(f"🤖 CONTEXT: Partial match - converted '{dtid}' to ID: {doc_id} (matched with '{name}')")
                                    break
                            
                            if partial_match:
                                converted_ids.append(partial_match)
                                needs_conversion = True
                            else:
                                logger.warning(f"🤖 CONTEXT: Could not convert document type name '{dtid}' to ID - no mapping found")
                                converted_ids.append(dtid)  # Keep original if no conversion found
                    else:
                        # This already looks like an ID, keep it
                        converted_ids.append(dtid)
                
                if needs_conversion:
                    enhanced["document_type_ids"] = converted_ids
                    logger.info(f"🤖 CONTEXT: Final document_type_ids after conversion: {enhanced['document_type_ids']}")
            
            # Auto-enhance with structured parameters if LLM didn't include them
            if not enhanced.get("location") and not enhanced.get("project_status") and not enhanced.get("years"):
                # Extract structured parameters from the original query
                original_query = context.get("original_query", enhanced.get("query", ""))
                extracted_params = self._extract_search_parameters_from_query(original_query)
                
                for param_name, param_value in extracted_params.items():
                    if param_name not in enhanced:
                        enhanced[param_name] = param_value
                        logger.info(f"🤖 CONTEXT: Auto-added {param_name}: {param_value}")
            
            # Prioritize user-provided parameters (always take precedence)
            if self.user_project_ids:
                enhanced["project_ids"] = self.user_project_ids
                logger.info(f"🤖 CONTEXT: Using user-provided project_ids: {enhanced['project_ids']}")
            
            if self.user_document_type_ids:
                enhanced["document_type_ids"] = self.user_document_type_ids
                logger.info(f"🤖 CONTEXT: Using user-provided document_type_ids: {enhanced['document_type_ids']}")
            
            if self.user_search_strategy:
                enhanced["search_strategy"] = self.user_search_strategy
                logger.info(f"🤖 CONTEXT: Using user-provided search_strategy: {enhanced['search_strategy']}")
            
            if self.user_ranking:
                enhanced["ranking"] = self.user_ranking
                logger.info(f"🤖 CONTEXT: Using user-provided ranking: {enhanced['ranking']}")
            
            # Prioritize user-provided new parameters
            if self.location:
                enhanced["location"] = self.location
                logger.info(f"🤖 CONTEXT: Using user-provided location: {enhanced['location']}")
            
            if self.user_project_status:
                enhanced["project_status"] = self.user_project_status
                logger.info(f"🤖 CONTEXT: Using user-provided project_status: {enhanced['project_status']}")
            
            if self.user_years:
                enhanced["years"] = self.user_years
                logger.info(f"🤖 CONTEXT: Using user-provided years: {enhanced['years']}")
        
        return enhanced

    def _deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate documents/chunks by creating unique identifiers.
        
        Args:
            documents: List of documents or document chunks to deduplicate
            
        Returns:
            List of unique documents sorted by relevance score
        """
        if not documents:
            return []
            
        document_map = {}
        
        def create_unique_id(doc):
            """Create a unique identifier for document or document chunk."""
            doc_id = doc.get("document_id", "")
            page_num = doc.get("page_number", "")
            content = doc.get("content", "")
            
            # For document chunks, use document_id + page_number + content hash for uniqueness
            # For documents, use just document_id
            if page_num or content:
                # This is likely a document chunk - use more specific identifier
                content_hash = str(hash(content[:100])) if content else ""
                return f"{doc_id}_{page_num}_{content_hash}"
            else:
                # This is likely a document - use document_id
                return doc_id
        
        # Add documents to map, keeping higher relevance scores
        for doc in documents:
            unique_id = create_unique_id(doc)
            if not unique_id:
                continue
                
            existing_doc = document_map.get(unique_id)
            
            if existing_doc:
                # Keep document with higher relevance score
                existing_score = existing_doc.get("relevance_score", 0)
                new_score = doc.get("relevance_score", 0)
                
                if new_score > existing_score:
                    document_map[unique_id] = doc
            else:
                # Add new document
                document_map[unique_id] = doc
        
        # Convert back to list and sort by relevance score (descending)
        deduplicated_documents = list(document_map.values())
        deduplicated_documents.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return deduplicated_documents
    
    def _update_execution_context(self, context: Dict[str, Any], tool_name: str, tool_result: Dict[str, Any]) -> None:
        """Update execution context with results from the current step.
        
        Args:
            context: Execution context to update
            tool_name: Name of the tool that was executed
            tool_result: Result from the tool execution
        """
        if not tool_result.get("success", False):
            return
            
        result_data = tool_result.get("result")
        
        if tool_name == "get_projects_list" and isinstance(result_data, list):
            # Extract project mappings for LLM to reference and select from
            for project in result_data:
                if isinstance(project, dict) and "project_id" in project and "project_name" in project:
                    project_id = project["project_id"]
                    project_name = project["project_name"]
                    context["project_name_to_id_mapping"][project_name.lower()] = project_id
                    logger.info(f"🤖 CONTEXT: Added project mapping: {project_name} -> {project_id}")
            
            # Store the full project list for LLM context
            context["available_projects"] = result_data
            
            # Format projects context for LLM visibility - SHOW ALL projects
            projects_context_lines = []
            for project in result_data:  # Show ALL projects
                if isinstance(project, dict) and "project_id" in project and "project_name" in project:
                    projects_context_lines.append(f"  - {project['project_name']}: {project['project_id']}")
            
            context["available_projects_context"] = f"""
AVAILABLE PROJECTS (select relevant project_ids):
{chr(10).join(projects_context_lines)}

Example usage: "project_ids": ["{result_data[0]['project_id'] if result_data else 'project_id_here'}"]
"""
            logger.info(f"🤖 CONTEXT: {len(result_data)} projects available for LLM selection")
        
        elif tool_name == "get_document_types" and isinstance(result_data, list):
            # Extract document type mappings for LLM to reference and select from
            for doc_type in result_data:
                if isinstance(doc_type, dict) and "document_type_id" in doc_type and "document_type_name" in doc_type:
                    doc_type_id = doc_type["document_type_id"]
                    doc_type_name = doc_type["document_type_name"]
                    context["document_type_name_to_id_mapping"][doc_type_name.lower()] = doc_type_id
                    logger.info(f"🤖 CONTEXT: Added document type mapping: {doc_type_name} -> {doc_type_id}")
            
            # Store the full document type list for LLM context
            context["available_document_types"] = result_data
            
            # Format document types context for LLM visibility - SHOW ALL types
            doc_types_context_lines = []
            for doc_type in result_data:  # Show ALL document types
                if isinstance(doc_type, dict) and "document_type_id" in doc_type and "document_type_name" in doc_type:
                    doc_name = doc_type['document_type_name']
                    doc_id = doc_type['document_type_id']
                    
                    # Include aliases if available
                    aliases = doc_type.get('aliases', [])
                    if aliases:
                        aliases_str = f" (aliases: {', '.join(aliases)})"
                    else:
                        aliases_str = ""
                    
                    # Include act if available for context
                    act = doc_type.get('act', '')
                    if act and act != '2018_act_terms':  # Don't show generic act
                        act_str = f" [{act}]"
                    else:
                        act_str = ""
                    
                    doc_types_context_lines.append(f"  - {doc_name}{aliases_str}{act_str}: {doc_id}")
            
            context["available_document_types_context"] = f"""
AVAILABLE DOCUMENT TYPES (select relevant document_type_ids):
{chr(10).join(doc_types_context_lines)}

Example usage: "document_type_ids": ["{result_data[0]['document_type_id'] if result_data else 'document_type_id_here'}"]
"""
            logger.info(f"🤖 CONTEXT: {len(result_data)} document types available for LLM selection")
        
        elif tool_name == "search" and isinstance(result_data, tuple) and len(result_data) >= 2:
            # Track search results for consolidation
            context["search_results"]["search_executions"] += 1
            
            # Extract documents and chunks from search results - now properly separated
            documents = result_data[0] if len(result_data) > 0 else []
            chunks = result_data[1] if len(result_data) > 1 else []
            # result_data[2] is the full API response (not needed for consolidation)
            
            if isinstance(documents, list):
                context["search_results"]["documents"].extend(documents)
                logger.info(f"🤖 CONTEXT: Added {len(documents)} documents from search execution {context['search_results']['search_executions']}")
                
                # Extract project IDs from search results if we don't have any yet
                if not context["discovered_project_ids"]:
                    for doc in documents:
                        if isinstance(doc, dict) and "project_id" in doc:
                            project_id = doc["project_id"]
                            if project_id not in context["discovered_project_ids"]:
                                context["discovered_project_ids"].append(project_id)
                                logger.info(f"🤖 CONTEXT: Discovered project_id from search: {project_id}")
            
            if isinstance(chunks, list):
                context["search_results"]["document_chunks"].extend(chunks)
                logger.info(f"🤖 CONTEXT: Added {len(chunks)} document chunks from search execution {context['search_results']['search_executions']}")
        
        elif tool_name == "validate_chunks_relevance":
            # Track filtered results from validation
            if "filtered_results" not in context:
                context["filtered_results"] = {"documents": [], "document_chunks": []}
            
            filtered_results = result_data.get("filtered_chunks", []) if isinstance(result_data, dict) else []
            
            if isinstance(filtered_results, list):
                # Determine if these are documents or chunks based on structure
                documents = []
                chunks = []
                
                for item in filtered_results:
                    if isinstance(item, dict):
                        # If it has content/page_number, it's likely a chunk
                        if "content" in item or "page_number" in item:
                            chunks.append(item)
                        else:
                            documents.append(item)
                
                context["filtered_results"]["documents"].extend(documents)
                context["filtered_results"]["document_chunks"].extend(chunks)
                
                logger.info(f"🔍 VALIDATION: Added {len(documents)} filtered documents + {len(chunks)} filtered chunks from validation")
        
        elif tool_name == "consolidate_results":
            # Perform consolidation - prioritize filtered results if available
            logger.info("🔗 AGENT CONSOLIDATION: Starting consolidation of search results...")
            
            # Use filtered results if available (from validation steps), otherwise use raw search results
            if "filtered_results" in context and (context["filtered_results"]["documents"] or context["filtered_results"]["document_chunks"]):
                all_documents = context["filtered_results"]["documents"]
                all_chunks = context["filtered_results"]["document_chunks"]
                logger.info(f"🔍 AGENT CONSOLIDATION: Using filtered results ({len(all_documents)} documents + {len(all_chunks)} chunks)")
            else:
                all_documents = context["search_results"]["documents"]
                all_chunks = context["search_results"]["document_chunks"]
                logger.info(f"🔍 AGENT CONSOLIDATION: Using raw search results ({len(all_documents)} documents + {len(all_chunks)} chunks) - no filtered results available")
            
            # Deduplicate documents and chunks
            consolidated_documents = self._deduplicate_documents(all_documents)
            consolidated_chunks = self._deduplicate_documents(all_chunks)
            
            context["consolidated_results"] = {
                "documents": consolidated_documents,
                "document_chunks": consolidated_chunks,
                "total_documents": len(consolidated_documents),
                "total_chunks": len(consolidated_chunks),
                "original_documents": len(all_documents),
                "original_chunks": len(all_chunks),
                "search_executions": context["search_results"]["search_executions"]
            }
            
            logger.info(f"🔗 AGENT CONSOLIDATION: {len(all_documents)} documents + {len(all_chunks)} chunks → {len(consolidated_documents)} unique documents + {len(consolidated_chunks)} unique chunks")
        
        elif tool_name == "summarize_results":
            # Perform summarization of consolidated results
            if not context["consolidated_results"]:
                logger.error("🤖 AGENT SUMMARY: No consolidated results available for summarization")
                return
            
            logger.info("📝 AGENT SUMMARY: Starting summarization of consolidated results...")
            
            try:
                from search_api.services.generation.factories import SummarizerFactory
                
                summarizer = SummarizerFactory.create_summarizer()
                
                # Combine documents and chunks for summarization
                all_results = context["consolidated_results"]["documents"] + context["consolidated_results"]["document_chunks"]
                
                if all_results:
                    summary_result = summarizer.summarize_search_results(
                        query=context["original_query"],
                        documents_or_chunks=all_results,
                        search_context={
                            "context": "Agent consolidation summary",
                            "search_strategy": "agent_multi_search", 
                            "total_documents": context["consolidated_results"]["total_documents"],
                            "total_chunks": context["consolidated_results"]["total_chunks"],
                            "search_executions": context["consolidated_results"]["search_executions"]
                        }
                    )
                    
                    context["summary_result"] = summary_result
                    logger.info(f"📝 AGENT SUMMARY: Generated summary using {summary_result.get('provider', 'unknown')} with confidence {summary_result.get('confidence', 0)}")
                else:
                    logger.warning("📝 AGENT SUMMARY: No results to summarize")
                    context["summary_result"] = {
                        "summary": "No relevant documents were found for the given query.",
                        "method": "empty_fallback",
                        "confidence": 0.0,
                        "documents_count": 0,
                        "provider": "agent_stub",
                        "model": "fallback"
                    }
                    
            except Exception as e:
                logger.error(f"📝 AGENT SUMMARY: Summarization failed: {e}")
                context["summary_result"] = {
                    "summary": "Summary generation failed due to an error.",
                    "method": "error_fallback", 
                    "confidence": 0.0,
                    "documents_count": len(all_results) if 'all_results' in locals() else 0,
                    "provider": "agent_stub",
                    "model": "fallback",
                    "error": str(e)
                }
    
    def _collect_verified_chunks(self, filter_steps: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and combine all validated chunks from filter steps.
        
        Args:
            filter_steps: List of filter step names to collect chunks from
            parameters: Clean parameters dict with execution context passed separately
            
        Returns:
            Dict with combined verified chunks, chunk names for logging, and verification metadata
        """
        # Defensive check for parameters
        if parameters is None:
            logger.error("🔗 VERIFY REDUCE: Parameters is None")
            return {"error": "Parameters is None", "success": False}
            
        # Get execution context from parameters if available
        context = parameters.get("_execution_context", {})
        if context is None:
            logger.error("🔗 VERIFY REDUCE: Execution context is None")
            return {"error": "Execution context is None", "success": False}
            
        step_results = context.get("step_results", {})
        
        all_verified_chunks = []
        validation_summaries = []
        total_original = 0
        total_kept = 0
        
        logger.info(f"🔗 VERIFY REDUCE: Collecting verified chunks from {len(filter_steps)} filter steps")
        
        # Find all validation/filter steps in step_results (more flexible matching)
        available_filter_steps = []
        for step_name in step_results.keys():
            if any(filter_step in step_name for filter_step in filter_steps) or step_name in filter_steps:
                available_filter_steps.append(step_name)
        
        if not available_filter_steps:
            # If no exact matches, try pattern matching for validation steps
            for step_name in step_results.keys():
                if "filter" in step_name.lower() or "validate" in step_name.lower():
                    available_filter_steps.append(step_name)
        
        chunks_removed = []
        chunks_kept = []
        
        for step_name in available_filter_steps:
            step_result = step_results[step_name]
            if step_result is None:
                continue
            if step_result.get("success") and step_result.get("result"):
                result_data = step_result["result"]
                
                # Extract validated chunks and validation info
                if isinstance(result_data, tuple) and len(result_data) >= 2:
                    validated_chunks, validation_info = result_data
                    if isinstance(validated_chunks, list):
                        # Track which chunks were kept by their document names
                        for chunk in validated_chunks:
                            chunk_name = chunk.get("document_name", chunk.get("document_display_name", "Unknown"))
                            if chunk_name not in chunks_kept:
                                chunks_kept.append(chunk_name)
                        
                        all_verified_chunks.extend(validated_chunks)
                        total_kept += len(validated_chunks)
                        
                    if isinstance(validation_info, dict):
                        validation_summaries.append(validation_info)
                        metrics = validation_info.get("validation_metrics", {})
                        total_original += metrics.get("total_received", 0)
                        
                        # Track removed chunks if available
                        if "removed_chunks" in validation_info:
                            for chunk in validation_info["removed_chunks"]:
                                chunk_name = chunk.get("document_name", chunk.get("document_display_name", "Unknown"))
                                if chunk_name not in chunks_removed:
                                    chunks_removed.append(chunk_name)
        
        # Log chunk changes concisely
        if chunks_kept:
            logger.info(f"🔗 CHUNKS KEPT ({len(chunks_kept)}): {', '.join(chunks_kept[:5])}{' ...' if len(chunks_kept) > 5 else ''}")
        if chunks_removed:
            logger.info(f"🔗 CHUNKS REMOVED ({len(chunks_removed)}): {', '.join(chunks_removed[:5])}{' ...' if len(chunks_removed) > 5 else ''}")
        elif available_filter_steps:
            logger.info(f"🔗 CHUNKS REMOVED: None (all chunks passed validation)")
        
        # Create summary of verification process
        verification_summary = {
            "total_filter_steps": len(available_filter_steps),
            "total_original_chunks": total_original,
            "total_verified_chunks": len(all_verified_chunks),
            "verification_rate": len(all_verified_chunks) / total_original if total_original > 0 else 0,
            "step_summaries": validation_summaries,
            "chunks_kept": chunks_kept,
            "chunks_removed": chunks_removed
        }
        
        logger.info(f"🔗 VERIFY REDUCE: Final result - {len(all_verified_chunks)} chunks verified for consolidation")
        
        return {
            "verified_chunks": all_verified_chunks,
            "verification_summary": verification_summary,
            "success": True
        }


def _filter_projects_by_decision_year(available_projects: list, query: str):
    """Return project IDs whose decisionDate falls within the temporal range stated in the query.

    Handles patterns:
      "after YYYY" / "since YYYY"           → year_from = YYYY
      "before YYYY"                          → year_to   = YYYY
      "between YYYY and YYYY"                → year_from, year_to
      "from YYYY to YYYY"                    → year_from, year_to
      "approved/certified/issued in YYYY"    → exact year
      "last N years" / "past N years"        → year_from = current_year - N
      "recent" / "recently" / "last year"   → year_from = current_year - 3

    Returns None  → no temporal pattern detected; caller should NOT filter.
    Returns list  → project IDs matching the range (empty list treated as None by caller).
    """
    import re
    from datetime import datetime

    current_year = datetime.now().year
    q = query.lower()
    year_from = None
    year_to = None

    # ── Explicit year patterns ────────────────────────────────────────────────
    m = re.search(r'\b(?:after|since)\s+(\d{4})\b', q)
    if m:
        year_from = int(m.group(1))

    m = re.search(r'\bbefore\s+(\d{4})\b', q)
    if m:
        year_to = int(m.group(1))

    m = re.search(r'\bbetween\s+(\d{4})\s+and\s+(\d{4})\b', q)
    if m:
        year_from, year_to = int(m.group(1)), int(m.group(2))

    m = re.search(r'\bfrom\s+(\d{4})\s+to\s+(\d{4})\b', q)
    if m:
        year_from, year_to = int(m.group(1)), int(m.group(2))

    if year_from is None and year_to is None:
        m = re.search(r'\b(?:approved|certified|certificate|issued|completed)\s+in\s+(\d{4})\b', q)
        if m:
            year_from = year_to = int(m.group(1))

    # ── Relative time patterns ────────────────────────────────────────────────
    if year_from is None and year_to is None:
        m = re.search(r'\b(?:last|past)\s+(\d+)\s+years?\b', q)
        if m:
            year_from = current_year - int(m.group(1))

    if year_from is None and year_to is None:
        if re.search(r'\b(?:recent|recently|latest|last\s+year|past\s+year)\b', q):
            year_from = current_year - 3

    if year_from is None and year_to is None:
        logger.info("🗓️ DECISION DATE FILTER: no temporal pattern detected — skipping")
        return None

    logger.info(f"🗓️ DECISION DATE FILTER: detected year_from={year_from} year_to={year_to} from query")

    # ── Diagnostic: sample the first project to verify metadata structure ─────
    if available_projects:
        sample = available_projects[0]
        sample_meta = sample.get('project_metadata') or {}
        sample_keys = list(sample_meta.keys())[:10] if isinstance(sample_meta, dict) else type(sample_meta).__name__
        projects_with_date = sum(
            1 for p in available_projects
            if isinstance(p.get('project_metadata'), dict)
            and (p['project_metadata'].get('decisionDate') or p['project_metadata'].get('decision_date'))
        )
        logger.info(
            f"🗓️ DECISION DATE FILTER: {len(available_projects)} total projects, "
            f"{projects_with_date} have decisionDate. "
            f"Sample metadata keys: {sample_keys}"
        )

    # ── Filter ────────────────────────────────────────────────────────────────
    matched_ids = []
    for proj in (available_projects or []):
        if not isinstance(proj, dict):
            continue
        meta = proj.get('project_metadata') or {}
        if not isinstance(meta, dict):
            continue
        raw_date = meta.get('decisionDate') or meta.get('decision_date') or ''
        if not raw_date:
            continue
        try:
            decision_year = int(str(raw_date)[:4])
        except (ValueError, TypeError):
            continue
        if year_from is not None and decision_year < year_from:
            continue
        if year_to is not None and decision_year > year_to:
            continue
        pid = proj.get('project_id')
        if pid:
            matched_ids.append(pid)

    logger.info(
        f"🗓️ DECISION DATE FILTER: {len(matched_ids)} projects match "
        f"year_from={year_from} year_to={year_to} → "
        f"{matched_ids[:5]}{'...' if len(matched_ids) > 5 else ''}"
    )
    return matched_ids if matched_ids else None


def _strip_temporal_phrase(semantic_query: str) -> str:
    """Remove project-approval temporal phrases from the semantic query.

    When _filter_projects_by_decision_year scopes the search to approved projects,
    the semantic query should focus on document content, not the approval date.
    Leaving "after 2010" in the query degrades vector match quality.
    """
    import re
    q = semantic_query

    patterns = [
        r'\b(?:approved|certified|certificate|issued|completed)\s+(?:after|since|before|in)\s+\d{4}\b',
        r'\b(?:after|since)\s+\d{4}\b',
        r'\bbefore\s+\d{4}\b',
        r'\bbetween\s+\d{4}\s+and\s+\d{4}\b',
        r'\bfrom\s+\d{4}\s+to\s+\d{4}\b',
        r'\b(?:last|past)\s+\d+\s+years?\b',
        r'\b(?:recent|recently|latest)\b',
    ]
    for pat in patterns:
        q = re.sub(pat, '', q, flags=re.IGNORECASE)

    q = re.sub(r'\s{2,}', ' ', q).strip()
    return q if q else semantic_query


def handle_agent_query(query: str, reason: str, llm_client=None, user_location: Optional[Dict[str, Any]] = None,
                      project_ids: Optional[List[str]] = None, document_type_ids: Optional[List[str]] = None, 
                      search_strategy: Optional[str] = None, ranking: Optional[Dict[str, Any]] = None,
                      project_status: Optional[str] = None, 
                      years: Optional[List[int]] = None) -> dict:
    """Handle agent-required queries with simplified parameter extraction flow.
    
    Args:
        query: The complex query that requires agent processing
        reason: Why the query was classified as agent-required
        llm_client: Optional LLM client for intelligent planning
        user_location: Optional user location data from request body
        project_ids: Optional user-provided project IDs 
        document_type_ids: Optional user-provided document type IDs
        search_strategy: Optional user-provided search strategy
        ranking: Optional user-provided ranking configuration        
        project_status: Optional project status parameter
        years: Optional years parameter
        
    Returns:
        Dict with agent processing results
    """
    import time
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("🤖 AGENT MODE ACTIVATED - SIMPLIFIED FLOW")
    logger.info(f"Query: {query}")
    logger.info(f"Reason: {reason}")
    
    # Validate that LLM client is provided (required for agent mode)
    if not llm_client:
        logger.error("🤖 AGENT: LLM client is required for agent mode")
        return {
            "error": "Agent mode requires LLM client for intelligent planning",
            "agent_results": [],
            "planning_method": "Failed - No LLM client",
            "execution_time": 0,
            "steps_executed": 0,
            "consolidated_summary": "Error: LLM client not provided for agent mode"
        }
    
    logger.info("=" * 60)
    
    # Per-step timing accumulators (ms)
    _step_timings = {
        "relevance_check_ms": 0,
        "parameter_extraction_ms": 0,
        "vector_search_ms": 0,
        "consolidation_ms": 0,
        "summary_generation_ms": 0,
    }
    # Flat method-level call trace
    _trace = []

    def _t_entry(method, module, duration_ms, detail="", is_external=False):
        _trace.append({
            "method": method,
            "module": module,
            "duration_ms": round(duration_ms, 2),
            "detail": detail,
            "is_external": is_external,
        })

    try:
        # STEPS 1 + 2a/2b run in parallel:
        #   - Step 1: LLM relevance check (skippable via DISABLE_RELEVANCE_CHECK=true)
        #   - Step 2a: Fetch projects list (cached 24h after first call)
        #   - Step 2b: Fetch document types (cached 24h after first call)
        # Running them concurrently saves ~1s on warm instances (relevance check latency).

        from search_api.services.generation.factories import ParameterExtractorFactory, QueryValidatorFactory
        from search_api.clients.vector_search_client import VectorSearchClient

        disable_relevance = os.getenv("DISABLE_RELEVANCE_CHECK", "false").lower() == "true"

        # Fast heuristic: skip LLM relevance check when query is obviously EAO-related.
        # Saves 3-4s per query when the relevance check would always pass anyway.
        # Triggers if: user already filtered by project, OR query contains EAO/project/doc terms.
        if not disable_relevance and project_ids:
            disable_relevance = True
            logger.info("⚡ RELEVANCE CHECK SKIPPED: frontend supplied project_ids → obviously relevant")
        elif not disable_relevance:
            _q_lower = query.lower()
            _eao_keywords = {
                "eao", "environmental assessment", "certificate", "schedule b", "schedule a",
                "conditions", "proponent", "compliance", "amendment", "assessment report",
                "working group", "mine", "mines", "mining", "pipeline", "lng", "hydroelectric",
                "transmission line", "dam", "reservoir", "highway", "port", "terminal",
                "application", "effects assessment", "ea certificate",
            }
            if any(kw in _q_lower for kw in _eao_keywords):
                disable_relevance = True
                logger.info("⚡ RELEVANCE CHECK SKIPPED: query contains obvious EAO/project keywords")

        available_projects = {}
        available_document_types = {}
        relevance_result = {"is_relevant": True}
        _proj_ms = 0.0
        _dt_ms = 0.0

        app = current_app._get_current_object()

        def _fetch_projects():
            with app.app_context():
                _t0 = time.time()
                data = VectorSearchClient.get_projects_list(include_metadata=True)
                return data, round((time.time() - _t0) * 1000, 2)

        def _fetch_doc_types():
            with app.app_context():
                _t0 = time.time()
                data = VectorSearchClient.get_document_types()
                return data, round((time.time() - _t0) * 1000, 2)

        def _check_relevance():
            with app.app_context():
                _t0 = time.time()
                checker = QueryValidatorFactory.create_validator()
                result = checker.validate_query_relevance(query)
                return result, round((time.time() - _t0) * 1000, 2)

        tasks = {"projects": _fetch_projects, "doc_types": _fetch_doc_types}
        if not disable_relevance:
            tasks["relevance"] = _check_relevance

        logger.info(f"🔍 STEP 1+2: Running {list(tasks.keys())} in parallel (DISABLE_RELEVANCE_CHECK={disable_relevance})")

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {name: executor.submit(fn) for name, fn in tasks.items()}
            for name, future in futures.items():
                try:
                    if name == "projects":
                        available_projects, _proj_ms = future.result()
                        _t_entry(
                            "VectorSearchClient.get_projects_list(include_metadata=True)",
                            "VectorSearchClient → GET /tools/projects",
                            _proj_ms,
                            f"Fetched {len(available_projects) if available_projects else 0} projects (cached after first call)",
                            is_external=True,
                        )
                    elif name == "doc_types":
                        available_document_types, _dt_ms = future.result()
                        _t_entry(
                            "VectorSearchClient.get_document_types()",
                            "VectorSearchClient → GET /tools/document-types",
                            _dt_ms,
                            f"Fetched {len(available_document_types) if available_document_types else 0} document types (cached after first call)",
                            is_external=True,
                        )
                    elif name == "relevance":
                        relevance_result, _rel_ms = future.result()
                        _step_timings["relevance_check_ms"] = _rel_ms
                        _t_entry(
                            "validate_query_relevance(query)",
                            "QueryValidator (LLM call)",
                            _rel_ms,
                            f"Result: {'relevant ✓' if relevance_result.get('is_relevant', True) else 'not relevant — early exit'}",
                        )
                except Exception as e:
                    logger.warning(f"🔍 Parallel task '{name}' failed: {e}")

        if disable_relevance:
            logger.info("🔍 STEP 1: Relevance check SKIPPED (DISABLE_RELEVANCE_CHECK=true)")
            _t_entry("validate_query_relevance(query)", "QueryValidator", 0, "SKIPPED via DISABLE_RELEVANCE_CHECK")
        elif not relevance_result.get("is_relevant", True):
            logger.info("🔍 AGENT: Query not relevant to EAO - returning early")
            return {
                "agent_results": [],
                "planning_method": "Early exit - not relevant",
                "execution_time": round((time.time() - start_time) * 1000, 2),
                "steps_executed": 1,
                "consolidated_summary": relevance_result.get("response", "This query appears to be outside the scope of EAO's mandate."),
                "early_exit": True,
                "exit_reason": "query_not_relevant",
                "step_timings": _step_timings,
                "trace": _trace,
            }
        else:
            logger.info(f"🔍 STEP 1: Query is relevant - continuing ({_step_timings['relevance_check_ms']}ms)")

        logger.info(f"🤖 STEP 2: Got {len(available_projects) if available_projects else 0} projects, {len(available_document_types) if available_document_types else 0} document types")

        # ── Early exits for generic informational queries ─────────────────────
        # These queries don't benefit from vector search — answer them directly.
        from search_api.services.search_handlers.ai_handler import AIHandler as _AIH

        if _AIH._is_generic_eao_query(query, available_projects):
            logger.info("🔍 AGENT: Generic EAO query — returning overview response (no vector search needed)")
            return {
                "agent_results": [],
                "planning_method": "Early exit - generic EAO informational query",
                "execution_time": round((time.time() - start_time) * 1000, 2),
                "steps_executed": 1,
                "consolidated_summary": _AIH._get_default_eao_response(),
                "early_exit": True,
                "exit_reason": "generic_informational",
                "step_timings": _step_timings,
                "trace": _trace,
            }

        if _AIH._is_project_count_query(query) and available_projects:
            logger.info("🔍 AGENT: Project count query — returning metadata response (no vector search needed)")
            return {
                "agent_results": [],
                "planning_method": "Early exit - project count query",
                "execution_time": round((time.time() - start_time) * 1000, 2),
                "steps_executed": 1,
                "consolidated_summary": _AIH._generate_project_count_response(available_projects),
                "early_exit": True,
                "exit_reason": "project_count_query",
                "step_timings": _step_timings,
                "trace": _trace,
            }

        # Typed count: "how many mines", "how many lng projects", etc.
        _typed_count_type = _AIH._is_typed_project_count_query(query)
        if _typed_count_type and available_projects:
            logger.info(f"🔍 AGENT: Typed count query (type='{_typed_count_type}') — returning metadata response")
            return {
                "agent_results": [],
                "planning_method": f"Early exit - typed count query ({_typed_count_type})",
                "execution_time": round((time.time() - start_time) * 1000, 2),
                "steps_executed": 1,
                "consolidated_summary": _AIH._generate_typed_count_response(available_projects, _typed_count_type),
                "early_exit": True,
                "exit_reason": "typed_count_query",
                "step_timings": _step_timings,
                "trace": _trace,
            }

        # Metadata fact: "who is the proponent for X?", "what phase is X in?", etc.
        _fact_type = _AIH._is_metadata_fact_query(query)
        if _fact_type and available_projects:
            _fact_answer = _AIH._answer_metadata_fact(query, _fact_type, available_projects)
            if _fact_answer:
                logger.info(f"🔍 AGENT: Metadata fact query (type='{_fact_type}') — answering from project metadata")
                return {
                    "agent_results": [],
                    "planning_method": f"Early exit - metadata fact query ({_fact_type})",
                    "execution_time": round((time.time() - start_time) * 1000, 2),
                    "steps_executed": 1,
                    "consolidated_summary": _fact_answer,
                    "early_exit": True,
                    "exit_reason": "metadata_fact_query",
                    "step_timings": _step_timings,
                    "trace": _trace,
                }

        # ── Early project detection (before LLM) ──────────────────────────────
        # Name-match and metadata-match reduce LLM extraction to a simple lookup
        # for queries like "Site C conditions" or "mines in northern BC".
        _early_project_ids = None
        if not project_ids and available_projects:
            _name_id, _name_score = _AIH._match_project_by_query_text(query, available_projects, return_score=True)
            if _name_id and _name_score >= 0.5:
                # Any decent name match beats metadata: avoids metadata over-matching
                # (e.g. "Site C dam" scoring 0.6 on name but matching water/wells projects on metadata)
                _early_project_ids = [_name_id]
                logger.info(f"🚀 AGENT EARLY DETECT: Name match '{_name_id}' (score={_name_score:.2f})")
            else:
                _meta_ids = _AIH._match_projects_by_metadata(query, available_projects)
                if _meta_ids:
                    _early_project_ids = _meta_ids
                    logger.info(f"🚀 AGENT EARLY DETECT: Metadata match → {len(_meta_ids)} projects")
                elif _name_id:
                    _early_project_ids = [_name_id]
                    logger.info(f"🚀 AGENT EARLY DETECT: Weak name match fallback '{_name_id}' (score={_name_score:.2f})")

        # Use parameter extractor to get optimized parameters
        # available_strategies is intentionally not fetched — the frontend always supplies
        # HYBRID_PARALLEL which takes precedence; the extractor has its own hardcoded fallback list.
        parameter_extractor = ParameterExtractorFactory.create_extractor()

        _t = time.time()
        extraction_result = parameter_extractor.extract_parameters(
            query=query,
            available_projects=available_projects,
            available_document_types=available_document_types,
            available_strategies=None,
            supplied_project_ids=project_ids or _early_project_ids,
            supplied_document_type_ids=document_type_ids,
            supplied_search_strategy=search_strategy,
            user_location=user_location,
            supplied_project_status=project_status,
            supplied_years=years
        )
        _llm_extract_ms = round((time.time() - _t) * 1000, 2)
        _step_timings["parameter_extraction_ms"] = round(
            _llm_extract_ms + _proj_ms + _dt_ms, 2
        )

        # Extract the optimized parameters
        optimized_project_ids = extraction_result.get('project_ids', [])
        optimized_document_type_ids = extraction_result.get('document_type_ids', [])
        optimized_search_strategy = extraction_result.get('search_strategy', 'HYBRID_PARALLEL')
        optimized_semantic_query = extraction_result.get('semantic_query', query)
        optimized_location = extraction_result.get('location')
        optimized_project_status = extraction_result.get('project_status')
        optimized_years = extraction_result.get('years', [])

        _proj_detail = f"{optimized_project_ids}" if optimized_project_ids else "none (all projects)"
        _dt_detail = f"{optimized_document_type_ids}" if optimized_document_type_ids else "none (all types)"
        _q_detail = f'"{optimized_semantic_query}"' if optimized_semantic_query != query else "(same as original)"
        _t_entry(
            "extract_parameters(query, projects, doc_types, strategies)",
            "ParameterExtractor (LLM call)",
            _llm_extract_ms,
            f"Projects: {_proj_detail} | Doc types: {_dt_detail} | Strategy: {optimized_search_strategy} | Rewritten query: {_q_detail}",
        )
        
        # Check for temporal decision-date patterns ("approved after 2010", "since 2018",
        # "between 2015 and 2020", "last 3 years", etc.) and filter available_projects by
        # their decisionDate field. This correctly handles project-approval temporal queries.
        #
        # Key design notes:
        # • The vector search "years" param is just appended to the query text as semantic
        #   context — it does NOT filter by document date. Large year lists corrupt the
        #   embedding, causing bad or empty results. We clear optimized_years when we apply
        #   the decision-date filter so the vector search isn't polluted.
        # • We run the date filter even when the LLM identified project IDs, then intersect:
        #   e.g. "mine projects approved after 2010" → LLM finds mine projects, date filter
        #   keeps only those certified after 2010, intersection is what the user wants.
        if available_projects:
            date_filtered_ids = _filter_projects_by_decision_year(available_projects, query)
            if date_filtered_ids:
                if optimized_project_ids:
                    # Intersect LLM's project selection with decision-date filtered projects
                    date_set = set(date_filtered_ids)
                    intersected = [p for p in optimized_project_ids if p in date_set]
                    if intersected:
                        optimized_project_ids = intersected
                        logger.info(f"🗓️ STEP 2: Decision-date filter intersected with LLM projects → {len(optimized_project_ids)} projects")
                    else:
                        # No overlap — trust the date filter (it's more precise for temporal queries)
                        optimized_project_ids = date_filtered_ids
                        logger.info(f"🗓️ STEP 2: Decision-date filter overrode LLM (no intersection) → {len(optimized_project_ids)} projects")
                else:
                    optimized_project_ids = date_filtered_ids
                    logger.info(f"🗓️ STEP 2: Decision-date filter applied → {len(optimized_project_ids)} projects scoped")
                # Clear years so a big LLM-generated list doesn't corrupt the vector query
                optimized_years = []
                # Strip the temporal phrase from the semantic query so vector search focuses
                # on document content, not the approval date string
                cleaned_query = _strip_temporal_phrase(optimized_semantic_query)
                if cleaned_query != optimized_semantic_query:
                    logger.info(f"🗓️ STEP 2: Stripped temporal phrase from semantic query: '{optimized_semantic_query}' → '{cleaned_query}'")
                    optimized_semantic_query = cleaned_query

        # ── HyDE (Hypothetical Document Embeddings) ──────────────────────────
        # WHAT IT DOES:
        #   Instead of embedding the user's short query for vector search, we ask
        #   the LLM to write a hypothetical EAO document passage that would answer
        #   the query. That passage is richer and closer in style to real documents,
        #   so pgvector finds better matches.
        #
        # HOW TO REVERT:
        #   Set env var USE_HYDE=false  (or delete the setting entirely).
        #   The code falls back to optimized_semantic_query unchanged — no other
        #   lines need to be touched.
        #
        # TRADE-OFF: one extra LLM call (~0.5–1s). Monitor via the trace panel.
        # ─────────────────────────────────────────────────────────────────────
        if os.getenv("USE_HYDE", "false").lower() == "true":
            _t_hyde = time.time()
            try:
                hyde_prompt = (
                    f"You are an expert in BC environmental assessment documents.\n"
                    f"Write a single paragraph (3-5 sentences) from an EAO environmental "
                    f"assessment document that would directly answer this query:\n\n"
                    f"\"{query}\"\n\n"
                    f"Write only the passage — no preamble, no headings, no explanation. "
                    f"Use technical EAO language: proponent, certificate, conditions, "
                    f"assessment, impacts, mitigation measures, regulatory requirements."
                )
                hyde_messages = [
                    {"role": "system", "content": "You generate hypothetical EAO document passages for semantic search."},
                    {"role": "user",   "content": hyde_prompt},
                ]
                hyde_response = llm_client.chat_completion(
                    messages=hyde_messages,
                    temperature=0.3,
                    max_tokens=200,
                )
                hyde_passage = (
                    hyde_response.choices[0].message.content.strip()
                    if hyde_response and hyde_response.choices
                    else None
                )
                if hyde_passage:
                    logger.info(f"🔮 HyDE: generated passage ({len(hyde_passage)} chars) replacing semantic query")
                    optimized_semantic_query = hyde_passage
                else:
                    logger.warning("🔮 HyDE: empty response — keeping original semantic query")
            except Exception as e:
                logger.warning(f"🔮 HyDE: LLM call failed ({e}) — keeping original semantic query")
            _hyde_ms = round((time.time() - _t_hyde) * 1000, 2)
            _t_entry(
                "generate_hyde_passage(query)",
                "HyDE — LLM generates hypothetical document passage",
                _hyde_ms,
                f"Query expanded from {len(query)} → {len(optimized_semantic_query)} chars for richer vector match",
            )
        # ── end HyDE ─────────────────────────────────────────────────────────

        logger.info(f"🤖 STEP 2: Parameter extraction complete:")
        logger.info(f"  - Project IDs: {optimized_project_ids}")
        logger.info(f"  - Document Type IDs: {optimized_document_type_ids}")
        logger.info(f"  - Search Strategy: {optimized_search_strategy}")
        logger.info(f"  - Semantic Query: '{optimized_semantic_query}'")
        logger.info(f"  - Location Filter (from query): {optimized_location}")
        logger.info(f"  - User Location (from browser): {user_location}")
        logger.info(f"  - Project Status: {optimized_project_status}")
        logger.info(f"  - Years: {optimized_years}")
        
        # STEP 3: Initialize Agent and Execute Search Plan
        logger.info("🤖 STEP 3: Initializing agent with optimized parameters...")
        
        # Get parallel search configuration
        parallel_searches_enabled = os.getenv("AGENT_PARALLEL_SEARCHES", "true").lower() == "true"
        max_parallel_workers = int(os.getenv("AGENT_MAX_PARALLEL_WORKERS", "4"))
        
        # Initialize agent with optimized parameters
        agent = VectorSearchAgent(
            llm_client=llm_client, 
            user_location=user_location,  # User's physical location from browser (passed through)
            project_ids=optimized_project_ids,  # Use extracted project IDs
            document_type_ids=optimized_document_type_ids,  # Use extracted document type IDs
            search_strategy=optimized_search_strategy,  # Use extracted strategy
            ranking=ranking,
            location=optimized_location,  # Geographic search filter extracted from query
            project_status=optimized_project_status,  # Use extracted project status
            years=optimized_years,  # Use extracted years
            parallel_searches_enabled=parallel_searches_enabled,
            max_parallel_workers=max_parallel_workers
        )
        
        # STEP 3: Execute Searches with Optimized Parameters
        logger.info("🤖 STEP 3: Executing searches with optimized parameters...")

        # Use the optimized semantic query and parameters for searches
        search_results = []
        tool_suggestions = []

        # One well-parameterised search is faster and cleaner than 3 noisy variations.
        # Parameter extraction already produced optimal project_ids/doc_type_ids/strategy.
        num_searches = 1
        
        # Execute multiple search variations with the optimized parameters
        search_queries = agent._generate_search_variations(optimized_semantic_query, num_searches)
        logger.info(f"🤖 STEP 3: Generated {len(search_queries)} search variations")
        
        # Helper function to build search parameters for a query
        def build_search_params(search_query):
            """Build search parameters with optimized values and user_location."""
            search_params = {
                "query": search_query,
                "search_strategy": optimized_search_strategy,
            }
            
            # Add non-None parameters (but always include user_location if provided)
            if optimized_project_ids:
                search_params["project_ids"] = optimized_project_ids
            if optimized_document_type_ids:
                search_params["document_type_ids"] = optimized_document_type_ids
            if optimized_location:
                search_params["location"] = optimized_location
            if optimized_project_status:
                search_params["project_status"] = optimized_project_status
            if optimized_years:
                search_params["years"] = optimized_years
            if ranking:
                search_params["ranking"] = ranking
                
            # ALWAYS include user_location when provided (even if None, let vector API handle it)
            search_params["user_location"] = user_location
            
            return search_params
        
        def execute_parallel_searches(search_queries, agent):
            """Execute searches in parallel using ThreadPoolExecutor."""
            # Capture the Flask app instance for use in worker threads
            app = current_app._get_current_object()
            
            def execute_single_search(search_data):
                i, search_query = search_data
                
                # Set up Flask application context for the worker thread using captured app
                try:
                    with app.app_context():
                        logger.info(f"🔍 Search {i+1}/{len(search_queries)}: '{search_query}'")
                        
                        # Build search parameters using shared function
                        search_params = build_search_params(search_query)
                        
                        try:
                            search_result = agent.execute_tool("search", search_params)
                            if search_result.get("success"):
                                logger.info(f"✅ Search {i+1} completed successfully")
                                return {
                                    "index": i,
                                    "query": search_query,
                                    "result": search_result,
                                    "step_name": f"search_{i+1}",
                                    "success": True
                                }
                            else:
                                logger.warning(f"❌ Search {i+1} failed: {search_result.get('error', 'Unknown error')}")
                                return {"index": i, "success": False, "error": search_result.get('error', 'Unknown error')}
                        except Exception as e:
                            logger.error(f"❌ Search {i+1} exception: {e}")
                            return {"index": i, "success": False, "error": str(e)}
                            
                except RuntimeError as e:
                    # Flask app context issue - fallback to sequential for this search
                    logger.warning(f"❌ Search {i+1} app context error, falling back to sequential: {e}")
                    return {"index": i, "success": False, "error": f"App context error: {str(e)}"}
            
            # Execute searches in parallel
            with ThreadPoolExecutor(max_workers=agent.max_parallel_workers) as executor:
                # Submit all search tasks
                future_to_search = {
                    executor.submit(execute_single_search, (i, query)): i 
                    for i, query in enumerate(search_queries)
                }
                
                # Collect results as they complete
                parallel_results = [None] * len(search_queries)
                for future in as_completed(future_to_search):
                    try:
                        result = future.result()
                        if result["success"]:
                            parallel_results[result["index"]] = result
                    except Exception as e:
                        logger.error(f"❌ Parallel search execution error: {e}")
                
                # Add successful results in original order
                return [r for r in parallel_results if r is not None]
        
        def execute_sequential_searches(search_queries, agent):
            """Execute searches sequentially."""
            search_results = []
            
            for i, search_query in enumerate(search_queries):
                logger.info(f"🔍 Search {i+1}/{len(search_queries)}: '{search_query}'")
                
                # Build search parameters using shared function
                search_params = build_search_params(search_query)
                
                try:
                    search_result = agent.execute_tool("search", search_params)
                    if search_result.get("success"):
                        search_results.append({
                            "query": search_query,
                            "result": search_result,
                            "step_name": f"search_{i+1}"
                        })
                        logger.info(f"✅ Search {i+1} completed successfully")
                    else:
                        logger.warning(f"❌ Search {i+1} failed: {search_result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"❌ Search {i+1} exception: {e}")
            
            return search_results
        
        # Determine execution mode (disable parallel for now due to Flask context issues)
        execution_mode = "parallel" if agent.parallel_searches_enabled and len(search_queries) > 1 else "sequential"
        
        logger.info(f"🤖 STEP 3: Executing searches in {execution_mode} mode")

        # Execute searches using appropriate strategy
        _t = time.time()
        if parallel_searches_enabled and len(search_queries) > 1:
            search_results = execute_parallel_searches(search_queries, agent)
        else:
            search_results = execute_sequential_searches(search_queries, agent)
        _step_timings["vector_search_ms"] = round((time.time() - _t) * 1000, 2)

        _filter_parts = []
        if optimized_project_ids:
            _filter_parts.append(f"projects={optimized_project_ids}")
        if optimized_document_type_ids:
            _filter_parts.append(f"doc_types={optimized_document_type_ids}")
        if optimized_location:
            _filter_parts.append(f"location={optimized_location}")
        if optimized_project_status:
            _filter_parts.append(f"status={optimized_project_status}")
        if optimized_years:
            _filter_parts.append(f"years={optimized_years}")
        _filter_str = ", ".join(_filter_parts) if _filter_parts else "none"
        _t_entry(
            f"VectorSearchClient.search(strategy={optimized_search_strategy})",
            "VectorSearchClient → POST /vector-search → vector-api → PostgreSQL pgvector",
            _step_timings["vector_search_ms"],
            f'Query: "{optimized_semantic_query}" | Filters: {_filter_str} | Searches: {len(search_results)}',
            is_external=True,
        )

        logger.info(f"🤖 STEP 3: Completed {len(search_results)} successful searches ({_step_timings['vector_search_ms']}ms)")

        # STEP 4: Consolidate Results
        logger.info("🤖 STEP 4: Consolidating search results...")
        _t = time.time()
        all_documents = []
        all_document_chunks = []
        
        for search_result in search_results:
            result_data = search_result["result"]["result"]
            if isinstance(result_data, tuple) and len(result_data) >= 2:
                documents, chunks = result_data[0], result_data[1]
                if documents:
                    all_documents.extend(documents)
                if chunks:
                    all_document_chunks.extend(chunks)
        
        logger.info(f"🤖 STEP 4: Collected {len(all_documents)} documents, {len(all_document_chunks)} chunks")
        
        # Remove duplicates - use content-based deduplication as fallback
        unique_documents = []
        unique_chunks = []
        seen_doc_ids = set()
        seen_chunk_content = set()
        
        for doc in all_documents:
            doc_id = doc.get('id') or doc.get('document_id')
            if doc_id and doc_id not in seen_doc_ids:
                unique_documents.append(doc)
                seen_doc_ids.add(doc_id)
        
        for chunk in all_document_chunks:
            # Try multiple possible ID field names, including document_id + page_number combo
            chunk_id = (chunk.get('id') or 
                       chunk.get('chunk_id') or 
                       chunk.get('document_chunk_id') or
                       chunk.get('_id'))
            
            # If no standard ID, create composite key from document_id + page_number
            if not chunk_id and isinstance(chunk, dict):
                doc_id = chunk.get('document_id')
                page_num = chunk.get('page_number')
                if doc_id and page_num is not None:
                    chunk_id = f"{doc_id}_{page_num}"
            
            # If we have an ID, use it for deduplication
            if chunk_id:
                if chunk_id not in seen_doc_ids:  # Reuse seen_doc_ids for chunk IDs too
                    unique_chunks.append(chunk)
                    seen_doc_ids.add(chunk_id)

            else:
                # Fallback: Use content-based deduplication
                chunk_content = ""
                if isinstance(chunk, dict):
                    chunk_content = (chunk.get('content') or 
                                   chunk.get('text') or 
                                   chunk.get('snippet') or 
                                   str(chunk))
                else:
                    chunk_content = str(chunk)
                
                # Create a hash of the content for deduplication
                content_hash = hash(chunk_content[:200])  # Use first 200 chars for hash
                if content_hash not in seen_chunk_content:
                    unique_chunks.append(chunk)
                    seen_chunk_content.add(content_hash)

        
        _step_timings["consolidation_ms"] = round((time.time() - _t) * 1000, 2)
        logger.info(f"🤖 STEP 4: After deduplication: {len(unique_documents)} documents, {len(unique_chunks)} chunks ({_step_timings['consolidation_ms']}ms)")
        _t_entry(
            "deduplicate_results(documents, chunks)",
            "agent_stub (in-memory)",
            _step_timings["consolidation_ms"],
            f"Output: {len(unique_documents)} documents, {len(unique_chunks)} chunks after deduplication",
        )

        # STEP 5: Generate Summary (skip if DISABLE_SUMMARY=true)
        _t = time.time()
        if os.getenv("DISABLE_SUMMARY", "false").lower() == "true":
            logger.info("🤖 STEP 5: Summary generation disabled via DISABLE_SUMMARY env var — skipping")
            final_summary = ""
        else:
            logger.info("🤖 STEP 5: Generating AI summary...")
            try:
                summary_result = agent.execute_tool("summarize_results", {
                    "query": query,
                    "include_metadata": True
                }, context={
                    "consolidated_results": {
                        "documents": unique_documents,
                        "document_chunks": unique_chunks
                    }
                })

                if summary_result.get("success"):
                    final_summary = summary_result["result"]
                    logger.info("✅ Summary generation completed successfully")
                else:
                    final_summary = "Summary generation failed"
                    logger.warning(f"❌ Summary generation failed: {summary_result.get('error')}")
            except Exception as e:
                logger.error(f"❌ Summary generation exception: {e}")
                final_summary = "Summary generation encountered an error"
        _step_timings["summary_generation_ms"] = round((time.time() - _t) * 1000, 2)
        _t_entry(
            "execute_tool('summarize_results', {query, chunks})",
            "SummarizerFactory (LLM call)",
            _step_timings["summary_generation_ms"],
            f"{'SKIPPED' if os.getenv('DISABLE_SUMMARY', 'false').lower() == 'true' else f'Summarised {len(unique_chunks)} chunks → natural-language response'}",
        )

        # STEP 6: Return Results
        execution_time = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"🤖 AGENT: Execution completed in {execution_time}ms")
        logger.info(f"🤖 AGENT: Final results: {len(unique_documents)} documents, {len(unique_chunks)} chunks")
        logger.info("=" * 60)
        
        # Build detailed execution summary for visibility
        search_execution_details = []
        for i, search_result in enumerate(search_results):
            result_data = search_result["result"]["result"]
            documents_count = len(result_data[0]) if isinstance(result_data, tuple) else 0
            chunks_count = len(result_data[1]) if isinstance(result_data, tuple) and len(result_data) >= 2 else 0
            
            search_execution_details.append({
                "search_number": i + 1,
                "query": search_result["query"],
                "parameters": {
                    "project_ids": optimized_project_ids,
                    "document_type_ids": optimized_document_type_ids,
                    "search_strategy": optimized_search_strategy,
                    "location": optimized_location,
                    "user_location": user_location,
                    "project_status": optimized_project_status,
                    "years": optimized_years,
                    "ranking": ranking
                },
                "results": {
                    "documents_returned": documents_count,
                    "chunks_returned": chunks_count,
                    "success": True
                },
                "execution_mode": execution_mode
            })

        logger.info(
            f"⏱️ AGENT STEPS | validate={_step_timings['relevance_check_ms']}ms | "
            f"extract={_step_timings['parameter_extraction_ms']}ms | "
            f"search={_step_timings['vector_search_ms']}ms | "
            f"consolidate={_step_timings['consolidation_ms']}ms | "
            f"summary={_step_timings['summary_generation_ms']}ms"
        )

        return {
            "agent_results": search_results,
            "planning_method": "Simplified parameter extraction + multi-search",
            "execution_time": execution_time,
            "steps_executed": 5,  # 1:validate, 2:extract, 3:search, 4:consolidate, 5:summarize
            "consolidated_summary": final_summary,
            "documents": unique_documents,
            "document_chunks": unique_chunks,
            "search_executions": len(search_results),
            "search_execution_details": search_execution_details,
            "tool_suggestions": tool_suggestions,
            "step_timings": _step_timings,
            "trace": _trace,
            "extracted_parameters": {
                "project_ids": optimized_project_ids,
                "document_type_ids": optimized_document_type_ids,
                "search_strategy": optimized_search_strategy,
                "semantic_query": optimized_semantic_query,
                "location": optimized_location,
                "user_location": user_location,
                "project_status": optimized_project_status,
                "years": optimized_years
            }
        }
        
    except Exception as e:
        logger.error(f"🤖 AGENT: Error in agent processing: {e}")
        logger.info("=" * 60)
        
        return {
            "agent_attempted": True,
            "agent_implemented": True,
            "query": query,
            "reason": reason,
            "error": str(e),
            "fallback_to_normal_flow": True
        }
