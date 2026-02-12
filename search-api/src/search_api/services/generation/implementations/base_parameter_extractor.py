"""
Base Parameter Extractor Implementation
Contains common logic for parameter extraction approach.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Union

from search_api.services.generation.abstractions.parameter_extractor import ParameterExtractor

logger = logging.getLogger(__name__)

class BaseParameterExtractor(ParameterExtractor):
    """Base implementation of parameter extractor with common logic."""
   
    def __init__(self, client):
        self.client = client
   
    def extract_parameters(
        self,
        query: str,
        available_projects: Optional[List] = None,
        available_document_types: Optional[List] = None,
        available_strategies: Optional[Dict] = None,
        supplied_project_ids: Optional[List[str]] = None,
        supplied_document_type_ids: Optional[List[str]] = None,
        supplied_search_strategy: Optional[str] = None,
        user_location: Optional[Dict] = None,
        supplied_location: Optional[Dict] = None,
        supplied_project_status: Optional[str] = None,
        supplied_years: Optional[list] = None,
        use_parallel: bool = True
    ) -> Dict[str, Any]:
        """Extract search parameters using parallel or sequential approach.
       
        Args:
            query: The natural language search query.
            available_projects: List of available projects from VectorSearchClient.get_projects_list().
            available_document_types: List of available document types from VectorSearchClient.get_document_types().
            available_strategies: Dict of available search strategies.
            supplied_project_ids: Already provided project IDs (skip LLM extraction if provided).
            supplied_document_type_ids: Already provided document type IDs (skip LLM extraction if provided).
            supplied_search_strategy: Already provided search strategy (skip LLM extraction if provided).
            user_location: User's location data for location-aware queries.
            supplied_location: Already provided location parameter (skip LLM extraction if provided).
            supplied_project_status: Already provided project status (skip LLM extraction if provided).
            supplied_years: Already provided years list (skip LLM extraction if provided).
            use_parallel: Whether to use parallel execution (default: True).
           
        Returns:
            Dict containing extracted parameters including temporal parameters.
        """
        logger.info("=== PARAMETER EXTRACTION START ===")
        logger.info(f"Query to extract from: '{query}'")
        logger.info(f"Use parallel execution: {use_parallel}")
       
        # Log available context data - SHOW ALL DATA (no truncation)
        logger.info("=== AVAILABLE CONTEXT DATA ===")
        if available_projects:
            logger.info(f"Available Projects Array ({len(available_projects)}):")
            for project in available_projects:  # Show ALL projects
                if isinstance(project, dict) and 'project_name' in project and 'project_id' in project:
                    logger.info(f"  - '{project['project_name']}' -> {project['project_id']}")
        else:
            logger.info("Available Projects: None provided")
           
        if available_document_types:
            logger.info(f"Available Document Types Array ({len(available_document_types)}):")
            for doc_type in available_document_types:  # Show ALL document types
                if isinstance(doc_type, dict) and 'document_type_id' in doc_type:
                    name = doc_type.get('document_type_name', 'Unknown')
                    aliases = doc_type.get('aliases', [])
                    logger.info(f"  - '{name}' (ID: {doc_type['document_type_id']}) - Aliases: {aliases}")
        else:
            logger.info("Available Document Types: None provided")
           
        if available_strategies:
            logger.info(f"Available Strategies ({len(available_strategies)}):")
            for name, description in available_strategies.items():
                logger.info(f"  - '{name}': {description}")
        else:
            logger.info("Available Strategies: None provided")
           
        # Log supplied parameters
        logger.info("=== SUPPLIED PARAMETERS ===")
        logger.info(f"Supplied Project IDs: {supplied_project_ids}")
        logger.info(f"Supplied Document Type IDs: {supplied_document_type_ids}")
        logger.info(f"Supplied Search Strategy: {supplied_search_strategy}")
        logger.info(f"User Location: {user_location}")
        logger.info(f"Supplied Location: {supplied_location}")
        logger.info(f"Supplied Project Status: {supplied_project_status}")
        logger.info(f"Supplied Years: {supplied_years}")
        logger.info("=== END CONTEXT DATA ===")
       
        available_projects_metadata = available_projects or []
       
        # Convert arrays to dict format for internal processing
        projects_dict = self._convert_projects_array_to_dict(available_projects)
        document_types_dict = self._convert_document_types_array_to_dict(available_document_types)
       
        if use_parallel:
            try:
                return self._extract_parameters_parallel(
                    query, projects_dict, available_projects_metadata, document_types_dict, available_strategies,
                    supplied_project_ids, supplied_document_type_ids, supplied_search_strategy,
                    user_location, supplied_location, supplied_project_status, supplied_years
                )
            except Exception as e:
                logger.warning(f"Parallel extraction failed, falling back to sequential: {e}")
                return self._extract_parameters_sequential(
                    query, projects_dict, available_projects_metadata, document_types_dict, available_strategies,
                    supplied_project_ids, supplied_document_type_ids, supplied_search_strategy,
                    user_location, supplied_location, supplied_project_status, supplied_years
                )
        else:
            return self._extract_parameters_sequential(
                query, projects_dict, available_projects_metadata, document_types_dict, available_strategies,
                supplied_project_ids, supplied_document_type_ids, supplied_search_strategy,
                user_location, supplied_location, supplied_project_status, supplied_years
            )
   
    def _extract_parameters_sequential(
        self,
        query: str,
        available_projects: Optional[Dict] = None,
        available_projects_metadata: Optional[List[Dict]] = None,
        available_document_types: Optional[Dict] = None,
        available_strategies: Optional[Dict] = None,
        supplied_project_ids: Optional[List[str]] = None,
        supplied_document_type_ids: Optional[List[str]] = None,
        supplied_search_strategy: Optional[str] = None,
        user_location: Optional[Dict] = None,
        supplied_location: Optional[Dict] = None,
        supplied_project_status: Optional[str] = None,
        supplied_years: Optional[list] = None
    ) -> Dict[str, Any]:
        """Extract search parameters using focused, sequential calls.
       
        Args:
            query: The natural language search query.
            available_projects: Dict of available projects {name: id}.
            available_document_types: Dict of available document types with aliases.
            available_strategies: Dict of available search strategies.
            supplied_project_ids: Already provided project IDs (skip LLM extraction if provided).
            supplied_document_type_ids: Already provided document type IDs (skip LLM extraction if provided).
            supplied_search_strategy: Already provided search strategy (skip LLM extraction if provided).
           
        Returns:
            Dict containing extracted parameters.
        """
        try:
            logger.info("Starting sequential parameter extraction")
           
            # Step 1: Extract project IDs (skip if already provided)
            if supplied_project_ids:
                project_ids = supplied_project_ids
                logger.info(f"Step 1 - Using supplied project IDs: {project_ids}")
            else:
                project_ids = self._extract_project_ids(query, available_projects_metadata or available_projects)
                logger.info(f"Step 1 - Extracted project IDs: {project_ids}")
           
            # Step 2: Extract document type IDs (skip if already provided)
            if supplied_document_type_ids:
                document_type_ids = supplied_document_type_ids
                logger.info(f"Step 2 - Using supplied document type IDs: {document_type_ids}")
            else:
                document_type_ids = self._extract_document_types(query, available_document_types)
                logger.info(f"Step 2 - Extracted document type IDs: {document_type_ids}")
           
            # Step 3: Extract search strategy (skip if already provided)
            if supplied_search_strategy:
                search_strategy = supplied_search_strategy
                logger.info(f"Step 3 - Using supplied search strategy: {search_strategy}")
            else:
                search_strategy = self._extract_search_strategy(query, available_strategies)
                logger.info(f"Step 3 - Extracted search strategy: {search_strategy}")
           
            # Step 4: Extract/optimize semantic query (usually always run for query optimization)
            semantic_query = self._extract_semantic_query(query)
            logger.info(f"Step 4 - Optimized semantic query: {semantic_query}")
           
            # Step 5: Extract temporal parameters (location, project_status, years) using LLM
            if supplied_location is not None or supplied_project_status is not None or supplied_years is not None:
                # Use supplied temporal parameters
                location = supplied_location
                project_status = supplied_project_status
                years = supplied_years
                logger.info(f"Step 5 - Using supplied temporal parameters: location={location}, status={project_status}, years={years}")
                temporal_sources = {
                    "location": "supplied" if supplied_location is not None else "fallback",
                    "project_status": "supplied" if supplied_project_status is not None else "fallback",
                    "years": "supplied" if supplied_years is not None else "fallback"
                }
            else:
                # Extract temporal and location parameters using LLM
                temporal_result = self._extract_temporal_and_location_parameters(query, user_location)
                location = temporal_result.get("location")
                project_status = temporal_result.get("project_status")
                years = temporal_result.get("years", [])
                logger.info(f"Step 5 - Extracted temporal and location parameters: location={location}, status={project_status}, years={years}")
                temporal_sources = {
                    "location": "llm_extracted" if location is not None else "fallback",
                    "project_status": "llm_extracted" if project_status is not None else "fallback",
                    "years": "llm_extracted" if years else "fallback"
                }
           
            # Combine results
            return {
                "project_ids": project_ids,
                "document_type_ids": document_type_ids,
                "search_strategy": search_strategy,
                "semantic_query": semantic_query,
                "location": location,
                "project_status": project_status,
                "years": years,
                "confidence": 0.8,
                "extraction_sources": {
                    "project_ids": "supplied" if supplied_project_ids else "llm_sequential",
                    "document_type_ids": "supplied" if supplied_document_type_ids else "llm_sequential",
                    "search_strategy": "supplied" if supplied_search_strategy else "llm_sequential",
                    "semantic_query": "llm_sequential",
                    **temporal_sources
                }
            }
           
        except Exception as e:
            logger.error(f"Sequential parameter extraction failed: {e}")
            return self._fallback_extraction(query, available_projects, available_document_types, available_strategies, supplied_project_ids, supplied_document_type_ids, supplied_search_strategy)
           
        except Exception as e:
            logger.error(f"Multi-step parameter extraction failed: {e}")
            return self._fallback_extraction(query, available_projects, available_document_types, available_strategies, supplied_project_ids, supplied_document_type_ids, supplied_search_strategy)
   
    def _extract_parameters_parallel(
        self,
        query: str,
        available_projects: Optional[Dict] = None,
        available_projects_metadata: Optional[List[Dict]] = None,
        available_document_types: Optional[Dict] = None,
        available_strategies: Optional[Dict] = None,
        supplied_project_ids: Optional[List[str]] = None,
        supplied_document_type_ids: Optional[List[str]] = None,
        supplied_search_strategy: Optional[str] = None,
        user_location: Optional[Dict] = None,
        supplied_location: Optional[Dict] = None,
        supplied_project_status: Optional[str] = None,
        supplied_years: Optional[list] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Extract search parameters using parallel LLM calls for maximum speed.
       
        Args:
            query: The natural language search query.
            available_projects: Dict of available projects {name: id}.
            available_document_types: Dict of available document types with aliases.
            available_strategies: Dict of available search strategies.
            supplied_project_ids: Already provided project IDs (skip LLM extraction if provided).
            supplied_document_type_ids: Already provided document type IDs (skip LLM extraction if provided).
            supplied_search_strategy: Already provided search strategy (skip LLM extraction if provided).
            timeout: Timeout in seconds for parallel execution.
           
        Returns:
            Dict containing extracted parameters.
        """
        try:
            logger.info("Starting parallel parameter extraction")
           
            # Prepare tasks for parallel execution
            tasks = []
            task_names = []
            projects_for_llm = []
            # Task 1: Extract project IDs (if not supplied)
            if not supplied_project_ids:
                if available_projects_metadata:
                    projects_for_llm = available_projects_metadata
                elif isinstance(available_projects, dict):
                    projects_for_llm = [
                        {"project_name": name, "project_id": pid, "project_metadata": {}}
                        for name, pid in available_projects.items()
                    ]
                else:
                    projects_for_llm = available_projects

                tasks.append(lambda projects=projects_for_llm: self._extract_project_ids(query, projects))
                task_names.append("project_ids")
           
            # Task 2: Extract document type IDs (if not supplied)
            if not supplied_document_type_ids:
                tasks.append(lambda: self._extract_document_types(query, available_document_types))
                task_names.append("document_type_ids")
           
            # Task 3: Extract search strategy (if not supplied)
            if not supplied_search_strategy:
                tasks.append(lambda: self._extract_search_strategy(query, available_strategies))
                task_names.append("search_strategy")
           
            # Task 4: Extract semantic query (always run for optimization)
            tasks.append(lambda: self._extract_semantic_query(query))
            task_names.append("semantic_query")
           
            # Task 5: Extract temporal and location parameters (if not supplied)
            if supplied_location is None or supplied_project_status is None or supplied_years is None:
                tasks.append(lambda: self._extract_temporal_and_location_parameters(query, user_location))
                task_names.append("temporal_and_location_parameters")
           
            # Execute tasks in parallel using ThreadPoolExecutor
            results = {}
           
            if tasks:
                with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
                    # Submit all tasks
                    future_to_name = {
                        executor.submit(task): name
                        for task, name in zip(tasks, task_names)
                    }
                   
                    # Collect results with timeout
                    for future in as_completed(future_to_name, timeout=timeout):
                        task_name = future_to_name[future]
                        try:
                            result = future.result()
                            results[task_name] = result
                            logger.info(f"Parallel task '{task_name}' completed: {result}")
                        except Exception as e:
                            logger.warning(f"Parallel task '{task_name}' failed: {e}")
                            # Use fallback for failed task
                            results[task_name] = self._get_fallback_for_task(
                                task_name, query, available_projects,
                                available_document_types, available_strategies
                            )
           
            # Extract temporal and location parameters from results or use supplied values
            temporal_result = results.get("temporal_and_location_parameters", {})
            location = supplied_location if supplied_location is not None else temporal_result.get("location")
            project_status = supplied_project_status if supplied_project_status is not None else temporal_result.get("project_status")
            years = supplied_years if supplied_years is not None else temporal_result.get("years", [])
           
            # Combine results with supplied values
            return {
                "project_ids": supplied_project_ids or results.get("project_ids", []),
                "document_type_ids": supplied_document_type_ids or results.get("document_type_ids", []),
                "search_strategy": supplied_search_strategy or results.get("search_strategy", "HYBRID_PARALLEL"),
                "semantic_query": results.get("semantic_query", query),
                "location": location,
                "project_status": project_status,
                "years": years,
                "confidence": 0.8,
                "extraction_sources": {
                    "project_ids": "supplied" if supplied_project_ids else "llm_parallel",
                    "document_type_ids": "supplied" if supplied_document_type_ids else "llm_parallel",
                    "search_strategy": "supplied" if supplied_search_strategy else "llm_parallel",
                    "semantic_query": "llm_parallel",
                    "location": "supplied" if supplied_location is not None else ("llm_parallel" if location is not None else "fallback"),
                    "project_status": "supplied" if supplied_project_status is not None else ("llm_parallel" if project_status is not None else "fallback"),
                    "years": "supplied" if supplied_years is not None else ("llm_parallel" if years else "fallback")
                }
            }
           
        except Exception as e:
            logger.error(f"Parallel parameter extraction failed: {e}")
            # Fallback to sequential extraction
            logger.info("Falling back to sequential extraction")
            return self._extract_parameters_sequential(
                query, available_projects, available_document_types, available_strategies,
                supplied_project_ids, supplied_document_type_ids, supplied_search_strategy
            )
   
    def _get_fallback_for_task(
        self,
        task_name: str,
        query: str,
        available_projects: Optional[Dict] = None,
        available_document_types: Optional[Dict] = None,
        available_strategies: Optional[Dict] = None
    ) -> Any:
        """Get fallback result for a specific failed task."""
        if task_name == "project_ids":
            return self._fallback_project_extraction(query, available_projects or {})
        elif task_name == "document_type_ids":
            return self._fallback_document_extraction(query, available_document_types or {})
        elif task_name == "search_strategy":
            return "HYBRID_PARALLEL"
        elif task_name == "semantic_query":
            return query
        else:
            return None
   
    def _extract_project_ids(self, query: str, available_projects: Optional[Union[Dict, List[Dict]]] = None) -> List[str]:
        """Extract project IDs from query using focused LLM call with validation and retry."""
        logger.info("=== PROJECT ID EXTRACTION START ===")
        logger.info(f"Query for project extraction: '{query}'")
       
        if not available_projects:
            logger.warning("No available projects provided - returning empty list")
            logger.info("=== PROJECT ID EXTRACTION END ===")
            return []
       
        logger.info(f"Available projects for matching ({len(available_projects)}):")
        if isinstance(available_projects, dict):
            for name, proj_id in available_projects.items():
                logger.info(f"  - '{name}' -> {proj_id}")
        elif isinstance(available_projects, list):
            for proj in available_projects:
                project_id = proj.get("project_id", "")
                project_name = proj.get("project_name", "")
                logger.info(f"  - '{project_name}' -> {project_id}")
        else:
            logger.warning("available_projects is neither dict nor list; cannot log contents")
       
        # Try LLM extraction with validation and retry
        for attempt in range(3):  # Maximum 3 attempts
            try:
                logger.info(f"Attempt {attempt + 1}/3 for project ID extraction")
                # Convert available_projects to a list of dicts if it's currently a dict
                if isinstance(available_projects, dict):
                    # Transform {name: id} -> [{"project_name": name, "project_id": id, "project_metadata": {}}]
                    projects_for_llm = [
                        {"project_name": name, "project_id": pid, "project_metadata": {}}
                        for name, pid in available_projects.items()
                    ]
                else:
                    # Already a list of dicts with metadata
                    projects_for_llm = available_projects

                # Pass projects_for_llm to _extract_project_ids_single_attempt
                result = self._extract_project_ids_single_attempt(query, projects_for_llm, attempt)
               
                # Validate the result quality
                if result:
                    logger.info(f"Project extraction successful on attempt {attempt + 1}: {result}")
                    logger.info("=== PROJECT ID EXTRACTION END ===")
                    return result
                else:
                    logger.warning(f"Project extraction attempt {attempt + 1} failed validation, will retry")
                   
            except Exception as e:
                logger.warning(f"Project extraction attempt {attempt + 1} failed with error: {e}")
                if attempt == 2:  # Last attempt
                    logger.error("All LLM attempts failed, using fallback")
                    break
       
        # All attempts failed, use fallback
        logger.warning("LLM project extraction failed all attempts, using fallback method")
        result = self._fallback_project_extraction(query, available_projects)
        logger.info(f"Fallback extraction result: {result}")
        logger.info("=== PROJECT ID EXTRACTION END ===")
        return result
   
    def _extract_project_ids_single_attempt(self, query: str, available_projects: List[Dict], attempt: int) -> List[str]:
        """Single attempt at project ID extraction using both project names and selected metadata context."""
        try:
            # ✅ Extract and format relevant fields from metadata for the LLM
            project_lines = []
            for proj in available_projects:
                project_id = proj.get("project_id", "")
                project_name = proj.get("project_name", "")
                meta = proj.get("project_metadata", {}) or {}

                # Only include selected fields (handle nested dicts safely)
                proponent_raw = meta.get("proponent", "")
                if isinstance(proponent_raw, dict):
                    proponent_name = proponent_raw.get("name", proponent_raw.get("company", ""))
                elif proponent_raw:
                    proponent_name = str(proponent_raw)
                else:
                    proponent_name = ""

                relevant_meta = {
                    "type": meta.get("type", ""),
                    "region": meta.get("region", ""),
                    "sector": meta.get("sector", ""),
                    "status": meta.get("status", ""),
                    "proponent": proponent_name,
                    "description": meta.get("description", ""),
                    "location": meta.get("location", "")
                }

                # Clean short text representation for the LLM
                project_lines.append(
                    f"- {project_name} (ID: {project_id}) | "
                    f"Type: {relevant_meta['type']}, Region: {relevant_meta['region']}, "
                    f"Sector: {relevant_meta['sector']}, Status: {relevant_meta['status']}, "
                    f"Proponent: {relevant_meta['proponent']}, "
                    f"Location: {relevant_meta['location']}, "
                    f"Description: {relevant_meta['description'][:200]}..."  # Trim long text
                )

            prompt = f"""
You are given a user query and a list of projects. Each project has the following metadata:
- project_id
- project_name
- proponent
- location
- region
- type
- sector
- status
- description

Task:
Return a ranked list of projects that match the query. You MUST follow these strict matching rules:

### CRITICAL MATCHING RULES (in priority order):

1. **EXACT PROJECT NAME MATCH IS HIGHEST PRIORITY**:
   - If the query mentions a specific project name (e.g., "Cariboo Gold", "Blackwater Gold", "KSM"),
     ONLY return projects whose name contains that EXACT phrase.
   - "Cariboo Gold" should ONLY match projects with "Cariboo Gold" in the name, NOT "Blackwater Gold".
   - "Blackwater Gold" should ONLY match "Blackwater Gold", NOT "Cariboo Gold".
   - Do NOT match projects just because they share a common word like "Gold", "Mine", "River", etc.

2. **DISAMBIGUATION RULE**:
   - When multiple projects share similar words (e.g., "X Gold" vs "Y Gold"), the FULL distinctive
     portion of the name must match.
   - "Cariboo" is distinctive. "Gold" is generic. Match on "Cariboo", not on "Gold".
   - Always prefer projects where MORE words from the query match the project name.

3. **PARTIAL/SEMANTIC MATCHING (only if no exact name match)**:
   - If no specific project name is mentioned, then use synonyms and related terms:
     - 'marine port facilities' → 'port', 'terminal', 'jetty', 'wharf', 'dock'
     - 'hydroelectric' → 'run-of-river', 'power plant', 'generating station'
     - 'energy storage' → 'LNG', 'gas storage', 'tank', 'facility'
   - Consider region, proponent, type, sector, and description for broader queries.

4. **CONFIDENCE SCORING**:
   - 0.95-1.0: Exact project name match (e.g., query mentions "Cariboo Gold" and project is "Cariboo Gold")
   - 0.80-0.94: Strong match on multiple distinctive terms
   - 0.60-0.79: Partial match with some relevant metadata
   - Below 0.60: Weak match, only generic terms match - DO NOT INCLUDE

For each project, return:
- project_id
- project_name
- confidence (0-1)
- reason (explain why this project is relevant)

### EXAMPLES:

Query: "get me the schedule b for Cariboo Gold"
Correct Response:
- Cariboo Gold Project | Confidence: 0.98 | Reason: Exact project name match "Cariboo Gold".
WRONG Response (DO NOT DO THIS):
- Blackwater Gold Project | Confidence: 0.85 | Reason: Contains "Gold" <- THIS IS WRONG!

Query: "documents for Blackwater Gold mine"
Correct Response:
- Blackwater Gold Project | Confidence: 0.98 | Reason: Exact project name match "Blackwater Gold".
WRONG Response (DO NOT DO THIS):
- Cariboo Gold Project | Confidence: 0.80 | Reason: Contains "Gold" <- THIS IS WRONG!

Query: "certificate for marine port facilities near Lower Mainland"
Projects:
- Tilbury Marine Jetty | Confidence: 0.95 | Reason: Marine port facility in Lower Mainland.
- Roberts Bank Terminal 2 | Confidence: 0.90 | Reason: Major port facility in Lower Mainland.

Query: "hydroelectric projects by BC Hydro"
Projects:
- Stave Falls Powerplant | Confidence: 0.95 | Reason: Hydro plant operated by BC Hydro.
- Waneta Generating Station Upgrade | Confidence: 0.85 | Reason: Co-proponent BC Hydro.

Available Projects:
{chr(10).join(project_lines)}

Query: "{query}"

IMPORTANT: Return ONLY projects with confidence >= 0.50. If the query mentions a specific project name,
that project MUST have the highest confidence. Do NOT return projects that only match on generic words.
Be generous with confidence scores when a distinctive project name (like "Brucejack", "Cariboo", etc.) appears in the query.
"""

            logger.info("=== PROJECT EXTRACTION PROMPT (METADATA-AWARE) ===")
            logger.info(f"Prompt: {prompt}")
            logger.info("=== END PROJECT EXTRACTION PROMPT ===")

            response = self._make_llm_call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
           
            logger.info("=== PROJECT EXTRACTION LLM RESPONSE ===")
            logger.info(f"Raw LLM Response: {response}")
           
            content = response["choices"][0]["message"]["content"].strip()
            logger.info(f"Content extracted from response: '{content}'")
            logger.info("=== END PROJECT EXTRACTION LLM RESPONSE ===")
           
            # Try to parse JSON response with confidence scores
            try:
                if content.startswith('{') and content.endswith('}'):
                    result = json.loads(content)
                    project_matches = result.get("project_matches", [])
                   
                    logger.info("=== PROJECT MATCHES ANALYSIS ===")
                    logger.info(f"Number of project matches returned: {len(project_matches)}")
                   
                    valid_ids = {p.get("project_id") for p in available_projects if p.get("project_id")}
                    # Extract project IDs from matches with confidence >= 0.7
                    matched_ids = []
                    for match in project_matches:
                        confidence = match.get("confidence", 0)
                        project_id = match.get("project_id")
                        project_name = match.get("project_name", "")
                        reason = match.get("reason", "")
                       
                        logger.info(f"Match: {project_name} (ID: {project_id}) - Confidence: {confidence} - Reason: {reason}")
                       
                        # Validate project ID exists in available projects
                        # Lowered threshold from 0.7 to 0.5 to handle large datasets better
                        if project_id in valid_ids and confidence >= 0.5:
                            matched_ids.append(project_id)
                            logger.info(f"  → ACCEPTED: {project_name} added to results (confidence: {confidence})")
                        else:
                            logger.info(f"  → REJECTED: Confidence too low ({confidence} < 0.5) or invalid ID")
                   
                    logger.info(f"Final matched project IDs: {matched_ids}")
                    logger.info("=== END PROJECT MATCHES ANALYSIS ===")
                    logger.info("=== PROJECT ID EXTRACTION END ===")
                    return matched_ids[:3]  # Limit to 3 for more focused results
                   
                elif content.startswith('[') and content.endswith(']'):
                    # Fallback: try old format
                    project_ids = json.loads(content)
                    valid_ids = {p["project_id"] for p in available_projects}
                    result = [pid for pid in project_ids if pid in valid_ids]
                    logger.warning(f"Using fallback array format, got {len(result)} project IDs: {result}")
                    logger.info("=== PROJECT ID EXTRACTION END ===")
                    return result[:3]  # Limit to 3 for focused results
                elif '|' in content and 'Confidence:' in content:
                    # Handle pipe-delimited format: "Project Name | Confidence: 0.95 | Reason: ..."
                    logger.info("Detected pipe-delimited LLM response format, parsing...")

                    # Parse the pipe-delimited format
                    parts = [p.strip() for p in content.split('|')]
                    if len(parts) >= 2:
                        project_name_from_llm = parts[0].strip()

                        # Extract confidence score
                        confidence = 0.0
                        for part in parts:
                            if 'Confidence:' in part:
                                try:
                                    conf_str = part.split('Confidence:')[1].strip()
                                    confidence = float(conf_str)
                                except (ValueError, IndexError):
                                    confidence = 0.0

                        logger.info(f"Parsed from pipe format: project='{project_name_from_llm}', confidence={confidence}")

                        # Find matching project by name (case-insensitive)
                        valid_ids = {p.get("project_id") for p in available_projects if p.get("project_id")}
                        matched_project_id = None
                        best_match_score = 0.0

                        # Debug: Show first few available projects to verify structure
                        if available_projects:
                            sample_project = available_projects[0]
                            logger.info(f"Sample project structure: {list(sample_project.keys())[:5] if isinstance(sample_project, dict) else 'not a dict'}")
                            logger.info(f"Searching {len(available_projects)} available projects for match...")

                        for project in available_projects:
                            # Support both 'project_name' (API format) and 'name' (dict format) keys
                            proj_name = (project.get("project_name") or project.get("name", "")).lower()
                            proj_id = project.get("project_id")
                            llm_name_lower = project_name_from_llm.lower()

                            if not proj_name:
                                continue  # Skip if no name found

                            # Exact match
                            if proj_name == llm_name_lower:
                                matched_project_id = proj_id
                                best_match_score = 1.0
                                logger.info(f"  Exact name match: '{proj_name}' -> ID: {proj_id}")
                                break

                            # Substring match (LLM name in project name or vice versa)
                            if llm_name_lower in proj_name or proj_name in llm_name_lower:
                                # Calculate match quality based on length overlap
                                overlap = len(set(llm_name_lower.split()) & set(proj_name.split()))
                                total_words = max(len(llm_name_lower.split()), len(proj_name.split()))
                                score = overlap / total_words if total_words > 0 else 0

                                if score > best_match_score:
                                    best_match_score = score
                                    matched_project_id = proj_id
                                    logger.info(f"  Substring match: '{proj_name}' score={score:.2f} -> ID: {proj_id}")

                        if matched_project_id and confidence >= 0.5:
                            logger.info(f"Pipe-delimited parsing successful: project_id={matched_project_id}, confidence={confidence}")
                            logger.info("=== PROJECT ID EXTRACTION END ===")
                            return [matched_project_id]
                        else:
                            logger.warning(f"Pipe-delimited parsing found match but confidence too low ({confidence} < 0.5) or no match found, using fallback")
                    else:
                        logger.warning("Pipe-delimited format not parseable, using fallback")

                    result = self._fallback_project_extraction(query, available_projects)
                    logger.info(f"Fallback extraction result: {result}")
                    logger.info("=== PROJECT ID EXTRACTION END ===")
                    return result
                else:
                    logger.warning("LLM response not in expected JSON format, using fallback")
                    result = self._fallback_project_extraction(query, available_projects)
                    logger.info(f"Fallback extraction result: {result}")
                    logger.info("=== PROJECT ID EXTRACTION END ===")
                    return result
                   
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                result = self._fallback_project_extraction(query, available_projects)
                logger.info(f"Fallback extraction result: {result}")
                logger.info("=== PROJECT ID EXTRACTION END ===")
                return result
               
        except Exception as e:
            logger.warning(f"Project ID extraction failed: {e}")
            return self._fallback_project_extraction(query, available_projects)
   
    def _extract_document_types(self, query: str, available_document_types: Optional[Dict] = None) -> List[str]:
        """Extract document type IDs from query using focused LLM call."""
        logger.info("=== DOCUMENT TYPE EXTRACTION START ===")
        logger.info(f"Query for document type extraction: '{query}'")
       
        if not available_document_types:
            logger.warning("No available document types provided - returning empty list")
            logger.info("=== DOCUMENT TYPE EXTRACTION END ===")
            return []
       
        logger.info(f"Available document types for matching ({len(available_document_types)}):")
        for doc_id, doc_data in available_document_types.items():  # Show ALL document types, no truncation
            name = doc_data.get('name', 'Unknown')
            aliases = doc_data.get('aliases', [])
            logger.info(f"  - '{name}' (ID: {doc_id}) - Aliases: {aliases}")
       
        try:
            # Build comprehensive document type info including aliases
            doc_context = []
            for doc_id, doc_data in available_document_types.items():
                name = doc_data.get('name', 'Unknown')
                aliases = doc_data.get('aliases', [])
                alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
                doc_context.append(f"- {name}{alias_text} (ID: {doc_id})")
           
            prompt = f"""
You are a **document type classification specialist** working for the **Environmental Assessment Office (EAO) of British Columbia**.

Your goal is to determine which **document types** (by ID) are relevant to the user’s query.

EAO document types include reports, letters, meeting notes, certificates, permits, applications, public comments, and many other project-related materials.

---

### 🗂 Available Document Types:
{chr(10).join(doc_context)}

Each document type name represents a category of EAO records. Many have multiple ways users might refer to them.

---

### 🔍 WHEN TO SELECT DOCUMENT TYPES (include IDs):
You should include **one or more document types** when the query clearly or indirectly refers to:
- A specific **type of document** (e.g., “letters”, “reports”, “applications”, “orders”, “meeting notes”)
- A **communication or submission** (“correspondence”, “comments”, “public input”, “feedback”)
- A **decision or approval** (“EAO order”, “project decision materials”, “certificate package”)
- A **plan or study** (“management plan”, “environmental report”, “impact study”)
- A **notice or news release** (“advertisement”, “notification”, “announcement”)
- A **presentation or technical document** (“slides”, “technical memo”, “scientific study”)
- A **package or grouping** of documents (e.g., “amendment documents”, “application package”)

✅ Be inclusive — if the query even *suggests* a document type or related term, include the most relevant types.

---

### 🚫 WHEN TO RETURN AN EMPTY ARRAY []
Return `[]` (no specific types) only if:
- The query is general or factual (e.g., “who is the proponent?”, “project location”, “status of the project”)
- The user is asking about processes, events, or outcomes rather than documents (e.g., “when was it approved?”, “what were the environmental impacts?”)
- The query requests “all documents” or “everything related to X”.

---

### 💡 Matching Rules and Synonyms

| Common User Terms | Match To Document Type |
|-------------------|------------------------|
| letter, correspondence, email | **Letter** |
| comments, feedback, submissions | **Comment/Submission**, **Comment Period** |
| meeting, minutes, notes | **Meeting Notes** |
| decision, approval, determination | **Decision Materials**, **Order** |
| application, form, submission materials | **Application Materials**, **Application Information Requirement** |
| plan, management plan, mitigation plan | **Plan**, **Management Plan** |
| report, study, technical paper, analysis | **Report/Study**, **Scientific Memo**, **Independent Memo** |
| certificate, EA certificate, permit, EAC | **Certificate Package**, **Order** |
| Schedule B, conditions, certificate conditions | **Certificate Package**, **Order** (Schedule B contains certificate conditions) |
| Schedule A, certified project description | **Certificate Package**, **Order** (Schedule A contains project description) |
| notification, announcement, advertisement | **Notification**, **Ad/News Release** |
| inspection, compliance check | **Inspection Record** |
| agreement, MOU | **Agreement** |
| amendment, revision, exception | **Amendment Package**, **Amendment Information**, **Exception Package** |
| project description, overview | **Project Description**, **Project Descriptions** |
| presentation, slides | **Presentation** |
| tracking, index | **Tracking Table** |

**IMPORTANT**: Schedule B and Schedule A are part of Environmental Assessment Certificates:
- **Schedule B** = Certificate Conditions (requirements the proponent must meet)
- **Schedule A** = Certified Project Description
When users ask about "Schedule B" or "conditions", match to **Certificate Package** or **Order** document types.

---

### 🧭 Reasoning Process (step-by-step)

1. **Interpret intent** – Is the query about a *type of document* or a *general topic*?  
2. **Extract related terms** – Identify nouns or phrases that hint at documents, records, or communications.  
3. **Match semantically** – Use synonyms and context clues (e.g., “emails” → “Letters”, “report on wildlife” → “Report/Study”).  
4. **Map to document type IDs** – Return the IDs for *all matching document types*.  
5. **If nothing fits or query is general**, return an empty list `[]`.

---

### 🧩 Examples

**Example 1**  
Query: “Letters between EAO and Ministry of Environment”  
→ Matches: `["Letter"]`

**Example 2**  
Query: “Technical reports about the KSM project”  
→ Matches: `["Report/Study", "Scientific Memo"]`

**Example 3**  
Query: “When was the environmental certificate issued?”  
→ Matches: `["Certificate Package", "Order"]`

**Example 4**  
Query: “What are the impacts of the project?”  
→ Matches: `[]` (general inquiry)

**Example 5**  
Query: “Meeting minutes with First Nations”  
→ Matches: `["Meeting Notes", "Agreement"]`

---

Now, analyze this query carefully and follow the reasoning steps.
Query: "{query}"

Return the **document type IDs** as a JSON array of strings (e.g., `["5cf00c03a266b7e1877504cb"]`), or `[]` if no specific document types apply.
"""

            logger.info("=== DOCUMENT TYPE EXTRACTION PROMPT ===")
            logger.info(f"Prompt: {prompt}")
            logger.info("=== END DOCUMENT TYPE EXTRACTION PROMPT ===")

            response = self._make_llm_call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
           
            logger.info("=== DOCUMENT TYPE EXTRACTION LLM RESPONSE ===")
            logger.info(f"Raw LLM Response: {response}")
           
            content = response["choices"][0]["message"]["content"].strip()
            logger.info(f"Content extracted from response: '{content}'")
            logger.info("=== END DOCUMENT TYPE EXTRACTION LLM RESPONSE ===")
           
            # Try to parse JSON response
            try:
                # Clean up the content and try to parse as JSON
                content_clean = content.strip()
                logger.info(f"Cleaned content for JSON parsing: '{content_clean}'")
               
                # Extract JSON from markdown code blocks if present
                if '```json' in content_clean:
                    # Find the JSON block between ```json and ```
                    start_marker = '```json'
                    end_marker = '```'
                    start_idx = content_clean.find(start_marker)
                    if start_idx != -1:
                        start_idx += len(start_marker)
                        end_idx = content_clean.find(end_marker, start_idx)
                        if end_idx != -1:
                            json_content = content_clean[start_idx:end_idx].strip()
                            logger.info(f"Extracted JSON from markdown blocks: '{json_content}'")
                            content_clean = json_content
               
                # Handle case where LLM provides explanation text followed by JSON array
                elif not content_clean.startswith('[') and not content_clean.startswith('{'):
                    # Look for JSON array at the end of explanation text
                    import re
                    # Find the last JSON array pattern in the content
                    array_matches = re.findall(r'\[(?:[^\[\]]*|\[[^\[\]]*\])*\]', content_clean)
                    if array_matches:
                        content_clean = array_matches[-1].strip()
                        logger.info(f"Extracted JSON array from explanation text: '{content_clean}'")
               
                # Try to parse directly as JSON (handles multiple formats)
                parsed_response = json.loads(content_clean)
               
                # Handle different response formats
                if isinstance(parsed_response, list):
                    # Direct array format: [] or ["id1", "id2"]
                    doc_type_ids = parsed_response
                elif isinstance(parsed_response, dict):
                    # Object format with result key: {"result": []}
                    if "result" in parsed_response:
                        doc_type_ids = parsed_response["result"]
                        logger.info(f"Extracted document type IDs from 'result' key: {doc_type_ids}")
                    else:
                        logger.warning(f"LLM response is dict but no 'result' key found: {parsed_response}")
                        result = self._fallback_document_extraction(query, available_document_types)
                        logger.info(f"Fallback extraction result: {result}")
                        logger.info("=== DOCUMENT TYPE EXTRACTION END ===")
                        return result
                else:
                    logger.warning(f"LLM response is not a list or dict, got: {type(parsed_response)}")
                    result = self._fallback_document_extraction(query, available_document_types)
                    logger.info(f"Fallback extraction result: {result}")
                    logger.info("=== DOCUMENT TYPE EXTRACTION END ===")
                    return result
               
                # Ensure the final result is a list
                if not isinstance(doc_type_ids, list):
                    logger.warning(f"Document type IDs is not a list after extraction, got: {type(doc_type_ids)}")
                    result = self._fallback_document_extraction(query, available_document_types)
                    logger.info(f"Fallback extraction result: {result}")
                    logger.info("=== DOCUMENT TYPE EXTRACTION END ===")
                    return result
               
                logger.info("=== DOCUMENT TYPE MATCHES ANALYSIS ===")
                logger.info(f"LLM returned document type IDs: {doc_type_ids}")
               
                # If empty array, LLM determined no specific document types needed
                if not doc_type_ids:
                    logger.info("LLM returned empty array - no document type filtering needed")
                    logger.info("=== END DOCUMENT TYPE MATCHES ANALYSIS ===")
                    logger.info("=== DOCUMENT TYPE EXTRACTION END ===")
                    return []
               
                # Validate that returned IDs are actually available
                valid_ids = []
                for dtid in doc_type_ids:
                    if dtid in available_document_types.keys():
                        doc_name = available_document_types[dtid].get('name', 'Unknown')
                        valid_ids.append(dtid)
                        logger.info(f"  ✓ Valid document type ID: {dtid} -> '{doc_name}'")
                    else:
                        logger.warning(f"  ✗ Invalid document type ID: {dtid} (not found in available types)")
               
                final_ids = valid_ids  # No artificial limit
                logger.info(f"Final document type IDs: {final_ids}")
                logger.info("=== END DOCUMENT TYPE MATCHES ANALYSIS ===")
                logger.info("=== DOCUMENT TYPE EXTRACTION END ===")
                return final_ids
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                result = self._fallback_document_extraction(query, available_document_types)
                logger.info(f"Fallback extraction result: {result}")
                logger.info("=== DOCUMENT TYPE EXTRACTION END ===")
                return result
               
        except Exception as e:
            logger.warning(f"Document type extraction failed: {e}")
            return self._fallback_document_extraction(query, available_document_types)
   
    def _extract_search_strategy(self, query: str, available_strategies: Optional[Dict] = None) -> str:
        """Extract search strategy using focused LLM call."""
        logger.info("=== SEARCH STRATEGY EXTRACTION START ===")
        logger.info(f"Query for search strategy extraction: '{query}'")
       
        try:
            strategies_list = list(available_strategies.keys()) if available_strategies else ["HYBRID_PARALLEL", "SEMANTIC_ONLY", "KEYWORD_ONLY"]
           
            logger.info(f"Available strategies: {strategies_list}")
           
            prompt = f"""
You are an expert search-strategy classifier.  
Your job is to select exactly one strategy from the list below based ONLY on the query.

Available Strategies: {', '.join(strategies_list)}

Decision Rules (follow in order):
1. DEFAULT → Choose "HYBRID_PARALLEL" unless there is a CLEAR and STRONG reason to choose another strategy.
2. Choose "KEYWORD_ONLY" ONLY when the user explicitly wants:
   - exact phrase matching,
   - literal text search,
   - quotes, operators, IDs, codes, or numbers.
3. Choose "SEMANTIC_ONLY" ONLY when the user asks for:
   - conceptual understanding,
   - thematic similarity,
   - idea-based lookup (not exact terms).
4. For all other situations (mixed, unclear, broad queries) return "HYBRID_PARALLEL".

Important:
- Return ONLY the strategy name as plain text, nothing else.
- Do NOT explain your reasoning.

Query: "{query}"
"""

            logger.info("=== SEARCH STRATEGY EXTRACTION PROMPT ===")
            logger.info(f"Prompt: {prompt}")
            logger.info("=== END SEARCH STRATEGY EXTRACTION PROMPT ===")

            response = self._make_llm_call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
           
            logger.info("=== SEARCH STRATEGY EXTRACTION LLM RESPONSE ===")
            logger.info(f"Raw LLM Response: {response}")
           
            content = response["choices"][0]["message"]["content"].strip().replace('"', '')
            logger.info(f"Content extracted from response: '{content}'")
            logger.info("=== END SEARCH STRATEGY EXTRACTION LLM RESPONSE ===")
           
            # Validate strategy
            if content in strategies_list:
                logger.info(f"Strategy '{content}' is valid - using it")
                logger.info("=== SEARCH STRATEGY EXTRACTION END ===")
                return content
            else:
                logger.warning(f"Strategy '{content}' not in available strategies, defaulting to HYBRID_PARALLEL")
                logger.info("=== SEARCH STRATEGY EXTRACTION END ===")
                return "HYBRID_PARALLEL"
               
        except Exception as e:
            logger.warning(f"Search strategy extraction failed: {e}")
            return "HYBRID_PARALLEL"

    def _extract_semantic_query(self, query: str) -> str:
        """
        Extract and optimize semantic query using focused LLM call.
        Preserves key regulatory, project, proponent, location, and environmental terms.
        """
        logger.info("=== SEMANTIC QUERY EXTRACTION START ===")
        logger.info(f"Original query for semantic optimization: '{query}'")
       
        try:
            prompt = f"""
You are an expert in environmental assessment and regulatory document search.
Your task is to extract the **core semantic search concepts** from the user's query so that it can be used to retrieve relevant projects or documents from the Environmental Assessment Office (EAO) system.

### Instructions:
- Focus on **main subject terms**: project names, proponents, locations, environmental topics, condition numbers, certificates, permits, approvals.
- Remove conversational or filler text like "can you get me", "show me", "reports for", etc.
- Keep the query concise (2-10 key terms maximum).
- Preserve important terms: project names, locations, environmental terms (e.g., “air quality”, “pipeline”, “marine port”, “transmission lines”), regulatory terms (e.g., “Schedule B”, “condition 1”, “certificate”, “permit”, “approval”).
- Avoid generic words like "document", "report", unless part of a specific entity name.
- If query mentions a **condition or schedule**, preserve both numbers and references.
- If query mentions a **facility or project type**, preserve it.
- If query mentions a **region or place**, keep it intact.
- Use synonyms and related terms where appropriate to retain meaning.

### Examples:
- "certificate for marine port facilities near Lower Mainland" → "certificate marine port facilities Lower Mainland"
- "get me condition 1 from Schedule B for Babkirk Secure Landfill" → "condition 1 Schedule B Babkirk Secure Landfill"
- "all permits issued to Tilbury Jetty LP for marine expansion" → "permit Tilbury Jetty LP marine expansion"
- "environmental assessment for Roberts Bank Terminal 2" → "environmental assessment Roberts Bank Terminal 2"
- "pipeline transmission line impacts on air quality in Lower Mainland" → "pipeline transmission line air quality Lower Mainland"
- "letters mentioning Nooaitch Indian Band" → "Nooaitch Indian Band"
- "reports on hydroelectric projects by BC Hydro" → "hydroelectric BC Hydro"
- "status of condition 3 for Wapiti Power Development" → "condition 3 Wapiti Power Development"
- "approvals for LNG storage facility near Squamish & Gibsons" → "approval LNG storage Squamish Gibsons"
- "mitigation plans for air emissions at Waneta Generating Station Upgrade" → "mitigation air emissions Waneta Generating Station Upgrade"

Original Query: "{query}"

Return ONLY the optimized semantic query (no quotes, no explanation).
"""

            logger.info("=== SEMANTIC QUERY EXTRACTION PROMPT ===")
            logger.info(f"Prompt: {prompt}")
            logger.info("=== END SEMANTIC QUERY EXTRACTION PROMPT ===")

            response = self._make_llm_call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
           
            logger.info("=== SEMANTIC QUERY EXTRACTION LLM RESPONSE ===")
            logger.info(f"Raw LLM Response: {response}")
           
            content = response["choices"][0]["message"]["content"].strip().replace('"', '')
            logger.info(f"Content extracted from response: '{content}'")
            logger.info("=== END SEMANTIC QUERY EXTRACTION LLM RESPONSE ===")
           
            # Basic validation - should be shorter and meaningful
            if len(content) > 0 and len(content) < len(query) * 1.5:
                logger.info(f"Semantic query optimization successful: '{query}' -> '{content}'")
                logger.info("=== SEMANTIC QUERY EXTRACTION END ===")
                return content
            else:
                logger.warning(f"Semantic query validation failed - content too long or empty, using original query")
                logger.info("=== SEMANTIC QUERY EXTRACTION END ===")
                return query
               
        except Exception as e:
            logger.warning(f"Semantic query extraction failed: {e}")
            logger.info("=== SEMANTIC QUERY EXTRACTION END ===")
            return query
   
    def _fallback_project_extraction(self, query: str, available_projects: Union[Dict, List]) -> List[str]:
        """Enhanced fallback project extraction with strict name matching and similarity scoring.

        Uses a scoring system that prioritizes:
        1. Exact project name matches (highest score)
        2. Distinctive word matches (project-specific identifiers)
        3. Penalizes matches on generic terms only
        """
        query_lower = query.lower()
        scored_projects = []

        # Common generic terms that should not drive matching
        # Includes industry terms, geographic regions, and generic project words
        generic_terms = {
            # Geographic/terrain features
            'mountain', 'river', 'creek', 'lake', 'park', 'resort', 'wind', 'reservoir',
            'island', 'valley', 'coast', 'coastal', 'bay', 'inlet', 'sound', 'strait',
            # Geographic regions (should not match on location alone)
            'mainland', 'lower', 'upper', 'interior', 'northern', 'southern', 'eastern', 'western',
            'north', 'south', 'east', 'west', 'central', 'vancouver', 'victoria', 'bc', 'british', 'columbia',
            # Industry terms (but NOT project-distinctive words like 'marine', 'tilbury', etc.)
            'project', 'mine', 'gold', 'copper', 'silver', 'coal', 'gas', 'oil', 'lng',
            'power', 'energy', 'terminal', 'port', 'pipeline', 'transmission', 'line',
            'facility', 'plant', 'station', 'expansion', 'upgrade', 'phase', 'development',
            # Additional generic terms
            'clean', 'hydro', 'electric', 'solar', 'thermal', 'nuclear',
            'storage', 'processing', 'refinery', 'smelter', 'mill', 'quarry',
            # Query terms that shouldn't drive matching
            'impact', 'impacts', 'effect', 'effects', 'salmon', 'fish', 'water', 'air',
            'environment', 'environmental', 'assessment', 'near', 'due', 'ports'
        }
        stop_words = {'the', 'and', 'or', 'of', 'in', 'at', 'to', 'for', 'with', 'by', 'a', 'an', 'get', 'me', 'show', 'find'}

        # Convert available_projects to consistent format
        projects_iter = []
        if isinstance(available_projects, dict):
            projects_iter = [(name, pid) for name, pid in available_projects.items()]
        elif isinstance(available_projects, list):
            for proj in available_projects:
                if isinstance(proj, dict):
                    projects_iter.append((proj.get('project_name', ''), proj.get('project_id', '')))

        for project_name, project_id in projects_iter:
            if not project_name or not project_id:
                continue

            project_name_lower = project_name.lower()
            score = 0.0
            match_reason = []

            # Priority 1: Exact full project name in query (highest score)
            if project_name_lower in query_lower:
                score = 1.0
                match_reason.append("exact_name_match")
            # Priority 2: Query contained in project name
            elif query_lower in project_name_lower and len(query_lower) > 5:
                score = 0.9
                match_reason.append("query_in_name")
            else:
                # Priority 3: Word-by-word matching with scoring
                project_words = set(project_name_lower.split())
                query_words = set(query_lower.split())

                # Remove stop words from both
                project_words_clean = project_words - stop_words
                query_words_clean = query_words - stop_words

                # Separate distinctive vs generic words
                distinctive_project_words = project_words_clean - generic_terms
                distinctive_query_words = query_words_clean - generic_terms

                generic_project_words = project_words_clean & generic_terms
                generic_query_words = query_words_clean & generic_terms

                # Calculate matches
                distinctive_matches = distinctive_project_words & distinctive_query_words
                generic_matches = generic_project_words & generic_query_words

                # Score based on distinctive matches (these are the key identifiers)
                if distinctive_matches:
                    # Calculate what percentage of distinctive project words matched
                    if distinctive_project_words:
                        distinctive_coverage = len(distinctive_matches) / len(distinctive_project_words)
                        score = 0.5 + (distinctive_coverage * 0.4)  # Range: 0.5 - 0.9
                        match_reason.append(f"distinctive_match:{distinctive_matches}")

                    # Bonus if generic terms also match (confirms context)
                    if generic_matches:
                        score += 0.05
                        match_reason.append(f"generic_support:{generic_matches}")
                elif generic_matches:
                    # ONLY generic matches - this is weak and often wrong
                    # Only score if it's a very short query targeting a type
                    if len(query_words_clean) <= 2:
                        score = 0.2  # Very low score
                        match_reason.append(f"generic_only:{generic_matches}")
                    # Otherwise, don't include - too likely to be wrong match

                # Check for substring matches of distinctive words (e.g., "Cariboo" in "Cariboo Gold")
                for dw in distinctive_query_words:
                    if len(dw) >= 4:  # Only check meaningful words
                        for pw in distinctive_project_words:
                            if dw in pw or pw in dw:
                                if score < 0.6:
                                    score = 0.6
                                    match_reason.append(f"substring_match:{dw}->{pw}")

                # Check for multi-word phrases in query that match project name
                # E.g., "cariboo gold" should strongly match "Cariboo Gold Project"
                for ngram_len in range(2, 5):  # Check 2-4 word ngrams
                    query_words_list = query_lower.split()
                    for i in range(len(query_words_list) - ngram_len + 1):
                        ngram = ' '.join(query_words_list[i:i + ngram_len])
                        if len(ngram) >= 5 and ngram in project_name_lower:
                            # Check if ngram is composed only of generic/location terms
                            ngram_words = set(ngram.split())
                            ngram_distinctive_words = ngram_words - generic_terms - stop_words

                            # If ngram has no distinctive words, reduce its weight significantly
                            # E.g., "lower mainland" matches many projects but isn't project-specific
                            if not ngram_distinctive_words:
                                # Generic-only ngram - much lower score
                                new_score = 0.4  # Weak signal
                                if new_score > score:
                                    score = new_score
                                    match_reason.append(f"ngram_match_generic:{ngram}")
                            else:
                                # Strong signal: multi-word phrase with distinctive words
                                coverage = len(ngram) / len(project_name_lower)
                                if coverage >= 0.5:
                                    new_score = 0.95 + (coverage * 0.05)
                                elif coverage >= 0.3:
                                    new_score = 0.85 + (coverage * 0.1)
                                else:
                                    new_score = 0.7 + (coverage * 0.1)
                                if new_score > score:
                                    score = new_score
                                    match_reason.append(f"ngram_match:{ngram}")

            # Only include projects with meaningful scores
            # Lowered threshold from 0.5 to 0.33 to handle projects with multiple distinctive words
            # e.g., "Pretium Brucejack" where only "brucejack" matches (score = 0.5)
            if score >= 0.33:
                scored_projects.append((project_id, project_name, score, match_reason))
                logger.debug(f"Fallback match: '{project_name}' score={score:.2f} reasons={match_reason}")

        # Sort by score descending
        scored_projects.sort(key=lambda x: x[2], reverse=True)

        # Log the scoring for debugging
        if scored_projects:
            logger.info(f"Fallback project scoring for query '{query}':")
            for pid, pname, score, reasons in scored_projects[:5]:
                logger.info(f"  - '{pname}' (ID: {pid}) Score: {score:.2f} Reasons: {reasons}")

        # Return top 3 project IDs
        return [p[0] for p in scored_projects[:3]]
   
    def _fallback_document_extraction(self, query: str, available_document_types: Dict) -> List[str]:
        """Fallback document type extraction using comprehensive alias matching."""
        logger.info("=== FALLBACK DOCUMENT TYPE EXTRACTION ===")
        query_lower = query.lower()
       
        # Check if this is a general information query that shouldn't filter document types
        general_query_indicators = [
            "who is", "what is", "when was", "where is", "how much", "how many",
            "main proponent", "project status", "tell me about", "information on",
            "what are the impacts", "what consultation", "all documents"
        ]
       
        if any(indicator in query_lower for indicator in general_query_indicators):
            logger.info(f"Query contains general information indicators - returning empty array")
            logger.info("=== END FALLBACK DOCUMENT TYPE EXTRACTION ===")
            return []
       
        # Only do text matching for queries that explicitly mention document types
        document_type_keywords = [
            "letter", "report", "memo", "correspondence", "transcript", "assessment", "presentation",
            "schedule", "condition", "certificate", "eac", "order", "package", "application", "agreement"
        ]
        if not any(keyword in query_lower for keyword in document_type_keywords):
            logger.info(f"Query does not mention specific document types - returning empty array")
            logger.info("=== END FALLBACK DOCUMENT TYPE EXTRACTION ===")
            return []
       
        matched_types = []
       
        for doc_id, doc_data in available_document_types.items():
            name = doc_data.get('name', '').lower()
            aliases = [alias.lower() for alias in doc_data.get('aliases', [])]
           
            # Check name
            if name and any(word in query_lower for word in name.split() if len(word) > 3):
                matched_types.append(doc_id)
                continue
           
            # Check all aliases - look for any that contain query terms or vice versa
            for alias in aliases:
                if alias in query_lower or any(term in alias for term in query_lower.split() if len(term) > 3):
                    matched_types.append(doc_id)
                    break
       
        logger.info(f"Fallback matched document types: {matched_types}")
        logger.info("=== END FALLBACK DOCUMENT TYPE EXTRACTION ===")
        return matched_types  # Return all matched types
   
   
    def _fallback_extraction(self, query: str, available_projects: Optional[Dict] = None,
                           available_document_types: Optional[Dict] = None,
                           available_strategies: Optional[Dict] = None,
                           supplied_project_ids: Optional[List[str]] = None,
                           supplied_document_type_ids: Optional[List[str]] = None,
                           supplied_search_strategy: Optional[str] = None) -> Dict[str, Any]:
        """Complete fallback extraction."""
        return {
            "project_ids": supplied_project_ids or self._fallback_project_extraction(query, available_projects or {}),
            "document_type_ids": supplied_document_type_ids or self._fallback_document_extraction(query, available_document_types or {}),
            "search_strategy": supplied_search_strategy or "HYBRID_PARALLEL",
            "semantic_query": query,
            "confidence": 0.2,
            "extraction_sources": {
                "project_ids": "supplied" if supplied_project_ids else "fallback",
                "document_type_ids": "supplied" if supplied_document_type_ids else "fallback",
                "search_strategy": "supplied" if supplied_search_strategy else "fallback",
                "semantic_query": "fallback"
            }
        }
   
    def _extract_temporal_and_location_parameters(self, query: str, user_location: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract temporal and location parameters (location, project_status, years) using LLM reasoning.
       
        This method intelligently determines whether to extract location based on query intent:
        - Extracts location for geographic search queries ("projects in Vancouver", "near me")
        - Skips location extraction for specific project queries ("For the X project...")
       
        Args:
            query: The search query to analyze
            user_location: User's physical location from browser (used only for "near me" queries)
           
        Returns:
            Dict containing temporal and location parameters: location, project_status, years
        """
        try:
            from datetime import datetime
            current_year = datetime.now().year
           
            # Check if this query is about a specific named project (should NOT extract location)
            query_lower = query.lower()
            specific_project_indicators = [
                'for the', 'about the', 'regarding the', 'concerning the',
                'project i want', 'project that', 'project,', 'in the project'
            ]
           
            is_specific_project_query = any(indicator in query_lower for indicator in specific_project_indicators)
           
            if is_specific_project_query:
                logger.info(f"Query targets specific project - skipping location extraction: '{query}'")
                # Only extract temporal parameters, not location
                prompt = f"""You are a temporal parameter extraction specialist. This query is about a SPECIFIC project, so do NOT extract location parameters. Only extract temporal parameters.

QUERY TO ANALYZE: "{query}"

Extract ONLY:
1. PROJECT STATUS: Look for project lifecycle indicators: "active", "completed", "recent", "ongoing", "historical", "current", "past", "future"
2. TEMPORAL/YEARS: Extract specific years, year ranges, or relative time expressions
   - Current year is {current_year}
   - Map relative terms to concrete years

Respond with ONLY a JSON object:
{{
    "location": null,
    "project_status": null_or_status_string,
    "years": [],
    "reasoning": "specific project query - no location extraction needed",
    "confidence": 0.0_to_1.0
}}
"""
            else:
                # Geographic search query - extract location parameters
                prompt = f"""You are a temporal and geographic parameter extraction specialist. Analyze the following search query to extract:

1. LOCATION PARAMETERS (MUST BE STRING FORMAT):
   - Look for geographic references, location names, or phrases like "near me", "local", "my area"
   - ALWAYS return location as a STRING in format "City, Region" or "Region" (e.g., "Langford, BC", "Peace River region", "Vancouver")
   - If user has provided location data and query contains "near me" or location references, extract the city/region from user location as a STRING
   - If query mentions specific location names, extract those as a STRING
   - DO NOT return location as a JSON object - ONLY strings like "Vancouver, BC" or "Peace River region"

2. PROJECT STATUS:
   - Look for project lifecycle indicators: "active", "completed", "recent", "ongoing", "historical", "current", "past", "future"
   - Map temporal words to appropriate status (e.g., "recently" -> "recent", "ongoing projects" -> "active")

3. TEMPORAL/YEARS:
   - Extract specific years, year ranges, or relative time expressions
   - Current year is {current_year}
   - Map relative terms to concrete years:
     * "recently", "lately" -> last 2-3 years including current [{current_year-2}, {current_year-1}, {current_year}]
     * "last N years" -> calculate range from current year
     * "this year", "current year" -> [{current_year}]
     * "past year" -> [{current_year-1}, {current_year}]
     * "since YYYY" -> range from specified year to current
     * Specific years or ranges -> extract as provided

USER LOCATION CONTEXT: {user_location if user_location else "No user location provided"}

QUERY TO ANALYZE: "{query}"

Respond with ONLY a JSON object in this exact format:
{{
    "location": null_or_string_like_"Vancouver_BC"_or_"Peace_River_region",
    "project_status": null_or_status_string,
    "years": [],
    "reasoning": "explanation of extraction logic",
    "confidence": 0.0_to_1.0
}}

EXAMPLES:
Query: "Show me recent projects near me"
User Location: {{"city": "Langford", "region": "British Columbia"}}
Response: {{"location": "Langford, British Columbia", "project_status": "recent", "years": [{current_year-2}, {current_year-1}, {current_year}], "reasoning": "User wants recent projects in their location - extracted city and region as string", "confidence": 0.9}}

Query: "Environmental reports from 2020-2022 in Peace River region"
Response: {{"location": "Peace River region", "project_status": null, "years": [2020, 2021, 2022], "reasoning": "Specific location and year range provided", "confidence": 0.95}}

Query: "Projects in Vancouver"
Response: {{"location": "Vancouver", "project_status": null, "years": [], "reasoning": "Query explicitly mentions Vancouver", "confidence": 0.95}}"""

            logger.info("=== TEMPORAL EXTRACTION PROMPT ===")
            logger.info(f"Prompt: {prompt}")
            logger.info("=== END TEMPORAL EXTRACTION PROMPT ===")

            response = self._make_llm_call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
           
            content = response["choices"][0]["message"]["content"].strip()
            logger.info(f"Raw temporal extraction response: {content}")
           
            # Parse JSON response
            result = json.loads(content)
           
            # Validate and clean the response
            temporal_params = {
                "location": result.get("location"),
                "project_status": result.get("project_status"),
                "years": result.get("years", []),
                "reasoning": result.get("reasoning", ""),
                "confidence": float(result.get("confidence", 0.0))
            }
           
            logger.info(f"Extracted temporal parameters: {temporal_params}")
            return temporal_params
           
        except Exception as e:
            logger.error(f"Error extracting temporal parameters: {e}")
            return {
                "location": None,
                "project_status": None,
                "years": [],
                "reasoning": f"LLM extraction failed: {e}",
                "confidence": 0.0
            }

    def _convert_projects_array_to_dict(self, projects_array: Optional[List]) -> Dict:
        """Convert projects array from VectorSearchClient.get_projects_list() to dict format.
       
        Args:
            projects_array: Array from VectorSearchClient.get_projects_list()
           
        Returns:
            Dict in format {project_name: project_id} for internal processing
        """
        if not projects_array:
            return {}
           
        projects_dict = {}
        if isinstance(projects_array, list):
            for project in projects_array:
                if isinstance(project, dict) and 'project_id' in project and 'project_name' in project:
                    projects_dict[project['project_name']] = project['project_id']
       
        logger.info(f"Converted {len(projects_array) if projects_array else 0} projects array to {len(projects_dict)} projects dict")
        return projects_dict
   
    def _convert_document_types_array_to_dict(self, document_types_array: Optional[List]) -> Dict:
        """Convert document types array from VectorSearchClient.get_document_types() to dict format.
       
        Args:
            document_types_array: Array from VectorSearchClient.get_document_types()
           
        Returns:
            Dict in format {doc_type_id: {name, aliases, act}} for internal processing
        """
        if not document_types_array:
            return {}
           
        document_types_dict = {}
        if isinstance(document_types_array, list):
            for doc_type in document_types_array:
                if isinstance(doc_type, dict) and 'document_type_id' in doc_type:
                    doc_type_id = doc_type['document_type_id']
                    document_types_dict[doc_type_id] = {
                        'name': doc_type.get('document_type_name', ''),
                        'aliases': doc_type.get('aliases', []),
                        'act': doc_type.get('act', '')
                    }
       
        logger.info(f"Converted {len(document_types_array) if document_types_array else 0} document types array to {len(document_types_dict)} document types dict")
        return document_types_dict

    def _make_llm_call(self, messages: List[Dict], temperature: float = 0.1) -> Dict[str, Any]:
        """Make LLM call - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _make_llm_call method")