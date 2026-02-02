"""Enhanced Agent with Planning, Self-Evaluation, and Refinement Capabilities.

Phase 1 Enhancement: Adds better reasoning, self-evaluation, and refinement loops
to the existing agent architecture.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of self-evaluation."""
    quality_score: float  # 0.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    coverage_score: float  # 0.0 to 1.0
    confidence_score: float  # 0.0 to 1.0
    needs_refinement: bool
    reasoning: str
    suggestions: List[str]


@dataclass
class PlanningResult:
    """Result of execution planning."""
    execution_plan: List[Dict[str, Any]]
    reasoning: str
    expected_outcomes: List[str]
    estimated_complexity: str  # "simple", "moderate", "complex"
    risk_factors: List[str]


class EnhancedAgentPlanner:
    """Enhanced planning module with chain-of-thought reasoning."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def create_enhanced_plan(self, query: str, reason: str, context: Dict[str, Any]) -> PlanningResult:
        """Create an execution plan with explicit reasoning and analysis.

        Args:
            query: The user query
            reason: Why agent mode was selected
            context: Current execution context

        Returns:
            PlanningResult with plan and reasoning
        """
        logger.info("🧠 ENHANCED PLANNER: Starting chain-of-thought planning...")

        # Step 1: Analyze query complexity and requirements
        analysis = self._analyze_query(query, context)
        logger.info(f"🧠 ENHANCED PLANNER: Query analysis complete - complexity: {analysis['complexity']}")

        # Step 2: Identify key entities and concepts
        entities = self._identify_entities(query)
        logger.info(f"🧠 ENHANCED PLANNER: Identified {len(entities.get('projects', []))} projects, "
                   f"{len(entities.get('document_types', []))} document types")

        # Step 3: Generate execution strategy
        strategy = self._generate_execution_strategy(query, analysis, entities, context)
        logger.info(f"🧠 ENHANCED PLANNER: Generated {strategy['search_count']} search strategies")

        # Step 4: Create detailed execution plan
        execution_plan = self._build_execution_plan(query, strategy, entities, context)

        planning_result = PlanningResult(
            execution_plan=execution_plan,
            reasoning=strategy['reasoning'],
            expected_outcomes=strategy['expected_outcomes'],
            estimated_complexity=analysis['complexity'],
            risk_factors=analysis.get('risk_factors', [])
        )

        logger.info(f"🧠 ENHANCED PLANNER: Planning complete - {len(execution_plan)} steps, "
                   f"complexity: {planning_result.estimated_complexity}")

        return planning_result

    def _analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query to understand complexity and requirements.

        Uses chain-of-thought prompting for deep analysis.
        """
        analysis_prompt = f"""Analyze this search query in detail:

Query: "{query}"

Perform a step-by-step analysis:

1. INTENT CLASSIFICATION
   - What is the user trying to find?
   - Is this a factual question, exploratory search, or comparison?
   - What type of answer would satisfy this query?

2. COMPLEXITY ASSESSMENT
   - Simple: Single concept, straightforward retrieval
   - Moderate: Multiple concepts, needs 2-3 searches
   - Complex: Multi-faceted, requires comprehensive analysis

3. INFORMATION REQUIREMENTS
   - What specific information is needed?
   - What entities are mentioned (projects, locations, document types)?
   - What temporal context is implied (recent, historical, specific years)?

4. SEARCH STRATEGY
   - How many different search angles are needed?
   - What semantic variations should be explored?
   - What filters should be applied?

5. RISK FACTORS
   - Are there ambiguous terms that might cause issues?
   - Is the query too broad or too narrow?
   - Are there potential data gaps?

Return analysis as JSON:
{{
    "intent": "factual_question" | "exploratory_search" | "comparison" | "aggregation",
    "complexity": "simple" | "moderate" | "complex",
    "key_concepts": ["concept1", "concept2"],
    "search_angles": ["angle1", "angle2"],
    "temporal_context": {{"type": "recent|historical|specific", "years": []}},
    "geographic_context": {{"mentioned": true/false, "locations": []}},
    "risk_factors": ["risk1", "risk2"],
    "reasoning": "step-by-step explanation of the analysis"
}}"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert query analyzer. Provide detailed, structured analysis in JSON format."},
                {"role": "user", "content": analysis_prompt}
            ]

            response = self.llm_client.chat_completion(messages, temperature=0.2, max_tokens=1000)

            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"].strip()
                content = self._clean_json_response(content)
                analysis = json.loads(content)
                logger.info(f"🧠 QUERY ANALYSIS: {analysis.get('reasoning', 'N/A')[:200]}...")
                return analysis

        except Exception as e:
            logger.error(f"🧠 QUERY ANALYSIS: Failed with error: {e}")

        # Fallback to simple heuristic analysis
        return {
            "intent": "exploratory_search",
            "complexity": "moderate",
            "key_concepts": [query],
            "search_angles": [query],
            "temporal_context": {"type": "any", "years": []},
            "geographic_context": {"mentioned": False, "locations": []},
            "risk_factors": ["Analysis unavailable - using fallback"],
            "reasoning": "Heuristic analysis due to LLM analysis failure"
        }

    def _identify_entities(self, query: str) -> Dict[str, List[str]]:
        """Identify key entities in the query.

        Args:
            query: User query

        Returns:
            Dictionary of identified entities
        """
        entity_prompt = f"""Extract entities from this query:

Query: "{query}"

Identify and extract:
1. Project names (e.g., "Air Liquide", "LNG Canada")
2. Document types (e.g., "assessment report", "certificate", "public comment")
3. Locations (e.g., "Vancouver", "Peace River", "Northern BC")
4. Organizations (e.g., "First Nations", "proponent", "stakeholder")
5. Topics (e.g., "water quality", "consultation", "impacts")
6. Temporal markers (e.g., "recent", "2023", "last year")

Return as JSON:
{{
    "projects": ["project1"],
    "document_types": ["type1"],
    "locations": ["location1"],
    "organizations": ["org1"],
    "topics": ["topic1"],
    "temporal": ["marker1"]
}}"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert entity extractor. Return only JSON, no explanations."},
                {"role": "user", "content": entity_prompt}
            ]

            response = self.llm_client.chat_completion(messages, temperature=0.1, max_tokens=500)

            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"].strip()
                content = self._clean_json_response(content)
                entities = json.loads(content)
                return entities

        except Exception as e:
            logger.error(f"🧠 ENTITY EXTRACTION: Failed with error: {e}")

        # Fallback: return empty entities
        return {
            "projects": [],
            "document_types": [],
            "locations": [],
            "organizations": [],
            "topics": [],
            "temporal": []
        }

    def _generate_execution_strategy(self, query: str, analysis: Dict[str, Any],
                                    entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution strategy based on analysis.

        Args:
            query: User query
            analysis: Query analysis result
            entities: Extracted entities
            context: Execution context

        Returns:
            Execution strategy
        """
        complexity = analysis.get('complexity', 'moderate')
        search_angles = analysis.get('search_angles', [query])

        # Determine number of searches based on complexity
        if complexity == 'simple':
            search_count = 1
        elif complexity == 'moderate':
            search_count = min(3, max(2, len(search_angles)))
        else:  # complex
            search_count = min(3, max(3, len(search_angles)))

        strategy = {
            "search_count": search_count,
            "search_strategies": self._create_search_strategies(query, search_angles, entities, search_count),
            "expected_outcomes": [
                f"Find documents related to {angle}" for angle in search_angles[:search_count]
            ],
            "reasoning": f"Query complexity: {complexity}. Will execute {search_count} searches covering: {', '.join(search_angles[:search_count])}"
        }

        return strategy

    def _create_search_strategies(self, query: str, angles: List[str],
                                 entities: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """Create specific search strategies for each angle.

        Args:
            query: Original query
            angles: Search angles to explore
            entities: Extracted entities
            count: Number of searches to create

        Returns:
            List of search strategies
        """
        strategies = []

        # Create diverse search queries based on angles and entities
        topics = entities.get('topics', [])
        projects = entities.get('projects', [])
        doc_types = entities.get('document_types', [])

        for i in range(count):
            if i < len(angles):
                # Use the identified angle
                search_query = angles[i]
            else:
                # Create variations by combining topics and entities
                if topics and i < len(topics) + len(angles):
                    search_query = f"{topics[i - len(angles)]} {query}"
                else:
                    search_query = query

            strategy = {
                "query": search_query,
                "parameters": {},
                "reasoning": f"Search angle {i+1}: {search_query}"
            }

            # Add location filter if mentioned
            if entities.get('locations'):
                strategy["parameters"]["location"] = entities['locations'][0]

            # Add temporal filter if mentioned
            if entities.get('temporal'):
                temporal = entities['temporal'][0].lower()
                if 'recent' in temporal or 'latest' in temporal:
                    strategy["parameters"]["project_status"] = "recent"
                    strategy["parameters"]["years"] = [2023, 2024]

            strategies.append(strategy)

        return strategies

    def _build_execution_plan(self, query: str, strategy: Dict[str, Any],
                            entities: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build the final execution plan.

        Args:
            query: User query
            strategy: Execution strategy
            entities: Extracted entities
            context: Execution context

        Returns:
            List of execution steps
        """
        plan = []

        # Step 1: Validation
        plan.append({
            "step_name": "validate_query",
            "tool": "validate_query_relevance",
            "parameters": {},
            "reasoning": "Validate query relevance to EAO scope"
        })

        # Step 2: Optional list retrieval (only if entities mentioned but not in context)
        if entities.get('projects') and not context.get('project_name_to_id_mapping'):
            plan.append({
                "step_name": "get_projects_list",
                "tool": "get_projects_list",
                "parameters": {},
                "reasoning": "Retrieve project list for ID mapping"
            })

        if entities.get('document_types') and not context.get('document_type_name_to_id_mapping'):
            plan.append({
                "step_name": "get_document_types",
                "tool": "get_document_types",
                "parameters": {},
                "reasoning": "Retrieve document types for filtering"
            })

        # Step 3: Search strategies
        filter_steps = []
        for i, search_strategy in enumerate(strategy['search_strategies']):
            step_name = f"search_strategy_{i+1}"
            plan.append({
                "step_name": step_name,
                "tool": "search",
                "parameters": search_strategy['parameters'] | {"query": search_strategy['query']},
                "reasoning": search_strategy['reasoning']
            })

            # Add validation step for each search
            filter_step_name = f"filter_{step_name}"
            plan.append({
                "step_name": filter_step_name,
                "tool": "validate_chunks_relevance",
                "parameters": {
                    "search_results": f"results_from_{step_name}",
                    "step_name": step_name
                },
                "reasoning": f"Filter {step_name} results for relevance"
            })
            filter_steps.append(filter_step_name)

        # Step 4: Verify and reduce
        plan.append({
            "step_name": "verify_reduce",
            "tool": "verify_reduce",
            "parameters": {"filter_steps": filter_steps},
            "reasoning": "Collect all validated chunks from filter steps"
        })

        # Step 5: Consolidation
        plan.append({
            "step_name": "consolidate_results",
            "tool": "consolidate_results",
            "parameters": {"merge_strategy": "deduplicate"},
            "reasoning": "Merge and deduplicate all verified results"
        })

        # Step 6: Summarization
        plan.append({
            "step_name": "summarize_results",
            "tool": "summarize_results",
            "parameters": {"include_metadata": True},
            "reasoning": "Generate comprehensive summary of findings"
        })

        return plan

    def _clean_json_response(self, content: str) -> str:
        """Clean LLM response to extract JSON."""
        content = content.replace("```json", "").replace("```", "").strip()

        while content.startswith('```'):
            newline_pos = content.find('\n')
            if newline_pos != -1:
                content = content[newline_pos + 1:].strip()
            else:
                content = content[3:].strip()

        while content.endswith('```'):
            content = content[:-3].strip()

        return content


class AgentSelfEvaluator:
    """Self-evaluation module for assessing search result quality."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def evaluate_results(self, query: str, results: Dict[str, Any],
                        execution_metrics: Dict[str, Any]) -> EvaluationResult:
        """Evaluate the quality of search results.

        Args:
            query: Original user query
            results: Search results to evaluate
            execution_metrics: Metrics from execution

        Returns:
            EvaluationResult with quality assessment
        """
        logger.info("📊 SELF-EVALUATION: Starting result quality assessment...")

        documents = results.get('documents', [])
        chunks = results.get('document_chunks', [])
        total_results = len(documents) + len(chunks)

        # Quick heuristic checks
        if total_results == 0:
            logger.warning("📊 SELF-EVALUATION: No results found - quality score: 0.0")
            return EvaluationResult(
                quality_score=0.0,
                relevance_score=0.0,
                coverage_score=0.0,
                confidence_score=0.0,
                needs_refinement=True,
                reasoning="No results found",
                suggestions=["Try broader search terms", "Remove restrictive filters", "Check if query is in scope"]
            )

        # Use LLM for deep evaluation
        evaluation = self._llm_evaluate_results(query, documents, chunks)

        # Calculate composite quality score
        quality_score = (
            evaluation.get('relevance_score', 0.5) * 0.4 +
            evaluation.get('coverage_score', 0.5) * 0.3 +
            evaluation.get('confidence_score', 0.5) * 0.3
        )

        needs_refinement = quality_score < 0.7 or total_results < 3

        result = EvaluationResult(
            quality_score=quality_score,
            relevance_score=evaluation.get('relevance_score', 0.5),
            coverage_score=evaluation.get('coverage_score', 0.5),
            confidence_score=evaluation.get('confidence_score', 0.5),
            needs_refinement=needs_refinement,
            reasoning=evaluation.get('reasoning', 'Evaluation complete'),
            suggestions=evaluation.get('suggestions', [])
        )

        logger.info(f"📊 SELF-EVALUATION: Quality score: {quality_score:.2f}, "
                   f"Needs refinement: {needs_refinement}")

        return result

    def _llm_evaluate_results(self, query: str, documents: List[Dict],
                             chunks: List[Dict]) -> Dict[str, Any]:
        """Use LLM to evaluate result quality.

        Args:
            query: User query
            documents: Retrieved documents
            chunks: Retrieved chunks

        Returns:
            Evaluation metrics and suggestions
        """
        # Prepare sample of results for evaluation
        sample_docs = documents[:3]
        sample_chunks = chunks[:5]

        evaluation_prompt = f"""Evaluate these search results for the given query:

Query: "{query}"

Documents found: {len(documents)}
Sample documents: {json.dumps([{{
    'name': d.get('document_name', 'Unknown'),
    'type': d.get('document_type', 'Unknown'),
    'relevance': d.get('similarity_score', 0)
}} for d in sample_docs], indent=2)}

Chunks found: {len(chunks)}
Sample chunks: {json.dumps([{{
    'content': c.get('content', '')[:200] + '...',
    'relevance': c.get('similarity_score', 0)
}} for c in sample_chunks], indent=2)}

Evaluate on these dimensions (0.0 to 1.0):

1. RELEVANCE: Do results actually address the query?
   - 1.0: Highly relevant, directly answers query
   - 0.5: Partially relevant, related but not direct
   - 0.0: Not relevant, off-topic

2. COVERAGE: Do results cover all aspects of the query?
   - 1.0: Comprehensive coverage of all query aspects
   - 0.5: Partial coverage, some aspects missing
   - 0.0: Poor coverage, major gaps

3. CONFIDENCE: How confident are you in these results?
   - 1.0: Very confident, high-quality evidence
   - 0.5: Moderately confident, some uncertainty
   - 0.0: Low confidence, questionable quality

4. SUGGESTIONS: What would improve these results?

Return as JSON:
{{
    "relevance_score": 0.0-1.0,
    "coverage_score": 0.0-1.0,
    "confidence_score": 0.0-1.0,
    "reasoning": "detailed explanation",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert search quality evaluator. Provide honest, critical assessment in JSON format."},
                {"role": "user", "content": evaluation_prompt}
            ]

            response = self.llm_client.chat_completion(messages, temperature=0.2, max_tokens=800)

            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"].strip()
                content = self._clean_json_response(content)
                evaluation = json.loads(content)
                logger.info(f"📊 LLM EVALUATION: {evaluation.get('reasoning', 'N/A')[:150]}...")
                return evaluation

        except Exception as e:
            logger.error(f"📊 LLM EVALUATION: Failed with error: {e}")

        # Fallback: heuristic evaluation
        return {
            "relevance_score": 0.6,
            "coverage_score": 0.6,
            "confidence_score": 0.5,
            "reasoning": "Heuristic evaluation - LLM evaluation unavailable",
            "strengths": [f"Found {len(documents) + len(chunks)} results"],
            "weaknesses": ["Unable to perform deep quality analysis"],
            "suggestions": ["Try refining search parameters"]
        }

    def _clean_json_response(self, content: str) -> str:
        """Clean LLM response to extract JSON."""
        content = content.replace("```json", "").replace("```", "").strip()

        while content.startswith('```'):
            newline_pos = content.find('\n')
            if newline_pos != -1:
                content = content[newline_pos + 1:].strip()
            else:
                content = content[3:].strip()

        while content.endswith('```'):
            content = content[:-3].strip()

        return content


class AgentRefiner:
    """Refinement module for improving low-quality results."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def create_refinement_plan(self, query: str, initial_results: Dict[str, Any],
                              evaluation: EvaluationResult) -> List[Dict[str, Any]]:
        """Create a plan to refine and improve results.

        Args:
            query: Original query
            initial_results: Initial search results
            evaluation: Evaluation of initial results

        Returns:
            List of refinement steps
        """
        logger.info("🔄 REFINEMENT: Creating refinement plan based on evaluation...")

        refinement_steps = []

        # Analyze weaknesses and create targeted refinement
        if evaluation.quality_score < 0.3:
            # Very poor results - try completely different approach
            logger.info("🔄 REFINEMENT: Quality very low - trying alternative search strategy")
            refinement_steps.extend(self._create_alternative_searches(query, evaluation))

        elif evaluation.quality_score < 0.7:
            # Moderate results - refine existing approach
            logger.info("🔄 REFINEMENT: Quality moderate - refining search parameters")
            refinement_steps.extend(self._create_parameter_refinements(query, evaluation))

        # Add expansion searches if coverage is low
        if evaluation.coverage_score < 0.6:
            logger.info("🔄 REFINEMENT: Coverage low - adding expansion searches")
            refinement_steps.extend(self._create_coverage_expansion(query, evaluation))

        logger.info(f"🔄 REFINEMENT: Created {len(refinement_steps)} refinement steps")

        return refinement_steps

    def _create_alternative_searches(self, query: str, evaluation: EvaluationResult) -> List[Dict[str, Any]]:
        """Create alternative search strategies for very poor results."""
        alternatives = []

        # Try broader search
        alternatives.append({
            "step_name": "refinement_broader_search",
            "tool": "search",
            "parameters": {
                "query": f"environmental assessment {query}",  # Add context
                "ranking": {"minScore": -10.0, "topN": 15}  # Lower threshold
            },
            "reasoning": "Broader search with lower relevance threshold"
        })

        # Try semantic variations
        alternatives.append({
            "step_name": "refinement_semantic_variation",
            "tool": "search",
            "parameters": {
                "query": query,
                "search_strategy": "SEMANTIC_ONLY"  # Force semantic search
            },
            "reasoning": "Semantic-only search for conceptual matches"
        })

        return alternatives

    def _create_parameter_refinements(self, query: str, evaluation: EvaluationResult) -> List[Dict[str, Any]]:
        """Refine search parameters based on evaluation."""
        refinements = []

        # Adjust ranking parameters
        refinements.append({
            "step_name": "refinement_adjusted_ranking",
            "tool": "search",
            "parameters": {
                "query": query,
                "ranking": {"minScore": -9.0, "topN": 12}
            },
            "reasoning": "Adjusted ranking to capture more relevant results"
        })

        return refinements

    def _create_coverage_expansion(self, query: str, evaluation: EvaluationResult) -> List[Dict[str, Any]]:
        """Expand coverage by trying different angles."""
        expansion = []

        # Use suggestions from evaluation if available
        for i, suggestion in enumerate(evaluation.suggestions[:2]):
            if "try" in suggestion.lower() or "search" in suggestion.lower():
                expansion.append({
                    "step_name": f"refinement_expansion_{i+1}",
                    "tool": "search",
                    "parameters": {"query": suggestion},
                    "reasoning": f"Coverage expansion: {suggestion}"
                })

        return expansion
