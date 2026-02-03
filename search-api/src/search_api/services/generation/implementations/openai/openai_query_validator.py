"""OpenAI query validator implementation."""

import json
import logging
from typing import Dict, Any, Optional
from .openai_client import OpenAIClient
from ...abstractions.query_validator import QueryValidator

logger = logging.getLogger(__name__)

# Knowledge base for generic EAO informational queries
EAO_KNOWLEDGE_BASE = """
The Environmental Assessment Office (EAO) is an independent office within the Government of British Columbia, Canada.

**What is the EAO?**
The Environmental Assessment Office (EAO) is responsible for conducting environmental assessments of major projects proposed in British Columbia. It operates under the Environmental Assessment Act (2018) and works to ensure that major projects are assessed for their potential environmental, economic, social, cultural, and health impacts.

**Key Functions:**
- Conducting environmental assessments of major projects (mining, energy, infrastructure, industrial)
- Issuing Environmental Assessment Certificates for approved projects
- Ensuring meaningful consultation with Indigenous nations
- Coordinating with federal and local governments on assessments
- Monitoring compliance with certificate conditions
- Managing amendments to existing certificates

**Types of Projects Assessed:**
- Mining projects (coal, metal, mineral, aggregate)
- Energy projects (LNG facilities, pipelines, power plants, transmission lines)
- Water management projects (dams, dykes, water diversion)
- Industrial projects (chemical plants, refineries)
- Transportation projects (ports, terminals, railways)
- Waste management projects (landfills, hazardous waste facilities)
- Resort/tourism developments over certain thresholds

**Environmental Assessment Process:**
1. **Early Engagement** - Proponent engages with Indigenous nations and the public
2. **Process Planning** - EAO determines scope and requirements
3. **Application Development** - Proponent prepares application with required studies
4. **Application Review** - Technical review and public comment period
5. **Referral to Ministers** - EAO recommendation to decision-makers
6. **Decision** - Ministers decide whether to issue certificate
7. **Post-Certificate** - Compliance monitoring and enforcement

**Key Documents:**
- Environmental Assessment Certificates (with schedules of conditions)
- Schedule A: Certified Project Description
- Schedule B: Conditions (requirements the proponent must meet)
- Application materials and supporting studies
- Consultation records with Indigenous nations
- Public comments and responses

**Contact Information:**
- Website: www.eao.gov.bc.ca
- Location: Victoria, British Columbia, Canada
"""


class OpenAIQueryValidator(QueryValidator):
    """OpenAI implementation of the query validator."""

    def __init__(self):
        """Initialize the OpenAI query validator."""
        self.client = OpenAIClient()

    def validate_query_relevance(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate if a query is relevant to EAO using OpenAI.

        Also classifies the query type to determine if it's:
        - generic_informational: General questions about EAO (e.g., "What is EAO?")
        - specific_search: Queries about specific projects/documents (e.g., "Schedule B for Cariboo Gold")
        - ambiguous: Could be either type

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
            - query_type: Type of query
            - generic_response: AI-generated response for generic queries

        Raises:
            Exception: If validation fails.
        """
        try:
            # Build the validation prompt
            prompt = self._build_validation_prompt(context)

            # Define the function schema for structured output
            tools = [{
                "type": "function",
                "function": {
                    "name": "validate_query_relevance",
                    "description": "Validate if a query is relevant to EAO and classify its type",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_relevant": {
                                "type": "boolean",
                                "description": "Whether the query is relevant to EAO scope"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence score for the relevance assessment"
                            },
                            "query_type": {
                                "type": "string",
                                "enum": ["generic_informational", "specific_search", "aggregation_summary", "ambiguous", "broad_category_search"],
                                "description": "Type of query: generic_informational (general questions about EAO), specific_search (queries about specific projects/documents that need document chunks returned), aggregation_summary (queries asking for counts, statistics, or summaries across documents - return AI summary only, no chunks), broad_category_search (queries asking about a category of projects like 'mining projects' or 'LNG projects'), or ambiguous"
                            },
                            "category_filter": {
                                "type": "string",
                                "enum": ["mining", "lng", "pipeline", "energy", "infrastructure", "water", "industrial", "transportation", "waste", "resort", "other"],
                                "description": "For broad_category_search queries, the category to filter projects by. Only set this if query_type is broad_category_search."
                            },
                            "reasoning": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of reasons for the relevance decision"
                            },
                            "recommendation": {
                                "type": "string",
                                "enum": ["proceed_with_search", "proceed_with_aggregation", "return_generic_response", "inform_user_out_of_scope", "return_category_list"],
                                "description": "Recommendation: proceed_with_search for specific queries (return chunks), proceed_with_aggregation for count/summary queries (return only AI summary, no chunks), return_generic_response for generic informational queries, return_category_list for broad category searches, inform_user_out_of_scope for irrelevant queries"
                            },
                            "suggested_response": {
                                "type": "string",
                                "description": "Optional response message for irrelevant queries"
                            }
                        },
                        "required": ["is_relevant", "confidence", "query_type", "reasoning", "recommendation"]
                    }
                }
            }]

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Validate this query: {query}"}
            ]

            logger.info(f"🔍 QUERY VALIDATOR: Validating query relevance using OpenAI function calling for: '{query}'")
            response = self.client.chat_completion(
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "validate_query_relevance"}},
                temperature=0.1
            )

            # Parse the function call response
            choice = response["choices"][0]
            if choice["message"].get("tool_calls"):
                tool_call = choice["message"]["tool_calls"][0]
                validation_result = json.loads(tool_call["function"]["arguments"])

                # Detailed logging for debugging query type detection
                query_type = validation_result.get("query_type", "unknown")
                recommendation = validation_result.get("recommendation", "unknown")
                logger.info(f"🔍 QUERY VALIDATOR: LLM returned query_type='{query_type}', recommendation='{recommendation}'")
                logger.info(f"🔍 QUERY VALIDATOR: Full validation result: {validation_result}")

                # If it's a generic informational query, generate a helpful response
                if validation_result.get("query_type") == "generic_informational":
                    generic_response = self._generate_generic_response(query)
                    validation_result["generic_response"] = generic_response
                    validation_result["recommendation"] = "return_generic_response"

                return validation_result
            else:
                logger.warning("No function call in response, using fallback validation")
                return self._fallback_validation(query, context)

        except Exception as e:
            logger.error(f"Query validation failed: {str(e)}")
            # Return fallback validation
            return self._fallback_validation(query, context)

    def _generate_generic_response(self, query: str) -> str:
        """Generate a helpful response for generic informational queries about EAO.

        Args:
            query: The user's generic query about EAO

        Returns:
            A helpful AI-generated response based on EAO knowledge base
        """
        try:
            prompt = f"""You are an expert on the Environmental Assessment Office (EAO) of British Columbia, Canada.
Answer the user's question based on the following knowledge base. Be helpful, concise, and informative.
If the question is not fully covered by the knowledge base, provide what information you can and suggest
they search for specific projects or documents for more detailed information.

KNOWLEDGE BASE:
{EAO_KNOWLEDGE_BASE}

USER QUESTION: {query}

Provide a clear, helpful response. Use markdown formatting for readability.
End with a suggestion to search for specific projects or documents if they need more detailed information."""

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]

            response = self.client.chat_completion(
                messages=messages,
                temperature=0.3
            )

            content = response["choices"][0]["message"]["content"]
            logger.info(f"Generated generic response for query: {query[:50]}...")
            return content

        except Exception as e:
            logger.error(f"Failed to generate generic response: {e}")
            # Return a fallback generic response
            return self._get_fallback_generic_response(query)
    
    def _get_fallback_generic_response(self, query: str) -> str:
        """Get a fallback generic response when LLM generation fails."""
        query_lower = query.lower()

        if "what is eao" in query_lower or "what is the eao" in query_lower:
            return """The **Environmental Assessment Office (EAO)** is an independent office within the Government of British Columbia, Canada.

The EAO is responsible for conducting environmental assessments of major projects proposed in BC, including mining projects, LNG facilities, pipelines, and major infrastructure developments.

**Key Functions:**
- Conducting environmental assessments of major projects
- Issuing Environmental Assessment Certificates
- Ensuring meaningful consultation with Indigenous nations
- Monitoring compliance with certificate conditions

For more specific information, try searching for a particular project or document type."""

        elif "environmental assessment" in query_lower:
            return """**Environmental Assessment in BC**

Environmental assessment is a process to identify and evaluate the potential effects of proposed major projects before they proceed. In British Columbia, this process is managed by the Environmental Assessment Office (EAO).

The process includes:
1. Early engagement with Indigenous nations and the public
2. Review of potential environmental, social, economic, and health effects
3. Public comment periods
4. Ministerial decision on whether to issue an Environmental Assessment Certificate

For specific project information, search for a project name or document type."""

        else:
            return """The **Environmental Assessment Office (EAO)** manages environmental assessments for major projects in British Columbia, Canada.

The EAO reviews projects including mining, energy (LNG, pipelines, power), infrastructure, and industrial developments to assess their potential environmental, social, and economic impacts.

For specific information, try searching for:
- A specific project name (e.g., "Cariboo Gold Project")
- Document types (e.g., "Schedule B", "Certificate", "Letters")
- Topics (e.g., "wildlife assessment", "consultation")"""

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "openai"

    def _build_validation_prompt(self, context: Optional[str] = None) -> str:
        """Build the validation prompt."""
        prompt = """You are an expert at determining if search queries are relevant to the Environmental Assessment Office (EAO) of British Columbia.

The EAO's scope includes:
- Environmental assessments of major projects in BC
- Mining projects, LNG facilities, pipelines, infrastructure
- Environmental reviews and regulatory processes
- Wildlife and habitat assessments
- Indigenous consultation and engagement
- Environmental certificates and approvals
- Project compliance and monitoring

**QUERY TYPE CLASSIFICATION:**

1. **GENERIC INFORMATIONAL** queries are general questions ABOUT the EAO or environmental assessment process:
   - "What is EAO?"
   - "How does environmental assessment work in BC?"
   - "What is the EAO's mandate?"
   - "What types of projects does EAO assess?"
   - "What is Schedule B?"
   - "How long does environmental assessment take?"
   - "What is an Environmental Assessment Certificate?"
   These should return a general informational response WITHOUT searching the database.

2. **SPECIFIC SEARCH** queries are looking for information about specific projects, documents, or content:
   - "Schedule B for Cariboo Gold" (specific project + document type)
   - "Does the Schedule B for Cariboo Gold have a condition that refers specifically to the Caribou?" (specific content question)
   - "Letters from Ministry of Environment about Blackwater Gold"
   - "What conditions are in the KSM Project certificate?"
   - "Documents mentioning Nooaitch Indian Band"
   - "What does the certificate say about fish habitat?"
   These should SEARCH the database and RETURN document chunks so users can see the specific content.

3. **AGGREGATION SUMMARY** queries ask for counts, statistics, or summaries across multiple documents:
   - "How many certificates were given in 2024?" (counting documents)
   - "How many projects have Schedule B conditions about caribou?" (counting/aggregating)
   - "What percentage of mining projects were approved?" (statistics)
   - "Give me a summary of all the certificate amendments in 2023" (aggregate summary)
   - "List all the projects that received certificates last year" (enumeration/counting)
   - "How many EAC documents mention salmon?" (counting mentions)
   These should SEARCH the database but only return an AI SUMMARY answer - NOT the document chunks.
   The user wants a summary/count answer, not to browse through individual documents.

4. **BROAD CATEGORY SEARCH** queries are asking about a CATEGORY of projects without specifying a particular one:
   - "Mining projects in BC" (wants a list of mining projects)
   - "LNG projects" (wants a list of LNG projects)
   - "What pipeline projects are there?" (wants a list of pipeline projects)
   - "Show me energy projects" (wants a list of energy projects)
   These should SEARCH the database for projects in that category and present a list.
   Set category_filter to the appropriate category: mining, lng, pipeline, energy, infrastructure, water, industrial, transportation, waste, resort, or other.

4. **AMBIGUOUS** queries could be any type - default to searching the database:
   - "Environmental monitoring reports" (general concept or specific search)
   These should proceed with database search.

**VALIDATION CRITERIA:**

RELEVANT queries relate to:
- Environmental assessments, projects, or processes
- EAO-regulated industries (mining, energy, infrastructure)
- Environmental impact studies or reports
- Regulatory documents, permits, or certificates
- Consultation processes or stakeholder engagement
- Project names, locations, or companies in BC
- General questions about how EAO works

IRRELEVANT queries are clearly about:
- Non-environmental topics (sports, entertainment, recipes, etc.)
- Areas completely outside BC environmental regulation
- Personal questions unrelated to environmental assessment

IMPORTANT: When in doubt about relevance, err on the side of RELEVANT.
When in doubt about query_type between generic and specific, choose "specific_search" or "ambiguous".

**RECOMMENDATION MAPPING:**
- For generic_informational queries → recommend "return_generic_response"
- For broad_category_search queries → recommend "return_category_list" and set category_filter appropriately
- For specific_search or ambiguous queries → recommend "proceed_with_search" (will return chunks)
- For aggregation_summary queries → recommend "proceed_with_aggregation" (AI summary only, no chunks)
- For IRRELEVANT queries → recommend "inform_user_out_of_scope"

**KEY DISTINCTION between specific_search and aggregation_summary:**
- specific_search: User wants to SEE the actual document content (e.g., "What does the Schedule B say about...")
- aggregation_summary: User wants a COUNT or SUMMARY answer (e.g., "How many certificates...", "What percentage...")"""

        if context:
            prompt += f"\n\nAdditional context: {context}"

        return prompt
    
    def _fallback_validation(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Provide fallback validation when LLM fails."""
        # Simple keyword-based validation as fallback
        query_lower = query.lower()
        context_lower = context.lower() if context else ""
        full_text = f"{query_lower} {context_lower}".strip()

        # Generic informational query patterns
        generic_patterns = [
            "what is eao", "what is the eao", "what does eao do", "what is environmental assessment",
            "how does environmental assessment work", "what is the environmental assessment office",
            "eao mandate", "eao process", "what types of projects", "how long does assessment take",
            "what is schedule b", "what is schedule a", "what is a certificate", "explain eao",
            "tell me about eao", "describe eao", "eao overview", "environmental assessment process"
        ]

        # Broad category search patterns - asking about categories of projects
        category_patterns = {
            "mining": ["mining projects", "mine projects", "mines in bc", "gold mines", "copper mines", "coal mines", "mineral projects", "mining operations"],
            "lng": ["lng projects", "lng facilities", "liquefied natural gas", "lng in bc", "natural gas projects"],
            "pipeline": ["pipeline projects", "pipelines in bc", "gas pipelines", "oil pipelines", "transmission pipelines"],
            "energy": ["energy projects", "power projects", "power plants", "electricity projects", "renewable energy", "hydro projects", "wind projects", "solar projects"],
            "infrastructure": ["infrastructure projects", "major infrastructure", "transportation infrastructure"],
            "water": ["water projects", "dam projects", "dams in bc", "water diversion", "reservoir projects", "dyke projects"],
            "industrial": ["industrial projects", "chemical plants", "refineries", "industrial facilities"],
            "transportation": ["port projects", "terminal projects", "railway projects", "ports in bc", "marine terminals"],
            "waste": ["waste projects", "landfill projects", "hazardous waste", "waste facilities"],
            "resort": ["resort projects", "tourism projects", "ski resorts", "resort developments"]
        }

        # Key EAO/environmental terms
        eao_terms = [
            "environmental assessment", "eao", "environmental", "mining", "project",
            "assessment", "environmental review", "lng", "pipeline", "wildlife",
            "habitat", "consultation", "certificate", "approval", "british columbia"
        ]

        # Document and process terms common in EAO database
        rag_terms = [
            "certificate", "correspondence", "report", "document", "letter",
            "application", "submission", "consultation", "band", "nation",
            "indigenous", "first nations", "permit", "monitoring"
        ]

        # Specific project/search indicators (user wants to SEE document content)
        specific_indicators = [
            "for the", "about the", "from the", "schedule b for", "schedule a for",
            "letters from", "documents for", "conditions for", "certificate for",
            "does the", "what does", "show me the", "find the", "refers to", "mention"
        ]

        # Aggregation/summary query patterns (user wants counts, statistics, summaries)
        aggregation_patterns = [
            "how many", "how much", "what percentage", "what percent",
            "count of", "total number", "number of", "list all",
            "summary of", "overview of", "statistics", "were given",
            "were issued", "were approved", "were rejected", "were completed",
            "in 2024", "in 2023", "in 2022", "in 2021", "in 2020",
            "last year", "this year", "per year", "annually",
            "across all", "all the", "total", "overall"
        ]

        # Clear non-EAO terms
        non_eao_terms = [
            "soccer", "football", "world cup", "movie", "music", "recipe",
            "shopping", "celebrity", "iphone", "restaurant", "vacation",
            "gaming", "netflix", "instagram", "facebook"
        ]

        # Check for broad category search patterns
        detected_category = None
        for category, patterns in category_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_category = category
                break

        # Check for generic informational patterns
        is_generic = any(pattern in query_lower for pattern in generic_patterns)

        # Check for aggregation/summary patterns
        is_aggregation = any(pattern in query_lower for pattern in aggregation_patterns)
        matched_aggregation = [p for p in aggregation_patterns if p in query_lower]
        if matched_aggregation:
            logger.info(f"🔍 FALLBACK: Aggregation patterns matched: {matched_aggregation}")

        # Check for specific search indicators
        has_specific_indicators = any(indicator in query_lower for indicator in specific_indicators)
        matched_specific = [i for i in specific_indicators if i in query_lower]
        if matched_specific:
            logger.info(f"🔍 FALLBACK: Specific indicators matched: {matched_specific}")

        eao_matches = sum(1 for term in eao_terms if term in full_text)
        rag_matches = sum(1 for term in rag_terms if term in full_text)
        non_eao_matches = sum(1 for term in non_eao_terms if term in query_lower)

        # Check for short query patterns
        is_short_query = len(query.split()) <= 3
        has_capital_letters = any(c.isupper() for c in query)

        if non_eao_matches > 0 and eao_matches == 0 and rag_matches == 0 and not is_short_query:
            # Clear non-EAO query with no environmental or document context
            is_relevant = False
            query_type = "ambiguous"
            confidence = 0.8
            reasoning = ["Non-EAO query detected", "No environmental or document-related keywords found"]
            recommendation = "inform_user_out_of_scope"
            suggested_response = "I'm designed to help with Environmental Assessment Office (EAO) related queries about environmental assessments, projects, and regulatory processes in British Columbia. Your question appears to be outside this scope. Please ask about environmental assessments, projects under review, or EAO processes."
            generic_response = None
            category_filter = None
        elif detected_category and not has_specific_indicators:
            # Broad category search - asking about a type of projects
            is_relevant = True
            query_type = "broad_category_search"
            confidence = 0.85
            reasoning = [f"Broad category search detected for '{detected_category}' projects", "Query asks about a category of projects without specifying a particular one"]
            recommendation = "return_category_list"
            suggested_response = None
            generic_response = None
            category_filter = detected_category
        elif is_generic and not has_specific_indicators and not is_aggregation:
            # Generic informational query about EAO
            is_relevant = True
            query_type = "generic_informational"
            confidence = 0.85
            reasoning = ["Generic informational query about EAO detected", "Query asks about EAO concepts/processes"]
            recommendation = "return_generic_response"
            suggested_response = None
            generic_response = self._get_fallback_generic_response(query)
            category_filter = None
        elif is_aggregation and not has_specific_indicators:
            # Aggregation/summary query - wants counts or summaries, not specific document content
            is_relevant = True
            query_type = "aggregation_summary"
            confidence = 0.85
            reasoning = ["Aggregation/summary query detected", "Query asks for counts, statistics, or summaries across documents"]
            recommendation = "proceed_with_aggregation"
            suggested_response = None
            generic_response = None
            category_filter = None
        elif has_specific_indicators or (rag_matches > 0 and has_capital_letters):
            # Specific search query
            is_relevant = True
            query_type = "specific_search"
            confidence = 0.8
            reasoning = ["Specific search indicators detected", "Query appears to target specific documents/projects"]
            recommendation = "proceed_with_search"
            suggested_response = None
            generic_response = None
            category_filter = None
        elif eao_matches > 0:
            # Contains EAO keywords - default to search
            is_relevant = True
            query_type = "ambiguous"
            confidence = min(0.8, 0.5 + (eao_matches * 0.1))
            reasoning = ["Environmental/EAO keywords detected", "Query appears relevant to EAO scope"]
            recommendation = "proceed_with_search"
            suggested_response = None
            generic_response = None
            category_filter = None
        elif rag_matches > 0:
            # Contains document/process terms
            is_relevant = True
            query_type = "specific_search"
            confidence = 0.7
            reasoning = ["Document/process terms detected", "Query appears to be searching for EAO-related documents"]
            recommendation = "proceed_with_search"
            suggested_response = None
            generic_response = None
            category_filter = None
        elif is_short_query:
            # Short queries are often proper nouns or specific search terms
            is_relevant = True
            query_type = "specific_search" if has_capital_letters else "ambiguous"
            confidence = 0.6 if has_capital_letters else 0.5
            reasoning = ["Short query detected - allowing for database search",
                        "Short queries may reference specific content in EAO documents"]
            recommendation = "proceed_with_search"
            suggested_response = None
            generic_response = None
            category_filter = None
        else:
            # Default to allowing for search
            is_relevant = True
            query_type = "ambiguous"
            confidence = 0.4
            reasoning = ["Allowing query for database search", "Query may reference content within EAO documents"]
            recommendation = "proceed_with_search"
            suggested_response = None
            generic_response = None
            category_filter = None

        result = {
            "is_relevant": is_relevant,
            "confidence": confidence,
            "query_type": query_type,
            "reasoning": reasoning,
            "recommendation": recommendation,
            "suggested_response": suggested_response
        }

        if generic_response:
            result["generic_response"] = generic_response

        if category_filter:
            result["category_filter"] = category_filter

        logger.info(f"🔍 FALLBACK: Final query_type='{query_type}', recommendation='{recommendation}', is_aggregation={is_aggregation}, has_specific_indicators={has_specific_indicators}")
        return result