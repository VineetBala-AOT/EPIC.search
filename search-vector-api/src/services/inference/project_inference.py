"""Project inference service for automatically detecting project references in search queries.

This module provides intelligent project detection capabilities that analyze user queries
to identify project names and other project-related entities. When users ask questions 
like "Coyote Hydrogen project zoning and land use?", the system can automatically 
infer which project(s) they're referring to, apply appropriate filtering, and clean 
the query to focus on actual search terms without requiring explicit project IDs.

The service includes:
1. Named entity recognition for project names and project-related terms
2. Fuzzy matching against known project names in the projects table
3. Confidence scoring for automatic project inference based solely on project name similarity
4. Query cleaning to remove identified project names and focus on actual search topics
5. Integration with the search pipeline for transparent project filtering

Note: Project inference is based exclusively on project names from the projects table,
not on proponent organizations, to ensure focused and accurate project detection.
"""

import re
import logging
import pandas as pd
import psycopg

from flask import current_app
from ..vector_store import VectorStore
from typing import List, Tuple, Dict, Any
from difflib import SequenceMatcher

class ProjectInferenceService:
    """Service for inferring project context from natural language queries.
    
    This service analyzes search queries to automatically detect project references
    and provides confident project ID suggestions when users mention specific projects
    by name without explicitly providing project IDs. Inference is based exclusively
    on project names from the projects table for focused and accurate matching.
    """
    
    def __init__(self):
        """Initialize the project inference service."""
        self.vector_store = VectorStore()
        self._project_cache = None
        self._cache_timestamp = None
        self.cache_ttl = 300  # 5 minutes cache TTL
    
    def infer_projects_from_query(self, query: str, confidence_threshold: float = 0.8) -> Tuple[List[str], float, Dict[str, Any]]:
        """Infer project IDs from a natural language query based on project names.
        
        Analyzes the query for project names and project-related terminology to suggest 
        relevant project IDs with confidence scoring. Matching is performed exclusively 
        against project names from the projects table.
        
        Args:
            query (str): The natural language search query
            confidence_threshold (float): Minimum confidence required for automatic inference (default: 0.8)
            
        Returns:
            tuple: A tuple containing:
                - List[str]: Inferred project IDs (empty if confidence too low)
                - float: Confidence score (0.0 to 1.0)
                - Dict[str, Any]: Inference metadata including matched entities and reasoning
        """
        inference_metadata = {
            "extracted_entities": [],
            "matched_projects": [],
            "reasoning": [],
            "method": "entity_matching"
        }
        
        try:
            # Extract potential project entities from the query
            entities = self._extract_project_entities(query)
            inference_metadata["extracted_entities"] = entities
            
            # Get available projects (with caching)
            projects_df = self._get_projects_cached()
            
            if projects_df.empty:
                logging.warning("No projects found in database for inference")
                return [], 0.0, inference_metadata
            
            # Match entities against known projects
            matched_projects = self._match_entities_to_projects(entities, projects_df)
            inference_metadata["matched_projects"] = matched_projects
            
            # Calculate confidence and select projects
            project_ids, confidence = self._calculate_confidence_and_select_projects(
                matched_projects, confidence_threshold
            )
            
            # Add reasoning to metadata
            if project_ids:
                inference_metadata["reasoning"] = [
                    f"Detected entity '{match['entity']}' matching project '{match['project_name']}' with similarity {match['similarity']:.3f}"
                    for match in matched_projects if match["project_id"] in project_ids
                ]
                logging.info(f"Project inference successful: {len(project_ids)} projects with confidence {confidence:.3f}")
            else:
                inference_metadata["reasoning"] = [
                    f"Confidence {confidence:.3f} below threshold {confidence_threshold}"
                ]
                logging.info(f"Project inference below confidence threshold: {confidence:.3f} < {confidence_threshold}")
            
            if not project_ids:
                # Fallback: Try matching by metadata fields if name match failed
                metadata_matches = self._match_projects_by_metadata(query)
                if metadata_matches:
                    # Define your confidence threshold
                    threshold = 0.8  # you can tune this value

                    # Filter out projects with similarity below threshold
                    high_confidence_matches = [
                        match for match in metadata_matches if match["similarity"] >= threshold
                    ]

                    if high_confidence_matches:
                        # Sort by similarity descending (optional, for consistency)
                        high_confidence_matches.sort(key=lambda x: x["similarity"], reverse=True)

                        # Collect all project IDs above threshold
                        project_ids = [match["project_id"] for match in high_confidence_matches]

                        # Optionally, calculate an average or max confidence score
                        confidence = sum(m["similarity"] for m in high_confidence_matches) / len(high_confidence_matches)

                        inference_metadata["matched_projects"].extend(high_confidence_matches)
                        inference_metadata["use_project_metadata"] = True
                else:
                    inference_metadata["use_project_metadata"] = False
            
            return project_ids, confidence, inference_metadata
            
        except Exception as e:
            logging.error(f"Error in project inference: {e}")
            inference_metadata["error"] = str(e)
            return [], 0.0, inference_metadata
    
    def _extract_project_entities(self, query: str) -> List[str]:
        """Extract potential project names from the query by matching against known project names.

        Uses a more targeted approach that only extracts entities that could plausibly
        match project names in the database, focusing on actual text spans from the query.

        IMPORTANT: Prioritizes multi-word phrases that contain DISTINCTIVE words
        (like "Cariboo Gold") over generic single words (like "Gold").

        Args:
            query (str): The search query text

        Returns:
            List[str]: List of n-grams from the query that might match project names
        """
        # Get known project names for pre-filtering
        projects_df = self._get_projects_cached()
        if projects_df.empty:
            return []

        # Generic terms that shouldn't be extracted alone
        generic_terms = {
            'project', 'development', 'pipeline', 'mine', 'dam', 'terminal', 'facility',
            'energy', 'gas', 'oil', 'hydro', 'south', 'north', 'east', 'west',
            'mountain', 'river', 'creek', 'island', 'point', 'road', 'bridge',
            'resort', 'park', 'extension', 'expansion', 'phase', 'gold', 'copper',
            'silver', 'coal', 'lng', 'power', 'plant', 'station', 'transmission',
            'line', 'port', 'upper', 'lower', 'new', 'old', 'wind', 'reservoir'
        }

        # Create a set of project name words for quick filtering
        project_name_words = set()
        # Also track distinctive words from project names (non-generic)
        distinctive_project_words = set()
        for name in projects_df['project_name'].tolist():
            name_words = set(word.lower() for word in re.findall(r'\b\w+\b', str(name)))
            project_name_words.update(name_words)
            distinctive_project_words.update(name_words - generic_terms)

        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        candidates = []
        candidate_scores = {}  # Track quality scores for candidates

        # First pass: Look for exact project name matches in the query
        for _, project in projects_df.iterrows():
            project_name = str(project.get("project_name", "")).lower().strip()
            if project_name and project_name in query_lower:
                candidates.append(project_name)
                candidate_scores[project_name] = 1.0  # Highest priority

        # Second pass: Generate n-grams prioritizing those with distinctive words
        for n in range(1, min(6, len(words) + 1)):  # 1 to 5 words
            for i in range(len(words) - n + 1):
                ngram_words = words[i:i+n]
                ngram = ' '.join(ngram_words)

                # Skip very short phrases
                if len(ngram) <= 3:
                    continue

                # Skip common non-project phrases
                if ngram.startswith(('i am', 'i have', 'looking for', 'please find', 'that refer', 'get me', 'show me')):
                    continue

                # For single words, ONLY accept if it's a DISTINCTIVE project word
                if n == 1:
                    if ngram in generic_terms:
                        continue  # Skip generic single words
                    if ngram not in distinctive_project_words:
                        continue  # Skip if not a distinctive project word

                # For multi-word phrases, require at least one distinctive word
                ngram_word_set = set(ngram_words)
                has_distinctive = bool(ngram_word_set & distinctive_project_words)
                has_project_word = bool(ngram_word_set & project_name_words)

                if has_distinctive:
                    # High priority: contains distinctive project word
                    candidates.append(ngram)
                    # Score based on number of distinctive words
                    distinctive_count = len(ngram_word_set & distinctive_project_words)
                    candidate_scores[ngram] = 0.8 + (0.05 * distinctive_count)
                elif has_project_word and n >= 2:
                    # Lower priority: multi-word with some project words
                    candidates.append(ngram)
                    candidate_scores[ngram] = 0.5

        # Extract specific patterns around project keywords with better precision
        project_keywords = ['project', 'pipeline', 'development', 'mine', 'dam', 'terminal', 'facility']
        for keyword in project_keywords:
            # More precise pattern matching for "[name] project" structures
            pattern = r'\b([a-zA-Z]+(?:\s+[a-zA-Z]+){0,3})\s+' + keyword + r'\b'
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                candidate = match.strip().lower()
                if len(candidate) > 3:
                    # Check if candidate contains distinctive words
                    candidate_word_set = set(candidate.split())
                    has_distinctive = bool(candidate_word_set & distinctive_project_words)

                    if has_distinctive:
                        candidates.append(candidate)
                        candidate_scores[candidate] = 0.9
                        # Also add the full "[name] project" phrase
                        full_phrase = f"{candidate} {keyword}".lower()
                        candidates.append(full_phrase)
                        candidate_scores[full_phrase] = 0.95

        # Exclusion list for common non-project terms
        excluded_terms = {
            'aboriginal groups', 'indigenous groups', 'first nations', 'environmental assessment',
            'british columbia', 'environmental protection', 'climate change', 'all correspondence',
            'please find', 'that refer', 'lheidli t enneh', 'schedule b', 'schedule a'
        }

        # Filter out excluded terms and very generic phrases
        filtered_candidates = []
        for candidate in candidates:
            if candidate not in excluded_terms and not any(excl in candidate for excl in excluded_terms):
                filtered_candidates.append(candidate)

        # Remove duplicates while preserving order, prioritize higher-scored candidates
        seen = set()
        entities = []
        # Sort by score (highest first) before deduplication
        scored_candidates = [(c, candidate_scores.get(c, 0.5)) for c in filtered_candidates]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        for candidate, score in scored_candidates:
            if candidate not in seen:
                seen.add(candidate)
                entities.append(candidate)

        logging.debug(f"Generated {len(entities)} candidate entities from query '{query}': {entities}")
        return entities
    
    def _get_projects_cached(self) -> pd.DataFrame:
        """Get all projects with caching for performance.
        
        Queries only the projects table and retrieves only project_id and project_name
        for project inference matching based solely on project names.
        
        Returns:
            pd.DataFrame: DataFrame containing project_id and project_name columns
        """
        import time
        current_time = time.time()
        
        # Check if cache is valid
        if (self._project_cache is not None and 
            self._cache_timestamp is not None and
            current_time - self._cache_timestamp < self.cache_ttl):
            return self._project_cache
        
        # Refresh cache
        try:
            with psycopg.connect(current_app.vector_settings.database_url) as conn:
                with conn.cursor() as cursor:
                    # Query the projects table directly for project inference
                    # Only retrieve project_id and project_name for name-based matching
                    query = """
                    SELECT DISTINCT 
                        project_id,
                        project_name
                    FROM projects 
                    WHERE project_id IS NOT NULL 
                        AND project_name IS NOT NULL
                        AND project_name != ''
                    ORDER BY project_name;
                    """
                    
                    cursor.execute(query)
                    results = cursor.fetchall()
                    
                    columns = ["project_id", "project_name"]
                    self._project_cache = pd.DataFrame(results, columns=columns)
                    self._cache_timestamp = current_time
                    
                    logging.debug(f"Cached {len(self._project_cache)} projects from projects table")
                    return self._project_cache
                    
        except Exception as e:
            logging.error(f"Error fetching projects for inference: {e}")
            # Return empty DataFrame on error
            self._project_cache = pd.DataFrame(columns=["project_id", "project_name"])
            return self._project_cache
    
    def _match_entities_to_projects(self, entities: List[str], projects_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Match extracted entities to known projects using strict matching on project names.

        This method performs case-insensitive matching against actual project names
        from the database, with emphasis on DISTINCTIVE words rather than generic terms.

        CRITICAL: When matching "Cariboo Gold", we should NOT match "Blackwater Gold"
        just because both contain "Gold". The distinctive part is "Cariboo" vs "Blackwater".

        Args:
            entities (List[str]): List of candidate entity strings (already lowercase)
            projects_df (pd.DataFrame): DataFrame with project_id and project_name columns

        Returns:
            List[Dict[str, Any]]: List of matches with similarity scores
        """
        matches = []

        # Generic terms that should NOT drive matching on their own
        generic_terms = {
            'project', 'mine', 'gold', 'copper', 'silver', 'coal', 'gas', 'oil', 'lng',
            'power', 'energy', 'terminal', 'port', 'pipeline', 'transmission', 'line',
            'facility', 'plant', 'station', 'expansion', 'upgrade', 'phase', 'development',
            'mountain', 'river', 'creek', 'lake', 'park', 'resort', 'wind', 'reservoir',
            'north', 'south', 'east', 'west', 'upper', 'lower', 'new', 'old'
        }

        for entity in entities:
            entity_lower = entity.lower().strip()
            entity_words = set(entity_lower.split())

            # Identify distinctive words in the entity (non-generic)
            distinctive_entity_words = entity_words - generic_terms

            for _, project in projects_df.iterrows():
                project_name = str(project.get("project_name", "")).lower().strip()

                # Skip empty names
                if not project_name or project_name == "nan":
                    continue

                project_words = set(project_name.split())
                distinctive_project_words = project_words - generic_terms

                # Calculate base similarity using SequenceMatcher
                base_similarity = SequenceMatcher(None, entity_lower, project_name).ratio()
                similarity = base_similarity
                match_reason = "sequence_match"

                # PRIORITY 1: Exact full name match (highest confidence)
                if entity_lower == project_name:
                    similarity = 1.0
                    match_reason = "exact_match"

                # PRIORITY 2: Entity is exact substring of project name
                elif entity_lower in project_name and len(entity_lower) >= 5:
                    # Calculate how much of the project name is covered
                    coverage = len(entity_lower) / len(project_name)
                    similarity = max(similarity, 0.85 + (coverage * 0.1))
                    match_reason = "substring_match"

                # PRIORITY 3: Project name is exact substring of entity
                elif project_name in entity_lower and len(project_name) >= 5:
                    similarity = max(similarity, 0.95)
                    match_reason = "reverse_substring"

                # PRIORITY 4: Distinctive word matching (KEY FOR DISAMBIGUATION)
                elif distinctive_entity_words and distinctive_project_words:
                    # Calculate overlap of distinctive words
                    distinctive_overlap = distinctive_entity_words & distinctive_project_words

                    if distinctive_overlap:
                        # Calculate what percentage of distinctive entity words matched
                        entity_coverage = len(distinctive_overlap) / len(distinctive_entity_words)
                        # Calculate what percentage of distinctive project words matched
                        project_coverage = len(distinctive_overlap) / len(distinctive_project_words)

                        # Use geometric mean for balanced scoring
                        combined_coverage = (entity_coverage * project_coverage) ** 0.5

                        # Strong match if most distinctive words overlap
                        if entity_coverage >= 0.8 and project_coverage >= 0.5:
                            similarity = max(similarity, 0.85 + (combined_coverage * 0.1))
                            match_reason = f"distinctive_match:{distinctive_overlap}"
                        elif entity_coverage >= 0.5:
                            similarity = max(similarity, 0.7 + (combined_coverage * 0.15))
                            match_reason = f"partial_distinctive:{distinctive_overlap}"
                    else:
                        # NO distinctive word overlap - this is likely a WRONG match
                        # Check if only generic terms match
                        generic_overlap = (entity_words & generic_terms) & (project_words & generic_terms)
                        if generic_overlap and not distinctive_overlap:
                            # PENALIZE: Only generic terms match (e.g., both have "Gold")
                            # This is the "Cariboo Gold" vs "Blackwater Gold" case
                            similarity = min(similarity, 0.3)  # Cap at very low similarity
                            match_reason = f"generic_only:{generic_overlap}"
                            logging.debug(f"Penalized generic-only match: '{entity}' vs '{project_name}' - only '{generic_overlap}' matched")

                # PRIORITY 5: All entity words are subset of project words
                elif entity_words and entity_words.issubset(project_words):
                    similarity = max(similarity, 0.8)
                    match_reason = "word_subset"

                # Only include matches with reasonable similarity
                if similarity >= 0.6:
                    matches.append({
                        "entity": entity,
                        "project_id": project["project_id"],
                        "project_name": project["project_name"],
                        "similarity": similarity,
                        "match_type": "fuzzy",
                        "match_reason": match_reason
                    })

        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        # Log top matches for debugging
        if matches:
            logging.info(f"Project matching results for entities {entities}:")
            for m in matches[:5]:
                logging.info(f"  - '{m['project_name']}' score={m['similarity']:.3f} reason={m.get('match_reason', 'unknown')}")

        logging.debug(f"Found {len(matches)} project matches with similarity >= 0.6")
        return matches
    
    def _calculate_confidence_and_select_projects(
        self, 
        matches: List[Dict[str, Any]], 
        confidence_threshold: float
    ) -> Tuple[List[str], float]:
        """Calculate overall confidence and select project IDs.
        
        Enhanced to better handle multiple project references in a single query.
        Now selects all projects with individually strong matches rather than 
        averaging confidence across all matches.
        
        Args:
            matches (List[Dict[str, Any]]): List of project matches with similarity scores
            confidence_threshold (float): Minimum confidence required
            
        Returns:
            Tuple[List[str], float]: Selected project IDs and confidence score
        """
        if not matches:
            return [], 0.0
        
        # Group matches by project_id and take the highest similarity for each project
        project_scores = {}
        for match in matches:
            project_id = match["project_id"]
            if project_id not in project_scores or match["similarity"] > project_scores[project_id]["similarity"]:
                project_scores[project_id] = match
        
        # Sort projects by their best similarity score
        top_matches = sorted(project_scores.values(), key=lambda x: x["similarity"], reverse=True)
        
        if not top_matches:
            return [], 0.0
        
        # For multi-project inference, evaluate each project individually
        # rather than averaging confidence across all matches
        high_confidence_projects = []
        individual_confidences = []
        
        # First pass: Look for very high similarity matches (0.95+) which indicate exact/near-exact matches
        excellent_matches = [match for match in top_matches if match["similarity"] >= 0.95]
        
        if excellent_matches:
            # If we have excellent matches, prefer those and be very selective
            # Only take the top 2 excellent matches to avoid over-inference
            for match in excellent_matches[:2]:
                high_confidence_projects.append(match["project_id"])
                individual_confidences.append(match["similarity"])
                logging.debug(f"Selected high-confidence project '{match['project_name']}' with similarity {match['similarity']:.3f}")
                
                # If we have a perfect or near-perfect match (0.98+), stop here
                if match["similarity"] >= 0.98:
                    logging.info(f"Found near-perfect project match, stopping inference: {match['project_name']} ({match['similarity']:.3f})")
                    break
        else:
            # Second pass: Look for good matches but be much more selective
            accepted_matches = 0
            for match in top_matches:
                individual_similarity = match["similarity"]
                entity = match.get("entity", "")
                entity_words = len(entity.split())
                
                # Very strict thresholds to prevent over-inference
                if entity_words >= 4 and individual_similarity >= 0.90:
                    # 4+ word entities (like "south anderson mountain resort") with very high similarity
                    individual_threshold = 0.90
                elif entity_words == 3 and individual_similarity >= 0.93:
                    # 3-word entities need very high similarity
                    individual_threshold = 0.93
                elif entity_words == 2 and individual_similarity >= 0.95:
                    # Two-word entities need near-perfect similarity
                    individual_threshold = 0.95
                else:
                    # Single words or low similarity - skip
                    continue
                
                if individual_similarity >= individual_threshold:
                    high_confidence_projects.append(match["project_id"])
                    individual_confidences.append(individual_similarity)
                    accepted_matches += 1
                    logging.debug(f"Selected project '{match['project_name']}' with similarity {individual_similarity:.3f} (entity: '{entity}', words: {entity_words})")
                
                # Very strict limit - only accept 1 project unless we have multiple excellent matches
                if accepted_matches >= 1:
                    break
        
        # Calculate overall confidence based on selected projects
        if high_confidence_projects:
            # For multiple projects, use the average of selected confidences
            # but apply a bonus for having multiple strong matches
            confidence = sum(individual_confidences) / len(individual_confidences)
            
            # Bonus for multiple strong matches (indicates explicit multi-project query)
            if len(high_confidence_projects) > 1 and all(c >= 0.8 for c in individual_confidences):
                confidence = min(1.0, confidence * 1.05)
                logging.debug(f"Applied multi-project bonus: {len(high_confidence_projects)} projects detected")
            
            # Ensure we meet the original confidence threshold
            if confidence >= confidence_threshold:
                return high_confidence_projects, confidence
            else:
                # If averaged confidence is too low, return only the best match
                best_match = top_matches[0]
                if best_match["similarity"] >= confidence_threshold:
                    return [best_match["project_id"]], best_match["similarity"]
                else:
                    return [], confidence
        else:
            # No individual matches met threshold
            best_similarity = top_matches[0]["similarity"] if top_matches else 0.0
            return [], best_similarity
        
    def clean_query_after_inference(self, query: str, inference_metadata: Dict[str, Any]) -> str:
        """Remove identified project names from the query to focus on actual search terms.
        
        After project inference identifies project names in the query, this method removes
        those project references so the search focuses on the actual topic rather than
        the project name itself. This prevents documents that mention the project name
        from being prioritized over documents about the actual search topic.
        
        Args:
            query (str): The original search query
            inference_metadata (Dict[str, Any]): Metadata from project inference containing extracted entities
            
        Returns:
            str: Cleaned query with project names removed
        """
        cleaned_query = query
        original_length = len(query)
        
        # Get the matched projects to identify which entities to remove
        matched_projects = inference_metadata.get("matched_projects", [])
        
        # Sort matched projects by similarity score (highest first) and get unique entities
        project_entities_to_remove = set()
        for match in matched_projects:
            entity = match.get("entity", "")
            similarity = match.get("similarity", 0.0)
            
            # Only remove entities with high similarity (0.85+) to be more aggressive
            if similarity >= 0.85 and len(entity) >= 5:
                project_entities_to_remove.add(entity.lower())
        
        # Remove project-related entities from the query
        for entity in project_entities_to_remove:
            # Pattern 1: Remove exact entity matches with word boundaries
            pattern1 = r'\b' + re.escape(entity) + r'\b'
            cleaned_query = re.sub(pattern1, '', cleaned_query, flags=re.IGNORECASE)
            
            # Pattern 2: Remove "the [entity]" patterns
            pattern2 = r'\bthe\s+' + re.escape(entity) + r'\b'
            cleaned_query = re.sub(pattern2, '', cleaned_query, flags=re.IGNORECASE)
            
            # Pattern 3: Remove "in the [entity]" patterns
            pattern3 = r'\bin\s+the\s+' + re.escape(entity) + r'\b'
            cleaned_query = re.sub(pattern3, '', cleaned_query, flags=re.IGNORECASE)
            
            # Pattern 4: Remove "[entity] project" patterns
            entity_base = entity.replace(' project', '').replace('project ', '').strip()
            if entity_base and len(entity_base) > 3:
                pattern4 = r'\b' + re.escape(entity_base) + r'\s+project\b'
                cleaned_query = re.sub(pattern4, '', cleaned_query, flags=re.IGNORECASE)
        
        # Additional patterns for common project references
        project_patterns = [
            r'\bin\s+the\s+[a-zA-Z\s]+\s+project\b',  # "in the [name] project"
            r'\bfor\s+the\s+[a-zA-Z\s]+\s+project\b',  # "for the [name] project"  
            r'\bthe\s+[a-zA-Z\s]+\s+project\b'  # "the [name] project"
        ]
        
        for pattern in project_patterns:
            cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and punctuation
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query)  # Multiple spaces to single space
        cleaned_query = re.sub(r'\s*[,;]\s*', ' ', cleaned_query)  # Remove commas/semicolons with spaces
        cleaned_query = cleaned_query.strip()
        
        # Clean up any remaining orphaned prepositions at the start
        cleaned_query = re.sub(r'^(in|for|of|at|on|with|by|from|to)\s+', '', cleaned_query, flags=re.IGNORECASE)
        cleaned_query = cleaned_query.strip()
        
        # If we removed more than 80% of the original query, it's too aggressive - keep original
        if len(cleaned_query) < original_length * 0.2:
            logging.warning(f"Project cleaning too aggressive (removed {100*(1-len(cleaned_query)/original_length):.0f}%), keeping original: '{query}'")
            return query
        
        # If the cleaned query is too short or empty, keep some of the original
        if len(cleaned_query.strip()) < 5:
            # Keep the original query but still log that we attempted cleaning
            logging.info(f"Cleaned query too short ('{cleaned_query}'), keeping original: '{query}'")
            return query
        
        if cleaned_query != query:
            logging.info(f"Project cleaning: '{query}' -> '{cleaned_query}'")
        
        return cleaned_query

    def _match_projects_by_metadata(self, query: str, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Matches projects based on metadata fields using fuzzy and exact term matching.

        Parameters:
            query (str): The user's query string used to search project metadata.
            similarity_threshold (float, default=0.8): Minimum normalized similarity score
                required for a project to be considered a match.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a matching project with:
                - 'entity': always 'metadata_match'
                - 'project_id': the unique project identifier
                - 'project_name': the name of the project
                - 'similarity': the normalized similarity score
                - 'match_type': always 'metadata'

        Method Details:
            - Uses weighted metadata fields: type, region, sector, status, proponent, description, location.
            - Uses SequenceMatcher for fuzzy term matching, with additional boosts for:
                * Exact token matches (EXACT_MATCH_BOOST)
                * Multi-term matches in the same field (MULTI_TERM_BOOST)
                * Entire field exact match (EXACT_FIELD_BOOST)
            - Ignores weak fuzzy matches below MIN_TERM_SIMILARITY to reduce false positives.
            - Normalizes cumulative similarity by total field weight and returns matches
            sorted in descending similarity order.
        """
        EXACT_MATCH_BOOST = 0.5
        MULTI_TERM_BOOST = 0.3  # extra boost if multiple query terms match in same field
        MIN_TERM_SIMILARITY = 0.3  # ignore weak fuzzy matches below this
        EXACT_FIELD_BOOST = 0.4    # boost if entire field matches exactly to a query term

        try:
            with psycopg.connect(current_app.vector_settings.database_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT project_id, project_name, project_metadata
                        FROM projects
                        WHERE project_metadata IS NOT NULL
                    """)
                    results = cursor.fetchall()

            matches = []
            query_terms = re.findall(r'\w+', query.lower())

            weighted_fields = {
                "type": 2.0,
                "region": 2.0,
                "sector": 2.0,
                "status": 2.0,
                "proponent": 1.5,
                "description": 1.0,
                "location": 1.0
            }

            for project_id, project_name, project_metadata in results:
                if not project_metadata:
                    continue
                try:
                    meta = project_metadata if isinstance(project_metadata, dict) else {}
                    cumulative_similarity = 0.0
                    total_weight = 0.0

                    for field, weight in weighted_fields.items():
                        value = None
                        if field == "proponent":
                            proponent = meta.get("proponent", {})
                            value = proponent.get("name")
                        else:
                            value = meta.get(field)

                        if not value:
                            continue

                        field_texts = [str(value).lower()] if not isinstance(value, list) else [str(v).lower() for v in value]
                        max_field_sim = 0.0

                        for field_text in field_texts:
                            field_terms = re.findall(r'\w+', field_text)
                            term_sims = []
                            matched_terms_count = 0

                            # Full-field exact match boost
                            if field_text.strip() in query.lower():
                                max_field_sim = 1.0 + EXACT_FIELD_BOOST
                                matched_terms_count = len(query_terms)
                                break

                            # Compute per-term fuzzy matches
                            for q_term in query_terms:
                                best_term_sim = max(
                                    (SequenceMatcher(None, q_term, f_term).ratio() for f_term in field_terms),
                                    default=0
                                )

                                if best_term_sim < MIN_TERM_SIMILARITY:
                                    continue

                                if q_term in field_text:
                                    best_term_sim += EXACT_MATCH_BOOST
                                    matched_terms_count += 1

                                term_sims.append(best_term_sim)

                            if term_sims:
                                field_sim = sum(term_sims) / len(term_sims)
                                # Add multi-term boost if more than 1 query term matched in this field
                                if matched_terms_count > 1:
                                    field_sim += MULTI_TERM_BOOST * (matched_terms_count - 1)
                                max_field_sim = max(max_field_sim, field_sim)

                        cumulative_similarity += max_field_sim * weight
                        total_weight += weight

                    normalized_similarity = cumulative_similarity / total_weight if total_weight > 0 else 0.0

                    if normalized_similarity >= similarity_threshold:
                        matches.append({
                            "entity": "metadata_match",
                            "project_id": project_id,
                            "project_name": project_name,
                            "similarity": normalized_similarity,
                            "match_type": "metadata"
                        })
                except Exception:
                    continue

            # Sort by cumulative similarity descending
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            logging.debug(f"Metadata matching found {len(matches)} projects")
            return matches

        except Exception as e:
            logging.error(f"Error during metadata project matching: {e}")
            return []

# Global instance for easy access
project_inference_service = ProjectInferenceService()
