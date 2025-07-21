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
            
            if not entities:
                logging.info(f"No project entities extracted from query: '{query}'")
                return [], 0.0, inference_metadata
            
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
            
            return project_ids, confidence, inference_metadata
            
        except Exception as e:
            logging.error(f"Error in project inference: {e}")
            inference_metadata["error"] = str(e)
            return [], 0.0, inference_metadata
    
    def _extract_project_entities(self, query: str) -> List[str]:
        """Extract potential project names from the query by matching against known project names.
        
        Uses a more targeted approach that only extracts entities that could plausibly 
        match project names in the database, focusing on actual text spans from the query.
        
        Args:
            query (str): The search query text
            
        Returns:
            List[str]: List of n-grams from the query that might match project names
        """
        # Get known project names for pre-filtering
        projects_df = self._get_projects_cached()
        if projects_df.empty:
            return []
        
        # Create a set of project name words for quick filtering
        project_name_words = set()
        for name in projects_df['project_name'].tolist():
            project_name_words.update(word.lower() for word in re.findall(r'\b\w+\b', str(name)))
        
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        candidates = []
        
        # Only generate n-grams that contain at least one word that appears in project names
        for n in range(1, min(6, len(words) + 1)):  # 1 to 5 words
            for i in range(len(words) - n + 1):
                ngram_words = words[i:i+n]
                ngram = ' '.join(ngram_words)
                
                # Skip very short phrases
                if len(ngram) <= 3:
                    continue
                
                # Skip common non-project phrases
                if ngram.startswith(('i am', 'i have', 'looking for', 'please find', 'that refer')):
                    continue
                
                # Skip overly generic single words that appear in many project names
                if n == 1 and ngram in {'project', 'development', 'pipeline', 'mine', 'dam', 'terminal', 'facility', 'energy', 'gas', 'oil', 'hydro'}:
                    continue
                
                # Only include if at least one word appears in known project names
                if any(word in project_name_words for word in ngram_words):
                    candidates.append(ngram)
        
        # Extract specific patterns around project keywords with better precision
        project_keywords = ['project', 'pipeline', 'development', 'mine', 'dam', 'terminal', 'facility']
        for keyword in project_keywords:
            # More precise pattern matching for "[name] project" structures
            pattern = r'\b([a-zA-Z]+(?:\s+[a-zA-Z]+){0,3})\s+' + keyword + r'\b'
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                candidate = match.strip().lower()
                if len(candidate) > 3:
                    # Verify this candidate contains project-related words
                    candidate_words = candidate.split()
                    if any(word in project_name_words for word in candidate_words):
                        candidates.append(candidate)
                        # Also add the full "[name] project" phrase
                        full_phrase = f"{candidate} {keyword}".lower()
                        candidates.append(full_phrase)
        
        # Exclusion list for common non-project terms
        excluded_terms = {
            'aboriginal groups', 'indigenous groups', 'first nations', 'environmental assessment',
            'british columbia', 'environmental protection', 'climate change', 'all correspondence',
            'please find', 'that refer', 'lheidli t enneh'
        }
        
        # Filter out excluded terms and very generic phrases
        filtered_candidates = []
        for candidate in candidates:
            if candidate not in excluded_terms and not any(excl in candidate for excl in excluded_terms):
                filtered_candidates.append(candidate)
        
        # Remove duplicates while preserving order
        seen = set()
        entities = []
        for candidate in filtered_candidates:
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
        """Match extracted entities to known projects using fuzzy matching on project names.
        
        This method performs case-insensitive fuzzy matching against actual project names
        from the database, providing much more accurate matching than pattern-based approaches.
        
        Args:
            entities (List[str]): List of candidate entity strings (already lowercase)
            projects_df (pd.DataFrame): DataFrame with project_id and project_name columns
            
        Returns:
            List[Dict[str, Any]]: List of matches with similarity scores
        """
        matches = []
        
        for entity in entities:
            entity_lower = entity.lower().strip()
            
            for _, project in projects_df.iterrows():
                project_name = str(project.get("project_name", "")).lower().strip()
                
                # Skip empty names
                if not project_name or project_name == "nan":
                    continue
                
                # Calculate similarity score using different methods
                similarity = SequenceMatcher(None, entity_lower, project_name).ratio()
                
                # Boost score for exact substring matches
                if entity_lower in project_name:
                    similarity = max(similarity, 0.8)
                
                # Boost score for reverse substring matches (project name in entity)
                if project_name in entity_lower:
                    similarity = max(similarity, 0.9)
                
                # Check for word-level matches (all words in entity match words in project name)
                entity_words = set(entity_lower.split())
                project_words = set(project_name.split())
                
                if entity_words and entity_words.issubset(project_words):
                    similarity = max(similarity, 0.85)
                
                # Use a lower threshold since we're generating more targeted candidates
                if similarity > 0.5:  # Lowered from 0.6
                    matches.append({
                        "entity": entity,
                        "project_id": project["project_id"],
                        "project_name": project["project_name"],
                        "similarity": similarity,
                        "match_type": "fuzzy"
                    })
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        logging.debug(f"Found {len(matches)} project matches with similarity > 0.5")
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
        
        for match in top_matches:
            individual_similarity = match["similarity"]
            
            # Apply individual project confidence thresholds
            # Use a lower threshold (0.75) to catch more multi-project scenarios
            individual_threshold = max(0.75, confidence_threshold - 0.05)
            
            if individual_similarity >= individual_threshold:
                high_confidence_projects.append(match["project_id"])
                individual_confidences.append(individual_similarity)
                
                logging.debug(f"Selected project '{match['project_name']}' with similarity {individual_similarity:.3f}")
        
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


# Global instance for easy access
project_inference_service = ProjectInferenceService()
