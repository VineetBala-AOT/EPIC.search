"""Document type inference service for automatically detecting document type references in search queries.

This module provides intelligent document type detection capabilities that analyze user queries
to identify document type names and related terms. When users ask questions like "I am looking
for the Inspection Record for project X" or "show me the Environmental Assessment reports",
the system can automatically infer which document type(s) they're referring to and apply
appropriate filtering.

The service includes:
1. Fuzzy matching against comprehensive document type alias dictionaries
2. Database-backed inference using document metadata, keywords, tags, and headings
3. Confidence scoring for automatic document type inference
4. Query cleaning to remove identified document type references
5. Integration with the search pipeline for transparent document type filtering
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from difflib import SequenceMatcher

class DocumentTypeInferenceService:
    """Service for inferring document type context from natural language queries.
    
    This service analyzes search queries to automatically detect document type references
    and provides confident document type ID suggestions when users mention specific document
    types by name without explicitly providing document type IDs.
    """
    
    def __init__(self):
        """Initialize the document type inference service."""
        pass
    
    def infer_document_types_from_query(self, query: str, confidence_threshold: float = 0.7) -> Tuple[List[str], float, Dict[str, Any]]:
        """Infer document type IDs from a natural language query.
        
        Analyzes the query for document type names and related terminology to suggest 
        relevant document type IDs with confidence scoring using fuzzy matching against
        comprehensive alias dictionaries.
        
        Args:
            query (str): The natural language search query
            confidence_threshold (float): Minimum confidence required for automatic inference (default: 0.7)
            
        Returns:
            tuple: A tuple containing:
                - List[str]: Inferred document type IDs (empty if confidence too low)
                - float: Confidence score (0.0 to 1.0)
                - Dict[str, Any]: Detailed inference metadata including entities and reasoning
        """
        # Get document type aliases
        try:
            # Import relative to the src root
            import sys
            import os
            src_path = os.path.join(os.path.dirname(__file__), '..', '..')
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            from utils.document_types import get_all_document_type_aliases
            document_type_aliases = get_all_document_type_aliases()
        except ImportError as e:
            logging.warning(f"Could not import document type aliases: {e}, using empty fallback")
            return [], 0.0, {"error": "Document type aliases not available"}
        
        # Extract potential document type terms from query
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Find matches using fuzzy matching
        matches = []
        
        for type_id, type_info in document_type_aliases.items():
            type_name = type_info["name"]
            aliases = type_info["aliases"]
            
            best_similarity = 0.0
            best_match = None
            match_type = None
            
            # Check for exact matches first (prioritize longer matches)
            # Sort aliases by length (longest first) to get most specific matches
            sorted_aliases = sorted(aliases, key=len, reverse=True)
            for alias in sorted_aliases:
                # Skip very short aliases for exact matching to prevent false positives
                # Require minimum 4 characters for exact matches unless it's a compound term
                if len(alias) < 4 and " " not in alias:
                    continue
                    
                # Use word boundary matching for exact matches
                pattern = r'\b' + re.escape(alias.lower()) + r'\b'
                if re.search(pattern, query_lower):
                    best_similarity = 1.0
                    best_match = alias
                    match_type = "exact"
                    break
            
            # If no exact match, try fuzzy matching
            if best_similarity < 1.0:
                for word in words:
                    # Skip very short words for fuzzy matching to avoid false positives
                    if len(word) < 4:  # Increased from 3 to 4
                        continue
                        
                    for alias in aliases:
                        # Skip very short aliases for fuzzy matching
                        if len(alias) < 4:  # Increased from 3 to 4
                            continue
                            
                        # Calculate similarity using SequenceMatcher
                        similarity = SequenceMatcher(None, word, alias.lower()).ratio()
                        
                        # For compound terms (containing spaces), only match if the word
                        # matches the entire alias or is very similar
                        if " " in alias:
                            # For compound aliases, require very high similarity or exact word match
                            if word.lower() == alias.lower() or similarity >= 0.95:
                                similarity = 1.0 if word.lower() == alias.lower() else similarity
                            else:
                                continue  # Skip compound aliases for partial matches
                        else:
                            # For single-word aliases, check substring matching
                            if len(word) >= 5 and len(alias) >= 5:
                                if word in alias.lower() or alias.lower() in word:
                                    similarity = max(similarity, 0.85)
                        
                        # Very high threshold for fuzzy matches to reduce false positives
                        # Also require minimum length for both word and alias to avoid generic matches
                        if (similarity > best_similarity and similarity >= 0.95 and  # Raised from 0.9 to 0.95
                            len(word) >= 6 and len(alias) >= 6):  # Require longer terms for fuzzy matching
                            best_similarity = similarity
                            best_match = f"{word} → {alias}"
                            match_type = "fuzzy"
            
            # Record significant matches
            if best_similarity >= 0.9:  # Updated threshold to match fuzzy matching
                matches.append({
                    "type_id": type_id,
                    "type_name": type_name,
                    "similarity": best_similarity,
                    "matched_term": best_match,
                    "match_type": match_type
                })
        
        # Sort matches by similarity (highest first)
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Determine the best match(es) and confidence
        if not matches:
            return [], 0.0, {
                "extracted_entities": [],
                "matched_document_types": [],
                "reasoning": ["No document type terms detected in query"],
                "method": "alias_fuzzy_matching"
            }
        
        # Take the highest scoring match(es)
        best_match = matches[0]
        confidence = best_match["similarity"]
        
        # Only return if confidence meets threshold
        if confidence >= confidence_threshold:
            # Get the primary match
            primary_match = best_match
            inferred_ids = [primary_match["type_id"]]
            
            # For overlapping document types, include both 2002 Act and 2018 Act versions
            # This handles cases where documents might be stored with either Act's document type
            document_type_aliases = get_all_document_type_aliases()
            primary_name = primary_match["type_name"]
            
            # Find all document types with the same name (overlapping types)
            overlapping_ids = []
            for type_id, type_info in document_type_aliases.items():
                if (type_info["name"] == primary_name and 
                    type_id != primary_match["type_id"]):
                    overlapping_ids.append(type_id)
                    logging.info(f"Also including overlapping document type: {type_id} ({type_info['name']})")
            
            if overlapping_ids:
                inferred_ids.extend(overlapping_ids)
                overlapping_note = f"Including all {primary_name} document types (both 2002 Act and 2018 Act versions)"
            else:
                overlapping_note = None
            
            logging.info(f"Document type inference: '{primary_match['matched_term']}' → {primary_match['type_name']} "
                        f"(IDs: {inferred_ids}) with {confidence:.3f} confidence")
            
            reasoning = [
                f"Detected document type '{primary_match['matched_term']}' matching '{primary_match['type_name']}' "
                f"with {confidence:.3f} confidence using {primary_match['match_type']} matching"
            ]
            if overlapping_note:
                reasoning.append(overlapping_note)
            
            return inferred_ids, confidence, {
                "extracted_entities": [primary_match["matched_term"]],
                "matched_document_types": [primary_match],
                "reasoning": reasoning,
                "method": "alias_fuzzy_matching"
            }
        else:
            return [], confidence, {
                "extracted_entities": [m["matched_term"] for m in matches[:3]],  # Top 3
                "matched_document_types": matches[:3],
                "reasoning": [
                    f"Best match '{best_match['matched_term']}' → {best_match['type_name']} "
                    f"with {confidence:.3f} confidence below threshold {confidence_threshold}"
                ],
                "method": "alias_fuzzy_matching"
            }

    def clean_query_after_inference(self, query: str, inference_metadata: Dict[str, Any]) -> str:
        """Clean the query by removing identified document type references.

        Args:
            query (str): The original search query
            inference_metadata (Dict[str, Any]): Metadata from the inference process

        Returns:
            str: Cleaned query with document type references removed
        """
        cleaned_query = query

        # Remove matched document type terms
        matched_types = inference_metadata.get("matched_document_types", [])
        for match in matched_types:
            matched_term = match.get("matched_term", "")
            if matched_term:
                if " → " in matched_term:
                    # For fuzzy matches, get the original word that was found in the query
                    original_word = matched_term.split(" → ")[0]
                    term_to_remove = original_word
                else:
                    # For exact matches, remove the matched alias term
                    term_to_remove = matched_term

                # Remove the term using word boundary matching (case insensitive)
                pattern = r'\b' + re.escape(term_to_remove) + r'\b'
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)

        # Clean up extra whitespace
        cleaned_query = ' '.join(cleaned_query.split())

        logging.debug(f"Query cleaning: '{query}' → '{cleaned_query}'")
        return cleaned_query

    def infer_from_database_metadata(
        self,
        query: str,
        project_ids: Optional[List[str]] = None,
        limit: int = 50
    ) -> Tuple[List[str], float, Dict[str, Any]]:
        """Infer document types by analyzing database document metadata.

        This method queries the database to find documents matching the query
        based on keywords, tags, and headings, then analyzes the document_type_id
        distribution to suggest relevant document types.

        Args:
            query: The search query text.
            project_ids: Optional list of project IDs to scope the search.
            limit: Maximum documents to analyze (default: 50).

        Returns:
            tuple: A tuple containing:
                - List[str]: Inferred document type IDs
                - float: Confidence score (0.0 to 1.0)
                - Dict[str, Any]: Detailed inference metadata
        """
        try:
            from flask import current_app
            import psycopg

            # Extract keywords and tags from query
            try:
                from services.keywords.query_keyword_extractor import get_keywords
                from services.tags.tag_extractor import get_tags
                raw_keywords = get_keywords(query)
                query_keywords = [kw for kw, score in raw_keywords] if raw_keywords else []
                query_tags = get_tags(query) or []
            except Exception as e:
                logging.warning(f"Failed to extract keywords/tags: {e}")
                query_keywords = query.lower().split()
                query_tags = []

            logging.info(f"Database document type inference - query: '{query}', keywords: {query_keywords[:5]}")

            # Build WHERE clause for document search
            where_conditions = ["TRUE"]
            params = []
            search_conditions = []

            if query_keywords:
                search_conditions.append("document_keywords ?| %s")
                params.append(query_keywords)
                search_conditions.append("document_headings ?| %s")
                params.append(query_keywords)

            if query_tags:
                search_conditions.append("document_tags ?| %s")
                params.append(query_tags)

            if search_conditions:
                where_conditions.append("(" + " OR ".join(search_conditions) + ")")

            if project_ids and len(project_ids) > 0:
                placeholders = ','.join(['%s'] * len(project_ids))
                where_conditions.append(f"project_id IN ({placeholders})")
                params.extend(project_ids)

            where_clause = " AND ".join(where_conditions)

            # Query for document type distribution
            documents_table = current_app.vector_settings.documents_table_name
            sql = f"""
            SELECT
                document_metadata->>'document_type_id' as doc_type_id,
                document_metadata->>'document_type' as doc_type_name,
                COUNT(*) as doc_count
            FROM {documents_table}
            WHERE {where_clause}
              AND document_metadata->>'document_type_id' IS NOT NULL
            GROUP BY document_metadata->>'document_type_id', document_metadata->>'document_type'
            ORDER BY doc_count DESC
            LIMIT %s
            """
            params.append(limit)

            conn_params = current_app.vector_settings.database_url
            with psycopg.connect(conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    results = cur.fetchall()

            if not results:
                logging.info("Database inference: No matching documents found")
                return [], 0.0, {
                    "method": "database_metadata",
                    "documents_analyzed": 0,
                    "reasoning": ["No documents matched the query keywords/tags"]
                }

            # Analyze document type distribution
            total_docs = sum(r[2] for r in results)
            doc_type_distribution = []
            for doc_type_id, doc_type_name, count in results:
                percentage = count / total_docs if total_docs > 0 else 0
                doc_type_distribution.append({
                    "type_id": doc_type_id,
                    "type_name": doc_type_name,
                    "count": count,
                    "percentage": round(percentage, 3)
                })

            # Determine confidence based on distribution concentration
            # If one type dominates (>50%), high confidence; otherwise lower
            top_type = doc_type_distribution[0] if doc_type_distribution else None
            if top_type and top_type["percentage"] >= 0.5:
                confidence = 0.8 + (top_type["percentage"] - 0.5) * 0.4  # 0.8-1.0
                inferred_ids = [top_type["type_id"]]
                reasoning = [f"Dominant document type: {top_type['type_name']} ({top_type['percentage']*100:.1f}% of {total_docs} matching docs)"]
            elif top_type and top_type["percentage"] >= 0.3:
                confidence = 0.6 + (top_type["percentage"] - 0.3) * 0.5  # 0.6-0.8
                # Include top 2 types if second is significant
                inferred_ids = [top_type["type_id"]]
                if len(doc_type_distribution) > 1 and doc_type_distribution[1]["percentage"] >= 0.2:
                    inferred_ids.append(doc_type_distribution[1]["type_id"])
                reasoning = [f"Primary document type: {top_type['type_name']} ({top_type['percentage']*100:.1f}%)"]
            else:
                # Low concentration - include top 3 types
                confidence = 0.4 + (top_type["percentage"] if top_type else 0) * 0.5
                inferred_ids = [d["type_id"] for d in doc_type_distribution[:3]]
                reasoning = ["Multiple document types found, no dominant type"]

            logging.info(f"Database inference: Found {len(inferred_ids)} types with {confidence:.2f} confidence")

            return inferred_ids, confidence, {
                "method": "database_metadata",
                "documents_analyzed": total_docs,
                "distribution": doc_type_distribution[:5],  # Top 5 for debugging
                "reasoning": reasoning,
                "query_keywords": query_keywords[:10],
                "query_tags": query_tags[:5]
            }

        except Exception as e:
            logging.error(f"Database document type inference failed: {e}")
            return [], 0.0, {
                "method": "database_metadata",
                "error": str(e),
                "reasoning": ["Database inference failed, falling back to alias matching"]
            }

    def infer_with_combined_methods(
        self,
        query: str,
        project_ids: Optional[List[str]] = None,
        confidence_threshold: float = 0.7
    ) -> Tuple[List[str], float, Dict[str, Any]]:
        """Infer document types using combined alias matching and database analysis.

        This method first tries alias-based matching, then enhances results with
        database metadata analysis for higher accuracy.

        Args:
            query: The search query text.
            project_ids: Optional project IDs to scope database search.
            confidence_threshold: Minimum confidence for inference.

        Returns:
            tuple: A tuple containing:
                - List[str]: Inferred document type IDs
                - float: Combined confidence score
                - Dict[str, Any]: Detailed inference metadata
        """
        # Step 1: Try alias-based matching
        alias_ids, alias_conf, alias_meta = self.infer_document_types_from_query(query, confidence_threshold)

        # Step 2: Try database-based matching
        db_ids, db_conf, db_meta = self.infer_from_database_metadata(query, project_ids)

        # Combine results
        if alias_ids and alias_conf >= confidence_threshold:
            # Alias matching succeeded with high confidence
            if db_ids and db_conf >= 0.5:
                # Database also found types - merge if they agree
                combined_ids = list(set(alias_ids + db_ids))
                combined_conf = max(alias_conf, db_conf)
                method = "combined_alias_and_database"
            else:
                combined_ids = alias_ids
                combined_conf = alias_conf
                method = "alias_matching"
        elif db_ids and db_conf >= confidence_threshold:
            # Only database matching succeeded
            combined_ids = db_ids
            combined_conf = db_conf
            method = "database_metadata"
        elif alias_ids or db_ids:
            # Lower confidence results available
            combined_ids = alias_ids if alias_ids else db_ids
            combined_conf = max(alias_conf, db_conf)
            method = "low_confidence_fallback"
        else:
            # No results from either method
            combined_ids = []
            combined_conf = 0.0
            method = "no_match"

        logging.info(f"Combined inference: {len(combined_ids)} types, {combined_conf:.2f} confidence, method={method}")

        return combined_ids, combined_conf, {
            "method": method,
            "alias_inference": alias_meta,
            "database_inference": db_meta,
            "reasoning": [
                f"Alias matching: {len(alias_ids)} types at {alias_conf:.2f} confidence",
                f"Database matching: {len(db_ids)} types at {db_conf:.2f} confidence",
                f"Final: {len(combined_ids)} types using {method}"
            ]
        }


# Create a singleton instance for easy importing
document_type_inference_service = DocumentTypeInferenceService()