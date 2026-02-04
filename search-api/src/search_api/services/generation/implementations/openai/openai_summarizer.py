"""OpenAI summarizer implementation."""

import logging
from typing import List, Dict, Any, Optional
from flask import current_app
from .openai_client import OpenAIClient
from ...abstractions.summarizer import Summarizer

logger = logging.getLogger(__name__)


class OpenAISummarizer(Summarizer):
    """OpenAI implementation of the summarizer."""
    
    def __init__(self):
        """Initialize the OpenAI summarizer."""
        self.client = OpenAIClient()
        
        # Load configuration with fallbacks
        self.temperature = getattr(current_app.config, 'LLM_TEMPERATURE', 0.3)
        self.max_tokens = getattr(current_app.config, 'LLM_MAX_TOKENS', 1000)
        self.max_context_length = getattr(current_app.config, 'LLM_MAX_CONTEXT_LENGTH', 8192)
    
    def summarize_search_results(
        self,
        query: str,
        documents_or_chunks: List[Dict[str, Any]],
        search_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Summarize search results using OpenAI.

        Args:
            query: Original search query
            documents_or_chunks: List of document/chunk dictionaries
            search_context: Additional context about the search (may include project_metadata)

        Returns:
            Dict containing summarization result
        """
        try:
            logger.info(f"Summarizing {len(documents_or_chunks)} documents/chunks using OpenAI")

            # Build context string including project metadata if available
            context = search_context.get('context') if search_context else None
            project_metadata = search_context.get('project_metadata') if search_context else None

            if project_metadata:
                logger.info(f"Project metadata available for summary: {project_metadata.get('project_name', 'unknown')}")

            # Use the existing summarize_documents method
            summary_text = self.summarize_documents(
                documents=documents_or_chunks,
                query=query,
                context=context,
                project_metadata=project_metadata
            )
            
            return {
                'summary': summary_text,
                'method': 'openai_summarization',
                'confidence': 0.8,  # Default confidence for OpenAI
                'documents_count': len(documents_or_chunks),
                'provider': self.client.get_provider_name(),
                'model': self.client.get_model_name()
            }
            
        except Exception as e:
            logger.error(f"OpenAI summarization failed: {str(e)}")
            return {
                'summary': f"Error generating summary: {str(e)}",
                'method': 'error_fallback',
                'confidence': 0.0,
                'documents_count': len(documents_or_chunks),
                'provider': 'openai',
                'model': 'unknown'
            }
    
    def summarize_documents(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        context: Optional[str] = None,
        project_metadata: Optional[Dict] = None
    ) -> str:
        """Summarize a list of documents in relation to a query using OpenAI.

        Args:
            documents: List of document dictionaries with content and metadata.
            query: The original search query for context.
            context: Optional additional context for summarization.
            project_metadata: Optional project metadata (description, status, etc.)

        Returns:
            str: A comprehensive summary of the documents.

        Raises:
            Exception: If summarization fails.
        """
        try:
            if not documents:
                return "No documents found to summarize."

            # Build the summarization prompt with project metadata
            prompt = self._build_summarization_prompt(query, context, project_metadata)

            # Prepare document content
            doc_content = self._prepare_document_content(documents)

            # Log metadata availability for debugging
            is_project_query = self._is_project_level_query(query) if project_metadata else False
            logger.info(f"Summarizer received project_metadata: {project_metadata is not None}, is_project_level_query: {is_project_query}")
            if project_metadata:
                logger.info(f"Project metadata keys: {list(project_metadata.keys())}, status='{project_metadata.get('status', '')}', description='{str(project_metadata.get('description', ''))[:80]}...'")

            # Build user message - include project metadata directly when available
            # for project-level queries so the LLM uses it as primary source
            user_content = f"Query: {query}\n\n"

            if project_metadata and self._is_project_level_query(query):
                # For project-level queries, present metadata as primary info
                user_content += "IMPORTANT: Use the following verified project information to answer the query accurately.\n\n"
                user_content += self._format_project_metadata_for_message(project_metadata)
                user_content += f"\n\nSupporting documents from the project:\n{doc_content}"
            else:
                user_content += f"Summarize the key regulatory findings, project implications, and compliance notes from these documents:\n{doc_content}"

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content}
            ]
            
            logger.info(f"Summarizing {len(documents)} documents using OpenAI")
            response = self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, 2000)  # Use config value but cap for summarization
            )
            
            summary = response["choices"][0]["message"]["content"]
            logger.info("Document summarization completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Document summarization failed: {str(e)}")
            # Return a basic fallback summary
            return self._fallback_summary(documents, query)
    
    def create_response(
        self,
        summary: str,
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a formatted response based on the summary using OpenAI.
        
        Args:
            summary: The document summary.
            query: The original search query.
            metadata: Optional metadata about the search.
            
        Returns:
            str: A well-formatted response.
            
        Raises:
            Exception: If response creation fails.
        """
        try:
            # Build the response formatting prompt
            prompt = self._build_response_prompt(metadata)
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Query: {query}\n\nSummary: {summary}\n\nCreate a comprehensive response."}
            ]
            
            logger.info("Creating formatted response using OpenAI")
            response = self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, 1500)  # Use config value but cap for response formatting
            )
            
            formatted_response = response["choices"][0]["message"]["content"]
            logger.info("Response formatting completed successfully")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Response creation failed: {str(e)}")
            # Return the summary as fallback
            return f"Based on the available documents, here's what I found:\n\n{summary}"
    
    def _is_project_level_query(self, query: str) -> bool:
        """Check if the query is asking about the project itself (overview, status, etc.)."""
        query_lower = query.lower()
        project_level_indicators = [
            "what is", "tell me about", "describe", "overview of", "summary of",
            "about the", "all about", "information on", "details of", "details about",
            "current status", "what status", "project status", "phase of",
            "current phase", "what phase", "who is the proponent", "proponent of",
            "where is", "location of", "what type", "type of project",
            "what region", "decision on", "ea decision", "eac decision",
            "when was", "decision date", "who owns", "who operates",
        ]
        return any(indicator in query_lower for indicator in project_level_indicators)

    def _format_project_metadata_for_message(self, project_metadata: Dict) -> str:
        """Format project metadata as a clear, structured block for the user message."""
        parts = ["--- VERIFIED PROJECT DATA (from official registry) ---"]
        field_map = [
            ("project_name", "Project Name"),
            ("description", "Description"),
            ("status", "Current Status/Phase"),
            ("type", "Project Type"),
            ("proponent", "Proponent"),
            ("region", "Region"),
            ("location", "Location"),
            ("ea_decision", "EA Decision"),
            ("decision_date", "Decision Date"),
        ]
        for key, label in field_map:
            value = project_metadata.get(key, "")
            if value:
                parts.append(f"{label}: {value}")
        parts.append("--- END PROJECT DATA ---")
        return "\n".join(parts)

    def _build_summarization_prompt(self, query: str, context: Optional[str] = None,
                                    project_metadata: Optional[Dict] = None) -> str:
        """Build a summarization prompt tailored for EAO content."""

        # Build project context section if metadata is available
        project_context = ""
        if project_metadata:
            project_parts = []
            if project_metadata.get("project_name"):
                project_parts.append(f"**Project Name:** {project_metadata['project_name']}")
            if project_metadata.get("description"):
                project_parts.append(f"**Project Description:** {project_metadata['description']}")
            if project_metadata.get("status"):
                project_parts.append(f"**Current Status/Phase:** {project_metadata['status']}")
            if project_metadata.get("type"):
                project_parts.append(f"**Project Type:** {project_metadata['type']}")
            if project_metadata.get("proponent"):
                project_parts.append(f"**Proponent:** {project_metadata['proponent']}")
            if project_metadata.get("region"):
                project_parts.append(f"**Region:** {project_metadata['region']}")
            if project_metadata.get("location"):
                project_parts.append(f"**Location:** {project_metadata['location']}")
            if project_metadata.get("ea_decision"):
                project_parts.append(f"**EA Decision:** {project_metadata['ea_decision']}")
            if project_metadata.get("decision_date"):
                project_parts.append(f"**Decision Date:** {project_metadata['decision_date']}")

            if project_parts:
                project_context = "\n\n**PROJECT INFORMATION (use this to provide accurate context in your summary):**\n" + "\n".join(project_parts)

        prompt = f"""You are an expert analyst specializing in Environmental Assessment Office (EAO) reports and regulatory documentation.
Your task is to create a concise summary that directly addresses the user's query.
{project_context}

CRITICAL RULES:
- When the user message contains "VERIFIED PROJECT DATA", you MUST use that data as your PRIMARY and AUTHORITATIVE source for project details (name, description, status, proponent, location, type, EA decision, decision date).
- DO NOT contradict or ignore the verified project data. State the facts from it directly.
- Use information from the documents only to SUPPLEMENT the verified project data, not to override it.
- If the query asks about project status, phase, or overview, your answer MUST reflect the verified project data.

Additional guidelines:
1. Focus only on information relevant to the query: "{query}"
2. Provide a short summary in 2-3 paragraphs maximum
3. Use professional, clear language suitable for stakeholders and project reviewers
4. Highlight regulatory considerations, project implications, and critical findings
5. Avoid lengthy legal or technical excerpts; summarize the essence

{f"Additional context: {context}" if context else ""}

Provide a concise summary tailored for EAO-related decision making that answers the query directly."""
        return prompt
    
    def _build_response_prompt(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Build a response prompt tailored for EAO-focused content."""
        prompt = """
You are a professional assistant summarizing Environmental Assessment Office (EAO) documents.
Your goal is to create a concise, focused response based on the summary.

Instructions:
1. Directly answer the user's question using information from the summary
2. Highlight regulatory, compliance, and project-relevant insights
3. Keep response to 1-2 short paragraphs maximum
4. Use clear, accessible language for decision makers and project stakeholders
5. Avoid headers, bullet points, or multiple sections unless critical
6. Get straight to the point with actionable or key information
"""

        if metadata:
            search_info = []
            if metadata.get("document_count"):
                search_info.append(f"searched {metadata['document_count']} documents")
            if metadata.get("project_ids"):
                search_info.append(f"across projects: {', '.join(metadata['project_ids'])}")
            if metadata.get("document_types"):
                search_info.append(f"document types: {', '.join(metadata['document_types'])}")

            if search_info:
                prompt += f"\n\nContext: This response is based on information from {', '.join(search_info)}."

        return prompt
    
    def _prepare_document_content(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare document content for summarization with EAO focus.

        Prioritizes sections relevant to environmental assessment, regulatory compliance,
        and project implications, and truncates less critical content to fit context limits.
        """
        content_parts = []

        # Reserve ~25% for prompt and response overhead
        available_tokens = int(self.max_context_length * 0.75)
        max_doc_length = min(3000, available_tokens // max(len(documents), 1))

        for i, doc in enumerate(documents, 1):
            title = doc.get("title", f"Document {i}")
            content = doc.get("content", "")
            doc_type = doc.get("document_type", "Unknown")
            section_info = doc.get("section", "")

            # Optional: Highlight EAO-specific sections
            eao_keywords = ["environmental assessment", "EAO", "compliance", "mitigation", "regulatory", "permit"]
            lines = content.splitlines()
            relevant_lines = [line for line in lines if any(k.lower() in line.lower() for k in eao_keywords)]

            # If relevant lines found, use them; else fallback to full content
            if relevant_lines:
                content = "\n".join(relevant_lines)
            else:
                content = "\n".join(lines[:500])  # Take first 500 lines/characters if no keyword match

            # Truncate content if exceeds max_doc_length
            if len(content) > max_doc_length:
                content = content[:max_doc_length] + "..."

            content_parts.append(
                f"Document {i}: {title} (Type: {doc_type}{', Section: ' + section_info if section_info else ''})\n{content}\n"
            )

        full_content = "\n".join(content_parts)

        # Safety check for overall context length
        if len(full_content) > available_tokens:
            full_content = full_content[:available_tokens] + "\n\n[Content truncated due to context limits...]"

        return full_content
    
    def _fallback_summary(self, documents: List[Dict[str, Any]], query: str) -> str:
        """Provide a basic fallback summary when LLM fails."""
        doc_count = len(documents)
        doc_types = list(set(doc.get("document_type", "Unknown") for doc in documents))
        
        summary = f"Found {doc_count} relevant documents related to '{query}'.\n\n"
        summary += f"Document types included: {', '.join(doc_types)}\n\n"
        
        # Include first few document titles
        for i, doc in enumerate(documents[:3], 1):
            title = doc.get("title", f"Document {i}")
            summary += f"- {title}\n"
        
        if len(documents) > 3:
            summary += f"... and {len(documents) - 3} more documents.\n"
        
        summary += "\nPlease review the individual documents for detailed information."
        
        return summary