"""Ollama summarizer implementation."""

import logging
from typing import List, Dict, Any, Optional
from flask import current_app
from .ollama_client import OllamaClient
from ...abstractions.summarizer import Summarizer

logger = logging.getLogger(__name__)


class OllamaSummarizer(Summarizer):
    """Ollama implementation of the summarizer."""
    
    def __init__(self):
        """Initialize the Ollama summarizer."""
        self.client = OllamaClient()
        
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
        """Summarize search results using Ollama.
        
        Args:
            query: Original search query
            documents_or_chunks: List of document/chunk dictionaries
            search_context: Additional context about the search
            
        Returns:
            Dict containing summarization result
        """
        try:
            logger.info(f"Summarizing {len(documents_or_chunks)} documents/chunks using Ollama")

            # Build context string including project metadata if available
            context = search_context.get('context') if search_context else None
            project_metadata = search_context.get('project_metadata') if search_context else None

            if project_metadata:
                logger.info(f"Project metadata available for summary: {project_metadata.get('project_name', 'unknown') if isinstance(project_metadata, dict) else 'multiple projects'}")

            # Use the existing summarize_documents method
            summary_text = self.summarize_documents(
                documents=documents_or_chunks,
                query=query,
                context=context,
                project_metadata=project_metadata
            )
            
            return {
                'summary': summary_text,
                'method': 'ollama_summarization',
                'confidence': 0.8,  # Default confidence for Ollama
                'documents_count': len(documents_or_chunks),
                'provider': self.client.get_provider_name(),
                'model': self.client.get_model_name()
            }
            
        except Exception as e:
            logger.error(f"Ollama summarization failed: {str(e)}")
            return {
                'summary': f"Error generating summary: {str(e)}",
                'method': 'error_fallback',
                'confidence': 0.0,
                'documents_count': len(documents_or_chunks),
                'provider': 'ollama',
                'model': 'unknown'
            }
    
    def summarize_documents(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        context: Optional[str] = None,
        project_metadata: Optional[Dict] = None
    ) -> str:
        """Summarize a list of documents in relation to a query using Ollama.

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
            
            # Prepare document content (with more aggressive truncation for Ollama)
            doc_content = self._prepare_document_content(documents)
            
            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Summarize the key regulatory findings, project implications, and compliance notes from these documents:\n{doc_content}"
                    )
                }
            ]
            
            logger.info(f"Summarizing {len(documents)} documents using Ollama")
            response = self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, 1500)  # Use config value but cap for Ollama summarization
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
        """Create a formatted response based on the summary using Ollama.
        
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
            
            logger.info("Creating formatted response using Ollama")
            response = self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, 1200)  # Use config value but cap for Ollama response formatting
            )
            
            formatted_response = response["choices"][0]["message"]["content"]
            logger.info("Response formatting completed successfully")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Response creation failed: {str(e)}")
            # Return the summary as fallback
            return f"Based on the available documents, here's what I found:\n\n{summary}"
    
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
Your task is to create a concise summary of the provided documents that directly addresses the user's query.
{project_context}

Key instructions:
1. Focus only on information relevant to the query: "{query}"
2. If the user asks about the project itself (e.g., "what is X project about", "tell me about X"), use the project information above to provide an accurate overview along with document findings
3. If the user asks about the project status or phase, use the Current Status/Phase from the project information above
4. Highlight regulatory considerations, project implications, and critical findings
5. Provide a short summary in 2-3 paragraphs maximum
6. Use professional, clear language suitable for stakeholders and project reviewers
7. Include references to document types, sections, or dates if critical
8. Avoid lengthy legal or technical excerpts; summarize the essence
9. Emphasize insights, decisions, or compliance-related points

{f"Additional context: {context}" if context else ""}

Provide a concise summary tailored for EAO-related decision making that answers the query directly."""

        return prompt
    
    def _build_response_prompt(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Build the response formatting prompt."""
        prompt = """You are a professional assistant helping users understand document information. 

Create a concise, focused response that:
1. Directly answers the user's question based on the summary
2. Uses clear, accessible language
3. Keeps the response to 1-2 short paragraphs maximum
4. Provides the most important information without extensive formatting
5. Avoids multiple sections, headers, or bullet points unless absolutely necessary
6. Gets straight to the point

Keep the response brief and informative."""
        
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
    
    def _prepare_document_content(
        self, 
        documents: List[Dict[str, Any]], 
        max_doc_length: int = None
    ) -> str:
        """Prepare document content for summarization with context-aware truncation for Ollama."""
        content_parts = []
        
        # Calculate reasonable limits based on context length
        # Reserve tokens for prompt, response, and overhead (~30% of context for Ollama)
        available_tokens = int(self.max_context_length * 0.7)
        
        # Use provided max_doc_length or calculate based on context
        if max_doc_length is None:
            max_doc_length = min(2000, available_tokens // max(len(documents), 1))
        
        for i, doc in enumerate(documents, 1):
            title = doc.get("title", f"Document {i}")
            content = doc.get("content", "")
            doc_type = doc.get("document_type", "Unknown")
            
            # Truncate content based on calculated limits
            if len(content) > max_doc_length:
                content = content[:max_doc_length] + "..."
            
            content_parts.append(f"Document {i}: {title} (Type: {doc_type})\n{content}\n")
        
        # Also limit total content length to avoid context overflow
        full_content = "\n".join(content_parts)
        
        if len(full_content) > available_tokens:
            # Truncate and add note
            full_content = full_content[:available_tokens] + "\n\n[Content truncated due to context length limits...]"
        
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