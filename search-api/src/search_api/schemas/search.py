"""Search schemas for request and response validation.

This module contains Marshmallow schemas for validating and serializing
search-related API requests and responses, including document structures
and search parameters.

The search request schema supports optional filtering parameters for enhanced
search capabilities while maintaining backward compatibility.
"""

from marshmallow import EXCLUDE, Schema, fields


class DocumentSchema(Schema):
    """Schema for document objects returned in search results.
    
    Defines the structure and validation for individual document items
    that are returned as part of search responses from the vector search API.
    Handles both document and document_chunks response formats.
    """

    document_id = fields.Str(data_key="document_id", metadata={"description": "Unique identifier for the document"})
    document_type = fields.Str(data_key="document_type", metadata={"description": "Type/category of the document"})
    document_name = fields.Str(data_key="document_name", metadata={"description": "Original name of the document"})
    document_saved_name = fields.Str(data_key="document_saved_name", metadata={"description": "Saved/processed name of the document"})
    document_date = fields.Str(data_key="document_date", metadata={"description": "Date of the document"})
    page_number = fields.Str(data_key="page_number", metadata={"description": "Page number within the document where content was found"})
    project_id = fields.Str(data_key="project_id", metadata={"description": "ID of the project this document belongs to"})
    project_name = fields.Str(data_key="project_name", metadata={"description": "Name of the project this document belongs to"})
    proponent_name = fields.Str(data_key="proponent_name", metadata={"description": "Name of the project proponent"})
    content = fields.Str(data_key="content", metadata={"description": "Relevant content/text extracted from the document"})
    s3_key = fields.Str(data_key="s3_key", metadata={"description": "S3 storage key for the document file"})
    relevance_score = fields.Float(data_key="relevance_score", metadata={"description": "Relevance score for this document/chunk"})
    search_mode = fields.Str(data_key="search_mode", metadata={"description": "Search mode used to find this result"})


class SearchResponseSchema(Schema):
    """Schema for search API response validation.
    
    Defines the structure for responses returned by the search endpoints.
    The response can contain either 'documents' (metadata-focused results) or 
    'document_chunks' (content-focused results) depending on the search type performed.
    """

    class Meta:  # pylint: disable=too-few-public-methods
        """Exclude unknown fields in the deserialized output."""

        unknown = EXCLUDE

    documents = fields.List(
        fields.Nested(DocumentSchema), 
        data_key="documents",
        required=False,
        allow_none=True,
        metadata={"description": "List of documents found by the search operation (metadata-focused results)"}
    )
    
    document_chunks = fields.List(
        fields.Nested(DocumentSchema), 
        data_key="document_chunks",
        required=False,
        allow_none=True,
        metadata={"description": "List of document chunks found by the search operation (content-focused results)"}
    )


class SearchRequestSchema(Schema):
    """Schema for search API request validation.
    
    Validates incoming search requests, ensuring required fields are present
    and optional parameters are properly formatted for the search operation.
    
    All parameters except 'question' are optional and maintain backward compatibility
    with existing API clients.
    """

    class Meta:  # pylint: disable=too-few-public-methods
        """Exclude unknown fields in the deserialized output."""

        unknown = EXCLUDE

    question = fields.Str(
        data_key="question", 
        required=True,
        metadata={"description": "The search query/question to find relevant documents for"}
    )
    projectIds = fields.List(
        fields.Str(), 
        data_key="projectIds", 
        required=False, 
        allow_none=True,
        metadata={"description": "Optional list of project IDs to filter search results by. If not provided, searches across all projects."}
    )
    documentTypeIds = fields.List(
        fields.Str(), 
        data_key="documentTypeIds", 
        required=False, 
        allow_none=True,
        metadata={"description": "Optional list of document type IDs to filter search results by. If not provided, searches across all document types."}
    )
    inference = fields.List(
        fields.Str(), 
        data_key="inference", 
        required=False, 
        allow_none=True,
        metadata={"description": "Optional list of inference types to enable (e.g., ['PROJECT', 'DOCUMENTTYPE']). If not provided, uses the vector search API's default inference settings."}
    )


class SimilaritySearchRequestSchema(Schema):
    """Schema for similarity search API request validation.
    Used for POST /api/search/similar to find similar documents.
    """
    class Meta:
        unknown = EXCLUDE

    document_id = fields.Str(data_key="documentId", required=True, metadata={"description": "The source document ID to find similarities for."})
    projectIds = fields.List(
        fields.Str(),
        data_key="projectIds",
        required=False,
        allow_none=True,
        metadata={"description": "Optional list of project IDs to filter similar documents."}
    )
    limit = fields.Int(required=False, allow_none=True, metadata={"description": "Maximum number of similar documents to return (default: 10)"})


class SimilarDocumentSchema(Schema):
    """Schema for individual similar document objects."""

    document_id = fields.Str()
    document_keywords = fields.List(fields.Str())
    document_tags = fields.List(fields.Str())
    document_headings = fields.List(fields.Str())
    project_id = fields.Str()
    similarity_score = fields.Float()
    created_at = fields.DateTime()


class SimilaritySearchMetricsSchema(Schema):
    """Schema for search metrics in similarity search responses."""

    embedding_retrieval_ms = fields.Float()
    similarity_search_ms = fields.Float()
    formatting_ms = fields.Float()
    total_search_ms = fields.Float()


class SimilaritySearchResponseSchema(Schema):
    """Schema for similarity search API response validation.
    
    Defines the structure for responses returned by the similarity search endpoint,
    containing the list of similar documents found and search metrics.
    """

    similar_documents = fields.List(fields.Nested(SimilarDocumentSchema))
    search_metrics = fields.Nested(SimilaritySearchMetricsSchema)
