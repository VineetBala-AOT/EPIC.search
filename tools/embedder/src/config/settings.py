import os
import multiprocessing

from functools import lru_cache
from dotenv import load_dotenv
from pydantic import BaseModel, Field

"""
Settings configuration module for the EPIC.search Embedder.

This module defines configuration classes using Pydantic models for type safety and validation.
It loads environment variables using dotenv and provides a cached settings instance
through the get_settings function.
"""

load_dotenv()

class EmbeddingModelSettings(BaseModel):
    """
    Configuration settings for the embedding model.
    
    Attributes:
        model_name (str): Name of the sentence transformer model to use for document embeddings
    """
    model_name: str = Field(default_factory=lambda:
        os.environ.get("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
    )

class KeywordExtractionSettings(BaseModel):
    """
    Configuration settings for the keyword extraction model.
    
    Attributes:
        model_name (str): Name of the sentence transformer model to use for keyword extraction
    """
    model_name: str = Field(default_factory=lambda:
        os.environ.get("KEYWORD_MODEL_NAME", "all-mpnet-base-v2")
    )

class LoggingDatabaseSettings(BaseModel):
    """
    Configuration for the logging database connection.
    
    Attributes:
        db_url (str): Database connection URL for the processing logs database
    """
    db_url: str = Field(default_factory=lambda: os.getenv("LOGS_DATABASE_URL"))

class VectorStoreSettings(BaseModel):
    """
    Configuration for the vector database.
    
    Attributes:
        db_url (str): Database connection URL for the vector database
        embedding_dimensions (int): Dimensionality of the embeddings
        auto_create_extension (bool): Whether to automatically create the pgvector extension
        reset_db (bool): Whether to drop and recreate all tables on startup (dev/test only)
    """
    db_url: str = Field(default_factory=lambda: os.getenv("VECTOR_DB_URL"))
    embedding_dimensions: int = os.environ.get("EMBEDDING_DIMENSIONS", 768)
    auto_create_extension: bool = Field(default_factory=lambda: os.environ.get("AUTO_CREATE_PGVECTOR_EXTENSION", "True").lower() in ("true", "1", "yes"))
    reset_db: bool = Field(default_factory=lambda: os.environ.get("RESET_DB", "False").lower() in ("true", "1", "yes"))

class ChunkSettings(BaseModel):
    """
    Configuration for document chunking behavior.
    
    Attributes:
        chunk_size (int): Size of text chunks in characters
        chunk_overlap (int): Number of characters to overlap between consecutive chunks
    """
    chunk_size: int = Field(default_factory=lambda: os.environ.get("CHUNK_SIZE", 1000))
    chunk_overlap: int = Field(default_factory=lambda: os.environ.get("CHUNK_OVERLAP", 200))


class MultiProcessingSettings(BaseModel):
    """
    Configuration for parallel processing.
    
    Attributes:
        files_concurrency_size (int): Number of documents to process in parallel (use all cores for server)
        chunk_insert_batch_size (int): Number of chunks to insert per database batch
        keyword_extraction_workers (int): Number of threads per document for keyword extraction
        keyword_extraction_mode (str): Mode for keyword extraction (standard, fast, simplified)
    """
    files_concurrency_size: int = Field(default_factory=lambda: _parse_files_concurrency())
    chunk_insert_batch_size: int = Field(default_factory=lambda: int(os.environ.get("CHUNK_INSERT_BATCH_SIZE", 25)))
    keyword_extraction_workers: int = Field(default_factory=lambda: _parse_keyword_workers())
    keyword_extraction_mode: str = Field(default_factory=lambda: os.environ.get("KEYWORD_EXTRACTION_MODE", "standard"))
    chunk_insert_batch_size: int = Field(default_factory=lambda: int(os.environ.get("CHUNK_INSERT_BATCH_SIZE", 25)))
    keyword_extraction_workers: int = Field(default_factory=lambda: _parse_keyword_workers())

def _parse_files_concurrency():
    """Parse FILES_CONCURRENCY_SIZE with intelligent auto-calculation"""
    env_value = os.environ.get("FILES_CONCURRENCY_SIZE", "")
    cpu_count = multiprocessing.cpu_count()
    
    # Handle empty or auto values with intelligent defaults
    if not env_value or env_value.lower() == 'auto':
        # For high-core systems, use half the cores to avoid over-parallelization
        if cpu_count >= 16:
            return cpu_count // 2
        else:
            return cpu_count
    elif env_value.lower() == 'auto-full':
        return cpu_count
    elif env_value.lower() == 'auto-conservative':
        return max(1, cpu_count // 4)
    
    try:
        # Try to parse as integer
        return int(env_value)
    except ValueError:
        # If parsing fails, fall back to intelligent auto
        print(f"[WARNING] Invalid FILES_CONCURRENCY_SIZE value '{env_value}', using auto calculation ({cpu_count // 2 if cpu_count >= 16 else cpu_count})")
        return cpu_count // 2 if cpu_count >= 16 else cpu_count

def _parse_keyword_workers():
    """Parse KEYWORD_EXTRACTION_WORKERS with intelligent auto-calculation"""
    env_value = os.environ.get("KEYWORD_EXTRACTION_WORKERS", "")
    cpu_count = multiprocessing.cpu_count()
    
    # Handle empty or auto values
    if not env_value or env_value.lower() == 'auto':
        # Conservative threading for KeyBERT bottleneck: 2 for high-core systems
        if cpu_count >= 16:
            return 2
        elif cpu_count >= 8:
            return 3
        else:
            return 4
    elif env_value.lower() == 'auto-aggressive':
        return 4
    elif env_value.lower() == 'auto-conservative':
        return 1
    
    try:
        # Try to parse as integer
        return int(env_value)
    except ValueError:
        # If parsing fails, fall back to intelligent auto
        default_workers = 2 if cpu_count >= 16 else (3 if cpu_count >= 8 else 4)
        print(f"[WARNING] Invalid KEYWORD_EXTRACTION_WORKERS value '{env_value}', using auto calculation ({default_workers})")
        return default_workers


class S3Settings(BaseModel):
    """
    Configuration for S3 storage connection.
    
    Attributes:
        bucket_name (str): Name of the S3 bucket
        region_name (str): S3 region for the S3 bucket
        access_key_id (str): S3 access key ID
        secret_access_key (str): S3 secret access key
        endpoint_uri (str): Endpoint URI for S3-compatible storage
    """
    bucket_name: str = Field(default_factory=lambda: os.getenv("S3_BUCKET_NAME"))
    region_name: str = Field(default_factory=lambda: os.getenv("S3_REGION_NAME"))
    access_key_id: str = Field(default_factory=lambda: os.getenv("S3_ACCESS_KEY_ID"))
    secret_access_key: str = Field(
        default_factory=lambda: os.getenv("S3_SECRET_ACCESS_KEY")
    )
    endpoint_uri: str = Field(default_factory=lambda: os.getenv("S3_ENDPOINT_URI"))

class DocumentSearchSettings(BaseModel):
    """
    Configuration for the document search API.
    
    Attributes:
        document_search_url (str): Base URL for the document search API
    """
    document_search_url: str = Field(default_factory=lambda: os.getenv("DOCUMENT_SEARCH_URL"))


class ApiPaginationSettings(BaseModel):
    """
    Configuration for API pagination settings.
    
    Attributes:
        project_page_size (int): Number of projects to fetch per API call
        documents_page_size (int): Number of documents to fetch per API call
    """
    project_page_size: int = Field(default_factory=lambda: int(os.environ.get("GET_PROJECT_PAGE", 1)))
    documents_page_size: int = Field(default_factory=lambda: int(os.environ.get("GET_DOCS_PAGE", 1000)))


class Settings(BaseModel):
    """
    Main settings class that combines all configuration categories.
    
    This class aggregates all the specialized setting classes into a single
    configuration object for the application.
    
    Attributes:
        embedding_model_settings (EmbeddingModelSettings): Settings for the embedding model
        keyword_extraction_settings (KeywordExtractionSettings): Settings for the keyword extraction model
        vector_store_settings (VectorStoreSettings): Settings for the vector database
        multi_processing_settings (MultiProcessingSettings): Settings for parallel processing
        s3_settings (S3Settings): Settings for S3 storage
        chunk_settings (ChunkSettings): Settings for document chunking
        logging_db_settings (LoggingDatabaseSettings): Settings for the logging database
        document_search_settings (DocumentSearchSettings): Settings for the document search API
    """
    embedding_model_settings: EmbeddingModelSettings = Field(
        default_factory=EmbeddingModelSettings
    )
    keyword_extraction_settings: KeywordExtractionSettings = Field(
        default_factory=KeywordExtractionSettings
    )
    vector_store_settings: VectorStoreSettings = Field(
        default_factory=VectorStoreSettings
    )
    multi_processing_settings: MultiProcessingSettings = Field(
        default_factory=MultiProcessingSettings
    )
    s3_settings: S3Settings = Field(default_factory=S3Settings)
    chunk_settings: ChunkSettings = Field(default_factory=ChunkSettings)
    logging_db_settings: LoggingDatabaseSettings = Field(
        default_factory=LoggingDatabaseSettings
    )
    document_search_settings: DocumentSearchSettings = Field(
        default_factory=DocumentSearchSettings
    )
    api_pagination_settings: ApiPaginationSettings = Field(
        default_factory=ApiPaginationSettings
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get the application settings, using cached values if available.
    
    This function is decorated with lru_cache to ensure settings are only loaded once
    per application instance, improving performance.
    
    Returns:
        Settings: The application settings object
    """
    settings = Settings()
    return settings
