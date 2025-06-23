# Copyright © 2024 Province of British Columbia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration management for the Vector Search API service.

This module centralizes all configuration settings for the application,
implementing a strongly-typed approach to configuration management through
specialized configuration classes. It handles:

1. Loading environment variables from .env files
2. Providing environment-specific configuration classes
3. Defining strongly-typed access to configuration values
4. Setting sensible defaults for all configuration parameters

The configuration is organized into logical groups (VectorSettings, SearchSettings, 
ModelSettings) to provide structured access to related settings throughout the application.
All components should access configuration through these settings classes rather than
directly reading environment variables.
"""

import os
import sys
from datetime import timedelta
from dotenv import find_dotenv, load_dotenv
from typing import Dict, Any

# Load all environment variables from .env file in the project root
load_dotenv(find_dotenv())


class VectorSettings:
    """Vector database configuration settings.
    
    Provides strongly-typed access to configuration parameters related to
    the vector database, including connection details, table names, and
    vector dimensions.
    
    This class encapsulates all database-related configuration to provide
    a clean interface for the application components that need these settings.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize vector settings from a configuration dictionary.
        
        Args:
            config_dict: A dictionary containing configuration values,
                        typically from Flask's app.config
        """
        self._config = config_dict
    
    @property
    def vector_table_name(self) -> str:
        """Get the name of the database table storing vector embeddings.
        
        Returns:
            str: The configured vector table name
        """
        return self._config.get("VECTOR_TABLE")
    
    @property
    def embedding_dimensions(self) -> int:
        """Get the dimensionality of the vector embeddings.
        
        Returns:
            int: The number of dimensions in the embedding vectors (default: 768)
        """
        return self._config.get("EMBEDDING_DIMENSIONS", 768)
    
    @property
    def database_url(self) -> str:
        """Get the PostgreSQL connection string for the vector database.
        
        Returns:
            str: The database connection URL
        """
        return self._config.get("VECTOR_DB_URL")
    
    @property
    def time_partition_interval(self) -> timedelta:
        """Get the time interval for database partitioning.
        
        Returns:
            timedelta: The time partition interval
        """
        return self._config.get("TIME_PARTITION_INTERVAL")


class SearchSettings:
    """Search configuration settings.
    
    Provides strongly-typed access to configuration parameters related to
    search operations, including result counts, batch sizes, and search limits.
    
    This class encapsulates search-specific settings to ensure consistent
    configuration across all search-related components.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize search settings from a configuration dictionary.
        
        Args:
            config_dict: A dictionary containing configuration values,
                        typically from Flask's app.config
        """
        self._config = config_dict
    
    @property
    def keyword_fetch_count(self) -> int:
        """Get the number of keyword search results to retrieve initially.
        
        Returns:
            int: The maximum number of keyword search results to fetch
        """
        return self._config.get("KEYWORD_FETCH_COUNT")
    
    @property
    def semantic_fetch_count(self) -> int:
        """Get the number of semantic search results to retrieve initially.
        
        Returns:
            int: The maximum number of semantic search results to fetch
        """
        return self._config.get("SEMANTIC_FETCH_COUNT")
    
    @property
    def top_record_count(self) -> int:
        """Get the number of top records to return after re-ranking.
        
        Returns:
            int: The number of final results to return to the client
        """
        return self._config.get("TOP_RECORD_COUNT")
    
    @property
    def reranker_batch_size(self) -> int:
        """Get the batch size for processing document pairs in the re-ranker.
        
        Returns:
            int: The batch size for the re-ranker (default: 8)
        """
        return self._config.get("RERANKER_BATCH_SIZE", 8)
    
    @property
    def min_relevance_score(self) -> float:
        """Get the minimum relevance score threshold for re-ranked results.
        
        Returns:
            float: The minimum relevance score (default: 0.0)
        """
        return float(self._config.get("MIN_RELEVANCE_SCORE", 0.0))


class ModelSettings:
    """Machine learning model configuration settings.
    
    Provides strongly-typed access to configuration parameters related to
    ML models used in the application, including model names and paths.
    
    This class encapsulates all model-specific configuration to allow for
    easy model swapping and configuration across the application.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize model settings from a configuration dictionary.
        
        Args:
            config_dict: A dictionary containing configuration values,
                        typically from Flask's app.config
        """
        self._config = config_dict
    
    @property
    def cross_encoder_model(self) -> str:
        """Get the name or path of the cross-encoder model for re-ranking.
        
        Returns:
            str: The cross-encoder model identifier
        """
        return self._config.get("CROSS_ENCODER_MODEL")
    
    @property
    def embedding_model_name(self) -> str:
        """Get the name or path of the embedding model for semantic search.
        
        Returns:
            str: The embedding model identifier
        """
        return self._config.get("EMBEDDING_MODEL_NAME")
    
    @property
    def keyword_model_name(self) -> str:
        """Get the name or path of the model used for keyword extraction.
        
        Returns:
            str: The keyword model identifier
        """
        return self._config.get("KEYWORD_MODEL_NAME")


def get_named_config(config_name: str = "development"):
    """Return the configuration object based on the name.

    :raise: KeyError: if an unknown configuration is requested
    """
    if config_name in ["production", "staging", "default"]:
        config = ProdConfig()
    elif config_name == "testing":
        config = TestConfig()
    elif config_name == "development":
        config = DevConfig()
    elif config_name == "docker":
        config = DockerConfig()
    else:
        raise KeyError(f"Unknown configuration '{config_name}'")
    return config


class _Config:  # pylint: disable=too-few-public-methods
    """Base class configuration that should set reasonable defaults for all the other configurations."""

    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

    SECRET_KEY = "a secret"

    TESTING = False
    DEBUG = False

    # Vector Database Configuration
    VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    TIME_PARTITION_INTERVAL = timedelta(days=7)

    # Search Configuration
    VECTOR_TABLE = os.getenv("VECTOR_TABLE", "document_tags")
    KEYWORD_FETCH_COUNT = int(os.getenv("KEYWORD_FETCH_COUNT", "100"))
    SEMANTIC_FETCH_COUNT = int(os.getenv("SEMANTIC_FETCH_COUNT", "100"))
    TOP_RECORD_COUNT = int(os.getenv("TOP_RECORD_COUNT", "10"))
    RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "8"))

    # ML Model Configuration
    CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
    KEYWORD_MODEL_NAME = os.getenv("KEYWORD_MODEL_NAME", "all-mpnet-base-v2")

    # Minimum relevance score for re-ranked results
    MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.0"))


class DevConfig(_Config):  # pylint: disable=too-few-public-methods
    """Dev Config."""

    TESTING = False
    DEBUG = True


class TestConfig(_Config):  # pylint: disable=too-few-public-methods
    """In support of testing only. Used by the pytest suite."""

    DEBUG = True
    TESTING = True

    # Override with test-specific settings if needed
    VECTOR_DB_URL = os.getenv("TEST_VECTOR_DB_URL", _Config.VECTOR_DB_URL)


class DockerConfig(_Config):  # pylint: disable=too-few-public-methods
    """Docker environment configuration."""
    
    # No specific overrides needed, since VECTOR_DB_URL will be provided in environment


class ProdConfig(_Config):  # pylint: disable=too-few-public-methods
    """Production Config."""

    SECRET_KEY = os.getenv("SECRET_KEY", None)

    if not SECRET_KEY:
        SECRET_KEY = os.urandom(24)
        print("WARNING: SECRET_KEY being set as a one-shot", file=sys.stderr)
