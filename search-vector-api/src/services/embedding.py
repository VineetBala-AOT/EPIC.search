"""Text embedding service for vector representation of queries and documents.

This module provides functionality for converting text into vector embeddings
using pretrained sentence transformer models. These embeddings enable semantic
search by representing text in a high-dimensional vector space where similar
meanings are close together, regardless of the specific words used.

The module implements lazy loading of the embedding model to optimize resource
usage, only loading the model when first needed. It uses the configured model
from the application settings and includes error handling with graceful fallbacks.
"""

from flask import current_app
import numpy as np
from typing import Union, List

_model = None


def preload_embedding_model() -> None:
    """Load the embedding model eagerly so first-request latency is avoided."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        model_name = current_app.model_settings.embedding_model_name
        _model = SentenceTransformer(model_name)


def get_embedding(texts: Union[str, List[str]]) -> np.ndarray:
    """Generate vector embeddings for the provided text(s).
    
    This function converts text strings into high-dimensional vector embeddings
    using a pre-trained sentence transformer model. The model is loaded 
    on the first call and reused for subsequent calls.
    
    Args:
        texts: Either a single text string or a list of text strings to embed
        
    Returns:
        np.ndarray: A numpy array of embeddings with shape (len(texts), embedding_dimensions)
        
    Note:
        In case of errors during embedding generation, a zero vector with the
        configured dimensions is returned as a fallback.
    """
    global _model
    
    # Initialize the model only on first call
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Use strongly typed configuration instead of environment variables
        model_name = current_app.model_settings.embedding_model_name
        _model = SentenceTransformer(model_name)
    
    # Convert single string to list if needed
    if isinstance(texts, str):
        texts = [texts]
            
    try:
        embeddings = _model.encode(texts, show_progress_bar=False)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        # Return zero embeddings as fallback
        # Get dimensions directly from configuration
        embedding_dimensions = current_app.vector_settings.embedding_dimensions
        return np.zeros((len(texts), embedding_dimensions))