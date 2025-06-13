"""
Helper module for advanced embeddings using SentenceTransformer.
This module is isolated to prevent Streamlit reload issues with PyTorch.
"""

import logging
from typing import List, Optional
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag for transformer availability
TRANSFORMER_AVAILABLE = False
transformer = None

def initialize_transformer() -> bool:
    """
    Initialize SentenceTransformer if available.
    
    Returns:
        bool: True if transformer is available and initialized
    """
    global TRANSFORMER_AVAILABLE, transformer
    
    if not TRANSFORMER_AVAILABLE:
        try:
            from sentence_transformers import SentenceTransformer
            transformer = SentenceTransformer('all-MiniLM-L6-v2')
            TRANSFORMER_AVAILABLE = True
            logger.info("Successfully initialized SentenceTransformer")
        except ImportError:
            logger.warning("SentenceTransformer not available. Please install with: pip install sentence-transformers")
            TRANSFORMER_AVAILABLE = False
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer: {str(e)}")
            TRANSFORMER_AVAILABLE = False
    
    return TRANSFORMER_AVAILABLE

def get_transformer_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    """
    Get embeddings using SentenceTransformer.
    
    Args:
        texts (List[str]): List of texts to embed
        
    Returns:
        Optional[np.ndarray]: Embeddings array or None if transformer is not available
    """
    if not TRANSFORMER_AVAILABLE:
        logger.warning("SentenceTransformer not available. Please initialize first.")
        return None
    
    try:
        if transformer is None:
            initialize_transformer()
        
        if transformer is not None:
            return transformer.encode(texts)
        return None
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return None

def is_transformer_available() -> bool:
    """
    Check if SentenceTransformer is available.
    
    Returns:
        bool: True if transformer is available
    """
    return TRANSFORMER_AVAILABLE 