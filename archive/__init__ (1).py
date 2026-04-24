"""
LLM Interface Module for ASIC-RAG-CHIMERA

Provides integration with local LLM (Qwen3-0.6B) for:
- Keyword extraction from queries
- RAG response generation
- Semantic understanding

Components:
    - QwenLoader: Model loading and management
    - KeywordExtractor: Semantic keyword extraction
    - ResponseGenerator: RAG-aware response generation
"""

from .qwen_loader import QwenLoader, ModelConfig
from .keyword_extractor import SemanticKeywordExtractor
from .response_generator import ResponseGenerator, GenerationConfig

__all__ = [
    "QwenLoader",
    "ModelConfig",
    "SemanticKeywordExtractor",
    "ResponseGenerator",
    "GenerationConfig",
]

__version__ = "0.1.0"
