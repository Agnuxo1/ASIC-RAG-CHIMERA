"""
Response Generator for ASIC-RAG

Generates answers using LLM with retrieved context:
- RAG-aware prompt construction
- Source citation
- Confidence scoring
- Streaming support
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Generator
import time


@dataclass
class GenerationConfig:
    """Configuration for response generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    include_sources: bool = True
    require_grounding: bool = True  # Only answer from context
    confidence_threshold: float = 0.5
    system_prompt: str = """You are a helpful assistant that answers questions based on the provided context documents. 
Always base your answers on the given context. If the context doesn't contain enough information to answer the question, say so clearly.
Be concise and accurate."""


@dataclass
class GeneratedResponse:
    """Generated response with metadata."""
    answer: str
    sources_used: List[Dict[str, Any]]
    confidence: float
    generation_time_ms: float
    tokens_generated: int
    grounded: bool  # Whether answer is grounded in context
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "sources_used": self.sources_used,
            "confidence": self.confidence,
            "generation_time_ms": self.generation_time_ms,
            "tokens_generated": self.tokens_generated,
            "grounded": self.grounded
        }


class ResponseGenerator:
    """
    RAG response generator using local LLM.
    
    Constructs prompts with retrieved context and generates
    grounded answers with source citations.
    
    Example:
        >>> generator = ResponseGenerator(llm_loader)
        >>> response = generator.generate(
        ...     query="What was Q3 revenue?",
        ...     context=retrieved_documents
        ... )
        >>> print(response.answer)
    """
    
    # Prompt templates
    RAG_PROMPT_TEMPLATE = """{system_prompt}

Context Documents:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the provided context documents
- If the answer is not in the context, say "I cannot find this information in the provided documents"
- Be specific and cite which document contains the information
- Keep your answer concise and focused

Answer:"""

    GROUNDING_CHECK_PROMPT = """Given the context and answer below, determine if the answer is fully grounded in the context.

Context:
{context}

Answer:
{answer}

Is this answer fully supported by the context? Reply only YES or NO."""

    def __init__(
        self,
        llm_loader,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize response generator.
        
        Args:
            llm_loader: QwenLoader instance
            config: Generation configuration
        """
        self.llm = llm_loader
        self.config = config or GenerationConfig()
        
        # Statistics
        self._total_generations = 0
        self._total_generation_time_ms = 0.0
    
    def generate(
        self,
        query: str,
        context: str,
        sources: Optional[List[Dict]] = None
    ) -> GeneratedResponse:
        """
        Generate response for query with context.
        
        Args:
            query: User question
            context: Retrieved context documents
            sources: Optional source metadata
            
        Returns:
            GeneratedResponse with answer and metadata
        """
        start_time = time.perf_counter()
        
        # Construct prompt
        prompt = self.RAG_PROMPT_TEMPLATE.format(
            system_prompt=self.config.system_prompt,
            context=context,
            query=query
        )
        
        # Generate answer
        answer = self.llm.generate(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature
        )
        
        # Check grounding if required
        grounded = True
        if self.config.require_grounding:
            grounded = self._check_grounding(context, answer)
        
        # Calculate confidence
        confidence = self._calculate_confidence(answer, context, grounded)
        
        # Identify sources used
        sources_used = self._identify_sources(answer, sources or [])
        
        generation_time = (time.perf_counter() - start_time) * 1000
        
        # Update statistics
        self._total_generations += 1
        self._total_generation_time_ms += generation_time
        
        return GeneratedResponse(
            answer=answer,
            sources_used=sources_used,
            confidence=confidence,
            generation_time_ms=generation_time,
            tokens_generated=len(answer.split()),  # Approximate
            grounded=grounded
        )
    
    def generate_stream(
        self,
        query: str,
        context: str
    ) -> Generator[str, None, None]:
        """
        Generate response with streaming output.
        
        Args:
            query: User question
            context: Retrieved context
            
        Yields:
            Generated tokens
        """
        prompt = self.RAG_PROMPT_TEMPLATE.format(
            system_prompt=self.config.system_prompt,
            context=context,
            query=query
        )
        
        for token in self.llm.generate_stream(prompt, self.config.max_new_tokens):
            yield token
    
    def _check_grounding(self, context: str, answer: str) -> bool:
        """Check if answer is grounded in context."""
        # Simple check: look for key answer terms in context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Filter common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                       "have", "has", "had", "do", "does", "did", "will", "would",
                       "could", "should", "may", "might", "must", "shall"}
        
        answer_key_words = answer_words - common_words
        
        if not answer_key_words:
            return True
        
        # Check overlap
        overlap = len(answer_key_words & context_words) / len(answer_key_words)
        
        return overlap > 0.3  # At least 30% of answer words in context
    
    def _calculate_confidence(
        self,
        answer: str,
        context: str,
        grounded: bool
    ) -> float:
        """Calculate confidence score for answer."""
        confidence = 1.0
        
        # Reduce confidence for ungrounded answers
        if not grounded:
            confidence *= 0.5
        
        # Reduce for "cannot find" type answers
        negative_phrases = [
            "cannot find", "not in the", "no information",
            "don't have", "unable to", "not provided"
        ]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in negative_phrases):
            confidence *= 0.3
        
        # Reduce for very short answers
        if len(answer.split()) < 5:
            confidence *= 0.7
        
        # Reduce for very long answers (might be hallucinating)
        if len(answer.split()) > 200:
            confidence *= 0.8
        
        return min(1.0, max(0.0, confidence))
    
    def _identify_sources(
        self,
        answer: str,
        sources: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Identify which sources were used in the answer."""
        used_sources = []
        answer_lower = answer.lower()
        
        for i, source in enumerate(sources):
            source_id = source.get("block_id", i)
            source_content = source.get("content", "").lower()
            source_name = source.get("source", f"Document {i+1}")
            
            # Check if source content appears in answer
            # Simple heuristic: check for common phrases
            if source_content:
                source_words = set(source_content.split())
                answer_words = set(answer_lower.split())
                
                overlap = len(source_words & answer_words)
                if overlap > 5:  # Significant overlap
                    used_sources.append({
                        "id": source_id,
                        "name": source_name,
                        "relevance": overlap / len(answer_words) if answer_words else 0
                    })
        
        # Sort by relevance
        used_sources.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        return used_sources
    
    def summarize_documents(
        self,
        documents: List[str],
        max_length: int = 200
    ) -> str:
        """
        Summarize multiple documents.
        
        Args:
            documents: List of document texts
            max_length: Maximum summary length in words
            
        Returns:
            Summary text
        """
        combined = "\n\n".join(documents)
        
        prompt = f"""Summarize the following documents in {max_length} words or less.
Focus on key facts, numbers, and main points.

Documents:
{combined}

Summary:"""
        
        return self.llm.generate(
            prompt,
            max_new_tokens=max_length * 2,  # Tokens != words
            temperature=0.5
        )
    
    def answer_with_citations(
        self,
        query: str,
        documents: List[Dict[str, str]]
    ) -> str:
        """
        Generate answer with inline citations.
        
        Args:
            query: User question
            documents: List of {"id": "...", "content": "..."}
            
        Returns:
            Answer with [1], [2] style citations
        """
        # Format context with citation numbers
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"[{i+1}] {doc.get('content', '')}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""{self.config.system_prompt}

Documents:
{context}

Question: {query}

Instructions:
- Answer the question using the provided documents
- Add citation numbers like [1], [2] after facts from specific documents
- Be accurate and concise

Answer:"""
        
        return self.llm.generate(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature
        )
    
    def get_statistics(self) -> Dict:
        """Get generation statistics."""
        return {
            "total_generations": self._total_generations,
            "total_generation_time_ms": self._total_generation_time_ms,
            "average_generation_time_ms": (
                self._total_generation_time_ms / self._total_generations
                if self._total_generations > 0 else 0.0
            )
        }


class ConversationalRAG:
    """
    Conversational RAG with chat history.
    """
    
    def __init__(self, response_generator: ResponseGenerator):
        self.generator = response_generator
        self.history: List[Dict[str, str]] = []
        self.max_history = 10
    
    def chat(
        self,
        query: str,
        context: str
    ) -> GeneratedResponse:
        """
        Chat with history context.
        
        Args:
            query: User message
            context: Retrieved documents
            
        Returns:
            Generated response
        """
        # Include recent history in prompt
        history_text = ""
        if self.history:
            recent = self.history[-self.max_history:]
            history_text = "Previous conversation:\n"
            for msg in recent:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"{role.capitalize()}: {content}\n"
            history_text += "\n"
        
        # Modify context to include history
        full_context = f"{history_text}Current context:\n{context}"
        
        # Generate response
        response = self.generator.generate(query, full_context)
        
        # Update history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response.answer})
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []


if __name__ == "__main__":
    print("Response Generator Demo")
    print("=" * 50)
    
    # Mock LLM for demo
    class MockLLM:
        def generate(self, prompt, **kwargs):
            if "revenue" in prompt.lower():
                return "Based on the context documents, the Q3 2024 revenue was $45.2 million, representing a 15% increase from Q2 2024. [1]"
            return "I found relevant information in the provided documents."
    
    mock_llm = MockLLM()
    generator = ResponseGenerator(mock_llm)
    
    # Test generation
    context = """
    [Document 1] Q3 2024 Financial Report
    Revenue for Q3 2024 reached $45.2 million, a 15% increase from Q2 2024.
    Net profit margin improved to 23%.
    
    [Document 2] Executive Summary
    The company showed strong growth across all divisions.
    Employee count grew to 450.
    """
    
    response = generator.generate(
        query="What was the Q3 2024 revenue?",
        context=context,
        sources=[
            {"block_id": 1, "source": "financial_report.pdf", "content": "Revenue $45.2 million"},
            {"block_id": 2, "source": "summary.pdf", "content": "Strong growth"}
        ]
    )
    
    print(f"\nQuery: What was the Q3 2024 revenue?")
    print(f"\nAnswer: {response.answer}")
    print(f"\nConfidence: {response.confidence:.2f}")
    print(f"Grounded: {response.grounded}")
    print(f"Generation time: {response.generation_time_ms:.2f} ms")
    print(f"Sources used: {response.sources_used}")
