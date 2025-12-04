"""
Semantic Keyword Extractor for ASIC-RAG

Uses LLM for intelligent keyword extraction:
- Understands context and semantics
- Generates synonyms and related terms
- Identifies named entities
- Extracts key concepts
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import re


@dataclass
class ExtractionResult:
    """Result of keyword extraction."""
    keywords: List[str]
    entities: List[str]
    concepts: List[str]
    synonyms: Dict[str, List[str]]
    confidence_scores: Dict[str, float]


class SemanticKeywordExtractor:
    """
    LLM-powered semantic keyword extractor.
    
    Uses the local Qwen model to intelligently extract
    keywords that capture document semantics.
    
    Example:
        >>> extractor = SemanticKeywordExtractor(llm_loader)
        >>> result = extractor.extract("Q3 financial report...")
        >>> print(result.keywords)
    """
    
    # Extraction prompt template
    EXTRACTION_PROMPT = """Extract keywords from the following text for a document search index.

Text:
{text}

Instructions:
1. Extract 10-20 important keywords that capture the main topics
2. Include specific terms, names, dates, and technical terms
3. Include both general concepts and specific details
4. Output keywords separated by commas, nothing else

Keywords:"""

    ENTITY_PROMPT = """Extract named entities from the following text.

Text:
{text}

Extract:
- Person names
- Organization names
- Product names
- Location names
- Dates and time periods

Output format: entity_type: entity_name (one per line)

Entities:"""

    def __init__(self, llm_loader, use_llm: bool = True):
        """
        Initialize extractor.
        
        Args:
            llm_loader: QwenLoader instance
            use_llm: Whether to use LLM (False = rule-based only)
        """
        self.llm = llm_loader
        self.use_llm = use_llm
        
        # Common domain terms for boosting
        self.domain_terms = {
            "finance": ["revenue", "profit", "loss", "budget", "quarterly", "fiscal", "earnings"],
            "legal": ["contract", "agreement", "clause", "liability", "compliance", "regulation"],
            "technical": ["api", "database", "server", "algorithm", "implementation", "architecture"],
            "hr": ["employee", "hiring", "performance", "salary", "benefits", "training"],
        }
    
    def extract(self, text: str, max_keywords: int = 20) -> ExtractionResult:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum keywords to extract
            
        Returns:
            ExtractionResult with keywords and metadata
        """
        if self.use_llm and self.llm:
            return self._extract_with_llm(text, max_keywords)
        else:
            return self._extract_rule_based(text, max_keywords)
    
    def _extract_with_llm(self, text: str, max_keywords: int) -> ExtractionResult:
        """Extract keywords using LLM."""
        # Truncate text if too long
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        # Extract keywords
        prompt = self.EXTRACTION_PROMPT.format(text=text)
        response = self.llm.generate(prompt, max_new_tokens=200, temperature=0.3)
        
        # Parse keywords
        keywords = self._parse_keywords(response)
        
        # Extract entities
        entity_prompt = self.ENTITY_PROMPT.format(text=text)
        entity_response = self.llm.generate(entity_prompt, max_new_tokens=200, temperature=0.3)
        entities = self._parse_entities(entity_response)
        
        # Combine and deduplicate
        all_keywords = list(dict.fromkeys(keywords + entities))[:max_keywords]
        
        # Generate synonyms for top keywords
        synonyms = self._generate_synonyms(all_keywords[:5])
        
        # Assign confidence scores
        confidence_scores = {kw: 1.0 - (i * 0.03) for i, kw in enumerate(all_keywords)}
        
        return ExtractionResult(
            keywords=all_keywords,
            entities=entities,
            concepts=self._extract_concepts(text),
            synonyms=synonyms,
            confidence_scores=confidence_scores
        )
    
    def _extract_rule_based(self, text: str, max_keywords: int) -> ExtractionResult:
        """Rule-based keyword extraction (fallback)."""
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', text.lower())
        
        # Remove stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "this", "that", "these", "those", "it", "its"
        }
        
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Count frequencies
        freq = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [w for w, _ in sorted_words[:max_keywords]]
        
        # Extract simple entities (capitalized words in original text)
        entities = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)))
        
        return ExtractionResult(
            keywords=keywords,
            entities=entities[:10],
            concepts=[],
            synonyms={},
            confidence_scores={kw: freq.get(kw, 1) / max(freq.values()) for kw in keywords}
        )
    
    def _parse_keywords(self, response: str) -> List[str]:
        """Parse keywords from LLM response."""
        # Handle comma-separated list
        keywords = []
        
        for part in response.split(','):
            keyword = part.strip().lower()
            # Clean up
            keyword = re.sub(r'[^\w\s]', '', keyword)
            keyword = keyword.strip()
            
            if keyword and len(keyword) > 1:
                keywords.append(keyword)
        
        return keywords
    
    def _parse_entities(self, response: str) -> List[str]:
        """Parse entities from LLM response."""
        entities = []
        
        for line in response.split('\n'):
            line = line.strip()
            if ':' in line:
                # Format: type: entity
                entity = line.split(':', 1)[1].strip()
                if entity:
                    entities.append(entity.lower())
            elif line and len(line) > 1:
                entities.append(line.lower())
        
        return list(set(entities))
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract high-level concepts from text."""
        concepts = []
        text_lower = text.lower()
        
        # Check for domain concepts
        for domain, terms in self.domain_terms.items():
            matches = sum(1 for term in terms if term in text_lower)
            if matches >= 2:
                concepts.append(domain)
        
        return concepts
    
    def _generate_synonyms(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Generate synonyms for keywords."""
        # Simple synonym mapping (expand with LLM if needed)
        synonym_map = {
            "revenue": ["income", "earnings", "sales"],
            "profit": ["gain", "earnings", "return"],
            "cost": ["expense", "expenditure", "spending"],
            "employee": ["worker", "staff", "personnel"],
            "contract": ["agreement", "deal", "arrangement"],
            "report": ["document", "analysis", "summary"],
            "increase": ["growth", "rise", "gain"],
            "decrease": ["decline", "drop", "reduction"],
        }
        
        synonyms = {}
        for kw in keywords:
            if kw in synonym_map:
                synonyms[kw] = synonym_map[kw]
        
        return synonyms
    
    def extract_query_keywords(self, query: str) -> List[str]:
        """
        Extract keywords optimized for search queries.
        
        Args:
            query: Search query
            
        Returns:
            List of search keywords
        """
        if self.use_llm and self.llm:
            prompt = f"""Extract search keywords from this query. Output only keywords separated by commas.

Query: {query}

Keywords:"""
            response = self.llm.generate(prompt, max_new_tokens=50, temperature=0.2)
            keywords = self._parse_keywords(response)
        else:
            # Simple extraction
            words = query.lower().split()
            stop_words = {"what", "is", "the", "how", "why", "when", "where", "who", "which", "a", "an"}
            keywords = [w for w in words if w not in stop_words and len(w) > 1]
        
        return keywords


class KeywordExpander:
    """
    Expands keywords with synonyms and related terms.
    """
    
    def __init__(self, llm_loader=None):
        self.llm = llm_loader
    
    def expand(self, keywords: List[str], max_expanded: int = 30) -> List[str]:
        """
        Expand keywords with related terms.
        
        Args:
            keywords: Original keywords
            max_expanded: Maximum total keywords
            
        Returns:
            Expanded keyword list
        """
        if not self.llm:
            return keywords
        
        prompt = f"""For each keyword, provide 2-3 related terms or synonyms.

Keywords: {', '.join(keywords)}

Output format: keyword -> related1, related2
"""
        response = self.llm.generate(prompt, max_new_tokens=200, temperature=0.3)
        
        # Parse response
        expanded = set(keywords)
        
        for line in response.split('\n'):
            if '->' in line:
                parts = line.split('->')
                if len(parts) == 2:
                    related = [t.strip().lower() for t in parts[1].split(',')]
                    expanded.update(related)
        
        return list(expanded)[:max_expanded]


if __name__ == "__main__":
    print("Semantic Keyword Extractor Demo")
    print("=" * 50)
    
    # Use without LLM for demo
    extractor = SemanticKeywordExtractor(None, use_llm=False)
    
    sample_text = """
    Q3 2024 Financial Report - Executive Summary
    
    Revenue for the third quarter of 2024 reached $45.2 million, representing
    a 15% increase compared to Q2 2024. The growth was primarily driven by
    our enterprise software division and the successful launch of ProductX.
    
    Operating expenses decreased by 10% due to cost optimization initiatives
    implemented by the finance team. Net profit margin improved to 23%.
    
    Key metrics:
    - Total revenue: $45.2 million
    - Customer acquisition cost: $125
    - Monthly active users: 2.3 million
    - Employee count: 450
    """
    
    result = extractor.extract(sample_text)
    
    print(f"\nExtracted Keywords ({len(result.keywords)}):")
    for i, kw in enumerate(result.keywords[:15]):
        score = result.confidence_scores.get(kw, 0)
        print(f"  {i+1}. {kw} (score: {score:.2f})")
    
    print(f"\nExtracted Entities ({len(result.entities)}):")
    for entity in result.entities[:10]:
        print(f"  - {entity}")
    
    print(f"\nDetected Concepts: {result.concepts}")
    
    # Query keyword extraction
    print("\n" + "=" * 50)
    query = "What was the revenue growth in Q3 2024?"
    query_keywords = extractor.extract_query_keywords(query)
    print(f"\nQuery: {query}")
    print(f"Search keywords: {query_keywords}")
