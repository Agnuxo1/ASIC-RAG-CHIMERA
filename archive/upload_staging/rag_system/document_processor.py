"""
Document Processor for ASIC-RAG

Handles document ingestion pipeline:
- Text extraction from multiple formats
- Keyword extraction for tagging
- Chunking for large documents
- Metadata extraction
- Block creation
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from pathlib import Path
from enum import Enum
import json
import os

from .knowledge_block import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory


class DocumentType(Enum):
    """Supported document types."""
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    UNKNOWN = "unknown"


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int = 4096  # Characters per chunk
    chunk_overlap: int = 512  # Overlap between chunks
    max_keywords: int = 50  # Maximum keywords per document
    min_keyword_length: int = 3  # Minimum keyword length
    extract_entities: bool = True  # Extract named entities
    generate_summary: bool = False  # Generate summary (requires LLM)
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".json", ".html", ".pdf"
    ])


@dataclass
class ProcessedDocument:
    """Result of document processing."""
    original_path: str
    content: str
    chunks: List[str]
    keywords: List[str]
    metadata: BlockMetadata
    document_type: DocumentType
    processing_time_ms: float
    word_count: int
    char_count: int
    
    def to_dict(self) -> Dict:
        return {
            "original_path": self.original_path,
            "document_type": self.document_type.value,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "chunk_count": len(self.chunks),
            "keyword_count": len(self.keywords),
            "processing_time_ms": self.processing_time_ms
        }


class KeywordExtractor:
    """
    Extracts keywords from text for tag generation.
    
    Uses TF-IDF-like scoring without external dependencies.
    """
    
    # Common stop words to filter out
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "i", "you", "he", "she", "it",
        "we", "they", "what", "which", "who", "whom", "whose", "where",
        "when", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "also", "now", "here"
    }
    
    def __init__(self, min_length: int = 3, max_keywords: int = 50):
        self.min_length = min_length
        self.max_keywords = max_keywords
        self._word_pattern = re.compile(r'\b[a-zA-Z][a-zA-Z0-9_]*\b')
    
    def extract(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords sorted by relevance
        """
        # Tokenize
        words = self._word_pattern.findall(text.lower())
        
        # Filter
        filtered = [
            w for w in words
            if len(w) >= self.min_length and w not in self.STOP_WORDS
        ]
        
        # Count frequencies
        freq: Dict[str, int] = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        
        # Score words (simple TF scoring)
        total_words = len(filtered)
        scored = [
            (word, count / total_words if total_words > 0 else 0)
            for word, count in freq.items()
        ]
        
        # Sort by score and return top keywords
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in scored[:self.max_keywords]]
    
    def extract_with_scores(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords with their scores."""
        words = self._word_pattern.findall(text.lower())
        filtered = [
            w for w in words
            if len(w) >= self.min_length and w not in self.STOP_WORDS
        ]
        
        freq: Dict[str, int] = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        
        total_words = len(filtered)
        scored = [
            (word, count / total_words if total_words > 0 else 0)
            for word, count in freq.items()
        ]
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.max_keywords]


class TextChunker:
    """
    Splits text into overlapping chunks for processing.
    """
    
    def __init__(
        self,
        chunk_size: int = 4096,
        overlap: int = 512,
        respect_sentences: bool = True
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sentences = respect_sentences
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        if self.respect_sentences:
            return self._chunk_by_sentences(text)
        else:
            return self._chunk_by_chars(text)
    
    def _chunk_by_chars(self, text: str) -> List[str]:
        """Simple character-based chunking."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.overlap
            
            if start + self.overlap >= len(text):
                break
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Sentence-aware chunking."""
        sentences = self._sentence_pattern.split(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Add overlap
        if self.overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Add end of previous chunk
                    prev_end = chunks[i-1][-self.overlap:]
                    chunk = prev_end + " " + chunk
                overlapped_chunks.append(chunk)
            return overlapped_chunks
        
        return chunks


class DocumentProcessor:
    """
    Main document processing pipeline for ASIC-RAG.
    
    Handles:
    - Document loading from various formats
    - Text extraction and cleaning
    - Keyword extraction for tagging
    - Chunking for large documents
    - Block creation for storage
    
    Example:
        >>> processor = DocumentProcessor()
        >>> result = processor.process_file("document.txt")
        >>> blocks = processor.create_blocks(result)
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize document processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.keyword_extractor = KeywordExtractor(
            min_length=self.config.min_keyword_length,
            max_keywords=self.config.max_keywords
        )
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )
        
        # Statistics
        self._documents_processed = 0
        self._total_processing_time_ms = 0.0
    
    def process_file(self, file_path: str) -> ProcessedDocument:
        """
        Process a document file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            ProcessedDocument with extracted data
        """
        start_time = time.perf_counter()
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect document type
        doc_type = self._detect_type(path)
        
        # Extract content
        content = self._extract_content(path, doc_type)
        
        # Extract keywords
        keywords = self.keyword_extractor.extract(content)
        
        # Create chunks
        chunks = self.chunker.chunk(content)
        
        # Create metadata
        metadata = self._create_metadata(path, doc_type)
        
        # Calculate statistics
        word_count = len(content.split())
        char_count = len(content)
        
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        
        self._documents_processed += 1
        self._total_processing_time_ms += processing_time_ms
        
        return ProcessedDocument(
            original_path=str(path.absolute()),
            content=content,
            chunks=chunks,
            keywords=keywords,
            metadata=metadata,
            document_type=doc_type,
            processing_time_ms=processing_time_ms,
            word_count=word_count,
            char_count=char_count
        )
    
    def process_text(
        self,
        text: str,
        source: str = "direct_input",
        category: BlockCategory = BlockCategory.GENERAL
    ) -> ProcessedDocument:
        """
        Process raw text content.
        
        Args:
            text: Text content
            source: Source identifier
            category: Document category
            
        Returns:
            ProcessedDocument with extracted data
        """
        start_time = time.perf_counter()
        
        # Extract keywords
        keywords = self.keyword_extractor.extract(text)
        
        # Create chunks
        chunks = self.chunker.chunk(text)
        
        # Create metadata
        metadata = BlockMetadata(
            category=category,
            source=source,
            created_at=time.time(),
            modified_at=time.time()
        )
        
        word_count = len(text.split())
        char_count = len(text)
        
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        
        self._documents_processed += 1
        self._total_processing_time_ms += processing_time_ms
        
        return ProcessedDocument(
            original_path=source,
            content=text,
            chunks=chunks,
            keywords=keywords,
            metadata=metadata,
            document_type=DocumentType.TEXT,
            processing_time_ms=processing_time_ms,
            word_count=word_count,
            char_count=char_count
        )
    
    def _detect_type(self, path: Path) -> DocumentType:
        """Detect document type from extension."""
        ext = path.suffix.lower()
        type_map = {
            ".txt": DocumentType.TEXT,
            ".md": DocumentType.MARKDOWN,
            ".json": DocumentType.JSON,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
            ".pdf": DocumentType.PDF,
        }
        return type_map.get(ext, DocumentType.UNKNOWN)
    
    def _extract_content(self, path: Path, doc_type: DocumentType) -> str:
        """Extract text content from document."""
        if doc_type == DocumentType.PDF:
            return self._extract_pdf(path)
        elif doc_type == DocumentType.HTML:
            return self._extract_html(path)
        elif doc_type == DocumentType.JSON:
            return self._extract_json(path)
        else:
            # Text-based formats
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF (placeholder)."""
        # In production, use PyPDF2 or pdfplumber
        # For now, return placeholder
        try:
            # Try to read as text (might work for text-based PDFs)
            with open(path, 'rb') as f:
                content = f.read()
                # Basic text extraction from PDF
                text_parts = []
                for match in re.finditer(rb'\(([^)]+)\)', content):
                    try:
                        text_parts.append(match.group(1).decode('utf-8', errors='ignore'))
                    except:
                        pass
                return ' '.join(text_parts) if text_parts else f"[PDF: {path.name}]"
        except Exception:
            return f"[PDF: {path.name}]"
    
    def _extract_html(self, path: Path) -> str:
        """Extract text from HTML."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove scripts and styles
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def _extract_json(self, path: Path) -> str:
        """Extract text from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Recursively extract text values
        def extract_strings(obj) -> List[str]:
            strings = []
            if isinstance(obj, str):
                strings.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    strings.extend(extract_strings(v))
            elif isinstance(obj, list):
                for item in obj:
                    strings.extend(extract_strings(item))
            return strings
        
        return ' '.join(extract_strings(data))
    
    def _create_metadata(self, path: Path, doc_type: DocumentType) -> BlockMetadata:
        """Create metadata from file."""
        stat = path.stat()
        
        # Try to infer category from path or content
        category = self._infer_category(path)
        
        return BlockMetadata(
            category=category,
            source=str(path.absolute()),
            created_at=stat.st_ctime,
            modified_at=stat.st_mtime,
            custom={
                "original_filename": path.name,
                "file_size": stat.st_size,
                "document_type": doc_type.value
            }
        )
    
    def _infer_category(self, path: Path) -> BlockCategory:
        """Infer category from file path."""
        path_str = str(path).lower()
        
        category_keywords = {
            BlockCategory.FINANCE: ["finance", "financial", "budget", "revenue", "accounting"],
            BlockCategory.LEGAL: ["legal", "contract", "agreement", "compliance", "law"],
            BlockCategory.TECHNICAL: ["technical", "tech", "api", "code", "engineering"],
            BlockCategory.HR: ["hr", "human", "employee", "personnel", "hiring"],
            BlockCategory.MARKETING: ["marketing", "market", "campaign", "brand", "sales"],
            BlockCategory.RESEARCH: ["research", "study", "analysis", "report", "paper"],
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in path_str for kw in keywords):
                return category
        
        return BlockCategory.GENERAL
    
    def create_blocks(
        self,
        processed_doc: ProcessedDocument,
        prev_hash: bytes = b'\x00' * 32
    ) -> List[KnowledgeBlock]:
        """
        Create knowledge blocks from processed document.
        
        Args:
            processed_doc: Processed document
            prev_hash: Hash of previous block in chain
            
        Returns:
            List of knowledge blocks (one per chunk)
        """
        blocks = []
        
        # Generate tag hashes
        tag_hashes = [
            hashlib.sha256(kw.encode('utf-8')).digest()
            for kw in processed_doc.keywords
        ]
        
        # Add category tag
        category_tag = hashlib.sha256(
            processed_doc.metadata.category.value.encode('utf-8')
        ).digest()
        if category_tag not in tag_hashes:
            tag_hashes.insert(0, category_tag)
        
        for i, chunk in enumerate(processed_doc.chunks):
            # Create header
            header = BlockHeader(
                prev_hash=prev_hash,
                timestamp=time.time(),
                nonce=i
            )
            
            # Create metadata for this chunk
            chunk_metadata = BlockMetadata(
                category=processed_doc.metadata.category,
                source=processed_doc.original_path,
                author=processed_doc.metadata.author,
                created_at=processed_doc.metadata.created_at,
                modified_at=processed_doc.metadata.modified_at,
                custom={
                    **processed_doc.metadata.custom,
                    "chunk_index": i,
                    "total_chunks": len(processed_doc.chunks)
                }
            )
            
            # Create block
            block = KnowledgeBlock(
                block_id=0,  # Will be assigned by storage
                header=header,
                tag_hashes=tag_hashes,
                metadata=chunk_metadata,
                content=chunk.encode('utf-8')
            )
            
            blocks.append(block)
            
            # Update prev_hash for chain linking
            prev_hash = header.block_hash
        
        return blocks
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            "documents_processed": self._documents_processed,
            "total_processing_time_ms": self._total_processing_time_ms,
            "average_processing_time_ms": (
                self._total_processing_time_ms / self._documents_processed
                if self._documents_processed > 0 else 0.0
            )
        }


class BatchProcessor:
    """
    Batch document processor for directories.
    """
    
    def __init__(self, processor: DocumentProcessor):
        self.processor = processor
    
    def process_directory(
        self,
        directory: str,
        recursive: bool = True
    ) -> List[ProcessedDocument]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Directory path
            recursive: Process subdirectories
            
        Returns:
            List of processed documents
        """
        path = Path(directory)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        results = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.processor.config.supported_extensions:
                    try:
                        result = self.processor.process_file(str(file_path))
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return results


if __name__ == "__main__":
    print("Document Processor Demo")
    print("=" * 50)
    
    processor = DocumentProcessor()
    
    # Process raw text
    sample_text = """
    Q3 2024 Financial Report
    
    Executive Summary:
    Revenue for Q3 2024 increased by 15% compared to Q2 2024.
    The finance team has successfully implemented cost reduction
    measures that resulted in a 10% decrease in operational expenses.
    
    Key Highlights:
    - Total revenue: $45.2 million
    - Net profit margin: 23%
    - Customer acquisition cost: $125
    - Monthly active users: 2.3 million
    
    The technical infrastructure upgrades completed in Q2 contributed
    to improved system performance and reduced downtime by 40%.
    """
    
    result = processor.process_text(
        sample_text,
        source="quarterly_report.txt",
        category=BlockCategory.FINANCE
    )
    
    print(f"\nProcessed Document:")
    print(f"  Source: {result.original_path}")
    print(f"  Type: {result.document_type.value}")
    print(f"  Words: {result.word_count}")
    print(f"  Characters: {result.char_count}")
    print(f"  Chunks: {len(result.chunks)}")
    print(f"  Processing time: {result.processing_time_ms:.2f} ms")
    
    print(f"\nExtracted Keywords ({len(result.keywords)}):")
    for kw in result.keywords[:10]:
        print(f"  - {kw}")
    
    # Create blocks
    blocks = processor.create_blocks(result)
    print(f"\nCreated {len(blocks)} knowledge block(s)")
    
    for i, block in enumerate(blocks):
        print(f"  Block {i}: {len(block.tag_hashes)} tags, {len(block.content)} bytes")
    
    # Statistics
    print("\n--- Processing Statistics ---")
    stats = processor.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
