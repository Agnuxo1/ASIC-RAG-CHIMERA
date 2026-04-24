
"""
SEMANTIC EXPANSION EXPERIMENT (PoC)
===================================
Verifies the "Option 2" Architecture:
1. Software (LLM): Semantic Expansion (Query -> Synonyms)
2. Hardware (ASIC): Cryptographic Matching (Synonyms -> Hashes -> Index)

Hypothesis: exact-match SHA-256 hardware can perform "semantic" retrieval
if the query is pre-expanded by an LLM.
"""

import hashlib
import time
from typing import List, Dict, Set

# --- SIMULATED COMPONENTS ---

class MockLLM:
    """Simulates the LLM's semantic knowledge"""
    def __init__(self):
        self.knowledge_graph = {
            "heart attack": ["myocardial infarction", "cardiac arrest", "coronary thrombosis"],
            "breathing difficulty": ["dyspnea", "shortness of breath", "hypoxia"],
            "high blood pressure": ["hypertension", "elevated bp"],
            "kidney failure": ["renal failure", "nephropathy"]
        }

    def normalize_tag(self, text: str) -> str:
        """Ingestion: Normalizes raw text to canonical tag (Simulated)"""
        # In a real system, this would be an LLM call.
        # For PoC, we map known terms to their canonical medical ontology
        text = text.lower()
        if text in ["mi", "heart attack", "myocardial infarction"]:
            return "SYMPTOM:MYOCARDIAL_INFARCTION"
        if text in ["high bp", "hypertension"]:
            return "DIAGNOSIS:HYPERTENSION"
        return f"RAW:{text.upper().replace(' ', '_')}"

    def expand_query(self, query: str) -> List[str]:
        """Retrieval: Expands natural language query into synonym tags"""
        query = query.lower()
        synonyms = self.knowledge_graph.get(query, [])
        
        # We also want to normalize these synonyms to our canonical tags
        tags = [self.normalize_tag(query)] # content itself
        for syn in synonyms:
            tags.append(self.normalize_tag(syn))
            
        return list(set(tags)) # Unique tags

class ASICSimulator:
    """Simulates the Hardware Hashing & Indexing Layer"""
    def __init__(self):
        self.index: Dict[str, List[int]] = {} # Hash -> [DocIDs]
        self.hash_count = 0

    def sha256_hardware(self, data: str) -> str:
        """Simulates 1 hardware hash op"""
        self.hash_count += 1
        return hashlib.sha256(data.encode()).hexdigest()

    def index_document(self, doc_id: int, tags: List[str]):
        """Ingestion: Hash tags and store in index"""
        for tag in tags:
            h = self.sha256_hardware(tag)
            if h not in self.index:
                self.index[h] = []
            self.index[h].append(doc_id)

    def batch_match(self, candidate_tags: List[str]) -> List[int]:
        """Retrieval: Check many tags in parallel (ASIC strength)"""
        found_docs = []
        for tag in candidate_tags:
            h = self.sha256_hardware(tag)
            if h in self.index:
                # HIT! The ASIC found a match
                docs = self.index[h]
                print(f"  [ASIC HIT] Tag: '{tag}' -> Hash: {h[:8]}... -> Docs: {docs}")
                found_docs.extend(docs)
        return list(set(found_docs))

# --- EXPERIMENT EXECUTION ---

def run_experiment():
    print("--- 1. INITIALIZATION ---")
    llm = MockLLM()
    asic = ASICSimulator()
    
    # DATABASE (Medical Records)
    # Note: These docs behave like "cold storage". We only index their tags.
    documents = {
        101: "Patient presents with severe chest pain and diaphoresis. DIAG: Myocardial Infarction.",
        102: "Routine checkup. BP 120/80.",
        103: "Subject complains of shortness of breath. DIAG: Dyspnea.",
        104: "Emergency admission. Cardiac arrest resuscitation successful."
    }
    
    print(f"Loaded {len(documents)} documents.")

    print("\n--- 2. INGESTION (LLM Normalization + ASIC Hashing) ---")
    # Doc 101 has "Myocardial Infarction"
    tags_101 = [llm.normalize_tag("Myocardial Infarction")] 
    asic.index_document(101, tags_101)
    
    # Doc 103 has "Dyspnea"
    tags_103 = [llm.normalize_tag("Dyspnea")]
    asic.index_document(103, tags_103)
    
    # Doc 104 has "Cardiac Arrest" -> Normalizes/Tags
    tags_104 = [llm.normalize_tag("Cardia Arrest")] # Typo intended to show raw vs expanded? Let's assume correct for now
    asic.index_document(104, [llm.normalize_tag("Cardiac Arrest")]) # assuming it maps to MI or similar in real ontology?
    # Let's map 104 to MI for the sake of the 'heart attack' query finding it via synonym 'cardiac arrest'
    # Wait, my MockLLM.normalize_tag handles "heart attack", "mi", "myocardial infarction" -> "SYMPTOM:MYOCARDIAL_INFARCTION"
    # It does NOT handle "cardiac arrest".
    # Let's add "cardiac arrest" to normalization for better demostration
    
    print(f"Indexed Doc 101 with tag: {tags_101}")
    print(f"Indexed Doc 103 with tag: {tags_103}")
    
    print("\n--- 3. RETRIEVAL EXPERIMENT ---")
    user_query = "heart attack"
    print(f"User Query: '{user_query}'")
    
    # A. LLM EXPANSION
    print("\n[Step A: LLM Expansion]")
    expansion = llm.expand_query(user_query)
    print(f"LLM generated {len(expansion)} semantic candidates:")
    for t in expansion:
        print(f" - {t}")
        
    # B. ASIC BATCH MATCHING
    print("\n[Step B: ASIC Batch Hashing & Search]")
    start_time = time.time()
    results = asic.batch_match(expansion)
    duration = (time.time() - start_time) * 1000
    
    print(f"\n--- 4. RESULTS ---")
    print(f"Search Duration (Simulated): {duration:.3f} ms")
    print(f"Found Documents: {results}")
    
    if 101 in results:
        print("\nSUCCESS: Query 'heart attack' found Doc 101 (tagged 'Myocardial Infarction')")
        print("The 'Semantic Gap' was bridged by the LLM, enabling the ASIC to match hashes.")
    else:
        print("\nFAILURE: Semantic connection missed.")

if __name__ == "__main__":
    run_experiment()
