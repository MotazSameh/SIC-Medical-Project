from typing import List, Optional
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.common.schemas import RetrievedChunk
from src.guidelines.vector_store import VectorStore
from src.guidelines.bm25_index import BM25Index


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        
        print("    HybridRetriever initialized")
        print(f"   Vector store: {self.vector_store.get_chunk_count()} chunks")
        print(f"   BM25 index: {self.bm25_index.get_corpus_size()} chunks")
    
    def retrieve_vector(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        return self.vector_store.search(query, top_k=top_k)
    
    def retrieve_bm25(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        return self.bm25_index.search(query, top_k=top_k)
    
    def retrieve_hybrid(self,query: str,top_k: int = 10,vector_weight: float = 0.7,
                        bm25_query: Optional[str] = None,) -> List[RetrievedChunk]:
        if not query or not query.strip():
            print("  Empty query provided, returning empty results")
            return []

        retrieval_depth = top_k * 2
        vector_results = self.retrieve_vector(query, top_k=retrieval_depth)
        bm25_results = self.retrieve_bm25(bm25_query if bm25_query else query, top_k=retrieval_depth   )
        
        if not vector_results and not bm25_results:
            return []
        if not vector_results:
            return bm25_results[:top_k]
        if not bm25_results:
            return vector_results[:top_k]

        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)

        hybrid_results = self._merge_and_rerank(vector_results,bm25_results,vector_weight,top_k)
        
        return hybrid_results
    
    def _normalize_scores(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    
        if not chunks:
            return chunks
        
        scores = [c.score for c in chunks]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for chunk in chunks:
                chunk.score = 1.0
            return chunks
        
        for chunk in chunks:
            chunk.score = (chunk.score - min_score) / (max_score - min_score)
        
        return chunks
    
    def _merge_and_rerank(self,vector_results: List[RetrievedChunk],bm25_results: List[RetrievedChunk],
                          vector_weight: float,top_k: int) -> List[RetrievedChunk]:
        merged = {}
        
        for chunk in vector_results:
            merged[chunk.chunk_id] = {'vector_score': chunk.score,'bm25_score': None,'chunk': chunk}
        
        for chunk in bm25_results:
            if chunk.chunk_id in merged:
                merged[chunk.chunk_id]['bm25_score'] = chunk.score
            else:
                merged[chunk.chunk_id] = {'vector_score': None,'bm25_score': chunk.score,'chunk': chunk}
        
        final_results = []
        
        for chunk_id, data in merged.items():
            v_score = data['vector_score']
            b_score = data['bm25_score']
            chunk = data['chunk']
            
            if v_score is not None and b_score is not None:
                final_score = (vector_weight * v_score) + ((1 - vector_weight) * b_score)
            elif v_score is not None:
                final_score = v_score
            else:
                final_score = b_score
            
            hybrid_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=final_score,
                rank=0, 
                retriever_type="hybrid",
                authority_family=chunk.authority_family,
                tier=chunk.tier,
                year=chunk.year,
                drug_names=chunk.drug_names
            )
            final_results.append(hybrid_chunk)
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        final_results = final_results[:top_k]
        
        for rank, chunk in enumerate(final_results, start=1):
            chunk.rank = rank
        
        return final_results


if __name__ == "__main__":
    from src.guidelines.vector_store import VectorStore
    from src.guidelines.bm25_index import BM25Index
    
    print("="*80)
    print("INITIALIZING HYBRID RETRIEVER")
    print("="*80)
    
    print("\n Loading Vector Store...")
    vector_store = VectorStore(persist_directory="data/chromadb",embedding_model_name="all-MiniLM-L6-v2")
    vector_store.load_chunks("data/processed/chunks.json")
    vector_store.create_or_load_collection()
    
    print("\n Loading BM25 Index...")
    bm25_index = BM25Index.load_from_disk("data/bm25_index.pkl")
    
    print("\n Initializing Hybrid Retriever...")
    hybrid = HybridRetriever(vector_store, bm25_index)
    
    print("\n" + "="*80)
    print("TESTING HYBRID RETRIEVAL")
    print("="*80)
    
    test_queries = ["warfarin side effects","aspirin contraindications",""]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f" Query: '{query}'")
        print(f"{'='*80}")
        
        if not query:
            print("   (Empty query test)")
        
        vector_results = hybrid.retrieve_vector(query, top_k=5)
        bm25_results = hybrid.retrieve_bm25(query, top_k=5)
        hybrid_results = hybrid.retrieve_hybrid(query, top_k=5, vector_weight=0.5)

        print(f"\n VECTOR (top-5):")
        if vector_results:
            for r in vector_results:
                print(f"   [{r.rank}] Score: {r.score:.4f} | {r.document_id[:30]}")
        else:
            print("   No results")
        
        print(f"\n BM25 (top-5):")
        if bm25_results:
            for r in bm25_results:
                print(f"   [{r.rank}] Score: {r.score:.4f} | {r.document_id[:30]}")
        else:
            print("   No results")
        
        print(f"\n HYBRID (50/50, top-5):")
        if hybrid_results:
            for r in hybrid_results:
                print(f"   [{r.rank}] Score: {r.score:.4f} | {r.document_id[:30]}")
        else:
            print("   No results")
    
        if hybrid_results:
            vector_ids = {r.chunk_id for r in vector_results}
            bm25_ids = {r.chunk_id for r in bm25_results}
            hybrid_ids = {r.chunk_id for r in hybrid_results}
            
            overlap = vector_ids & bm25_ids
            print(f"\n Overlap Analysis:")
            print(f"   Vector ∩ BM25: {len(overlap)} chunks")
            print(f"   Hybrid unique sources:")
            print(f"      From vector only: {len(hybrid_ids & vector_ids - bm25_ids)}")
            print(f"      From BM25 only: {len(hybrid_ids & bm25_ids - vector_ids)}")
            print(f"      From both: {len(hybrid_ids & overlap)}")
    
    print(f"\n{'='*80}")
    print(" Hybrid retrieval testing complete")
    print(f"{'='*80}")
