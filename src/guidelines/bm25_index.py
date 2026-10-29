import json
import pickle
import string
import time
from pathlib import Path
from typing import List, Optional, Tuple
from collections import Counter

from rank_bm25 import BM25Okapi

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.common.schemas import Chunk, RetrievedChunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []
        self.token_stats = {}
        
        print(f"Initialized BM25Index (k1={k1}, b={b})")
    
    def load_chunks(self, chunks_json_path: str) -> List[Chunk]:
        print(f"\nLoading chunks from {chunks_json_path}...")
        with open(chunks_json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = []
        for chunk_data in chunks_data:
            chunk = Chunk(**chunk_data)
            chunks.append(chunk)
        
        print(f"✅ Loaded {len(chunks)} chunks")
        self.chunks = chunks
        return chunks
    
    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def build_index(self):
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_chunks() first.")
        
        print(f"\nBuilding BM25 index...")
        start_time = time.time()
        
        print(f"Tokenizing {len(self.chunks)} chunks...")
        self.tokenized_corpus = []
        for chunk in self.chunks:
            tokens = self.tokenize(chunk.text)
            self.tokenized_corpus.append(tokens)
        
        print(f"Computing BM25 scores (k1={self.k1}, b={self.b})...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        elapsed = time.time() - start_time
        print(f"✅ BM25 index built in {elapsed:.2f}s")
        
        self._log_token_stats()
    
    def _log_token_stats(self):
    
        print(f"\n{'='*80}")
        print(f"📊 TOKEN DISTRIBUTION STATS (Task 4.4.1)")
        print(f"{'='*80}")
        
        all_tokens = [token for doc_tokens in self.tokenized_corpus for token in doc_tokens]
        token_counts = Counter(all_tokens)
        
        tokens_per_chunk = [len(doc_tokens) for doc_tokens in self.tokenized_corpus]
        
        print(f"\n📈 Corpus Stats:")
        print(f"   Total tokens: {len(all_tokens):,}")
        print(f"   Unique tokens (vocabulary): {len(token_counts):,}")
        print(f"   Avg tokens/chunk: {len(all_tokens) / len(self.tokenized_corpus):.1f}")
        print(f"   Min tokens/chunk: {min(tokens_per_chunk)}")
        print(f"   Max tokens/chunk: {max(tokens_per_chunk)}")
        print(f"   Median tokens/chunk: {sorted(tokens_per_chunk)[len(tokens_per_chunk)//2]}")
        
        print(f"\n🔝 Top 20 Most Frequent Tokens:")
        for i, (token, count) in enumerate(token_counts.most_common(20), 1):
            print(f"   {i:2d}. '{token}' → {count:,} occurrences")
        
        print(f"\n⚠️  SUSPICIOUS TOKEN DETECTION (Task 4.4.2):")
        
        standalone_punct = {t: token_counts[t] for t in token_counts if t in string.punctuation}
        print(f"\n   Standalone punctuation tokens: {len(standalone_punct)}")
        if standalone_punct:
            for token, count in sorted(standalone_punct.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"      '{token}' → {count} occurrences")
        
        single_char = {t: token_counts[t] for t in token_counts if len(t) == 1 and t not in string.punctuation}
        print(f"\n   Single-character tokens: {len(single_char)}")
        if single_char:
            for token, count in sorted(single_char.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"      '{token}' → {count} occurrences")
        
        very_long = {t: token_counts[t] for t in token_counts if len(t) > 25}
        print(f"\n   Very long tokens (>25 chars): {len(very_long)}")
        if very_long:
            for token, count in sorted(very_long.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      '{token[:30]}...' ({len(token)} chars) → {count} occurrences")
        
        trailing_punct = {t: token_counts[t] for t in token_counts if t and len(t) > 1 and t[-1] in string.punctuation}
        print(f"\n   Tokens with trailing punctuation: {len(trailing_punct)}")
        if trailing_punct:
            for token, count in sorted(trailing_punct.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"      '{token}' → {count} occurrences")
        
        print(f"\n{'='*80}")
        
        self.token_stats = {
            'total_tokens': len(all_tokens),
            'unique_tokens': len(token_counts),
            'avg_tokens_per_chunk': len(all_tokens) / len(self.tokenized_corpus),
            'top_20_tokens': token_counts.most_common(20),
            'standalone_punct_count': len(standalone_punct),
            'single_char_count': len(single_char),
            'very_long_count': len(very_long),
            'trailing_punct_count': len(trailing_punct)
        }
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if not query or not query.strip():
            print("⚠️  Empty query provided, returning empty results")
            return []
        
        query_tokens = self.tokenize(query)
        
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
        top_indices = [i for i in top_indices if scores[i] > 0]
        
        if not top_indices:
            return []
        
        top_scores = [scores[i] for i in top_indices]
        min_score = min(top_scores)
        max_score = max(top_scores)
        
        if max_score > min_score:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in top_scores]
        else:
            normalized_scores = [1.0] * len(top_scores)
        
        retrieved_chunks = []
        for rank, (idx, score) in enumerate(zip(top_indices, normalized_scores), start=1):
            chunk = self.chunks[idx]
            retrieved_chunk = self._chunk_to_retrieved_chunk(chunk, score, rank)
            retrieved_chunks.append(retrieved_chunk)
        
        return retrieved_chunks
    
    def _chunk_to_retrieved_chunk(self, chunk: Chunk, score: float, rank: int) -> RetrievedChunk:
        
        return RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            score=score,
            rank=rank,
            retriever_type="bm25",
            authority_family=chunk.authority_family,
            tier=chunk.tier,
            year=chunk.year,
            drug_names=chunk.drug_names
        )
    
    def get_corpus_size(self) -> int:
        return len(self.chunks)
    
    def save_to_disk(self, filepath: str = "data/bm25_index.pkl"):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving BM25 index to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ BM25 index saved ({file_size:.2f} MB)")
    
    @staticmethod
    def load_from_disk(filepath: str = "data/bm25_index.pkl") -> 'BM25Index':
        print(f"\nLoading BM25 index from {filepath}...")
        with open(filepath, 'rb') as f:
            index = pickle.load(f)
        
        print(f"✅ BM25 index loaded")
        print(f"   Corpus size: {index.get_corpus_size()} chunks")
        print(f"   Vocabulary: {index.token_stats.get('unique_tokens', 'N/A'):,} unique tokens")
        return index


if __name__ == "__main__":
    import os
    
    index_path = "data/bm25_index.pkl"
    
    if os.path.exists(index_path):
        print(f"Found existing index at {index_path}")
        user_input = input("Load from disk? (y/n): ").strip().lower()
        
        if user_input == 'y':
            bm25_index = BM25Index.load_from_disk(index_path)
        else:
            bm25_index = BM25Index(k1=1.5, b=0.75)
            chunks = bm25_index.load_chunks("data/processed/chunks.json")
            bm25_index.build_index()
            bm25_index.save_to_disk(index_path)
    else:
        bm25_index = BM25Index(k1=1.5, b=0.75)
        chunks = bm25_index.load_chunks("data/processed/chunks.json")
        bm25_index.build_index()
        bm25_index.save_to_disk(index_path)
    
    print("\n" + "="*80)
    print("TESTING BM25 SEARCH")
    print("="*80)
    
    test_queries = [
        ("warfarin side effects", "In-corpus drug"),
        ("aspirin contraindications", "Out-of-corpus drug"),
        ("", "Empty query")
    ]
    
    for query, description in test_queries:
        print(f"\n📝 Query: '{query}' ({description})")
        results = bm25_index.search(query, top_k=5)
        
        if results:
            print(f"   Found {len(results)} results:")
            for r in results[:3]:  
                print(f"   [{r.rank}] Score: {r.score:.4f} | {r.authority_family} | {r.document_id}")
                print(f"       {r.text[:100]}...")
        else:
            print(f"   No results returned")
    
    print(f"\n✅ BM25 index validation complete")
    print(f"   Total chunks indexed: {bm25_index.get_corpus_size()}")
