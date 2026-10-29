import json
import time
from pathlib import Path
from typing import List, Optional
from dataclasses import asdict

import chromadb
from src.common import settings_config
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.common.schemas import Chunk, RetrievedChunk


class VectorStore:
    def __init__(self,persist_directory: str = "data/vector_db/chromadb",
                 embedding_model_name: str = "intfloat/multilingual-e5-base",
                 collection_name: str = "drug_chunks"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        self._is_e5_model = "e5" in embedding_model_name.lower()
        self.passage_prefix = "passage: " if self._is_e5_model else ""
        self.query_prefix = "query: " if self._is_e5_model else ""

        print(f"Initializing ChromaDB at {self.persist_directory}...")
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded ({self.embedding_dim} dimensions)")
        if self._is_e5_model:
            print("   e5 model detected — applying 'passage: '/'query: ' prefixes automatically")

        self.collection = None
        self.chunks_loaded = []

    def load_chunks(self, chunks_json_path: str) -> List[Chunk]:
    
        print(f"\nLoading chunks from {chunks_json_path}...")
        with open(chunks_json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        chunks = []
        for chunk_data in chunks_data:
            chunk = Chunk(**chunk_data)
            chunks.append(chunk)

        print(f" Loaded {len(chunks)} chunks")
        self.chunks_loaded = chunks
        return chunks

    def _generate_embeddings(self,texts: List[str],batch_size: int = 32,is_query: bool = False,) -> List[List[float]]:

        prefix = self.query_prefix if is_query else self.passage_prefix
        prefixed_texts = [f"{prefix}{t}" for t in texts]

        print(f"\nGenerating embeddings for {len(texts)} texts...")
        start_time = time.time()

        embeddings = []
        for i in tqdm(range(0, len(prefixed_texts), batch_size), desc="Embedding batches"):
            batch = prefixed_texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True, 
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings.tolist())

        elapsed = time.time() - start_time
        avg_time = (elapsed / len(texts)) * 1000 if texts else 0.0
        print(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s ({avg_time:.2f}ms/chunk)")
        return embeddings

    def create_or_load_collection(self, reset: bool = False):
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"  Deleted existing collection: {self.collection_name}")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

        count = self.collection.count()
        print(f" Collection '{self.collection_name}' ready (current count: {count})")

    def add_chunks(self, chunks: Optional[List[Chunk]] = None):
        
        if chunks is None:
            chunks = self.chunks_loaded

        if not chunks:
            raise ValueError("No chunks to add. Call load_chunks() first.")

        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_or_load_collection() first.")

        print(f"\nAdding {len(chunks)} chunks to ChromaDB...")

        ids = []
        texts = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.id)
            texts.append(chunk.text)

            metadata = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "authority_family": chunk.authority_family,
                "tier": chunk.tier,
                "year": chunk.year if chunk.year else 0,
                "drug_names": json.dumps(chunk.drug_names),
                "start_page": chunk.start_page if chunk.start_page is not None else -1,
                "end_page": chunk.end_page if chunk.end_page is not None else -1,
            }
            metadatas.append(metadata)

        embeddings = self._generate_embeddings(texts, is_query=False)

        print(f"Storing chunks in ChromaDB...")
        self.collection.upsert(ids=ids,embeddings=embeddings,documents=texts,metadatas=metadatas)

        print(f"Upserted {len(chunks)} chunks to collection")
        print(f"Total chunks in collection: {self.collection.count()}")

    def search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_or_load_collection() first.")

        if not query or not query.strip():
            print("  Empty query provided, returning empty results")
            return []

        query_embedding = self._generate_embeddings([query], is_query=True)[0]

        results = self.collection.query(query_embeddings=[query_embedding],n_results=top_k)

        retrieved_chunks = []

        if results['ids'] and results['ids'][0]:
            for rank, (chunk_id, distance, metadata, text) in enumerate(
                zip(results['ids'][0],results['distances'][0],
                    results['metadatas'][0],results['documents'][0]),
                start=1):
                retrieved_chunk = self._chunk_to_retrieved_chunk(
                    chunk_id=chunk_id,
                    distance=distance,
                    metadata=metadata,
                    text=text,
                    rank=rank
                )
                retrieved_chunks.append(retrieved_chunk)

        return retrieved_chunks

    def _chunk_to_retrieved_chunk(self,chunk_id: str,distance: float,metadata: dict,text: str,rank: int) -> RetrievedChunk:
        
        score = max(0.0, min(1.0, 1.0 - distance))

        start_page = metadata.get('start_page', -1)
        end_page = metadata.get('end_page', -1)

        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=metadata['document_id'],
            text=text,
            score=score,
            rank=rank,
            retriever_type="vector",
            authority_family=metadata['authority_family'],
            tier=metadata['tier'],
            year=metadata['year'] if metadata['year'] != 0 else None,
            drug_names=json.loads(metadata['drug_names']),
            start_page=start_page if start_page != -1 else None,
            end_page=end_page if end_page != -1 else None,
        )

    def get_chunk_count(self) -> int:
        
        if self.collection is None:
            return 0
        return self.collection.count()



if __name__ == "__main__":
    vs = VectorStore(
        persist_directory="data/vector_db/chromadb",
        embedding_model_name="intfloat/multilingual-e5-base"
    )

    chunks = vs.load_chunks("data/processed/chunks.json")

    vs.create_or_load_collection(reset=True)
    vs.add_chunks()

    print("\n" + "="*80)
    print("TESTING VECTOR SEARCH")
    print("="*80)

    test_queries = [
        ("warfarin side effects", "In-corpus drug"),
        ("aspirin contraindications", "Out-of-corpus drug"),
        ("", "Empty query")
    ]

    for query, description in test_queries:
        print(f"\n Query: '{query}' ({description})")
        results = vs.search(query, top_k=5)

        if results:
            print(f"   Found {len(results)} results:")
            for r in results[:3]:
                page_info = f" | p.{r.start_page}" if r.start_page is not None else ""
                print(f"   [{r.rank}] Score: {r.score:.4f} | {r.authority_family} | {r.document_id}{page_info}")
                print(f"       {r.text[:100]}...")
        else:
            print(f"   No results returned")

    print(f"\n Vector store validation complete")
    print(f"   Total chunks indexed: {vs.get_chunk_count()}")