from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = (r"data/arabic/arabic_chunks.json")

CHROMA_DIR = BASE_DIR / "vector_db" / "chromadb"

COLLECTION_NAME = "arabic_medical_qa"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

BATCH_SIZE = 64


class ArabicVectorStore:

    def __init__(self,chunks_file: Path = CHUNKS_FILE,chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,model_name: str = EMBEDDING_MODEL_NAME,):

        self.chunks_file = Path(chunks_file)

        self.chroma_dir = Path(chroma_dir)

        self.collection_name = collection_name

        self.model_name = model_name

        self.model = None

        self.client = None

        self.collection = None


    def load_model(self):

        if self.model is None:

            print(f"Loading embedding model: "f"{self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            print("Embedding model loaded successfully.")

        return self.model

    def initialize_chroma(self):

        self.chroma_dir.mkdir(parents=True,exist_ok=True,)

        print(f"Initializing ChromaDB:\n"f"{self.chroma_dir.resolve()}")

        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,

                metadata={
                    "description": ("Arabic medical Q&A ""vector database"),
                    "language": "ar",
                    "source_type": "arabic_qa",
                    "embedding_model": ( self.model_name),
                    "distance": "cosine",
                },
            )
        )

        print(f"Collection: {self.collection_name}" )

        print(f"Existing documents: "f"{self.collection.count():,}")

        return self.collection

    def load_chunks(self) -> List[Dict[str, Any]]:

        if not self.chunks_file.exists():
            raise FileNotFoundError("Arabic chunks file not found:\n"f"{self.chunks_file.resolve()}")

        print(f"\nLoading chunks:\n"f"{self.chunks_file.resolve()}")

        with open(self.chunks_file,"r",encoding="utf-8",) as f:
            chunks = json.load(f)

        if not isinstance(chunks, list):
            raise ValueError("arabic_chunks.json must contain a JSON list.")

        print(f"Loaded {len(chunks):,} chunks.")

        return chunks

    @staticmethod
    def validate_chunk(chunk: Dict[str, Any]) -> None:

        required_fields = {"id","document_id", "text",}

        missing = (required_fields - set(chunk.keys()))

        if missing:
            raise ValueError("Chunk is missing required fields: "f"{sorted(missing)}")

        if not str(chunk["id"]).strip():
            raise ValueError("Chunk ID cannot be empty.")

        if not str(chunk["text"]).strip():
            raise ValueError(f"Chunk {chunk['id']} has empty text.")


    @staticmethod
    def build_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:

        metadata = {"document_id": str(chunk.get("document_id", "")),
                    "language": str(chunk.get("language", "ar")),
                    "source_type": str(chunk.get("source_type","arabic_qa",)),
                    "authority_family": str(chunk.get("authority_family","ARABIC_QA",)),
                    "category": str(chunk.get("category", "")),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "token_count": int(chunk.get("token_count", 0)),}
        return metadata

    def embed_documents(self,texts: List[str],) -> List[List[float]]:

        model = self.load_model()
        formatted_texts = [f"passage: {text}" for text in texts]
        embeddings = model.encode(formatted_texts,batch_size=BATCH_SIZE,show_progress_bar=True,normalize_embeddings=True,)

        return embeddings.tolist()

    def build(self):

        print("\n" + "=" * 60)
        print("BUILDING ARABIC VECTOR DATABASE")
        print("=" * 60)

        chunks = self.load_chunks()
        if not chunks:
            raise ValueError("No chunks found.")

        print("\nValidating chunks...")

        for chunk in chunks:
            self.validate_chunk(chunk)

        print("All chunks passed validation.")

        self.load_model()

        self.initialize_chroma()

        ids = [str(chunk["id"])for chunk in chunks]

        documents = [str(chunk["text"])for chunk in chunks]

        metadatas = [self.build_metadata(chunk)for chunk in chunks]

        print(f"\nGenerating embeddings " f"for {len(documents):,} chunks...")

        embeddings = self.embed_documents(documents)

        print("\nStoring vectors in ChromaDB...")

        for start in range(0,len(ids),BATCH_SIZE,):

            end = min(start + BATCH_SIZE,len(ids),)

            self.collection.upsert(ids=ids[start:end],documents=documents[start:end],
                                   embeddings=embeddings[start:end],metadatas=metadatas[start:end],)

            print(f"Stored "f"{end:,}/{len(ids):,}")

        total = self.collection.count()

        print("\n" + "=" * 60)
        print("VECTOR DATABASE READY")
        print("=" * 60)

        print(f"Collection: {self.collection_name}")
        print(f"Total vectors: {total:,}")
        print(f"Embedding model: "f"{self.model_name}")
        print(f"Database path: "f"{self.chroma_dir.resolve()}")
        print("=" * 60)

    def search(self,query: str,top_k: int = 5,) -> Dict[str, Any]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if self.collection is None:
            self.initialize_chroma()

        model = self.load_model()

        query_embedding = model.encode([f"query: {query}"],normalize_embeddings=True,).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents","metadatas","distances",],)

        return results

    def stats(self) -> Dict[str, Any]:

        if self.collection is None:
            self.initialize_chroma()

        return {"collection_name": (self.collection_name),
                "total_chunks": (self.collection.count()),
                "embedding_model": (self.model_name),
                "database_path": str(self.chroma_dir.resolve()),}


def main():
    store = ArabicVectorStore()
    store.build()
    stats = store.stats()
    print("\nFinal statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()