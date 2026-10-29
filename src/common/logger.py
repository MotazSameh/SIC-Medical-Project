"""
Logging infrastructure for API queries and retrieval results
"""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List


# Create logs directory
LOG_DIR = Path("logs/api")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("evidence_rag_api")


class RetrievalLogger:
    """
    Logs retrieval queries and results for future training/evaluation
    
    Discipline: Log every query with:
    - Query text
    - Retriever used
    - Latency
    - Top drugs retrieved
    - Top chunk IDs
    - Score range
    
    Future use: Training data, evaluation metrics, query analysis
    """
   
    def __init__(self, log_file: str = "retrieval_log.jsonl"):
        self.log_file = LOG_DIR / log_file
        logger.info(f"[OK] Retrieval logger initialized: {self.log_file}")
    
    def log_retrieval(
        self,
        query: str,
        retriever_type: str,
        latency_ms: float,
        top_k: int,
        results: List[dict]
    ):
        """
        Log retrieval event for future analysis
        
        Args:
            query: User query text
            retriever_type: vector | bm25 | hybrid
            latency_ms: Retrieval latency in milliseconds
            top_k: Number of results requested
            results: List of retrieved chunk dictionaries
        """
        # Extract unique drugs from results
        top_drugs = []
        seen_drugs = set()
        for result in results:
            for drug in result.get('drug_names', []):
                if drug not in seen_drugs:
                    top_drugs.append(drug)
                    seen_drugs.add(drug)
                if len(top_drugs) >= 5:  # Max 5 drugs
                    break
            if len(top_drugs) >= 5:
                break
        
        # Build log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retriever_type": retriever_type,
            "latency_ms": round(latency_ms, 2),
            "top_k": top_k,
            "result_count": len(results),
            "top_drugs_retrieved": top_drugs,
            "top_3_chunk_ids": [r.get('chunk_id') for r in results[:3]],
            "score_range": {
                "min": round(min([r.get('score', 0) for r in results]), 4) if results else 0,
                "max": round(max([r.get('score', 0) for r in results]), 4) if results else 0
            }
        }
        
        # Write to JSONL file (append mode)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Also log to console
        logger.info(
            f"[LOG] RETRIEVAL | Query: '{query[:60]}...' | "
            f"Retriever: {retriever_type} | "
            f"Latency: {latency_ms:.1f}ms | "
            f"Drugs: {top_drugs[:3]}"
        )
    
    def log_error(self, query: str, retriever_type: str, error: str):
        """Log retrieval errors"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retriever_type": retriever_type,
            "error": str(error),
            "event_type": "error"
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.error(f"[ERROR] RETRIEVAL ERROR | Query: '{query}' | Error: {error}")


# Global retrieval logger instance
retrieval_logger = RetrievalLogger()
