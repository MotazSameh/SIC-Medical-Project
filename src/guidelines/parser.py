import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from llama_parse import LlamaParse
from src.common.schemas import ParsedDocument


class ParsingError(Exception):
    pass

class DocumentParser:
    def __init__(self, api_key: str, result_type: str = "markdown"):

        self.parser = LlamaParse(api_key=api_key,result_type=result_type,verbose=True)
        self.result_type = result_type
        
    def parse_pdf(self, pdf_path: str) -> ParsedDocument:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise ParsingError(f"PDF not found: {pdf_path}")
        
        metadata = self.extract_metadata_from_path(str(pdf_path))
        
        max_retries = 3
        retry_delay = 2
        
        parse_errors = []
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                documents = self.parser.load_data(str(pdf_path))
                
                if not documents:
                    raise ParsingError("LlamaParse returned empty document list")

                page_texts = [doc.text for doc in documents]        # NEW
                parsed_markdown = "\n\n".join(page_texts) 
                
                if not parsed_markdown.strip():
                    raise ParsingError("Parsed markdown is empty")
                
                parse_duration = time.time() - start_time
                token_count = len(parsed_markdown.split())
                page_count = len(documents)
                estimated_table_count = self.estimate_table_count(parsed_markdown)
                
                parsed_doc = ParsedDocument(
                    document_id=pdf_path.stem,
                    source_path=str(pdf_path),
                    authority_family=metadata["authority_family"],
                    tier=metadata["tier"],
                    year=metadata["year"],
                    drug_names=metadata["drug_names"],
                    raw_text=parsed_markdown,
                    parsed_markdown=parsed_markdown,
                    token_count=token_count,
                    page_count=page_count,
                    estimated_table_count=estimated_table_count,
                    parsing_method="llamaparse_markdown",
                    parse_duration_seconds=round(parse_duration, 2),
                    parse_errors=parse_errors,
                    parse_timestamp=datetime.now().isoformat(),
                    page_texts=page_texts,
                )
                
                return parsed_doc
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                parse_errors.append(error_msg)
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise ParsingError(
                        f"Failed to parse {pdf_path.name} after {max_retries} attempts. "
                        f"Last error: {str(e)}"
                    )
    
    def estimate_table_count(self, markdown_text: str) -> int:
        table_rows = 0
        for line in markdown_text.split('\n'):
            if line.count('|') >= 3:
                table_rows += 1
        return table_rows
    
    def extract_metadata_from_path(self, pdf_path: str) -> Dict:
 
        path = Path(pdf_path)
        filename = path.stem 
        
        parts = filename.split('_')
        authority_prefix = parts[0].upper()
        
        authority_map = {
            "FDA": ("FDA", 1),
            "NICE": ("NICE", 1),
            "WHO": ("WHO", 2)
        }
        
        authority_family, tier = authority_map.get(
            authority_prefix, 
            ("UNKNOWN", 3)
        )
        
        year_match = re.search(r'\d{4}', filename)
        year = int(year_match.group()) if year_match else 0
        
        locked_drugs = [
            "atorvastatin", "lisinopril", "amoxicillin", "ciprofloxacin",
            "ibuprofen", "paracetamol", "acetaminophen", "metformin", "warfarin"
        ]
        
        drug_names = []
        filename_lower = filename.lower()
        for drug in locked_drugs:
            if drug in filename_lower:
                if drug == "acetaminophen":
                    drug_names.append("paracetamol")
                else:
                    drug_names.append(drug)
        
        drug_names = list(set(drug_names))
        
        return {
            "authority_family": authority_family,
            "tier": tier,
            "year": year,
            "drug_names": drug_names if drug_names else ["unknown"]
        }

