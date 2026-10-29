# Medical AI Agent — Multi-RAG

A bilingual, evidence-grounded medical AI Agent that combines specialized RAG pipelines for diagnosis-oriented questions with official medical guidelines and drug information.

The system is designed as a **Multi-RAG Agent**, rather than a single RAG pipeline. An LLM-based Router analyzes the user's intent, selects the appropriate knowledge path, retrieves relevant evidence, and generates a grounded response in the user's original language.

> **Medical Disclaimer:** This project is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Any decision regarding medication, dosage, or treatment must be made by a licensed healthcare professional.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [How the Router Works](#how-the-router-works)
- [Bilingual and Cross-Lingual Retrieval](#bilingual-and-cross-lingual-retrieval)
- [Retrieval Architecture](#retrieval-architecture)
- [Evidence-Grounded Generation](#evidence-grounded-generation)
- [Knowledge Sources](#knowledge-sources)
- [Multi-RAG Design](#multi-rag-design)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [API](#api)
- [Trust Score](#trust-score)
- [Safety and Medical Grounding](#safety-and-medical-grounding)
- [Running the Project](#running-the-project)
- [Future Improvements](#future-improvements)
- [Medical Disclaimer](#medical-disclaimer)

---

## Project Overview

The system separates medical knowledge into two specialized RAG pipelines because diagnosis-oriented questions and official medical guidance require different knowledge sources and retrieval strategies.

### 1. Diagnosis RAG

The Diagnosis RAG is designed for patient-style medical and diagnosis-oriented questions.

- Approximately **47,000 medical questions**
- Covers **92 medical specialties**
- Supports general medical consultation and diagnosis-oriented queries
- Uses semantic retrieval to find relevant medical question-and-answer evidence

### 2. Medical Guidelines & Drug RAG

The Guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information.

- **12 medical guideline files**
- Covers areas such as:
  - Cardiovascular diseases
  - Hypertension
  - Diabetes
  - Other medical conditions
- **8 detailed drug studies**
- Uses evidence from official medical sources such as **FDA** and **NICE**
- Uses **hybrid retrieval** combining Dense Vector Search and BM25

The two RAG pipelines remain independent, while the Agent and Router operate above them.

---

## System Architecture

The overall workflow is:

```text
                           USER QUERY
                                |
                                v
                    +------------------------+
                    |    MEDICAL AI AGENT     |
                    |                        |
                    |    LLM-based Router     |
                    |    + Translation Layer  |
                    +-----------+------------+
                                |
                         Intent Detection
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
          DIAGNOSIS RAG                GUIDELINES RAG
                 |                             |
          47K Questions                 12 Guidelines
          92 Specialties               + 8 Drug Studies
                 |                       FDA / NICE
                 |                             |
                 v                             v
        Dense Retrieval               Dense + BM25
                 |                             |
                 +--------------+--------------+
                                |
                                v
                       Retrieved Evidence
                                |
                                v
                      Evidence Normalization
                                |
                                v
                         Grounded LLM
                                |
                                v
                    Answer + Sources + Trust Score
```

The architecture is designed so that each knowledge path can use the retrieval strategy most appropriate for its source material.

---

## How the Router Works

The Router is the main decision-making component of the Agent.

It does **not** select a path based only on whether the user speaks Arabic or English. Instead, an LLM classifier analyzes the intent of the query and determines which knowledge source should be used.

```text
User Query
    |
    v
LLM Router
    |
    +-- Diagnosis Intent
    |       |
    |       +--> Diagnosis RAG
    |
    +-- Medical Guidance / Drug Intent
    |       |
    |       +--> Guidelines & Drug RAG
    |
    +-- Requires Both
            |
            +--> Both RAG Pipelines
```

The routing decision can be represented as:

```json
{
  "source": "diagnosis | guidelines | both",
  "language": "ar | en"
}
```

This allows the Agent to separate **query intent** from **response language**.

---

## Bilingual and Cross-Lingual Retrieval

The Agent supports both **Arabic and English**.

If the user's language differs from the language required by the selected knowledge source, the system translates the **retrieval query only**.

For example:

```text
Arabic User Query
       |
       v
  LLM Router
       |
       v
Translate Query -> English
       |
       v
English Medical Retrieval
       |
       v
Retrieved Evidence
       |
       v
Answer generated in Arabic
```

The user's original language is preserved for the final response.

---

## Retrieval Architecture

### Diagnosis RAG

The diagnosis path uses semantic retrieval to identify medically relevant question-and-answer evidence from the medical corpus.

```text
Query
  |
  v
Embedding
  |
  v
Vector Search
  |
  v
Relevant Medical Q&A
  |
  v
Evidence
```

### Guidelines & Drug RAG

The official-guidance path uses hybrid retrieval, combining semantic and lexical search:

```text
                     Query
                       |
             +---------+---------+
             |                   |
             v                   v
       Dense Search            BM25
     Semantic Search       Lexical Search
             |                   |
             +---------+---------+
                       |
                       v
               Score Normalization
                       |
                       v
                 Weighted Fusion
                       |
                       v
                Relevant Evidence
```

This approach combines **semantic similarity** with **keyword/lexical matching**.

---

## Evidence-Grounded Generation

The Agent is designed to answer from retrieved evidence rather than relying on unsupported model knowledge.

The generation layer follows these principles:

1. Answer using retrieved evidence.
2. Avoid unsupported medical claims.
3. Indicate when evidence is insufficient.
4. Preserve source attribution.
5. Prefer authoritative guideline evidence when sources conflict.
6. Return the answer in the user's original language.

Conceptually:

```text
Retrieved Evidence
       |
       v
  Grounded LLM
       |
       +-- Evidence sufficient
       |        |
       |        v
       |      Answer
       |
       +-- Evidence insufficient
                |
                v
          Decline / Explain
```

---

## Knowledge Sources

### Diagnosis Knowledge Base

- Approximately **47,000 medical questions**
- **92 medical specialties**
- Patient-oriented medical consultation content

### Official Medical Knowledge Base

- **12 medical guideline documents**
- **8 detailed drug studies**
- FDA sources
- NICE sources

The project keeps these knowledge domains separate so that each source can maintain its own ingestion, indexing, retrieval, and authority characteristics.

---

## Multi-RAG Design

The project follows a specialized Multi-RAG architecture:

```text
                         Medical AI Agent
                                |
                              Router
                                |
                  +-------------+-------------+
                  |                           |
                  v                           v
           Diagnosis RAG                Guidelines RAG
                  |                           |
          Medical Q&A Corpus              FDA / NICE
                  |                           |
          Dense Retrieval               Dense + BM25
                  |                           |
                  +-------------+-------------+
                                |
                                v
                         Evidence Layer
                                |
                                v
                           Grounded LLM
```

This architecture allows each RAG pipeline to be optimized for its own domain instead of putting every medical source into one undifferentiated retrieval system.

---

## Key Features

- Medical AI Agent
- LLM-based query routing
- Multi-RAG architecture
- Diagnosis-oriented RAG
- Official medical guidelines and drug RAG
- Arabic and English support
- Cross-lingual retrieval
- Dense semantic retrieval
- BM25 lexical retrieval
- Hybrid retrieval
- Evidence-grounded generation
- Source attribution
- Trust / confidence score
- Groundedness checks to reduce hallucination
- Bilingual RTL/LTR frontend
- FastAPI backend

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Agent / Routing | LLM-based Router |
| LLM | Groq |
| Embeddings | `intfloat/multilingual-e5-base` |
| Vector Database | ChromaDB |
| Semantic Retrieval | Dense Vector Search |
| Lexical Retrieval | BM25 (`rank-bm25`) |
| Hybrid Retrieval | Dense + BM25 Fusion |
| PDF Parsing | LlamaParse |
| Backend | FastAPI + Pydantic |
| Frontend | HTML / CSS / JavaScript |
| Data Processing | pandas |
| Text Splitting | LangChain Text Splitters |

---

## Project Structure

```text
project-root/
|
+-- data/
|   +-- raw/
|   |   +-- nice/
|   |   +-- fda/
|   |
|   +-- arabic/
|   |   +-- data.csv
|   |   +-- data_cleaned.csv
|   |   +-- arabic_chunks.json
|   |
|   +-- processed/
|   |   +-- chunks.json
|   |   +-- guidelines/
|   |
|   +-- vector_db/
|   |   +-- chromadb/
|   |
|   +-- bm25_index.pkl
|
+-- src/
|   +-- common/
|   |   +-- schemas.py
|   |   +-- evidence.py
|   |   +-- documents.py
|   |   +-- settings_config.py
|   |   +-- logger.py
|   |
|   +-- guidelines/
|   |   +-- parser.py
|   |   +-- chunker.py
|   |   +-- cleaning.py
|   |   +-- vector_store.py
|   |   +-- bm25_index.py
|   |   +-- hybrid_retriever.py
|   |   +-- Chatbot.py
|   |
|   +-- arabic/
|   |   +-- cleaning_arabic.py
|   |   +-- chunk_arabic.py
|   |   +-- vector_db_arabic.py
|   |   +-- Chatbot_arabic.py
|   |
|   +-- router.py
|   +-- main.py
|   |
|   +-- api/
|       +-- main.py
|       +-- models.py
|
+-- frontend/
|   +-- index.html
|
+-- requirements.txt
+-- README.md
```

---

## API

### `POST /api/chat`

Main Agent endpoint.

#### Example Request

```json
{
  "message": "ما هي التحذيرات الخاصة بهذا الدواء؟",
  "chat_history": []
}
```

#### Example Response

```json
{
  "answer": "...",
  "sources": [],
  "routing": {
    "source": "guidelines",
    "language": "ar"
  },
  "trust_score": 0.88,
  "trust_label": "high"
}
```

### `GET /api/health`

Checks whether the API is running.

### `GET /api/system-info`

Returns information about the models and technologies used by the system.

---

## Trust Score

The system provides a confidence indicator based on retrieval strength.

```text
High
 |
 +-- Strong retrieved evidence

Medium
 |
 +-- Moderate evidence

Low
 |
 +-- Weak evidence
```

The goal is to avoid presenting weakly supported medical information as a confident answer.

---

## Safety and Medical Grounding

The project is intentionally designed around evidence-bound generation.

The system should not treat the LLM's internal knowledge as the primary medical source.

Instead, the workflow is:

```text
Question
   |
   v
Routing
   |
   v
Retrieval
   |
   v
Evidence
   |
   v
Grounded Generation
   |
   v
Answer + Sources
```

If the retrieved evidence is insufficient, the system is designed to avoid confidently inventing an answer.

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/amira2162005/dawk.git
cd dawk
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
LLAMA_CLOUD_API_KEY=your_llamacloud_api_key
```

### 6. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then open:

```text
http://localhost:8000
```

---

## Future Improvements

Possible future improvements include:

- Cross-encoder reranking
- Better query intent classification
- More medical specialties
- More official clinical guidelines
- Page-level PDF citations
- Improved evaluation datasets
- Automated RAG evaluation
- Better uncertainty estimation
- More robust multilingual retrieval
- Agent observability and tracing

---

## Medical Disclaimer

This system is an informational and educational AI project.

It is **not intended to**:

- Diagnose patients
- Replace physicians
- Prescribe medication
- Determine personalized dosage
- Replace emergency medical care

For any real medical condition, medication decision, dosage, or treatment plan, consult a qualified healthcare professional.

---

## Project

**Medical AI Agent — Multi-RAG**

Built with a focus on:

**AI Agents · RAG · Multi-RAG · NLP · Information Retrieval · Medical AI · LLMs · Evidence Grounding**
