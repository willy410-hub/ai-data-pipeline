# Production AI-Data Pipeline & Vector Ingestion Engine

A production-grade ETL and Vector data pipeline built in Python, designed to enforce strict data quality quarantine, structured document chunking, and high-performance semantic retrieval using FAISS.

## Key Architectural Highlights

* **Data Quality Quarantine Layer:** Strict validation using Pydantic schemas to quarantine corrupted logs, prevent schema drift, and guarantee input integrity before downstream processing.
* **Structured Chunking & Metadata Enrichment:** Batched text chunking with token estimation and rich metadata tagging (`record_id`, `server_id`, `chunk_id`).
* **Vector Indexing & Semantic Search:** Dense embeddings generation via SentenceTransformers with indexing directly in FAISS (`IndexFlatL2`) for sub-millisecond retrieval.
* **Comprehensive Test Suite:** Fully verified with `pytest` covering quarantine validation, edge-case rejection, and semantic search accuracy.

## Project Structure

* `models.py`: Pydantic data contracts for raw ingestion and processed chunks.
* `pipeline.py`: Core ETL engine, chunking logic, and FAISS vector indexer.
* `test_pipeline.py`: Automated end-to-end unit and integration test suite.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run test suite
pytest test_pipeline.py -v