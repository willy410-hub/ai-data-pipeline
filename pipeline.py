import uuid
import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from pydantic import ValidationError
from models import RawLogRecord, ProcessedDocumentChunk

class ProductionAIDataPipeline:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        # Flat L2 Index for Vector Search
        self.vector_index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunk_metadata: List[Dict[str, Any]] = []

    def extract_and_quarantine(self, raw_records: List[Dict[str, Any]]) -> Tuple[List[RawLogRecord], List[Dict[str, Any]]]:
        valid_records: List[RawLogRecord] = []
        quarantine_zone: List[Dict[str, Any]] = []

        for item in raw_records:
            try:
                validated = RawLogRecord(**item)
                valid_records.append(validated)
            except ValidationError as ve:
                quarantine_zone.append({
                    "raw_payload": item,
                    "validation_errors": ve.errors()
                })

        return valid_records, quarantine_zone

    def transform_and_chunk(self, records: List[RawLogRecord], chunk_size_words: int = 20) -> List[ProcessedDocumentChunk]:
        chunks: List[ProcessedDocumentChunk] = []

        for record in records:
            structured_context = f"Server {record.server_id} reported {record.log_level}: {record.message}"
            words = structured_context.split()

            # Sliding/Batched Chunking
            for i in range(0, len(words), chunk_size_words):
                chunk_slice = " ".join(words[i:i + chunk_size_words])
                chunk_obj = ProcessedDocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    record_id=record.record_id,
                    server_id=record.server_id,
                    chunk_text=chunk_slice,
                    token_estimate=len(chunk_slice.split())
                )
                chunks.append(chunk_obj)

        return chunks

    def generate_embeddings_and_index(self, chunks: List[ProcessedDocumentChunk]) -> int:
        if not chunks:
            return 0

        texts = [c.chunk_text for c in chunks]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Normalize for cosine similarity if needed, or add directly for L2 distance
        self.vector_index.add(embeddings.astype(np.float32))

        for idx, chunk in enumerate(chunks):
            chunk.embedding = embeddings[idx].tolist()
            self.chunk_metadata.append({
                "chunk_id": chunk.chunk_id,
                "record_id": chunk.record_id,
                "server_id": chunk.server_id,
                "chunk_text": chunk.chunk_text
            })

        return len(chunks)

    def vector_search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        query_vector = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.vector_index.search(query_vector.astype(np.float32), top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunk_metadata):
                item = self.chunk_metadata[idx].copy()
                item["distance"] = float(dist)
                results.append(item)

        return results