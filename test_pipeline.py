import pytest
from datetime import datetime
from pipeline import ProductionAIDataPipeline

@pytest.fixture
def pipeline():
    return ProductionAIDataPipeline()

def test_data_quarantine_and_validation(pipeline):
    raw_input_batch = [
        {
            "record_id": "rec_001",
            "timestamp": datetime.utcnow().isoformat(),
            "server_id": "srv-us-east-1",
            "log_level": "ERROR",
            "message": "Database connection timeout after 3 retries",
            "cpu_utilization": 88.5,
            "memory_utilization": 92.0
        },
        {
            # Invalid record: bad log level & message too short
            "record_id": "rec_002",
            "timestamp": datetime.utcnow().isoformat(),
            "server_id": "srv-eu-central-1",
            "log_level": "UNKNOWN_SEVERITY",
            "message": "err",
            "cpu_utilization": 150.0 # Out of bounds
        }
    ]

    valid_records, quarantine = pipeline.extract_and_quarantine(raw_input_batch)

    assert len(valid_records) == 1
    assert valid_records[0].record_id == "rec_001"
    assert len(quarantine) == 1
    assert quarantine[0]["raw_payload"]["record_id"] == "rec_002"

def test_chunking_and_vector_indexing(pipeline):
    sample_data = [
        {
            "record_id": "rec_101",
            "timestamp": datetime.utcnow().isoformat(),
            "server_id": "srv-cluster-a",
            "log_level": "CRITICAL",
            "message": "Memory heap allocation exhausted in worker process causing crash",
            "cpu_utilization": 99.1,
            "memory_utilization": 99.8
        },
        {
            "record_id": "rec_102",
            "timestamp": datetime.utcnow().isoformat(),
            "server_id": "srv-cluster-b",
            "log_level": "INFO",
            "message": "Routine health check passed successfully across all pods",
            "cpu_utilization": 12.0,
            "memory_utilization": 30.5
        }
    ]

    valid_records, _ = pipeline.extract_and_quarantine(sample_data)
    chunks = pipeline.transform_and_chunk(valid_records, chunk_size_words=10)

    assert len(chunks) >= 2

    indexed_count = pipeline.generate_embeddings_and_index(chunks)
    assert indexed_count == len(chunks)

    # Test Semantic Search
    query = "out of memory heap crash"
    search_results = pipeline.vector_search(query, top_k=1)

    assert len(search_results) == 1
    assert search_results[0]["record_id"] == "rec_101"