from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator

class RawLogRecord(BaseModel):
    record_id: str = Field(..., description="Unique record identifier")
    timestamp: datetime = Field(..., description="Event timestamp")
    server_id: str = Field(..., description="Target server ID")
    log_level: str = Field(..., description="Log severity (INFO, WARNING, ERROR, CRITICAL)")
    message: str = Field(..., min_length=5, description="Log payload message")
    cpu_utilization: Optional[float] = Field(None, ge=0.0, le=100.0)
    memory_utilization: Optional[float] = Field(None, ge=0.0, le=100.0)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in allowed:
            raise ValueError(f"Invalid log_level: {v}. Must be one of {allowed}")
        return upper_v

class ProcessedDocumentChunk(BaseModel):
    chunk_id: str
    record_id: str
    server_id: str
    chunk_text: str
    token_estimate: int
    embedding: Optional[List[float]] = None