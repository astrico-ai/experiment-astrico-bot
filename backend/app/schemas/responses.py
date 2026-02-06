"""
Pydantic response schemas for FastAPI.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class FindingResponse(BaseModel):
    """API response schema for a finding."""
    hypothesis_id: int
    hypothesis_description: str
    is_interesting: bool
    insight: Optional[str] = None
    business_impact: Optional[str] = None
    recommended_action: Optional[str] = None
    supporting_data: Optional[Dict[str, Any]] = None
    confidence: str = "medium"


class ExplorationResponse(BaseModel):
    """API response schema for exploration results."""
    run_id: str
    findings: List[FindingResponse]
    iterations: int
    truncated: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_queries_executed: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    num_interesting_findings: int = 0
    error_message: Optional[str] = None


class ExploreRequest(BaseModel):
    """Request schema for exploration endpoint."""
    num_hypotheses: Optional[int] = Field(default=10, ge=1, le=30)
    focus_area: Optional[str] = Field(default=None, description="Specific area to focus on (e.g., 'customer churn', 'margin analysis')")


class SchemaResponse(BaseModel):
    """API response schema for database schema."""
    tables: Dict[str, List[dict]]
    business_context: str
    total_rows: Optional[Dict[str, int]] = None
