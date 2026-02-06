"""
Hypothesis data schema.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Hypothesis:
    """Represents a testable hypothesis about the data."""

    id: int
    description: str  # What we're looking for
    business_rationale: str  # Why it matters
    sql_query: str  # ClickHouse query to test it
    expected_insight_type: str  # anomaly | pattern | trend | correlation

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "business_rationale": self.business_rationale,
            "sql_query": self.sql_query,
            "expected_insight_type": self.expected_insight_type
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Hypothesis':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            business_rationale=data["business_rationale"],
            sql_query=data["sql_query"],
            expected_insight_type=data["expected_insight_type"]
        )
