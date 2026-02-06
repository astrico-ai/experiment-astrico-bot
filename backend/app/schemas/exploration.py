"""
Exploration result schemas.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import uuid

from .finding import Finding


@dataclass
class ExplorationResult:
    """Result of a complete data exploration run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    findings: List[Finding] = field(default_factory=list)
    iterations: int = 0
    truncated: bool = False  # True if hit max iterations
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_queries_executed: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "findings": [f.to_dict() for f in self.findings],
            "iterations": self.iterations,
            "truncated": self.truncated,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_queries_executed": self.total_queries_executed,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "error_message": self.error_message,
            "num_interesting_findings": len([f for f in self.findings if f.is_interesting])
        }

    def add_finding(self, finding: Finding):
        """Add a finding to results."""
        self.findings.append(finding)

    def get_interesting_findings(self) -> List[Finding]:
        """Get only interesting findings, sorted by confidence."""
        interesting = [f for f in self.findings if f.is_interesting]
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(interesting, key=lambda f: confidence_order.get(f.confidence, 3))


@dataclass
class SchemaMetadata:
    """Metadata about database schema for LLM context."""

    tables: dict  # Table name -> column info
    business_context: str  # Business domain context
    total_rows: Optional[dict] = None  # Table name -> row count
    date_range: Optional[dict] = None  # Table name -> (min_date, max_date)

    def to_context_string(self) -> str:
        """
        Format metadata as a string for LLM context.

        Returns:
            str: Formatted schema context
        """
        context = "## DATABASE SCHEMA\n\n"

        for table_name, columns in self.tables.items():
            context += f"### Table: {table_name}\n"

            if self.total_rows and table_name in self.total_rows:
                context += f"Total rows: {self.total_rows[table_name]:,}\n"

            if self.date_range and table_name in self.date_range:
                min_date, max_date = self.date_range[table_name]
                context += f"Date range: {min_date} to {max_date}\n"

            context += "\nColumns:\n"
            for col in columns:
                context += f"  - {col['name']}: {col['type']}"
                if col.get('description'):
                    context += f" - {col['description']}"
                if col.get('unit'):
                    context += f" (Unit: {col['unit']})"
                context += "\n"
            context += "\n"

        context += f"\n## BUSINESS CONTEXT\n\n{self.business_context}\n"

        return context
