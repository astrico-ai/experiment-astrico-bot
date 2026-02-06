"""
Finding data schema.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Finding:
    """Represents an interesting insight discovered from data exploration."""

    hypothesis_id: int
    hypothesis_description: str
    is_interesting: bool
    insight: Optional[str] = None  # Plain language explanation
    business_impact: Optional[str] = None  # Quantified impact
    recommended_action: Optional[str] = None  # What to do about it
    supporting_data: Optional[Dict[str, Any]] = None  # Additional data
    confidence: str = "medium"  # high | medium | low

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_description": self.hypothesis_description,
            "is_interesting": self.is_interesting,
            "insight": self.insight,
            "business_impact": self.business_impact,
            "recommended_action": self.recommended_action,
            "supporting_data": self.supporting_data,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Finding':
        """Create from dictionary."""
        return cls(
            hypothesis_id=data["hypothesis_id"],
            hypothesis_description=data["hypothesis_description"],
            is_interesting=data["is_interesting"],
            insight=data.get("insight"),
            business_impact=data.get("business_impact"),
            recommended_action=data.get("recommended_action"),
            supporting_data=data.get("supporting_data"),
            confidence=data.get("confidence", "medium")
        )
