"""
Tests for DataExplorer.
"""
import pytest
from app.schemas.exploration import SchemaMetadata
from app.schemas.metadata import MetadataLoader
from app.llm.explorer import DataExplorer


def test_metadata_loader():
    """Test metadata loading."""
    # Load example metadata
    metadata = {
        "test_table": [
            {
                "name": "id",
                "type": "integer",
                "unit": None,
                "description": "Primary key"
            }
        ]
    }

    business_context = "Test context"

    schema_metadata = SchemaMetadata(
        tables=metadata,
        business_context=business_context
    )

    assert schema_metadata.tables == metadata
    assert schema_metadata.business_context == business_context


def test_schema_metadata_to_context():
    """Test schema metadata conversion to context string."""
    metadata = {
        "test_table": [
            {
                "name": "id",
                "type": "integer",
                "unit": None,
                "description": "Primary key"
            }
        ]
    }

    schema_metadata = SchemaMetadata(
        tables=metadata,
        business_context="Test context"
    )

    context = schema_metadata.to_context_string()

    assert "test_table" in context
    assert "id" in context
    assert "integer" in context
    assert "Test context" in context


def test_explorer_initialization():
    """Test explorer initialization."""
    metadata = {
        "test_table": [
            {
                "name": "id",
                "type": "integer",
                "unit": None,
                "description": "Primary key"
            }
        ]
    }

    schema_metadata = SchemaMetadata(
        tables=metadata,
        business_context="Test context"
    )

    explorer = DataExplorer(schema_metadata=schema_metadata)

    assert explorer.schema_metadata == schema_metadata
    assert explorer.llm_client is not None
    assert explorer.sql_executor is not None


# Note: Full integration tests require:
# 1. A test database with sample data
# 2. OpenAI API key configured
# 3. Mock LLM responses for deterministic testing
# These should be added in a separate test_integration.py file
