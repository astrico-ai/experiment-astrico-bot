"""
Test script for RAG pipeline.

This script tests the complete RAG flow:
1. Index schemas
2. Analyze queries
3. Retrieve relevant schemas
4. Build minimal context

Usage:
    python test_rag.py
"""
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.rag.pipeline import get_rag_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_rag_pipeline():
    """Test the RAG pipeline with sample queries."""
    logger.info("=" * 60)
    logger.info("TESTING RAG PIPELINE")
    logger.info("=" * 60)

    # Test queries
    test_queries = [
        "What was total sales volume in FY24?",
        "Show me revenue by region",
        "Compare Q1 vs Q2 performance",
        "Forecast next month volume",
        "Which customers contributed most to growth?",
    ]

    try:
        # Initialize pipeline
        logger.info("\n1. Initializing RAG pipeline...")
        pipeline = get_rag_pipeline()

        # Get stats
        stats = pipeline.get_stats()
        logger.info(f"\n2. RAG System Stats:")
        logger.info(f"   - Indexed Tables: {stats['indexed_tables']}")
        logger.info(f"   - Indexed Columns: {stats['indexed_columns']}")
        logger.info(f"   - Indexed Relationships: {stats['indexed_relationships']}")

        # Test each query
        logger.info("\n3. Testing queries...")
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n   Query {i}: {query}")

            result = pipeline.process_query(query)

            logger.info(f"   - Query Type: {result['metadata']['query_type']}")
            logger.info(f"   - Requires Forecast: {result['metadata']['requires_forecast']}")
            logger.info(f"   - Retrieved Tables: {result['metadata']['retrieved_tables']}")
            logger.info(f"   - Retrieved Columns: {result['metadata']['retrieved_columns_count']}")
            logger.info(f"   - Estimated Tokens: {result['metadata']['estimated_tokens']}")
            logger.info(f"   - Context Length: {len(result['context'])} chars")

        logger.info("\n" + "=" * 60)
        logger.info("RAG PIPELINE TEST COMPLETE")
        logger.info("=" * 60)

        # Show sample context
        logger.info("\n4. Sample RAG Context Output:")
        logger.info("-" * 60)
        sample_result = pipeline.process_query("What was total sales volume?")
        logger.info(sample_result['context'][:1000] + "...")
        logger.info("-" * 60)

        return True

    except Exception as e:
        logger.error(f"\nRAG pipeline test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_rag_pipeline()
    sys.exit(0 if success else 1)
