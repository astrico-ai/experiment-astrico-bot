"""
RAG Initialization Script

Run this script to index all schemas into the vector database.
Should be run once during deployment or when schemas are updated.

Usage:
    python init_rag.py
"""
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.rag.schema_indexer import get_schema_indexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def initialize_rag():
    """Initialize RAG system by indexing all schemas."""
    logger.info("=" * 60)
    logger.info("RAG SYSTEM INITIALIZATION")
    logger.info("=" * 60)

    try:
        # Get schema indexer
        logger.info("Initializing schema indexer...")
        indexer = get_schema_indexer()

        # Index all schemas
        logger.info("Indexing all schemas from metadata directory...")
        indexer.index_all_schemas()

        # Get stats
        stats = indexer.get_stats()

        logger.info("=" * 60)
        logger.info("RAG INITIALIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Indexed Tables:         {stats['tables']}")
        logger.info(f"Indexed Columns:        {stats['columns']}")
        logger.info(f"Indexed Relationships:  {stats['relationships']}")
        logger.info("=" * 60)
        logger.info("RAG system is ready for use!")

        return True

    except Exception as e:
        logger.error(f"RAG initialization failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = initialize_rag()
    sys.exit(0 if success else 1)
