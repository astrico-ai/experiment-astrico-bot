"""
RAG Pipeline - Orchestrates the complete RAG flow.
"""
import logging
from typing import Dict, Any
from .query_analyzer import get_query_analyzer
from .retriever import get_schema_retriever
from .context_builder import get_context_builder

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline that orchestrates:
    1. Query Analysis (extract intent)
    2. Schema Retrieval (vector search)
    3. Context Building (minimal prompt assembly)
    """

    def __init__(self):
        """Initialize RAG pipeline."""
        self.query_analyzer = get_query_analyzer()
        self.schema_retriever = get_schema_retriever()
        self.context_builder = get_context_builder()

        logger.info("[RAG] Pipeline initialized and ready")

    def process_query(
        self,
        query: str,
        top_k_tables: int = 3,
        top_k_columns: int = 15
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.

        Args:
            query: User's natural language query
            top_k_tables: Number of tables to retrieve
            top_k_columns: Number of columns to retrieve

        Returns:
            Dict containing:
            - context: Minimal formatted context for LLM prompt
            - sql_rules: SQL-specific rules
            - metadata: Analysis and retrieval metadata
        """
        logger.info(f"[RAG] Processing query: {query[:100]}...")

        # Stage 1: Analyze query intent
        query_analysis = self.query_analyzer.analyze(query)

        logger.info(f"[RAG] Query type: {query_analysis['query_type']}, "
                   f"requires_forecast: {query_analysis['requires_forecast']}")

        # Stage 2: Retrieve relevant schemas using analyzed keywords
        search_query = query_analysis['search_keywords']

        retrieved_schemas = self.schema_retriever.retrieve_relevant_schemas(
            query=search_query,
            top_k_tables=top_k_tables,
            top_k_columns=top_k_columns
        )

        # Stage 3: Build minimal context
        context = self.context_builder.build_context(
            query_analysis=query_analysis,
            retrieved_schemas=retrieved_schemas
        )

        # Build SQL-specific rules
        sql_rules = self.context_builder.build_sql_rules(retrieved_schemas)

        # Estimate token savings
        token_estimate = self.context_builder.estimate_token_count(context)

        result = {
            "context": context,
            "sql_rules": sql_rules,
            "metadata": {
                "query_type": query_analysis['query_type'],
                "requires_forecast": query_analysis['requires_forecast'],
                "requires_insights": query_analysis['requires_insights'],
                "retrieved_tables": [t['table_name'] for t in retrieved_schemas['tables']],
                "retrieved_columns_count": len(retrieved_schemas['columns']),
                "relationships_count": len(retrieved_schemas['relationships']),
                "estimated_tokens": token_estimate,
                "original_query": query
            }
        }

        logger.info(f"[RAG] Pipeline complete: {token_estimate} tokens, "
                   f"{len(retrieved_schemas['tables'])} tables retrieved")

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        indexer_stats = self.schema_retriever.indexer.get_stats()

        return {
            "indexed_tables": indexer_stats['tables'],
            "indexed_columns": indexer_stats['columns'],
            "indexed_relationships": indexer_stats['relationships'],
            "status": "ready"
        }


# Global instance
rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """Get or create global RAG pipeline instance."""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline
