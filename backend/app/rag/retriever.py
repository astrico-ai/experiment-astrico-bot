"""
Schema Retriever - Multi-stage retrieval for relevant schemas.
"""
import logging
from typing import List, Dict, Any, Set
from .schema_indexer import get_schema_indexer

logger = logging.getLogger(__name__)


class SchemaRetriever:
    """
    Multi-stage schema retrieval system.

    Stage 1: Table-level retrieval (coarse)
    Stage 2: Column-level retrieval (fine)
    Stage 3: Relationship detection (joins)
    """

    def __init__(self):
        """Initialize retriever."""
        self.indexer = get_schema_indexer()
        logger.info("[RAG] Schema retriever initialized")

    def retrieve_relevant_schemas(
        self,
        query: str,
        top_k_tables: int = 3,
        top_k_columns: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve relevant schemas for a query.

        Args:
            query: User's natural language query
            top_k_tables: Number of tables to retrieve
            top_k_columns: Number of columns to retrieve

        Returns:
            Dict with tables, columns, and relationships
        """
        logger.info(f"[RAG] Retrieving schemas for query: {query}")

        # Stage 1: Retrieve relevant tables
        relevant_tables = self._retrieve_tables(query, top_k=top_k_tables)

        # Stage 2: Retrieve relevant columns
        relevant_columns = self._retrieve_columns(query, top_k=top_k_columns)

        # Stage 2.5: Ensure each retrieved table has at least some columns
        # (prevents inconsistency where table is retrieved but has no columns)
        relevant_columns = self._backfill_table_columns(relevant_tables, relevant_columns, query)

        # Stage 3: Get relationships for join paths
        relationships = self._retrieve_relationships(relevant_tables)

        # Build result
        result = {
            "tables": relevant_tables,
            "columns": relevant_columns,
            "relationships": relationships,
            "query": query
        }

        logger.info(f"[RAG] Retrieved {len(relevant_tables)} tables, "
                   f"{len(relevant_columns)} columns, "
                   f"{len(relationships)} relationships")

        # Print EXACT retrieved schemas
        logger.info("=" * 80)
        logger.info("[RAG-RETRIEVED] EXACT TABLES:")
        for table in relevant_tables:
            logger.info(f"  - {table['table_name']} (score: {table['relevance_score']:.2f})")

        logger.info("[RAG-RETRIEVED] EXACT COLUMNS:")
        for col in relevant_columns:
            logger.info(f"  - {col['table']}.{col['column']} ({col['type']}, unit: {col.get('unit', 'none')}) [score: {col['relevance_score']:.2f}]")

        logger.info("[RAG-RETRIEVED] RELATIONSHIPS:")
        for rel in relationships:
            logger.info(f"  - {rel['table']}.{rel['join_column']}")
        logger.info("=" * 80)

        return result

    def _retrieve_tables(self, query: str, top_k: int = 3) -> List[Dict]:
        """Stage 1: Retrieve relevant tables."""
        # Encode query using configured embedding provider
        query_embedding = self.indexer.encode(query)

        # Query table collection with embeddings
        results = self.indexer.table_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        tables = []
        if results['ids'] and results['ids'][0]:
            for i, table_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0

                tables.append({
                    "table_name": table_id,
                    "relevance_score": 1 - distance,  # Convert distance to similarity
                    "metadata": metadata
                })

        return tables

    def _retrieve_columns(self, query: str, top_k: int = 10) -> List[Dict]:
        """Stage 2: Retrieve relevant columns."""
        # Encode query using configured embedding provider
        query_embedding = self.indexer.encode(query)

        # Query column collection with embeddings
        results = self.indexer.column_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        columns = []
        if results['ids'] and results['ids'][0]:
            for i, col_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0

                columns.append({
                    "column_id": col_id,
                    "table": metadata.get("table", ""),
                    "column": metadata.get("column", ""),
                    "type": metadata.get("type", ""),
                    "unit": metadata.get("unit", ""),
                    "description": metadata.get("description", ""),
                    "relevance_score": 1 - distance
                })

        return columns

    def _backfill_table_columns(
        self,
        tables: List[Dict],
        columns: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Ensure each retrieved table has at least some columns.
        Prevents inconsistency where table is retrieved but has no columns shown.

        Args:
            tables: Retrieved tables from Stage 1
            columns: Retrieved columns from Stage 2
            query: Original query

        Returns:
            Columns list with backfilled entries for tables missing columns
        """
        # Count columns per table
        table_column_count = {}
        for table in tables:
            table_column_count[table['table_name']] = 0

        for col in columns:
            table_name = col['table']
            if table_name in table_column_count:
                table_column_count[table_name] += 1

        # Backfill tables with 0 columns
        backfilled_columns = columns.copy()
        min_columns_per_table = 3  # Ensure at least 3 columns per table

        for table in tables:
            table_name = table['table_name']
            if table_column_count[table_name] < min_columns_per_table:
                # Need to add columns for this table
                needed = min_columns_per_table - table_column_count[table_name]
                logger.info(f"[RAG-BACKFILL] Table {table_name} has {table_column_count[table_name]} columns, adding {needed} more")

                # Query columns specifically for this table
                table_query = f"{query} {table_name}"
                table_query_embedding = self.indexer.encode(table_query)
                results = self.indexer.column_collection.query(
                    query_embeddings=[table_query_embedding],
                    n_results=needed + 5  # Get extra in case some are already included
                )

                if results['ids'] and results['ids'][0]:
                    added = 0
                    for i, col_id in enumerate(results['ids'][0]):
                        if added >= needed:
                            break

                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                        # Only add if it's from the target table and not already in list
                        if metadata.get("table") == table_name:
                            col_id_check = f"{metadata.get('table')}.{metadata.get('column')}"
                            existing_ids = [f"{c['table']}.{c['column']}" for c in backfilled_columns]

                            if col_id_check not in existing_ids:
                                distance = results['distances'][0][i] if results['distances'] else 0
                                backfilled_columns.append({
                                    "column_id": col_id,
                                    "table": metadata.get("table", ""),
                                    "column": metadata.get("column", ""),
                                    "type": metadata.get("type", ""),
                                    "unit": metadata.get("unit", ""),
                                    "description": metadata.get("description", ""),
                                    "relevance_score": 1 - distance
                                })
                                added += 1
                                logger.info(f"[RAG-BACKFILL] Added column {table_name}.{metadata.get('column')}")

        return backfilled_columns

    def _retrieve_relationships(self, tables: List[Dict]) -> List[Dict]:
        """Stage 3: Retrieve join relationships between tables."""
        if len(tables) <= 1:
            return []  # No joins needed for single table

        # Get all table names
        table_names = [t["table_name"] for t in tables]

        # Build query for relationship search
        relationship_query = f"join {' '.join(table_names)}"
        relationship_embedding = self.indexer.encode(relationship_query)

        # Query relationship collection
        results = self.indexer.relationship_collection.query(
            query_embeddings=[relationship_embedding],
            n_results=10
        )

        relationships = []
        if results['ids'] and results['ids'][0]:
            for i, rel_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                # Only include relationships for retrieved tables
                if metadata.get("table") in table_names:
                    relationships.append({
                        "relationship_id": rel_id,
                        "table": metadata.get("table", ""),
                        "join_column": metadata.get("join_column", ""),
                        "type": metadata.get("relationship_type", "")
                    })

        return relationships


# Global instance
schema_retriever = None

def get_schema_retriever() -> SchemaRetriever:
    """Get or create global schema retriever instance."""
    global schema_retriever
    if schema_retriever is None:
        schema_retriever = SchemaRetriever()
    return schema_retriever
