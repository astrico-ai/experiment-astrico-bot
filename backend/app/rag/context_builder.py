"""
Context Builder - Assembles minimal, focused prompts from retrieved schemas.
"""
import logging
from typing import Dict, List, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds minimal context for LLM prompts using RAG-retrieved schemas.

    Goal: Reduce 27k char prompts to 5-10k chars by including only relevant schemas.
    """

    def __init__(self, metadata_dir: str = "metadata"):
        """Initialize context builder."""
        # Resolve metadata directory - check multiple locations
        metadata_path = Path(metadata_dir)

        # If relative path, try multiple locations
        if not metadata_path.is_absolute():
            # Try current directory
            if metadata_path.exists():
                self.metadata_dir = metadata_path
            # Try parent directory (for backend/app/rag structure)
            elif (Path.cwd().parent / metadata_dir).exists():
                self.metadata_dir = Path.cwd().parent / metadata_dir
            # Try project root (../../ from backend/)
            elif (Path(__file__).parent.parent.parent.parent / metadata_dir).exists():
                self.metadata_dir = Path(__file__).parent.parent.parent.parent / metadata_dir
            else:
                # Default to provided path
                self.metadata_dir = metadata_path
        else:
            self.metadata_dir = metadata_path

        # Load business context (always include this - it's small)
        self.business_context = self._load_business_context()

        logger.info("[RAG] Context builder initialized")

    def _load_business_context(self) -> str:
        """Load business context file."""
        business_file = self.metadata_dir / "business_context.json"

        if not business_file.exists():
            logger.warning("[RAG] Business context file not found")
            return ""

        try:
            with open(business_file, 'r') as f:
                context = json.load(f)

            # Format business context
            formatted = "**Business Context:**\n"
            formatted += f"Domain: {context.get('domain', 'Unknown')}\n"
            formatted += f"Company: {context.get('company', 'Unknown')}\n"

            if 'key_metrics' in context:
                formatted += "\nKey Metrics:\n"
                for metric in context['key_metrics']:
                    formatted += f"- {metric}\n"

            return formatted

        except Exception as e:
            logger.error(f"[RAG] Error loading business context: {e}")
            return ""

    def build_context(
        self,
        query_analysis: Dict[str, Any],
        retrieved_schemas: Dict[str, Any]
    ) -> str:
        """
        Build minimal context from query analysis and retrieved schemas.

        Args:
            query_analysis: Output from QueryAnalyzer
            retrieved_schemas: Output from SchemaRetriever

        Returns:
            Formatted context string for LLM prompt
        """
        context_parts = []

        # 1. Business context (always included)
        if self.business_context:
            context_parts.append(self.business_context)

        # 2. Query type and intent
        context_parts.append(self._format_query_intent(query_analysis))

        # 3. Relevant tables (coarse-grained)
        if retrieved_schemas.get("tables"):
            context_parts.append(self._format_tables(retrieved_schemas["tables"]))

        # 4. Relevant columns (fine-grained)
        if retrieved_schemas.get("columns"):
            context_parts.append(self._format_columns(retrieved_schemas["columns"]))

        # 5. Join relationships (if multi-table query)
        if retrieved_schemas.get("relationships"):
            context_parts.append(self._format_relationships(retrieved_schemas["relationships"]))

        # Combine all parts
        full_context = "\n\n".join(context_parts)

        logger.info(f"[RAG] Built context: {len(full_context)} chars "
                   f"(from {len(retrieved_schemas.get('tables', []))} tables, "
                   f"{len(retrieved_schemas.get('columns', []))} columns)")

        # Print EXACT RAG context
        logger.info("=" * 80)
        logger.info("[RAG-CONTEXT-FULL] EXACT CONTEXT BEING SENT:")
        logger.info("=" * 80)
        logger.info(full_context)
        logger.info("=" * 80)

        return full_context

    def _format_query_intent(self, analysis: Dict) -> str:
        """Format query intent section."""
        intent = f"**Query Intent:**\n"
        intent += f"Type: {analysis['query_type']}\n"

        if analysis.get('requires_forecast'):
            intent += "Requires: Time series forecasting\n"

        if analysis.get('requires_insights'):
            intent += "Requires: Analytical insights\n"

        if analysis.get('dimensions'):
            intent += f"Group by: {', '.join(analysis['dimensions'])}\n"

        return intent

    def _format_tables(self, tables: List[Dict]) -> str:
        """Format table-level schemas."""
        if not tables:
            return ""

        formatted = "**Relevant Tables:**\n"

        for table in tables[:3]:  # Top 3 tables only
            name = table['table_name']
            score = table['relevance_score']
            metadata = table.get('metadata', {})

            formatted += f"\n- **{name}** (relevance: {score:.2f})\n"

            if metadata.get('key_columns'):
                key_cols = metadata['key_columns'].split(',')
                formatted += f"  Key columns: {', '.join(key_cols[:5])}\n"

            if metadata.get('column_count'):
                formatted += f"  Total columns: {metadata['column_count']}\n"

        return formatted

    def _format_columns(self, columns: List[Dict]) -> str:
        """Format column-level schemas."""
        if not columns:
            return ""

        formatted = "**Relevant Columns:**\n"

        # Group columns by table
        by_table = {}
        for col in columns[:15]:  # Top 15 columns only
            table = col['table']
            if table not in by_table:
                by_table[table] = []
            by_table[table].append(col)

        # Format by table
        for table, table_cols in by_table.items():
            formatted += f"\n{table}:\n"

            for col in table_cols:
                col_name = col['column']
                col_type = col['type']
                col_unit = col.get('unit', '')
                col_desc = col.get('description', '')

                formatted += f"  - {col_name} ({col_type}"
                if col_unit and col_unit != 'none':
                    formatted += f", unit: {col_unit}"
                formatted += ")\n"

                if col_desc:
                    formatted += f"    {col_desc}\n"

        return formatted

    def _format_relationships(self, relationships: List[Dict]) -> str:
        """Format join relationships."""
        if not relationships:
            return ""

        formatted = "**Join Relationships:**\n"

        for rel in relationships[:5]:  # Top 5 relationships
            table = rel['table']
            join_col = rel['join_column']
            rel_type = rel.get('type', 'potential_join')

            formatted += f"- {table}.{join_col} ({rel_type})\n"

        return formatted

    def build_sql_rules(self, retrieved_schemas: Dict[str, Any]) -> str:
        """
        Build SQL-specific rules based on retrieved schemas.

        Returns:
            SQL rules and constraints for the LLM
        """
        rules = []

        # Extract unique table names
        tables = set()
        if retrieved_schemas.get("tables"):
            tables.update(t['table_name'] for t in retrieved_schemas['tables'])

        if retrieved_schemas.get("columns"):
            tables.update(c['table'] for c in retrieved_schemas['columns'])

        if tables:
            rules.append(f"**Available Tables:** {', '.join(tables)}")

        # Add critical SQL rules
        rules.append("""
**SQL Rules:**
- Use ClickHouse SQL syntax
- NO SQL COMMENTS ALLOWED (no -- or /* */)
- ONE SQL STATEMENT ONLY (no semicolons to separate statements)
- Date functions: toStartOfMonth(), toStartOfWeek(), today()
- Type casting: toDate(), toInt32(), toFloat64()
- ROUND function: ROUND(expression, 2) - no type casting needed
- Always use explicit JOIN conditions
- Use meaningful column aliases
- Date columns: billing_date, invoice_date
        """.strip())

        return "\n\n".join(rules)

    def estimate_token_count(self, context: str) -> int:
        """
        Rough estimate of token count (1 token ≈ 4 chars).

        Args:
            context: The context string

        Returns:
            Estimated token count
        """
        return len(context) // 4


# Global instance
context_builder = None

def get_context_builder() -> ContextBuilder:
    """Get or create global context builder instance."""
    global context_builder
    if context_builder is None:
        context_builder = ContextBuilder()
    return context_builder
