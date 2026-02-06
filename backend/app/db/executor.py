"""
Safe SQL query execution with validation.
"""
import re
import logging
from typing import Dict, List, Any, Optional
from clickhouse_connect.driver.exceptions import ClickHouseError

from .connection import db
from ..config import settings

logger = logging.getLogger(__name__)


class QueryValidationError(Exception):
    """Raised when a query fails validation."""
    pass


class SQLExecutor:
    """Executes SQL queries with safety validation."""

    # Dangerous SQL keywords that should be blocked
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER',
        'TRUNCATE', 'EXEC', 'CREATE', 'GRANT', 'REVOKE'
    ]

    @staticmethod
    def validate_query(sql: str) -> bool:
        """
        Validate that LLM-generated SQL is safe to execute.

        Args:
            sql: SQL query to validate

        Returns:
            bool: True if query is safe

        Raises:
            QueryValidationError: If query is unsafe
        """
        if not sql or not sql.strip():
            raise QueryValidationError("Empty query")

        sql_upper = sql.upper().strip()

        # Must be a SELECT statement
        if not sql_upper.startswith('SELECT') and not sql_upper.startswith('WITH'):
            raise QueryValidationError("Only SELECT queries are allowed")

        # Check for dangerous keywords
        for keyword in SQLExecutor.DANGEROUS_KEYWORDS:
            # Use word boundaries to avoid false positives (e.g., "SELECTED" shouldn't match)
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, sql_upper):
                raise QueryValidationError(f"Dangerous keyword detected: {keyword}")

        # Block SQL comments that might hide malicious code
        if '--' in sql or '/*' in sql:
            raise QueryValidationError("SQL comments are not allowed")

        # Block multiple statements (semicolon check)
        if sql.count(';') > 1 or (sql.count(';') == 1 and not sql.strip().endswith(';')):
            raise QueryValidationError("Multiple SQL statements are not allowed")

        return True

    @staticmethod
    def execute_query(sql: str, max_rows: int = 1000) -> Dict[str, Any]:
        """
        Execute a validated SQL query and return results.

        Args:
            sql: SQL query to execute
            max_rows: Maximum number of rows to return

        Returns:
            dict: Query results with metadata
                {
                    "success": bool,
                    "rows": List[dict],
                    "row_count": int,
                    "columns": List[str],
                    "error": Optional[str],
                    "query": str
                }
        """
        result = {
            "success": False,
            "rows": [],
            "row_count": 0,
            "columns": [],
            "error": None,
            "query": sql
        }

        try:
            # Validate query first
            SQLExecutor.validate_query(sql)

            # Add LIMIT if not present (safety measure)
            sql_upper = sql.upper()
            if 'LIMIT' not in sql_upper:
                sql = sql.rstrip(';') + f' LIMIT {max_rows}'

            # Execute query
            with db.get_connection() as client:
                query_result = client.query(sql)

                # Convert to list of dicts using named_results() - must convert generator to list
                result["rows"] = list(query_result.named_results())
                result["row_count"] = len(result["rows"])
                result["columns"] = query_result.column_names
                result["success"] = True

                logger.info(f"Query executed successfully. Returned {result['row_count']} rows.")

        except QueryValidationError as e:
            result["error"] = f"Validation error: {str(e)}"
            logger.warning(f"Query validation failed: {e}")

        except ClickHouseError as e:
            result["error"] = f"Database error: {str(e)}"
            logger.error(f"Query execution failed: {e}")

        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error during query execution: {e}")

        return result

    @staticmethod
    def format_result_for_llm(result: Dict[str, Any], max_rows_display: int = 50) -> str:
        """
        Format query results in a compact format suitable for LLM context.

        Args:
            result: Query result from execute_query()
            max_rows_display: Maximum rows to include in formatted output

        Returns:
            str: Formatted result string
        """
        if not result["success"]:
            return f"Query failed: {result['error']}"

        if result["row_count"] == 0:
            return "Query returned 0 rows (no data found)"

        # Format header
        output = f"Query returned {result['row_count']} rows.\n"
        output += f"Columns: {', '.join(result['columns'])}\n\n"

        # Format rows (limit display)
        rows_to_display = min(result["row_count"], max_rows_display)
        output += "Sample data:\n"

        for i, row in enumerate(result["rows"][:rows_to_display]):
            output += f"Row {i+1}: {row}\n"

        if result["row_count"] > max_rows_display:
            output += f"\n... ({result['row_count'] - max_rows_display} more rows not shown)"

        return output


# Global executor instance
executor = SQLExecutor()
