"""
Database connection management for ClickHouse.
"""
import clickhouse_connect
from contextlib import contextmanager
from typing import Generator
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages ClickHouse database connections."""

    def __init__(self):
        """
        Initialize database connection manager.
        Uses ClickHouse connection parameters from settings.
        """
        self.host = settings.clickhouse_host
        self.port = settings.clickhouse_port
        self.database = settings.clickhouse_database
        self.username = settings.clickhouse_username
        self.password = settings.clickhouse_password
        self._test_connection()

    def _test_connection(self):
        """Test database connection on initialization."""
        try:
            client = self.get_client()
            result = client.command("SELECT 1")
            logger.info(f"ClickHouse connection successful to {self.host}:{self.port}/{self.database}")
        except Exception as e:
            logger.error(f"ClickHouse connection failed: {e}")
            raise

    def get_client(self):
        """
        Get a ClickHouse client connection.

        Returns:
            clickhouse_connect.driver.Client: ClickHouse client

        Example:
            client = db.get_client()
            result = client.query("SELECT * FROM table")
        """
        try:
            client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                database=self.database,
                username=self.username,
                password=self.password,
                settings={
                    'max_execution_time': settings.query_timeout_seconds
                }
            )
            return client
        except Exception as e:
            logger.error(f"ClickHouse connection error: {e}")
            raise

    @contextmanager
    def get_connection(self) -> Generator:
        """
        Context manager for database connections.

        Yields:
            clickhouse_connect.driver.Client: ClickHouse client

        Example:
            with db.get_connection() as client:
                result = client.query("SELECT * FROM table")
        """
        client = None
        try:
            client = self.get_client()
            yield client
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if client:
                client.close()

    def get_table_info(self) -> dict:
        """
        Get information about all tables in the database.

        Returns:
            dict: Table information including columns and types
        """
        query = f"""
            SELECT
                table,
                name as column_name,
                type as data_type,
                default_kind as is_nullable
            FROM system.columns
            WHERE database = '{self.database}'
            ORDER BY table, position
        """

        with self.get_connection() as client:
            result = client.query(query)
            results = result.named_results()

        # Organize by table
        tables = {}
        for row in results:
            table = row['table']
            if table not in tables:
                tables[table] = []
            tables[table].append({
                'column': row['column_name'],
                'type': row['data_type'],
                'nullable': row['is_nullable']
            })

        return tables


# Global database instance
db = DatabaseConnection()
