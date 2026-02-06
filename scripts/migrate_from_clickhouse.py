#!/usr/bin/env python3
"""
Migrate data from ClickHouse to PostgreSQL.

This script:
1. Connects to ClickHouse source database (READ ONLY)
2. Reads customer_master, material_master, and sales data
3. Creates PostgreSQL tables
4. Copies data to PostgreSQL

⚠️ SAFE: Original ClickHouse database is NOT modified - only reads data!

Usage:
    python scripts/migrate_from_clickhouse.py --help
    python scripts/migrate_from_clickhouse.py --inspect  # Just inspect, don't migrate
    python scripts/migrate_from_clickhouse.py --tables customer_master  # Migrate one table
    python scripts/migrate_from_clickhouse.py --all  # Migrate all tables
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

# ClickHouse connector
try:
    import clickhouse_connect
except ImportError:
    print("❌ ClickHouse driver not installed. Run: pip install clickhouse-connect")
    sys.exit(1)

# PostgreSQL connector
import psycopg
from psycopg.rows import dict_row

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClickHouseToPostgres:
    """Migrate data from ClickHouse to PostgreSQL."""

    def __init__(
        self,
        ch_host: str,
        ch_port: int,
        ch_database: str,
        ch_user: str,
        ch_password: str,
        pg_connection_string: str
    ):
        """Initialize migration."""
        self.ch_client = clickhouse_connect.get_client(
            host=ch_host,
            port=ch_port,
            database=ch_database,
            username=ch_user,
            password=ch_password
        )
        self.pg_conn_string = pg_connection_string

        logger.info(f"ClickHouse: {ch_host}:{ch_port}/{ch_database}")
        logger.info(f"PostgreSQL: {pg_connection_string.split('@')[1] if '@' in pg_connection_string else 'configured'}")

    def test_connections(self):
        """Test both database connections."""
        try:
            # Test ClickHouse
            result = self.ch_client.query("SELECT 1")
            logger.info("✓ ClickHouse connection successful")
        except Exception as e:
            logger.error(f"✗ ClickHouse connection failed: {e}")
            raise

        try:
            # Test PostgreSQL
            with psycopg.connect(self.pg_conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            logger.info("✓ PostgreSQL connection successful")
        except Exception as e:
            logger.error(f"✗ PostgreSQL connection failed: {e}")
            raise

    def inspect_clickhouse_table(self, table_name: str) -> Dict[str, Any]:
        """
        Inspect a ClickHouse table structure.

        Returns:
            dict: Table metadata including columns and sample data
        """
        logger.info(f"Inspecting ClickHouse table: {table_name}")

        # Get table structure
        describe_query = f"DESCRIBE TABLE {table_name}"
        describe_result = self.ch_client.query(describe_query)
        columns = [(row[0], row[1]) for row in describe_result.result_rows]

        # Get row count
        count_query = f"SELECT count(*) FROM {table_name}"
        count_result = self.ch_client.query(count_query)
        row_count = count_result.result_rows[0][0]

        # Get sample data
        sample_query = f"SELECT * FROM {table_name} LIMIT 5"
        sample_result = self.ch_client.query(sample_query)

        return {
            "table_name": table_name,
            "columns": columns,
            "row_count": row_count,
            "sample_data": sample_result
        }

    def clickhouse_to_postgres_type(self, ch_type: str) -> str:
        """Convert ClickHouse data type to PostgreSQL type."""
        type_mapping = {
            'String': 'TEXT',
            'FixedString': 'VARCHAR(500)',
            'Int8': 'SMALLINT',
            'Int16': 'SMALLINT',
            'Int32': 'INTEGER',
            'Int64': 'BIGINT',
            'UInt8': 'SMALLINT',
            'UInt16': 'INTEGER',
            'UInt32': 'BIGINT',
            'UInt64': 'BIGINT',
            'Float32': 'REAL',
            'Float64': 'DOUBLE PRECISION',
            'Decimal': 'NUMERIC',
            'Date': 'DATE',
            'DateTime': 'TIMESTAMP',
            'DateTime64': 'TIMESTAMP',
            'Bool': 'BOOLEAN',
            'UUID': 'UUID',
        }

        # Handle nullable types
        if ch_type.startswith('Nullable('):
            inner_type = ch_type[9:-1]
            return self.clickhouse_to_postgres_type(inner_type)

        # Handle LowCardinality
        if ch_type.startswith('LowCardinality('):
            inner_type = ch_type[15:-1]
            return self.clickhouse_to_postgres_type(inner_type)

        # Handle decimal with precision
        if ch_type.startswith('Decimal'):
            return 'NUMERIC'

        # Check base type
        for ch_prefix, pg_type in type_mapping.items():
            if ch_type.startswith(ch_prefix):
                return pg_type

        # Default to TEXT
        logger.warning(f"Unknown ClickHouse type: {ch_type}, using TEXT")
        return 'TEXT'

    def create_postgres_table(self, table_name: str, columns: List[tuple]) -> str:
        """
        Generate CREATE TABLE SQL for PostgreSQL.

        Args:
            table_name: Name of the table
            columns: List of (name, type, ...) tuples from ClickHouse

        Returns:
            str: CREATE TABLE SQL statement
        """
        col_definitions = []

        for col in columns:
            col_name = col[0]
            ch_type = col[1]
            pg_type = self.clickhouse_to_postgres_type(ch_type)

            col_definitions.append(f'    "{col_name}" {pg_type}')

        # Join column definitions outside f-string to avoid backslash issue
        cols_joined = ',\n'.join(col_definitions)
        create_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id SERIAL PRIMARY KEY,
{cols_joined},
    migrated_at TIMESTAMP DEFAULT NOW()
);
"""
        return create_sql

    def migrate_table(
        self,
        table_name: str,
        batch_size: int = 10000,
        create_table: bool = True
    ):
        """
        Migrate a table from ClickHouse to PostgreSQL.

        Args:
            table_name: Name of the table to migrate
            batch_size: Number of rows per batch
            create_table: Whether to create the table first
        """
        logger.info(f"=" * 60)
        logger.info(f"Migrating table: {table_name}")
        logger.info(f"=" * 60)

        # Inspect table
        metadata = self.inspect_clickhouse_table(table_name)
        logger.info(f"Source table has {metadata['row_count']:,} rows")
        logger.info(f"Columns: {len(metadata['columns'])}")

        # Create PostgreSQL table
        if create_table:
            create_sql = self.create_postgres_table(table_name, metadata['columns'])
            logger.info(f"Creating PostgreSQL table...")

            with psycopg.connect(self.pg_conn_string) as pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute(create_sql)
                    pg_conn.commit()
            logger.info("✓ Table created in PostgreSQL")

        # Get column names (excluding our added columns)
        column_names = [col[0] for col in metadata['columns']]
        columns_sql = ', '.join([f'"{col}"' for col in column_names])

        # Migrate data in batches
        offset = 0
        total_migrated = 0

        with psycopg.connect(self.pg_conn_string) as pg_conn:
            while True:
                # Fetch batch from ClickHouse (READ ONLY)
                query = f"SELECT {columns_sql} FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
                result = self.ch_client.query(query)
                rows = result.result_rows

                if not rows:
                    break

                # Convert datetime objects to strings for PostgreSQL
                processed_rows = []
                for row in rows:
                    processed_row = []
                    for val in row:
                        # Handle None/NULL values
                        if val is None:
                            processed_row.append(None)
                        else:
                            processed_row.append(val)
                    processed_rows.append(tuple(processed_row))

                # Insert into PostgreSQL
                placeholders = ', '.join(['%s'] * len(column_names))
                insert_sql = f"""
                    INSERT INTO {table_name} ({columns_sql})
                    VALUES ({placeholders})
                """

                with pg_conn.cursor() as cur:
                    cur.executemany(insert_sql, processed_rows)
                    pg_conn.commit()

                total_migrated += len(rows)
                logger.info(f"Migrated {total_migrated:,} / {metadata['row_count']:,} rows ({total_migrated/metadata['row_count']*100:.1f}%)")

                offset += batch_size

        logger.info(f"✓ Migration complete: {total_migrated:,} rows migrated")

        # Create indexes
        self.create_indexes(table_name, column_names)

    def create_indexes(self, table_name: str, columns: List[str]):
        """Create useful indexes on the table."""
        logger.info("Creating indexes...")

        indexes = []

        # Common index patterns
        date_columns = [col for col in columns if 'date' in col.lower()]
        code_columns = [col for col in columns if 'code' in col.lower() or col.endswith('_id')]

        for col in date_columns:
            indexes.append(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{col.replace(" ", "_")} ON {table_name}("{col}")')

        for col in code_columns[:5]:  # Limit to first 5 code columns
            indexes.append(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{col.replace(" ", "_")} ON {table_name}("{col}")')

        if indexes:
            with psycopg.connect(self.pg_conn_string) as pg_conn:
                with pg_conn.cursor() as cur:
                    for idx_sql in indexes:
                        try:
                            logger.info(f"  Creating index...")
                            cur.execute(idx_sql)
                        except Exception as e:
                            logger.warning(f"  Index creation failed (may already exist): {e}")
                    pg_conn.commit()
            logger.info(f"✓ Created/verified {len(indexes)} indexes")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate data from ClickHouse to PostgreSQL (READ ONLY on source)"
    )

    # ClickHouse connection
    parser.add_argument('--ch-host', default='13.126.145.174', help='ClickHouse host')
    parser.add_argument('--ch-port', type=int, default=8123, help='ClickHouse port')
    parser.add_argument('--ch-database', default='veedol_sales', help='ClickHouse database')
    parser.add_argument('--ch-user', default='default', help='ClickHouse username')
    parser.add_argument('--ch-password', default='', help='ClickHouse password')

    # PostgreSQL connection
    parser.add_argument('--pg-url', help='PostgreSQL connection URL (e.g., postgresql://user:pass@localhost/dbname)')

    # Migration options
    parser.add_argument('--tables', nargs='+', help='Tables to migrate (space-separated)')
    parser.add_argument('--all', action='store_true', help='Migrate all tables')
    parser.add_argument('--inspect', action='store_true', help='Just inspect tables, don\'t migrate')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for migration')
    parser.add_argument('--test', action='store_true', help='Test connections only')

    args = parser.parse_args()

    # Get PostgreSQL URL from env if not provided
    if not args.pg_url and not args.inspect:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        args.pg_url = os.getenv('DATABASE_URL')

        if not args.pg_url:
            print("❌ PostgreSQL URL required. Use --pg-url or set DATABASE_URL in .env")
            sys.exit(1)

    # Initialize migrator
    migrator = ClickHouseToPostgres(
        ch_host=args.ch_host,
        ch_port=args.ch_port,
        ch_database=args.ch_database,
        ch_user=args.ch_user,
        ch_password=args.ch_password,
        pg_connection_string=args.pg_url if args.pg_url else ''
    )

    # Test connections
    if args.test:
        migrator.test_connections()
        print("\n✓ Connection test successful!")
        return

    # Determine tables to process
    if args.all:
        # Get all tables from ClickHouse
        tables_query = "SHOW TABLES"
        result = migrator.ch_client.query(tables_query)
        tables = [row[0] for row in result.result_rows]
        logger.info(f"Found {len(tables)} tables in ClickHouse: {tables}")
    elif args.tables:
        tables = args.tables
    else:
        # Default: just list available tables
        tables_query = "SHOW TABLES"
        result = migrator.ch_client.query(tables_query)
        available_tables = [row[0] for row in result.result_rows]
        print(f"\n📋 Available tables in ClickHouse:")
        for i, table in enumerate(available_tables, 1):
            print(f"  {i}. {table}")
        print(f"\nUsage:")
        print(f"  --inspect               : Inspect tables without migrating")
        print(f"  --tables customer_master: Migrate specific table")
        print(f"  --all                   : Migrate all tables")
        return

    # Inspect or migrate
    if args.inspect:
        for table in tables:
            try:
                metadata = migrator.inspect_clickhouse_table(table)
                print(f"\n{'='*60}")
                print(f"Table: {table}")
                print(f"{'='*60}")
                print(f"Rows: {metadata['row_count']:,}")
                print(f"\nColumns ({len(metadata['columns'])}):")
                for col in metadata['columns']:
                    print(f"  - {col[0]}: {col[1]}")

                # Save to JSON for reference
                output_file = f"metadata/inspect_{table}.json"
                Path("metadata").mkdir(exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump({
                        'table': table,
                        'row_count': metadata['row_count'],
                        'columns': [{'name': c[0], 'type': c[1]} for c in metadata['columns']]
                    }, f, indent=2)
                print(f"\n✓ Metadata saved to: {output_file}")

            except Exception as e:
                logger.error(f"Failed to inspect {table}: {e}")
    else:
        # Migrate tables
        print("\n⚠️  Starting migration (ClickHouse will NOT be modified)")
        for table in tables:
            try:
                migrator.migrate_table(table, batch_size=args.batch_size)
            except Exception as e:
                logger.error(f"Failed to migrate {table}: {e}", exc_info=True)

        print("\n" + "="*60)
        print("✓ Migration Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run metadata generation: python scripts/generate_metadata.py")
        print("2. Test exploration: python scripts/run_exploration.py")


if __name__ == "__main__":
    main()
