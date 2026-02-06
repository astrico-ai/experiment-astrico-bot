"""
Migrate multiple tables from ClickHouse to PostgreSQL
"""
import psycopg2
import requests
import json
from typing import List, Dict, Any

# Source: ClickHouse
CLICKHOUSE_HOST = "10.1.1.22"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_DB = "veedol_sales"
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = ""

# Target: PostgreSQL
POSTGRES_URL = "postgresql://sanujphilip@localhost:5432/lubricant_sales"

# Tables to migrate
TABLES_TO_MIGRATE = [
    "customer_master_veedol",
    "material_master_veedol",
    "sales_invoices_veedol"
]


def clickhouse_query(query: str) -> List[Dict[str, Any]]:
    """Execute query on ClickHouse via HTTP interface."""
    url = f"http://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/"
    params = {
        "database": CLICKHOUSE_DB,
        "user": CLICKHOUSE_USER,
        "password": CLICKHOUSE_PASSWORD,
        "query": query,
        "default_format": "JSONEachRow"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    # Parse JSON lines
    results = []
    for line in response.text.strip().split('\n'):
        if line:
            results.append(json.loads(line))

    return results


def get_clickhouse_schema(table_name: str) -> List[Dict[str, str]]:
    """Get table schema from ClickHouse."""
    query = f"DESCRIBE TABLE {table_name}"
    return clickhouse_query(query)


def get_clickhouse_data(table_name: str) -> List[Dict[str, Any]]:
    """Get all data from ClickHouse table."""
    query = f"SELECT * FROM {table_name}"
    return clickhouse_query(query)


def clickhouse_to_postgres_type(ch_type: str) -> str:
    """Convert ClickHouse data type to PostgreSQL."""
    type_mapping = {
        'UInt8': 'SMALLINT',
        'UInt16': 'INTEGER',
        'UInt32': 'BIGINT',
        'UInt64': 'NUMERIC',
        'Int8': 'SMALLINT',
        'Int16': 'SMALLINT',
        'Int32': 'INTEGER',
        'Int64': 'BIGINT',
        'Float32': 'REAL',
        'Float64': 'DOUBLE PRECISION',
        'String': 'TEXT',
        'FixedString': 'VARCHAR',
        'Date': 'DATE',
        'DateTime': 'TIMESTAMP',
        'DateTime64': 'TIMESTAMP',
        'Decimal': 'NUMERIC',
    }

    # Handle nullable types
    if ch_type.startswith('Nullable('):
        inner_type = ch_type[9:-1]
        return clickhouse_to_postgres_type(inner_type)

    # Handle decimal with precision
    if ch_type.startswith('Decimal'):
        return 'NUMERIC'

    # Check exact match
    for ch_prefix, pg_type in type_mapping.items():
        if ch_type.startswith(ch_prefix):
            return pg_type

    # Default to TEXT if unknown
    return 'TEXT'


def create_postgres_table(cursor, table_name: str, schema: List[Dict[str, str]]) -> None:
    """Create table in PostgreSQL based on ClickHouse schema."""
    # Drop if exists
    cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

    # Build CREATE TABLE statement
    columns = []
    for col in schema:
        col_name = col['name']
        ch_type = col['type']
        pg_type = clickhouse_to_postgres_type(ch_type)
        columns.append(f'"{col_name}" {pg_type}')

    create_sql = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(columns) + "\n)"
    cursor.execute(create_sql)


def insert_data(cursor, table_name: str, data: List[Dict[str, Any]]) -> None:
    """Insert data into PostgreSQL table."""
    if not data:
        print("   No data to insert")
        return

    # Get column names from first row
    columns = list(data[0].keys())
    col_names = ', '.join([f'"{col}"' for col in columns])
    placeholders = ', '.join(['%s'] * len(columns))

    insert_sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"

    # Prepare data rows
    rows = []
    for row in data:
        rows.append([row[col] for col in columns])

    # Batch insert
    cursor.executemany(insert_sql, rows)
    print(f"   Inserted {len(rows):,} rows")


def migrate_table(cursor, table_name: str) -> None:
    """Migrate a single table from ClickHouse to PostgreSQL."""
    print(f"\n{'='*70}")
    print(f"Migrating: {table_name}")
    print('='*70)

    # Step 1: Get schema
    print("1. Fetching schema from ClickHouse...")
    schema = get_clickhouse_schema(table_name)
    print(f"   Found {len(schema)} columns")

    # Step 2: Get data
    print("2. Fetching data from ClickHouse...")
    data = get_clickhouse_data(table_name)
    print(f"   Fetched {len(data):,} rows")

    # Step 3: Create table
    print("3. Creating table in PostgreSQL...")
    create_postgres_table(cursor, table_name, schema)
    print("   Table created")

    # Step 4: Insert data
    print("4. Inserting data into PostgreSQL...")
    insert_data(cursor, table_name, data)

    print(f"✓ {table_name} migration complete!")


def main():
    """Main migration function."""
    print("\n" + "="*70)
    print("MIGRATING TABLES FROM CLICKHOUSE TO POSTGRESQL")
    print("="*70)

    try:
        # Connect to PostgreSQL
        print("\nConnecting to PostgreSQL...")
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()
        print("Connected successfully\n")

        # Migrate each table
        for table_name in TABLES_TO_MIGRATE:
            try:
                migrate_table(cursor, table_name)
                conn.commit()
            except Exception as e:
                print(f"\n✗ Error migrating {table_name}: {e}")
                conn.rollback()
                continue

        # Final verification
        print("\n" + "="*70)
        print("MIGRATION SUMMARY")
        print("="*70)
        for table_name in TABLES_TO_MIGRATE:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"✓ {table_name}: {count:,} rows")

        cursor.close()
        conn.close()

        print("\n✓ All migrations complete!")

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
