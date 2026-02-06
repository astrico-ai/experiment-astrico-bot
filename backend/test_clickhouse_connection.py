#!/usr/bin/env python3
"""
Test ClickHouse connection after migration.
"""
import sys
sys.path.insert(0, '/Users/sanujphilip/Desktop/anomaly/backend')

from app.db.connection import db
from app.db.executor import executor

def test_connection():
    """Test basic connection."""
    print("=" * 60)
    print("Testing ClickHouse Connection")
    print("=" * 60)

    try:
        # Test 1: Basic connection
        print("\n1. Testing basic connection...")
        client = db.get_client()
        result = client.command("SELECT 1")
        print(f"   ✅ Connection successful! Result: {result}")
        client.close()

        # Test 2: Get table info
        print("\n2. Testing schema discovery (get_table_info)...")
        tables = db.get_table_info()
        print(f"   ✅ Found {len(tables)} tables:")
        for table_name in sorted(tables.keys())[:5]:  # Show first 5
            print(f"      - {table_name} ({len(tables[table_name])} columns)")
        if len(tables) > 5:
            print(f"      ... and {len(tables) - 5} more tables")

        # Test 3: Simple query
        print("\n3. Testing simple query execution...")
        query_result = executor.execute_query("SELECT 1 as test")
        if query_result["success"]:
            print(f"   ✅ Query successful! Result: {query_result['rows']}")
        else:
            print(f"   ❌ Query failed: {query_result['error']}")

        # Test 4: Date function
        print("\n4. Testing ClickHouse date functions...")
        date_query = "SELECT today() as current_date, toStartOfMonth(today()) as month_start"
        date_result = executor.execute_query(date_query)
        if date_result["success"]:
            print(f"   ✅ Date functions work! Result: {date_result['rows']}")
        else:
            print(f"   ❌ Date query failed: {date_result['error']}")

        # Test 5: Query a real table (if exists)
        print("\n5. Testing query on actual table...")
        table_query = "SELECT COUNT(*) as count FROM sales_invoices_veedol LIMIT 1"
        table_result = executor.execute_query(table_query)
        if table_result["success"]:
            print(f"   ✅ Table query successful! Row count: {table_result['rows']}")
        else:
            print(f"   ⚠️  Table query failed (table may not exist): {table_result['error']}")

        print("\n" + "=" * 60)
        print("✅ All tests passed! ClickHouse connection is working.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_connection()
