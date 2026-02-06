# ClickHouse to PostgreSQL Migration Guide

## Overview

This guide will help you copy your data from ClickHouse (13.126.145.174:8123) to a new PostgreSQL database.

**⚠️ IMPORTANT: Your original ClickHouse database will NOT be modified - this is READ ONLY!**

---

## Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This installs:
- `clickhouse-driver` - to read from ClickHouse
- `psycopg2-binary` - to write to PostgreSQL

---

## Step 2: Set Up PostgreSQL

You need a PostgreSQL database to migrate TO. Choose one option:

### Option A: Local PostgreSQL (Recommended for Testing)

```bash
# Install PostgreSQL (macOS)
brew install postgresql@15
brew services start postgresql@15

# Create database
createdb lubricant_sales

# Your connection URL:
# postgresql://YOUR_USERNAME@localhost:5432/lubricant_sales
```

### Option B: Cloud PostgreSQL (Recommended for Production)

Popular options:
- **Supabase** (free tier): https://supabase.com
- **Neon** (free tier): https://neon.tech
- **AWS RDS**, **Google Cloud SQL**, etc.

Get your connection URL, looks like:
```
postgresql://user:password@host:5432/database
```

### Option C: Docker PostgreSQL (Quick Setup)

```bash
cd backend
docker-compose up -d db

# Your connection URL:
# postgresql://insights:insights123@localhost:5432/lubricant_sales
```

---

## Step 3: Configure Environment

Add PostgreSQL URL to your `.env`:

```bash
# Edit .env
DATABASE_URL=postgresql://user:password@localhost:5432/lubricant_sales
```

---

## Step 4: Test Connections

```bash
# Test both ClickHouse and PostgreSQL connections
python scripts/migrate_from_clickhouse.py --test
```

Expected output:
```
✓ ClickHouse connection successful
✓ PostgreSQL connection successful

✓ Connection test successful!
```

---

## Step 5: Inspect Available Tables

See what tables are in your ClickHouse database:

```bash
python scripts/migrate_from_clickhouse.py
```

This will list all available tables.

To inspect a specific table:

```bash
python scripts/migrate_from_clickhouse.py --inspect --tables customer_master
```

Output:
```
Table: customer_master
Rows: 1,234
Columns:
  - customer_code: String
  - customer_name: String
  - state: String
  - region: String
  ...

✓ Metadata saved to: metadata/inspect_customer_master.json
```

---

## Step 6: Migrate Data

### Migrate One Table (Start Here)

```bash
# Migrate just customer_master first
python scripts/migrate_from_clickhouse.py --tables customer_master
```

### Migrate Multiple Tables

```bash
# Migrate customer and material master
python scripts/migrate_from_clickhouse.py --tables customer_master material_master
```

### Migrate All Tables

```bash
# Migrate everything
python scripts/migrate_from_clickhouse.py --all
```

---

## What Happens During Migration

```
1. Connects to ClickHouse (READ ONLY)
2. Reads table structure and data
3. Creates matching PostgreSQL table
4. Copies data in batches (10,000 rows at a time)
5. Creates indexes for performance
6. Shows progress as it runs
```

Example output:
```
============================================================
Migrating table: customer_master
============================================================
Source table has 1,234 rows
Columns: 15
Creating PostgreSQL table...
✓ Table created in PostgreSQL
Migrated 10,000 / 1,234,567 rows (0.8%)
Migrated 20,000 / 1,234,567 rows (1.6%)
...
✓ Migration complete: 1,234,567 rows migrated
Creating indexes...
✓ Created 5 indexes
```

---

## Step 7: Verify Migration

Check your PostgreSQL database:

```bash
psql $DATABASE_URL

# List tables
\dt

# Count rows
SELECT COUNT(*) FROM customer_master;

# Sample data
SELECT * FROM customer_master LIMIT 5;
```

---

## Common Issues & Solutions

### Error: "clickhouse-driver not installed"
```bash
pip install clickhouse-driver
```

### Error: "PostgreSQL connection failed"
- Check DATABASE_URL in .env
- Verify PostgreSQL is running: `pg_isready`
- Test connection: `psql $DATABASE_URL -c "SELECT 1"`

### Error: "ClickHouse connection failed"
- Check network connectivity to 13.126.145.174
- Verify port 8123 is accessible
- Check username/password

### Migration is slow
- Normal for large tables (millions of rows)
- Increase batch size: `--batch-size 50000`
- Check network speed between servers

### Table already exists
- The script uses `CREATE TABLE IF NOT EXISTS`
- It will skip creation but still insert data
- To start fresh, drop the table first:
  ```sql
  DROP TABLE customer_master;
  ```

---

## Advanced Options

```bash
# Custom ClickHouse connection
python scripts/migrate_from_clickhouse.py \
  --ch-host 13.126.145.174 \
  --ch-port 8123 \
  --ch-database veedol_sales \
  --ch-user default \
  --ch-password "" \
  --tables customer_master

# Custom batch size
python scripts/migrate_from_clickhouse.py \
  --tables sales_data \
  --batch-size 50000

# Custom PostgreSQL URL
python scripts/migrate_from_clickhouse.py \
  --pg-url postgresql://user:pass@host/db \
  --tables customer_master
```

---

## Next Steps After Migration

1. **Generate Metadata**
   ```bash
   # Create metadata JSON files for LLM
   python scripts/generate_metadata.py
   ```

2. **Test Exploration**
   ```bash
   # Run autonomous exploration
   python scripts/run_exploration.py --verbose
   ```

3. **Start API**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

---

## Safety Notes

✅ **Safe Operations:**
- Script only reads from ClickHouse
- No DELETE, UPDATE, or DROP on source
- Can run multiple times safely
- Original data unchanged

⚠️ **Be Aware:**
- Migration creates NEW tables in PostgreSQL
- If table exists, data is appended (may create duplicates)
- Large tables take time (millions of rows = hours)

---

## Estimated Migration Times

Based on network speed and data volume:

| Rows        | Time (approx) |
|-------------|---------------|
| 1,000       | < 1 minute    |
| 10,000      | 1-2 minutes   |
| 100,000     | 5-10 minutes  |
| 1,000,000   | 30-60 minutes |
| 10,000,000  | 3-6 hours     |

*Times vary based on network speed and server performance*

---

## Questions?

- Check logs for detailed error messages
- Use `--help` flag for all options
- Refer to README.md for general documentation
