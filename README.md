# Autonomous Insights Engine

An LLM-powered system that automatically discovers anomalies, patterns, and insights in B2B sales data **without users needing to ask questions**. The system acts like an "autonomous analyst" - it explores the data, generates hypotheses, tests them, and surfaces what's interesting.

## Core Concept

Instead of "ask me anything" (traditional NLQ), this is **"let me tell you what you should be paying attention to"** - proactive intelligence.

The LLM (GPT-5.2 with reasoning):
1. Receives database schema + metadata + business context
2. Autonomously explores data using SQL queries via tool calling
3. Analyzes results and follows interesting leads
4. Synthesizes findings with business impact and recommended actions

## Tech Stack

- **Backend:** Python + FastAPI
- **Database:** PostgreSQL
- **LLM:** GPT-5.2 (Thinking) via OpenAI API
  - Model: `gpt-5.2`
  - Reasoning effort: `high` for deeper exploration
  - 400K context window, native tool calling
- **Architecture:** Agentic tool-calling pattern

## Project Structure

```
autonomous-insights/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Configuration
│   │   ├── db/                  # Database connection & SQL execution
│   │   ├── llm/                 # LLM client & exploration engine
│   │   ├── schemas/             # Data models & metadata
│   │   └── api/                 # API routes
│   └── requirements.txt
├── metadata/                    # Database metadata JSON files
│   ├── primary_sales.json
│   └── business_context.txt
├── scripts/
│   └── run_exploration.py       # CLI for testing
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.9+
- PostgreSQL database
- OpenAI API key (for GPT-5.2)

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/your_database

# OpenAI API
OPENAI_API_KEY=sk-...

# LLM Settings
LLM_MODEL=gpt-5.2
REASONING_EFFORT=high
MAX_EXPLORATION_ITERATIONS=15

# API
API_PORT=8000
DEBUG=False
```

### 4. Set Up Metadata

The system needs metadata to understand your database schema. Create JSON files in `metadata/` directory:

**Example: `metadata/primary_sales.json`**
```json
{
  "billing_date": {
    "type": "date",
    "unit": "YYYY-MM-DD",
    "description": "Date of billing transaction"
  },
  "customer_name": {
    "type": "varchar(100)",
    "unit": null,
    "description": "Name of the customer"
  },
  "gsv": {
    "type": "numeric",
    "unit": "₹",
    "description": "Gross Sales Value before discounts"
  }
}
```

**Business Context: `metadata/business_context.txt`**
```
This is a B2B lubricant distribution company...
[Business priorities, sales characteristics, key signals to watch]
```

## Usage

### Option 1: CLI Script (Recommended for Testing)

```bash
# Basic exploration
python scripts/run_exploration.py

# Focused exploration
python scripts/run_exploration.py --focus customer_churn

# Custom iterations
python scripts/run_exploration.py --iterations 20 --verbose
```

**Available focus areas:**
- `customer_churn` - Customer retention and churn risk analysis
- `margin_analysis` - Margin and profitability patterns
- `product_performance` - Product performance and mix analysis
- `regional_analysis` - Geographic performance comparison

### Option 2: FastAPI Server

```bash
# Start server
cd backend
python -m app.main

# Or with uvicorn
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

**API Endpoints:**

```bash
# Health check
GET /health

# Get loaded schema
GET /api/v1/schema

# Run exploration
POST /api/v1/explore
{
  "num_hypotheses": 10,
  "focus_area": "customer_churn"
}

# Reload metadata
POST /api/v1/schema
```

## How It Works

### Agentic Exploration Pattern

The system uses GPT-5.2 with tool calling for autonomous exploration:

```
1. LLM receives schema + business context
2. LLM reasons about what to explore first
3. LLM calls execute_sql tool with query
4. System executes query, returns results
5. LLM analyzes results, decides next query
6. Repeat until complete (or max iterations)
7. LLM synthesizes findings with business impact
```

### Example Exploration Flow

```
GPT-5.2: [Reasoning] Let me first understand the data landscape...
         [Tool Call] execute_sql("SELECT COUNT(*), MIN(billing_date)...")

System:  Returned 700,000 rows, date range 2024-04-01 to 2025-11-30

GPT-5.2: [Reasoning] Good dataset. Let me check for churn signals...
         [Tool Call] execute_sql("WITH customer_baseline AS (...)")

System:  Found 23 customers with >50% drop in order frequency

GPT-5.2: [Reasoning] Significant churn risk. Let me quantify revenue impact...
         [Tool Call] execute_sql("SELECT region, SUM(revenue)...")

...continues exploring autonomously...

GPT-5.2: [Final Response] KEY FINDINGS:
         1. 🔴 HIGH PRIORITY - 23 customers showing churn signals
            Business Impact: ₹45L revenue at risk
            Recommended Action: Immediate outreach to North region customers
```

## Configuration

### LLM Settings

- **Model:** `gpt-5.2` - Latest GPT-5 model with reasoning
- **Reasoning Effort:** `high` or `xhigh` for exploration
  - `none/minimal/low` - Fast, basic reasoning
  - `medium` - Balanced
  - `high/xhigh` - Deep reasoning (recommended for exploration)
- **Max Iterations:** Prevents infinite exploration loops (default: 15)

### Safety Features

- **SQL Validation:** Only SELECT queries allowed, dangerous keywords blocked
- **Query Timeout:** 30-second timeout on all queries
- **Read-only Access:** Use read-only database user
- **Result Limiting:** Automatic LIMIT added to queries

## Example Findings

The LLM might discover insights like:

1. **Churn Risk Detection**
   - "23 customers with order gaps >2x their historical average"
   - Impact: ₹45L revenue at risk
   - Action: Immediate outreach to North region customers

2. **Margin Erosion**
   - "Product XYZ margin declined 15% in Q3"
   - Impact: ₹12L margin loss
   - Action: Review pricing and discount policies

3. **Regional Divergence**
   - "North region YoY growth -8% vs company avg +12%"
   - Impact: Regional underperformance
   - Action: Investigate regional competitive issues

## Development

### Running Tests

```bash
cd backend
pytest
```

### Adding New Metadata

1. Create JSON file in `metadata/` directory (e.g., `customer_master.json`)
2. Follow the schema format (column_name -> type/unit/description)
3. Restart the server or reload schema via API

### Extending Focus Areas

Edit `backend/app/llm/prompts.py` and add new focus area instructions:

```python
FOCUS_AREA_INSTRUCTIONS = {
    "your_new_area": """
    Focus instructions for the LLM...
    """
}
```

## Cost Optimization

GPT-5.2 pricing:
- Input: $1.75/1M tokens
- Output: $14/1M tokens
- **90% discount on cached inputs** (subsequent runs cheaper)

Typical exploration costs:
- First run: $0.50 - $2.00
- Subsequent runs: $0.05 - $0.20 (with caching)

**Tips:**
- Use `reasoning_effort=medium` for cheaper runs
- Reduce `max_iterations` for faster/cheaper exploration
- Results are typically worth the cost (finding one churn risk can save thousands)

## Troubleshooting

### "No metadata loaded"
- Ensure metadata JSON files exist in `metadata/` directory
- Check JSON syntax with a validator

### "Database connection failed"
- Verify DATABASE_URL in `.env`
- Check PostgreSQL is running: `pg_isready`
- Verify database credentials

### "LLM request failed"
- Check OPENAI_API_KEY is valid
- Verify API quota/billing
- Try reducing reasoning_effort or max_iterations

### "Query timeout"
- Increase QUERY_TIMEOUT_SECONDS in `.env`
- Check for expensive queries in logs
- Optimize database indexes

## Production Considerations

For production deployment:

1. **Security**
   - Use read-only database user
   - Enable query validation
   - Rate limit API endpoints
   - Add authentication

2. **Monitoring**
   - Log all queries and results
   - Track exploration costs
   - Monitor LLM performance

3. **Optimization**
   - Cache exploration results
   - Schedule regular runs
   - Implement feedback loop (user ratings)

4. **Scaling**
   - Use Redis for result caching
   - Add background job queue
   - Deploy with Docker/K8s

## Future Enhancements

- [ ] Scheduled runs (daily/weekly exploration)
- [ ] Feedback loop (thumbs up/down improves ranking)
- [ ] Deep dive mode (investigate root causes)
- [ ] Alerting (email/Slack for high-priority findings)
- [ ] Historical comparison (this run vs last run)
- [ ] Multi-tenant support
- [ ] Frontend dashboard with insight cards
- [ ] Natural language follow-up questions

## License

MIT

## Support

For issues or questions, please open a GitHub issue.
