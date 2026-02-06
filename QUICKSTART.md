# Quick Start Guide

Get the Autonomous Insights Engine running in 5 minutes.

## Prerequisites

- Python 3.9+
- PostgreSQL database with your data
- OpenAI API key

## Installation

```bash
# 1. Clone/navigate to project
cd autonomous-insights

# 2. Install dependencies
cd backend
pip install -r requirements.txt

# 3. Set up environment
cp ../.env.example .env
# Edit .env with your database URL and OpenAI API key

# 4. Add your metadata
# Create JSON files in metadata/ directory describing your tables
# See metadata/primary_sales.json as an example

# 5. Run exploration!
python ../scripts/run_exploration.py
```

## Minimum Configuration

Your `.env` file needs just these two lines to start:

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/your_db
OPENAI_API_KEY=sk-your-key-here
```

## First Run

```bash
# Test with verbose output
python scripts/run_exploration.py --verbose

# Focus on a specific area
python scripts/run_exploration.py --focus customer_churn

# Limit iterations for faster testing
python scripts/run_exploration.py --iterations 5
```

## Expected Output

```
================================================================================
                    AUTONOMOUS INSIGHTS ENGINE
               LLM-Powered Data Exploration
================================================================================
Model: gpt-5.2
Reasoning Effort: high
================================================================================

📂 Loading metadata...
✓ Loaded metadata for 1 tables: primary_sales

🤖 Initializing explorer...
✓ Explorer ready

🔍 Starting autonomous exploration...

================================================================================
EXPLORATION COMPLETE - Run ID: abc-123
================================================================================
Iterations: 8
Queries Executed: 12 (✓ 11 / ✗ 1)
Duration: 45.3s

📊 DISCOVERED 3 INTERESTING FINDINGS
================================================================================

1. 🔴 HIGH PRIORITY - Confidence: HIGH
--------------------------------------------------------------------------------
Finding: 23 customers showing strong churn signals with order frequency
dropping >50% in the last 3 months compared to historical baseline

💰 Business Impact: ₹45.2L revenue at risk based on trailing 12-month spend

✅ Recommended Action: Immediate outreach to affected customers, prioritizing
North region where 15 of 23 at-risk customers are located

...
```

## Troubleshooting

### Import Error
```bash
# Make sure you're in the backend directory when running
cd backend
python -m app.main
```

### Database Connection Failed
```bash
# Test your database connection
psql $DATABASE_URL -c "SELECT 1"
```

### API Key Error
```bash
# Verify your OpenAI API key
echo $OPENAI_API_KEY
```

## Next Steps

1. **Add More Metadata**: Expand your metadata JSON files with more tables and columns
2. **Try Different Focus Areas**: customer_churn, margin_analysis, product_performance
3. **Run the API Server**: `uvicorn app.main:app --reload` and visit `http://localhost:8000/docs`
4. **Customize Prompts**: Edit `backend/app/llm/prompts.py` for your domain
5. **Schedule Regular Runs**: Set up cron job or scheduler for daily insights

## Common Use Cases

### Daily Churn Monitoring
```bash
# Add to crontab for daily 8am run
0 8 * * * cd /path/to/project && python scripts/run_exploration.py --focus customer_churn >> logs/daily_churn.log 2>&1
```

### Weekly Business Review
```bash
# Comprehensive weekly exploration
python scripts/run_exploration.py --iterations 25 > reports/weekly_$(date +%Y%m%d).txt
```

### API Integration
```python
import requests

response = requests.post("http://localhost:8000/api/v1/explore", json={
    "focus_area": "margin_analysis",
    "num_hypotheses": 15
})

findings = response.json()
```

## Performance Tips

- **First run is slower** (~2 min) - subsequent runs are faster due to LLM caching
- **Use focus areas** to get targeted insights faster
- **Lower iterations** (5-10) for quick checks, higher (15-25) for deep dives
- **Cost:** $0.50-$2 per run initially, $0.05-$0.20 after caching kicks in

## Getting Help

- Check logs in the console output
- Use `--verbose` flag for detailed logging
- Review README.md for full documentation
- Check `/docs` endpoint when API is running

Happy exploring! 🚀
