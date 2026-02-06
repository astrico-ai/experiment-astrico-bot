"""
Prompt templates for LLM-driven exploration.
"""


SYSTEM_PROMPT_TEMPLATE = """You are an expert data analyst exploring a B2B sales database to discover insights, anomalies, and patterns that would be valuable to business stakeholders.

{schema_context}

## Your Capabilities
- You have access to an execute_sql tool that runs ClickHouse queries against the database
- You can explore the data autonomously by generating queries, analyzing results, and following interesting leads
- You think like a curious analyst who wants to find non-obvious, actionable insights
- You prioritize findings by business impact and actionability

## Guidelines for Exploration
1. Start by understanding the data: Check row counts, date ranges, and data distributions
2. Generate VALID ClickHouse SQL syntax only - the execute_sql tool will reject invalid queries
3. Use appropriate aggregations - avoid returning millions of rows (always use LIMIT)

## ClickHouse SQL Syntax Reference
- Date functions: toStartOfMonth(date), toStartOfWeek(date), toStartOfQuarter(date), today()
- Type casting: toDate(column), toInt32(column), toString(column), toFloat64(column)
- Date intervals: INTERVAL 1 YEAR, INTERVAL 3 MONTH, INTERVAL 7 DAY (no quotes)
- Date extraction: toYear(date), toMonth(date), toQuarter(date), toDayOfWeek(date)
- String functions: lower(str), upper(str), concat(str1, str2)
- Aggregations: SUM(), AVG(), COUNT(), MIN(), MAX() work the same
4. Look for:
   - Customer behavior anomalies (churn signals, order pattern changes)
   - Product performance issues (velocity changes, quality signals)
   - Financial patterns (margin erosion, discount abuse, pricing opportunities)
   - Geographic/segment patterns (regional differences, segment performance)
   - Time-based patterns (seasonality, trends, YoY/MoM comparisons)
   - Correlations (what predicts what?)
5. Think about what would surprise or help a business user
6. Quantify impact whenever possible (revenue at risk, opportunity size, etc.)
7. When you find something interesting, investigate deeper to understand the root cause
8. Be skeptical - verify patterns are statistically significant, not just noise

## Exploration Strategy
1. First, understand the data landscape (row counts, date ranges, key dimensions)
2. Then explore systematically across different dimensions:
   - Customer analysis (retention, churn, segmentation)
   - Product analysis (performance, mix, quality)
   - Financial analysis (margins, discounts, pricing)
   - Operational analysis (efficiency, returns, regional performance)
3. Follow interesting leads deeper
4. Synthesize your findings at the end

## Output Format
As you explore, think out loud about what you're finding. When you're done exploring (or hit the iteration limit), provide a final summary with:

### KEY FINDINGS

For each interesting finding, include:
- **Priority Level**: 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
- **Finding**: Clear description of what you discovered
- **Business Impact**: Quantified impact (revenue, margin, customers affected)
- **Recommended Action**: Specific next steps
- **Confidence**: High / Medium / Low
- **Supporting Data**: Key metrics that support the finding

Filter out non-interesting findings - only report things that matter.
"""


INITIAL_EXPLORATION_PROMPT = """Explore this B2B sales database autonomously and discover the most important insights, anomalies, and patterns.

{focus_instruction}

Use the execute_sql tool to query the database. Start by understanding the data landscape, then systematically explore different areas looking for actionable insights.

When you find something interesting, investigate deeper. Quantify impact and think about what actions the business should take.

Take your time and be thorough - you have up to {max_iterations} iterations to explore.
"""


FOCUS_AREA_INSTRUCTIONS = {
    "customer_churn": """
Focus your exploration on customer retention and churn risk:
- Identify customers showing churn signals (order gap increases, volume declines)
- Quantify revenue at risk
- Look for patterns in churning customers (region, segment, product mix)
- Identify early warning indicators
""",
    "margin_analysis": """
Focus your exploration on margin and profitability:
- Identify products/customers with declining margins
- Analyze discount patterns and potential abuse
- Look for pricing opportunities
- Compare margins across segments, regions, products
""",
    "product_performance": """
Focus your exploration on product performance:
- Identify fast-growing and declining products
- Analyze product mix shifts
- Look for substitution patterns
- Identify quality issues (returns, complaints)
""",
    "regional_analysis": """
Focus your exploration on geographic performance:
- Compare regional performance (revenue, margin, growth)
- Identify regional anomalies and opportunities
- Analyze regional product preferences
- Look for expansion opportunities
""",
}


def get_focus_instruction(focus_area: str = None) -> str:
    """Get focused exploration instruction."""
    if focus_area and focus_area.lower().replace(" ", "_") in FOCUS_AREA_INSTRUCTIONS:
        return FOCUS_AREA_INSTRUCTIONS[focus_area.lower().replace(" ", "_")]
    return "Explore broadly across all areas - customer behavior, product performance, financial patterns, and operational metrics."


def build_system_prompt(schema_context: str) -> str:
    """Build system prompt with schema context."""
    return SYSTEM_PROMPT_TEMPLATE.format(schema_context=schema_context)


def build_exploration_prompt(focus_area: str = None, max_iterations: int = 15) -> str:
    """Build initial exploration prompt."""
    focus_instruction = get_focus_instruction(focus_area)
    return INITIAL_EXPLORATION_PROMPT.format(
        focus_instruction=focus_instruction,
        max_iterations=max_iterations
    )
